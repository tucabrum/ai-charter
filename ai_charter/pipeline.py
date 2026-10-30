"""Pipeline completo: áudio -> stems -> notas -> charts -> pasta pronta para o jogo."""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

from ai_charter import stem_sources
from ai_charter.analysis import adtof
from ai_charter.analysis.drums import HOP as DRUM_HOP
from ai_charter.analysis.drums import ONSET_LATENCY as DRUM_ONSET_LATENCY
from ai_charter.analysis.drums import FEATURE_NAMES as DRUM_FEATURES
from ai_charter.analysis.drums import TOM as DRUM_TOM
from ai_charter.analysis.drums import TOM_LOW as DRUM_TOM_LOW
from ai_charter.analysis.drums import TOM_MID as DRUM_TOM_MID
from ai_charter.analysis.drums import TOMS as DRUM_TOMS
from ai_charter.analysis.drums import analyze_kit, detect_drum_hits, ring_features
from ai_charter.analysis.pitched import detect_pitched_notes
from ai_charter.analysis.sections import find_preview_start, find_sections
from ai_charter.analysis.tempo import TempoMap, estimate_tempo_map
from ai_charter.analysis.transcribe import transcribe_lyrics
from ai_charter.analysis.vocals import detect_sung_notes, detect_voice_segments
from ai_charter.charting.common import (
    Candidate, TrackChart, dedupe, quantize_runs, quantize_tick, star_power_for_phrases, star_power_windows,
)
from ai_charter.charting.chords import label_chords
from ai_charter.charting.drum_reduce import bpm_from_tempo
from ai_charter.charting.drum_repeats import consolidate_repeats
from ai_charter.charting.drums import build_drum_track
from ai_charter.charting.frets import build_fret_track
from ai_charter.charting.guitar_select import GuitarSource, merge_guitars
from ai_charter.charting.lyrics import read_lyrics
from ai_charter.charting.sanitize import sanitize
from ai_charter.charting.vocals import VocalTrack, build_aligned_track, build_vocal_track
from ai_charter.config import Rules, get_genre, load_rules
from ai_charter.drum_model import MODEL_PATH as BASIC_DRUM_MODEL
from ai_charter.drum_model import OUTPUTS as MODEL_OUTPUTS
from ai_charter.drum_model import (CYM_LANES, TOM_LANES, CymLaneModel, DrumModel, TomLaneModel, cym_lane_inputs,
                                   tom_lane_inputs)
from ai_charter.export import audio, lead_in
from ai_charter.export.confidence import NAME as CONFIDENCE_NAME
from ai_charter.export.confidence import write_confidence
from ai_charter.export.chart_writer import SongInfo, write_chart
from ai_charter.export.midi_writer import write_midi
from ai_charter.export.song_ini import write_song_ini
from ai_charter.separation import separate

ANALYSIS_SR = 22050   # taxa usada para instrumentos com altura e para o tempo
DRUMS_SR = 44100      # bateria precisa dos agudos (pratos)

# Arquivo de áudio que o Clone Hero/YARG espera para cada stem.
STEM_FILES = {
    "other": ("song.ogg", "MusicStream"),
    "guitar": ("guitar.ogg", "GuitarStream"),
    "bass": ("bass.ogg", "BassStream"),
    "drums": ("drums.ogg", "DrumStream"),
    "keys": ("keys.ogg", "KeysStream"),
    "vocals": ("vocals.ogg", "VocalStream"),
}
ALL_INSTRUMENTS = ("guitar", "bass", "drums", "keys")


@dataclass
class Options:
    input: Path
    output_dir: Path
    separator: str = "demucs"
    stems_dir: Path | None = None
    rules_path: Path | None = None
    genre: str | None = None       # predefinição de regras (ai_charter/genres/<gênero>.toml)
    reference: Path | None = None  # partitura (MIDI ou Guitar Pro) que guia a transcrição (reference.py)
    reference_tracks: dict[int, str] | None = None   # pista da partitura -> instrumento (None = palpite)
    reference_fidelity: str | dict[str, str] = "seguir"   # corrigir, completar ou seguir (ou um por instrumento)
    reference_sections: list[tuple[int, int, int]] | None = None   # pista escolhida à mão por trecho
    reference_anchors: list[tuple[int, float]] | None = None   # compasso da partitura -> segundos, à mão
    cifra: Path | None = None      # cifra (Cifra Club): letra e vocabulário de acordes (cifra.py)
    cifra_lyrics: bool = True      # sem outra letra, os vocais usam a letra da cifra
    lead_in: float = 3.0           # segundos mínimos antes da 1ª nota (compassos de silêncio; 0 = sem)
    instruments: tuple[str, ...] = ALL_INSTRUMENTS
    stem_sources: dict[str, str] | None = None   # instrumento -> stem de onde ele sai (stem_sources.py)
    tempo_mode: str = "auto"
    title: str | None = None
    artist: str | None = None
    lyrics: Path | None = None     # letra (texto, uma linha por frase); sem ela, não há vocal
    auto_lyrics: bool = False      # sem letra: transcreve com o Whisper (rascunho)
    output_format: str = "mid"     # "mid" (com vocais) ou "chart"
    make_zip: bool = False
    keep_work: bool = False


@dataclass
class Result:
    song_dir: Path
    tracks: dict[str, TrackChart] = field(default_factory=dict)
    vocals: VocalTrack | None = None
    guitar_sources: list[str] = field(default_factory=list)  # guitarra escolhida em cada compasso
    bpm: float = 0.0
    zip_path: Path | None = None
    lead_in: float = 0.0   # segundos de silêncio postos antes da música (export/lead_in.py)


def _safe_name(text: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", text).strip() or "song"


def _stem_energy(paths: list[Path], mix_rms: float) -> float:
    """Volume do stem (o mais alto, se houver vários arquivos) relativo à mix."""
    levels = []
    for path in paths:
        y, _ = librosa.load(path, sr=ANALYSIS_SR, mono=True)
        levels.append(np.sqrt(np.mean(y ** 2)))
    return float(max(levels) / (mix_rms + 1e-12))


def _describe_runs(names: list[str]) -> str:
    """["a", "a", "b"] -> "a (compassos 1-2), b (compasso 3)"."""
    parts, start = [], 0
    for i in range(1, len(names) + 1):
        if i == len(names) or names[i] != names[start]:
            span = f"compasso {start + 1}" if i - start == 1 else f"compassos {start + 1}-{i}"
            parts.append(f"{names[start]} ({span})")
            start = i
    return ", ".join(parts)


def _to_candidates(detected, tempo: TempoMap, rules: Rules) -> list[Candidate]:
    res, triplets = rules.general.resolution, rules.general.allow_triplets
    cands = [
        Candidate(
            tick=quantize_tick(tempo.time_to_tick(n.time), res, triplets),
            strength=n.strength,
            end_tick=int(tempo.time_to_tick(n.end)),
            pitch=n.pitch,
            voices=n.voices,
            chord_scores=n.chord_scores,
            lowest_by_class=n.lowest_by_class,
            fifth_ratio=n.fifth_ratio,
            third_ratio=n.third_ratio,
        )
        for n in detected
    ]
    return dedupe(cands)


def estimate_tuning(mix: np.ndarray, sr: int) -> float:
    """Afinação da gravação, em frações de semitom (-0.5 a 0.5) em relação ao Lá 440 Hz.

    Muitas bandas gravam um pouco fora do padrão (ex.: -0.4 = 40 centésimos abaixo).
    É medida na mixagem inteira: os instrumentos concordam entre si, e a voz,
    que desafina mais, não serve de régua sozinha.
    """
    return float(librosa.estimate_tuning(y=mix, sr=sr))


def chart_pitched(stems: list[Path], instrument: str, tempo: TempoMap, rules: Rules,
                  sp_windows, log=print, tuning: float = 0.0, guide=None) -> tuple[TrackChart, list[str]]:
    """Chart de guitarra/baixo/teclado. Com várias guitarras, escolhe uma por compasso.

    `guide` (opcional, ver reference.py) corrige altura e acordes das notas detectadas com uma partitura."""
    cfg = rules.detection[instrument]
    sources = []
    for stem in stems:
        y, sr = librosa.load(stem, sr=ANALYSIS_SR, mono=True)
        cands = _to_candidates(detect_pitched_notes(y, sr, cfg, tuning), tempo, rules)
        if cfg.recognize_chords:
            label_chords(cands, rules.general.resolution, cfg)
        cands, simplified = sanitize(cands, y, sr, tempo, rules.general.resolution, rules.sanitize)
        if simplified:
            log(f"    {instrument} ({stem.stem}): {len(simplified)} compasso(s) confuso(s) simplificado(s): "
                + ", ".join(str(m + 1) for m in simplified))
        if guide is not None:
            cands = guide(cands, stem.stem if len(stems) > 1 else "", log)
        sources.append(GuitarSource(stem.stem, cands, y, sr))
    cands, chosen = merge_guitars(sources, tempo, rules.general.resolution, rules.guitar_selection)
    track = build_fret_track(cands, rules.frets, rules.general.resolution, cfg.lane_window, sp_windows)
    return track, chosen


def chart_vocals(stem: Path, lyrics: Path, tempo: TempoMap, rules: Rules, sp_windows,
                 tuning: float = 0.0) -> VocalTrack:
    y, sr = librosa.load(stem, sr=ANALYSIS_SR, mono=True)
    sung = detect_sung_notes(y, sr, rules.vocals, tuning)
    lines = read_lyrics(lyrics, rules.vocals.language)
    def listen():   # só chamado numa música mal ouvida: o Whisper diz onde cada frase é cantada
        try:
            from ai_charter.analysis.transcribe import anchor_words

            print("    vocals: letra mal ouvida; conferindo com o Whisper...")
            return anchor_words(stem, rules.vocals.whisper_model)
        except Exception as err:
            print(f"    vocals: sem o Whisper para conferir a letra ({str(err).splitlines()[0]})")
            return None

    try:   # alinhamento forçado da letra (precisa do torchaudio e do modelo MMS_FA)
        from ai_charter.analysis.align import align_syllables

        spans = align_syllables(y, sr, lines, listen)
    except Exception as err:
        print(f"    vocals: alinhamento forçado indisponível ({err}); usando o alinhador antigo")
        segments = detect_voice_segments(y, sr, rules.vocals, rules.vocals.phrase_gap_beats * 60 / tempo.average_bpm)
        vocals = build_vocal_track(sung, segments, lines, tempo, rules.general.resolution,
                                   rules.general.allow_triplets, rules.vocals)
    else:
        vocals = build_aligned_track(spans, sung, lines, tempo, rules.general.resolution, rules.vocals)
    vocals.star_power = star_power_for_phrases(vocals.phrases, sp_windows)
    return vocals


def continues_pattern(tick: int, ticks: set[int], resolution: int) -> bool:
    """Se o golpe continua o ritmo que a peça está tocando ali: há golpes firmes dela no intervalo
    mais comum entre os seus golpes vizinhos, antes e depois (ou dois antes, ou dois depois).

    Ex.: um surdo de 6 em 6 semicolcheias que some debaixo do bumbo volta; já um tom no meio de
    toms de 4 em 4 semicolcheias não entra (seria outro ritmo)."""
    near = sorted(t for t in ticks if abs(t - tick) <= 8 * resolution)
    gaps = [b - a for a, b in zip(near, near[1:]) if b - a <= 2 * resolution]
    if not gaps:
        return False
    step = max(set(gaps), key=gaps.count)
    before, after = tick - step in ticks, tick + step in ticks
    return (before and after) or (before and tick - 2 * step in ticks) or (after and tick + 2 * step in ticks)


def model_drum_candidates(y: np.ndarray, sr: int, tempo: TempoMap, rules: Rules, model,
                          adtof_act: np.ndarray | None = None, doubt: dict | None = None) -> list[Candidate]:
    """Bateria pelo modelo treinado com charts oficiais: retrato acústico de cada ataque + lugar na
    levada -> probabilidade de cada peça; entra o que passa do limiar daquela peça. Para "tom", qual
    tom é vem dos sons do kit da música (o de maior presença no ataque), pela altura de cada um."""
    res, triplets = tempo.resolution, rules.general.allow_triplets   # a grade do mapa de tempo (480 nas charts do Rock Band)
    uses_adtof = getattr(model, "uses_adtof", False)
    # Com o ADTOF, os golpes que ele ouve e o detector não marcou (ex.: chimbal junto do bumbo) entram também
    extra = adtof.hit_times(adtof_act) if uses_adtof else None
    kit = analyze_kit(y, sr, rules.drum_detection, extra)
    if not len(kit.onsets):
        return []
    ticks = np.asarray(tempo.time_to_tick(kit.onsets * DRUM_HOP / sr - DRUM_ONSET_LATENCY), dtype=float)
    adtof_feats = None
    if uses_adtof:
        adtof_feats = adtof.at_times(adtof_act, kit.onsets * DRUM_HOP / sr - DRUM_ONSET_LATENCY)
    lane_model = TomLaneModel.load() if uses_adtof else None
    cym_model = CymLaneModel.load() if uses_adtof else None
    # Medidas de ressonância (altura do tom, sustentação dos pratos), só se algum modelo usa
    ring = (ring_features(y, sr, kit.onsets)
            if getattr(model, "uses_ring", False) or getattr(lane_model, "uses_ring", False)
            or cym_model is not None else None)
    probs = model.predict_song(kit.features, ticks, res, adtof_feats, ring)
    trust = rules.drum_detection.adtof_kick_trust
    if uses_adtof and trust > 0:   # pedal duplo rápido: o modelo duvida, o ADTOF ouve
        k = MODEL_OUTPUTS.index("kick")
        times = kit.onsets * DRUM_HOP / sr
        confirmed = times[probs[:, k] >= model.thresholds[k]]
        # Só colado (< 100 ms) a um bumbo que o modelo já confirmou: uma rajada, não um bumbo solto
        near = np.array([len(confirmed) > 0 and np.min(np.abs(confirmed - t)) < 0.1 for t in times], dtype=bool)
        sure = (adtof_feats[:, adtof.CLASSES.index("bumbo")] >= trust) & near
        probs[sure, k] = np.maximum(probs[sure, k], model.thresholds[k])
    # A cor de cada tom: aprendida das charts oficiais (TomLaneModel), se houver; senão, pela altura
    lane_probs = (lane_model.predict(tom_lane_inputs(kit.features, ticks, res, adtof_feats,
                                                     ring if lane_model.uses_ring else None))
                  if lane_model is not None else None)
    # A cor de cada prato grande (azul = condução, verde = ataque): aprendida das charts (CymLaneModel)
    cym_probs = (cym_model.predict(cym_lane_inputs(kit.features, ticks, res, adtof_feats, ring))
                 if cym_model is not None else None)
    big = [MODEL_OUTPUTS.index(p) for p in CYM_LANES]
    seen: dict[tuple[int, str], Candidate] = {}
    hit = np.flatnonzero((probs >= model.thresholds).any(axis=1))   # ataques que viram nota
    quantized = dict(zip(hit.tolist(), quantize_runs(ticks[hit], res, triplets)))
    if doubt is not None:   # dúvida de cada golpe: alguma peça com chance perto do limiar de decisão
        near = np.exp(-(((probs - model.thresholds) / 0.1) ** 2)).max(axis=1)
        for j in np.flatnonzero(near >= 0.05):
            tick = quantized.get(int(j), quantize_tick(ticks[j], res, triplets))
            doubt[tick] = max(doubt.get(tick, 0.0), float(near[j]))
    for j in hit:
        tick = quantized[int(j)]
        for k, piece in enumerate(MODEL_OUTPUTS):
            if probs[j, k] < model.thresholds[k]:
                continue
            if cym_probs is not None and k in big:   # um prato grande só, na cor que o modelo de pratos escolhe
                if k != next(b for b in big if probs[j, b] >= model.thresholds[b]):
                    continue
                piece = CYM_LANES[int(cym_probs[j].argmax())]
            if piece == DRUM_TOM:
                piece = (TOM_LANES[int(lane_probs[j].argmax())] if lane_probs is not None
                         else which_tom(kit, j, rules.drum_detection.floor_tom_max_hz))
            key = (tick, piece)
            if key not in seen or probs[j, k] > seen[key].strength:
                seen[key] = Candidate(tick, float(probs[j, k]), piece=piece)
    cands = sorted(seen.values(), key=lambda c: c.tick)
    if rules.drum_detection.repeat_similarity > 0:   # trechos que se repetem saem iguais
        cands = consolidate_repeats(cands, tempo.bar_starts() * res, res, rules.drum_detection.repeat_similarity)
    return cands


def which_tom(kit, j: int, floor_tom_max_hz: float) -> str:
    """O tom do kit que mais aparece no ataque j; sem nenhum, pela altura do próprio golpe."""
    best, piece = 0.0, None
    for k, sound in enumerate(kit.sounds):
        tom = sound.piece if sound.piece in DRUM_TOMS else sound.tom_as if sound.tom_as in DRUM_TOMS else None
        score = kit.explained[k][j] * min(kit.strength[k][j], 1.0)
        if tom and score > best:
            best, piece = score, tom
    if piece:
        return piece
    pitch = 60.0 * 2.0 ** kit.features[j, DRUM_FEATURES.index("altura_graves")]
    return DRUM_TOM_LOW if pitch < floor_tom_max_hz else DRUM_TOM_MID


def chart_drums(stem: Path, tempo: TempoMap, rules: Rules, sp_windows, mix: Path | None = None,
                adtof_act: np.ndarray | None = None) -> TrackChart:
    """A pista de bateria a partir do stem. `mix` (a música inteira) ou `adtof_act` (as saídas do
    ADTOF já calculadas) alimentam o modelo com ADTOF; sem eles, vale o modelo básico."""
    y, sr = librosa.load(stem, sr=DRUMS_SR, mono=True)
    model = DrumModel.load() if rules.drum_detection.use_model else None
    if model is not None and getattr(model, "uses_adtof", False) and adtof_act is None:
        if mix is not None:
            adtof_act = adtof.activations(librosa.load(mix, sr=adtof.SR, mono=False)[0], adtof.SR)
        if adtof_act is None:
            model = DrumModel.load(BASIC_DRUM_MODEL)
    if model is not None:
        doubt: dict[int, float] = {}
        track = build_drum_track(model_drum_candidates(y, sr, tempo, rules, model, adtof_act, doubt), rules.drums,
                                 tempo.resolution, sp_windows, rules.general.drum_fill_every_measures,
                                 bpm_at=bpm_from_tempo(tempo))
        track.doubt = doubt
        return track
    hits = detect_drum_hits(y, sr, rules.drum_detection)
    res, triplets = tempo.resolution, rules.general.allow_triplets
    strong: dict[tuple[int, str], Candidate] = {}
    weak: dict[tuple[int, str], Candidate] = {}
    for h in hits:
        tick = quantize_tick(tempo.time_to_tick(h.time), res, triplets)
        seen, key = (weak if h.weak else strong), (tick, h.piece)
        if key not in seen or h.strength > seen[key].strength:
            seen[key] = Candidate(tick, h.strength, piece=h.piece)
    # Golpe fraco (ex.: surdo mascarado pelo bumbo) entra só se continuar um padrão da mesma peça
    by_piece: dict[str, set[int]] = {}
    for tick, piece in strong:
        by_piece.setdefault(piece, set()).add(tick)
    for (tick, piece), c in weak.items():
        if (tick, piece) not in strong and continues_pattern(tick, by_piece.get(piece, set()), res):
            strong[(tick, piece)] = c
    cands = sorted(strong.values(), key=lambda c: c.tick)
    return build_drum_track(cands, rules.drums, res, sp_windows,
                            rules.general.drum_fill_every_measures, bpm_at=bpm_from_tempo(tempo))


def run(opts: Options, log=print) -> Result:
    audio.require_ffmpeg()
    rules = load_rules(opts.rules_path, opts.genre)
    res = rules.general.resolution
    tempo_mode = opts.tempo_mode
    if opts.genre:
        genre = get_genre(opts.genre)
        log(f"Regras do gênero: {genre.name}")
        if tempo_mode == "auto" and genre.tempo:
            tempo_mode = genre.tempo

    tags = audio.read_tags(opts.input)
    title = opts.title or tags.get("title") or opts.input.stem
    artist = opts.artist or tags.get("artist") or "Unknown Artist"
    song_dir = opts.output_dir / _safe_name(f"{artist} - {title}")
    song_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(tempfile.mkdtemp(prefix="ai-charter-"))
    try:
        log("1/5 Convertendo o áudio...")
        mix_wav = work_dir / "mix.wav"
        audio.to_wav(opts.input, mix_wav)
        mix, _ = librosa.load(mix_wav, sr=ANALYSIS_SR, mono=True)
        duration = len(mix) / ANALYSIS_SR
        mix_rms = float(np.sqrt(np.mean(mix ** 2)))

        log("2/5 Detectando o andamento (BPM)...")
        tempo = estimate_tempo_map(mix, ANALYSIS_SR, res, tempo_mode)
        log(f"    BPM médio: {tempo.average_bpm:.1f}")
        tuning = estimate_tuning(mix, ANALYSIS_SR)
        if abs(tuning) >= 0.1:
            log(f"    afinação da gravação: {tuning * 100:+.0f} centésimos de semitom (compensada)")

        log(f"3/5 Separando os stems (modo: {opts.separator})...")
        stems = separate(opts.separator, mix_wav, work_dir, opts.stems_dir)

        cifra = None
        if opts.cifra is not None:
            from dataclasses import replace

            from ai_charter.cifra import prepare
            cifra, vocabulary = prepare(opts.cifra, mix, ANALYSIS_SR, log)
            if vocabulary:   # o reconhecimento de acordes só escolhe entre os acordes da cifra
                rules = replace(rules, detection={name: replace(cfg, allowed_chords=vocabulary)
                                                  for name, cfg in rules.detection.items()})
            if Path(opts.cifra).resolve() != (song_dir / "cifra.txt").resolve():
                shutil.copyfile(opts.cifra, song_dir / "cifra.txt")   # fica junto, para o Acrescentar

        guides = {}
        if opts.reference is not None:
            from ai_charter.reference import load_guides
            guides = load_guides(opts.reference, opts.reference_tracks, mix, ANALYSIS_SR, tempo, rules, log,
                                 opts.reference_fidelity, opts.reference_sections,
                                 opts.reference_anchors)

        log("4/5 Gerando as charts...")
        last_tick = int(tempo.time_to_tick(duration))
        sp_windows = star_power_windows(last_tick, res, rules.general.starpower_every_measures,
                                        rules.general.starpower_phrase_beats)
        tracks: dict[str, TrackChart] = {}
        guitar_sources: list[str] = []
        sources = stem_sources.validate(opts.stem_sources or {})
        for instrument in opts.instruments:
            stem = stems.get(sources.get(instrument, instrument))
            if stem is None:
                log(f"    {instrument}: sem stem, pulando")
                continue
            if instrument in sources:
                log(f"    {instrument}: usando o stem {sources[instrument]}")
            energy = _stem_energy(stem, mix_rms)
            if energy < rules.general.min_stem_energy:
                log(f"    {instrument}: stem quase silencioso ({energy:.3f}), pulando")
                continue
            if instrument == "drums":
                track = chart_drums(stem[0], tempo, rules, sp_windows, mix=mix_wav)
                if "drums" in guides:   # a partitura corrige as peças da pista pronta (reference.py)
                    track = guides["drums"](track, log, sp_windows)
            else:
                track, chosen = chart_pitched(stem, instrument, tempo, rules, sp_windows, log, tuning,
                                              guide=guides.get(instrument))
                if len(stem) > 1:
                    guitar_sources = chosen
                    log(f"    {instrument}: faixas usadas: {_describe_runs(chosen)}")
            counts = ", ".join(f"{d} {len(n)}" for d, n in track.notes.items())
            log(f"    {instrument}: {counts} notas")
            tracks[instrument] = track

        vocals = None
        lyrics = opts.lyrics
        if lyrics is None and opts.input.with_suffix(".txt").exists():
            lyrics = opts.input.with_suffix(".txt")  # letra com o mesmo nome da música
        if lyrics is None and cifra is not None and cifra.lyrics and opts.cifra_lyrics:
            from ai_charter.cifra import lyrics_text
            lyrics = song_dir / "letra.txt"   # a letra da cifra
            lyrics.write_text(lyrics_text(cifra), encoding="utf-8")
            log(f"    vocals: letra tirada da cifra (salva em {lyrics.name})")
        if "vocals" not in stems:
            log("    vocals: sem stem de voz, pulando")
        elif opts.output_format != "mid":
            log("    vocals: o formato .chart não tem faixa de vocal (use --format mid), pulando")
        else:
            if lyrics is None and opts.auto_lyrics:
                lyrics = song_dir / "letra_automatica.txt"
                log(f"    vocals: transcrevendo a letra com o Whisper (modelo {rules.vocals.whisper_model})...")
                lines = transcribe_lyrics(stems["vocals"][0], rules.vocals.language,
                                          rules.vocals.whisper_model,
                                          gap_seconds=rules.vocals.phrase_gap_beats * 60 / tempo.average_bpm)
                lyrics.write_text("\n".join(lines) + "\n", encoding="utf-8")
                log(f"    vocals: rascunho da letra salvo em {lyrics} — revise e rode de novo com --lyrics")
            if lyrics is None:
                log("    vocals: sem letra (use --lyrics letra.txt ou --auto-lyrics), pulando")
            else:
                vocals = chart_vocals(stems["vocals"][0], lyrics, tempo, rules, sp_windows, tuning)
                log(f"    vocals: {len(vocals.notes)} notas em {len(vocals.phrases)} frases"
                    + (f" ({len(vocals.spoken)} sílabas faladas, sem nota)" if vocals.spoken else ""))

        log("5/5 Exportando para o Clone Hero / YARG...")
        # Compassos de silêncio antes da 1ª nota, só nos arquivos: a análise acima não muda.
        lead = lead_in.plan(tempo, lead_in.first_tick(tracks, vocals), opts.lead_in)
        if lead.bars:
            log(f"    entrada: {lead.bars} compasso(s) de silêncio ({lead.seconds:.1f} s) antes da música")
        stem_entries: dict[str, str] = {}
        # O áudio acompanha a troca: o stem charteado vai no lugar do instrumento, e o dele no do stem.
        swapped = {t: s for t, s in sources.items() if t in tracks}
        if swapped and opts.separator != "none":
            stems = stem_sources.arrange(stems, swapped)
            log("    áudio trocado de lugar: " + ", ".join(f"{STEM_FILES[t][0]} <-> {STEM_FILES[s][0]}"
                                                       for t, s in stem_sources.layout(swapped).items() if t in swapped))
        if opts.separator == "none":
            # Sem stems de verdade: a música inteira vai no song.ogg.
            audio.to_ogg(mix_wav, song_dir / "song.ogg", delay=lead.seconds)
            stem_entries["MusicStream"] = "song.ogg"
        else:
            for stem_name, (filename, key) in STEM_FILES.items():
                if stem_name in stems:
                    audio.mix_to_ogg(stems[stem_name], song_dir / filename, delay=lead.seconds)
                    stem_entries[key] = filename

        info = SongInfo(
            name=title, artist=artist, album=tags.get("album", ""), genre=tags.get("genre", ""),
            year=(tags.get("date") or tags.get("year") or "")[:4],
            preview_start=find_preview_start(mix, ANALYSIS_SR) + lead.seconds, song_length=duration + lead.seconds,
            stems=stem_entries,
        )
        sections = (find_sections(mix, ANALYSIS_SR, tempo, rules.general.section_seconds)
                    if rules.general.sections else [])
        if opts.output_format == "mid":
            write_midi(song_dir / "notes.mid", title, tempo, tracks, sections, vocals, last_tick)
            lead_in.shift_file(song_dir / "notes.mid", lead.ticks)
        else:
            write_chart(song_dir / "notes.chart", info, tempo, tracks, sections)
            lead_in.shift_file(song_dir / "notes.chart", lead.ticks)
        write_song_ini(song_dir / "song.ini", info, tracks, vocals)
        if "drums" in tracks:   # compassos de bateria em que o modelo ficou em dúvida, para o editor
            write_confidence(song_dir / CONFIDENCE_NAME, tracks["drums"].doubt,
                             tempo.bar_starts() * tempo.resolution, lead.ticks)
        if swapped and opts.separator != "none":   # o Acrescentar precisa saber que os áudios estão trocados
            stem_sources.record(song_dir, swapped)
        audio.extract_cover(opts.input, song_dir / "album.png")

        result = Result(song_dir=song_dir, tracks=tracks, vocals=vocals,
                        guitar_sources=guitar_sources, bpm=tempo.average_bpm, lead_in=lead.seconds)
        if opts.make_zip:
            result.zip_path = Path(shutil.make_archive(str(song_dir), "zip", song_dir.parent, song_dir.name))
        return result
    finally:
        if opts.keep_work:
            log(f"Arquivos temporários mantidos em {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)
