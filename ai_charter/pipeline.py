"""Pipeline completo: áudio -> stems -> notas -> charts -> pasta pronta para o jogo."""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

from ai_charter.analysis.drums import detect_drum_hits
from ai_charter.analysis.pitched import detect_pitched_notes
from ai_charter.analysis.sections import find_preview_start, find_sections
from ai_charter.analysis.tempo import TempoMap, estimate_tempo_map
from ai_charter.analysis.transcribe import transcribe_lyrics
from ai_charter.analysis.vocals import detect_sung_notes
from ai_charter.charting.common import (
    Candidate, TrackChart, dedupe, quantize_tick, star_power_for_phrases, star_power_windows,
)
from ai_charter.charting.chords import label_chords
from ai_charter.charting.drums import build_drum_track
from ai_charter.charting.frets import build_fret_track
from ai_charter.charting.guitar_select import GuitarSource, merge_guitars
from ai_charter.charting.lyrics import read_lyrics
from ai_charter.charting.sanitize import sanitize
from ai_charter.charting.vocals import VocalTrack, build_vocal_track
from ai_charter.config import Rules, load_rules
from ai_charter.export import audio
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
    instruments: tuple[str, ...] = ALL_INSTRUMENTS
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


def chart_pitched(stems: list[Path], instrument: str, tempo: TempoMap, rules: Rules,
                  sp_windows, log=print) -> tuple[TrackChart, list[str]]:
    """Chart de guitarra/baixo/teclado. Com várias guitarras, escolhe uma por compasso."""
    cfg = rules.detection[instrument]
    sources = []
    for stem in stems:
        y, sr = librosa.load(stem, sr=ANALYSIS_SR, mono=True)
        cands = _to_candidates(detect_pitched_notes(y, sr, cfg), tempo, rules)
        if cfg.recognize_chords:
            label_chords(cands, rules.general.resolution, cfg)
        cands, simplified = sanitize(cands, y, sr, tempo, rules.general.resolution, rules.sanitize)
        if simplified:
            log(f"    {instrument} ({stem.stem}): {len(simplified)} compasso(s) confuso(s) simplificado(s): "
                + ", ".join(str(m + 1) for m in simplified))
        sources.append(GuitarSource(stem.stem, cands, y, sr))
    cands, chosen = merge_guitars(sources, tempo, rules.general.resolution, rules.guitar_selection)
    track = build_fret_track(cands, rules.frets, rules.general.resolution, cfg.lane_window, sp_windows)
    return track, chosen


def chart_vocals(stem: Path, lyrics: Path, tempo: TempoMap, rules: Rules, sp_windows) -> VocalTrack:
    y, sr = librosa.load(stem, sr=ANALYSIS_SR, mono=True)
    sung = detect_sung_notes(y, sr, rules.vocals)
    lines = read_lyrics(lyrics, rules.vocals.language)
    vocals = build_vocal_track(sung, lines, tempo, rules.general.resolution,
                               rules.general.allow_triplets, rules.vocals)
    vocals.star_power = star_power_for_phrases(vocals.phrases, sp_windows)
    return vocals


def chart_drums(stem: Path, tempo: TempoMap, rules: Rules, sp_windows) -> TrackChart:
    y, sr = librosa.load(stem, sr=DRUMS_SR, mono=True)
    hits = detect_drum_hits(y, sr, rules.drum_detection)
    res, triplets = rules.general.resolution, rules.general.allow_triplets
    cands: list[Candidate] = []
    seen: dict[tuple[int, str], Candidate] = {}
    for h in hits:
        tick = quantize_tick(tempo.time_to_tick(h.time), res, triplets)
        key = (tick, h.piece)
        if key not in seen or h.strength > seen[key].strength:
            seen[key] = Candidate(tick, h.strength, piece=h.piece)
    cands = sorted(seen.values(), key=lambda c: c.tick)
    return build_drum_track(cands, rules.drums, res, sp_windows,
                            rules.general.drum_fill_every_measures)


def run(opts: Options, log=print) -> Result:
    audio.require_ffmpeg()
    rules = load_rules(opts.rules_path)
    res = rules.general.resolution

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
        tempo = estimate_tempo_map(mix, ANALYSIS_SR, res, opts.tempo_mode)
        log(f"    BPM médio: {tempo.average_bpm:.1f}")

        log(f"3/5 Separando os stems (modo: {opts.separator})...")
        stems = separate(opts.separator, mix_wav, work_dir, opts.stems_dir)

        log("4/5 Gerando as charts...")
        last_tick = int(tempo.time_to_tick(duration))
        sp_windows = star_power_windows(last_tick, res, rules.general.starpower_every_measures,
                                        rules.general.starpower_phrase_beats)
        tracks: dict[str, TrackChart] = {}
        guitar_sources: list[str] = []
        for instrument in opts.instruments:
            stem = stems.get(instrument)
            if stem is None:
                log(f"    {instrument}: sem stem, pulando")
                continue
            energy = _stem_energy(stem, mix_rms)
            if energy < rules.general.min_stem_energy:
                log(f"    {instrument}: stem quase silencioso ({energy:.3f}), pulando")
                continue
            if instrument == "drums":
                track = chart_drums(stem[0], tempo, rules, sp_windows)
            else:
                track, chosen = chart_pitched(stem, instrument, tempo, rules, sp_windows, log)
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
                vocals = chart_vocals(stems["vocals"][0], lyrics, tempo, rules, sp_windows)
                log(f"    vocals: {len(vocals.notes)} notas em {len(vocals.phrases)} frases")

        log("5/5 Exportando para o Clone Hero / YARG...")
        stem_entries: dict[str, str] = {}
        if opts.separator == "none":
            # Sem stems de verdade: a música inteira vai no song.ogg.
            audio.to_ogg(mix_wav, song_dir / "song.ogg")
            stem_entries["MusicStream"] = "song.ogg"
        else:
            for stem_name, (filename, key) in STEM_FILES.items():
                if stem_name in stems:
                    audio.mix_to_ogg(stems[stem_name], song_dir / filename)
                    stem_entries[key] = filename

        info = SongInfo(
            name=title, artist=artist, album=tags.get("album", ""), genre=tags.get("genre", ""),
            year=(tags.get("date") or tags.get("year") or "")[:4],
            preview_start=find_preview_start(mix, ANALYSIS_SR), song_length=duration,
            stems=stem_entries,
        )
        sections = (find_sections(mix, ANALYSIS_SR, tempo, rules.general.section_seconds)
                    if rules.general.sections else [])
        if opts.output_format == "mid":
            write_midi(song_dir / "notes.mid", title, tempo, tracks, sections, vocals, last_tick)
        else:
            write_chart(song_dir / "notes.chart", info, tempo, tracks, sections)
        write_song_ini(song_dir / "song.ini", info, tracks, vocals)
        audio.extract_cover(opts.input, song_dir / "album.png")

        result = Result(song_dir=song_dir, tracks=tracks, vocals=vocals,
                        guitar_sources=guitar_sources, bpm=tempo.average_bpm)
        if opts.make_zip:
            result.zip_path = Path(shutil.make_archive(str(song_dir), "zip", song_dir.parent, song_dir.name))
        return result
    finally:
        if opts.keep_work:
            log(f"Arquivos temporários mantidos em {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)
