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
from ai_charter.charting.common import Candidate, TrackChart, dedupe, quantize_tick, star_power_windows
from ai_charter.charting.drums import build_drum_track
from ai_charter.charting.frets import build_fret_track
from ai_charter.config import Rules, load_rules
from ai_charter.export import audio
from ai_charter.export.chart_writer import SongInfo, write_chart
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
    make_zip: bool = False
    keep_work: bool = False


@dataclass
class Result:
    song_dir: Path
    tracks: dict[str, TrackChart] = field(default_factory=dict)
    bpm: float = 0.0
    zip_path: Path | None = None


def _safe_name(text: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", text).strip() or "song"


def _stem_energy(path: Path, mix_rms: float) -> float:
    y, _ = librosa.load(path, sr=ANALYSIS_SR, mono=True)
    return float(np.sqrt(np.mean(y ** 2)) / (mix_rms + 1e-12))


def _to_candidates(detected, tempo: TempoMap, rules: Rules) -> list[Candidate]:
    res, triplets = rules.general.resolution, rules.general.allow_triplets
    cands = [
        Candidate(
            tick=quantize_tick(tempo.time_to_tick(n.time), res, triplets),
            strength=n.strength,
            end_tick=int(tempo.time_to_tick(n.end)),
            pitch=n.pitch,
            voices=n.voices,
        )
        for n in detected
    ]
    return dedupe(cands)


def chart_pitched(stem: Path, instrument: str, tempo: TempoMap, rules: Rules,
                  sp_windows) -> TrackChart:
    y, sr = librosa.load(stem, sr=ANALYSIS_SR, mono=True)
    cfg = rules.detection[instrument]
    detected = detect_pitched_notes(y, sr, cfg)
    cands = _to_candidates(detected, tempo, rules)
    return build_fret_track(cands, rules.frets, rules.general.resolution, cfg.lane_window, sp_windows)


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
    return build_drum_track(cands, rules.drums, res, sp_windows)


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
                track = chart_drums(stem, tempo, rules, sp_windows)
            else:
                track = chart_pitched(stem, instrument, tempo, rules, sp_windows)
            counts = ", ".join(f"{d} {len(n)}" for d, n in track.notes.items())
            log(f"    {instrument}: {counts} notas")
            tracks[instrument] = track

        log("5/5 Exportando para o Clone Hero / YARG...")
        stem_entries: dict[str, str] = {}
        if opts.separator == "none":
            # Sem stems de verdade: a música inteira vai no song.ogg.
            audio.to_ogg(mix_wav, song_dir / "song.ogg")
            stem_entries["MusicStream"] = "song.ogg"
        else:
            for stem_name, (filename, key) in STEM_FILES.items():
                if stem_name in stems:
                    audio.to_ogg(stems[stem_name], song_dir / filename)
                    stem_entries[key] = filename

        info = SongInfo(
            name=title, artist=artist, album=tags.get("album", ""), genre=tags.get("genre", ""),
            year=(tags.get("date") or tags.get("year") or "")[:4],
            preview_start=find_preview_start(mix, ANALYSIS_SR), song_length=duration,
            stems=stem_entries,
        )
        sections = (find_sections(mix, ANALYSIS_SR, tempo, rules.general.section_seconds)
                    if rules.general.sections else [])
        write_chart(song_dir / "notes.chart", info, tempo, tracks, sections)
        write_song_ini(song_dir / "song.ini", info, tracks)
        audio.extract_cover(opts.input, song_dir / "album.png")

        result = Result(song_dir=song_dir, tracks=tracks, bpm=tempo.average_bpm)
        if opts.make_zip:
            result.zip_path = Path(shutil.make_archive(str(song_dir), "zip", song_dir.parent, song_dir.name))
        return result
    finally:
        if opts.keep_work:
            log(f"Arquivos temporários mantidos em {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)
