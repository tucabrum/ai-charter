"""Acrescenta instrumentos a uma chart que já existe, sem mexer nas outras pistas.

Uso:
  ai-charter-add "D:\\Clone Hero\\Songs\\Banda - Música" --instruments vocals --lyrics letra.txt
  ai-charter-add "pasta da música" --instruments keys,drums --replace

Usa o que já está na pasta da música: o mapa de tempo do notes.mid (as notas novas
caem nas mesmas batidas das pistas que já existem), as frases de Star Power das outras
partes e os stems (guitar.ogg, bass.ogg, drums.ogg, keys.ogg, vocals.ogg). Por isso não
precisa da música original nem do Demucs.

Uma pista que já existe só é refeita com --replace. Nesse caso, os compassos travados no
editor (notes.locks.json) continuam como estavam. Antes de gravar, a chart anterior é
copiada para notes.before-add.mid. O song.ini ganha a dificuldade das partes novas.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import mido

from ai_charter import revision

CHART_NAME = "notes.mid"
BEFORE_ADD_NAME = "notes.before-add.mid"
LYRICS_NAMES = ("letra.txt", "lyrics.txt")   # letra guardada na pasta (a interface salva letra.txt)
INSTRUMENTS = ("guitar", "bass", "drums", "keys", "vocals")
TRACK_NAMES = {"guitar": "PART GUITAR", "bass": "PART BASS", "drums": "PART DRUMS",
               "keys": "PART KEYS", "vocals": "PART VOCALS"}


@dataclass
class AugmentResult:
    added: list[str] = field(default_factory=list)            # pistas gravadas
    skipped: dict[str, str] = field(default_factory=dict)     # instrumento -> motivo
    undone: int = 0                                           # notas desfeitas por travas


def existing_parts(folder: Path) -> list[str]:
    """Instrumentos que o notes.mid da pasta já tem."""
    midi = mido.MidiFile(Path(folder) / CHART_NAME)
    names = {t.name for t in midi.tracks}
    return [i for i, name in TRACK_NAMES.items() if name in names or (i == "guitar" and "T1 GEMS" in names)]


def _find_lyrics(folder: Path, lyrics: Path | None) -> Path | None:
    if lyrics is not None:
        return lyrics
    return next((folder / n for n in LYRICS_NAMES if (folder / n).is_file()), None)


def _load_mix(files: list[Path], sr: int):
    """Soma de todos os áudios da pasta (a música inteira, para a afinação e o volume)."""
    import librosa
    import numpy as np

    signals = [librosa.load(f, sr=sr, mono=True)[0] for f in files]
    mix = np.zeros(max(len(s) for s in signals), dtype=np.float32)
    for s in signals:
        mix[:len(s)] += s
    return mix


def _star_power_windows(midi: mido.MidiFile) -> list[tuple[int, int]]:
    """Frases de Star Power das partes que já existem (as novas usam as mesmas)."""
    for track in midi.tracks:
        if track.name in TRACK_NAMES.values() and track.name != "PART VOCALS":
            notes, _ = revision._split(track)
            windows = sorted({(n.start, n.end - n.start) for n in notes if n.pitch == revision.STAR_POWER})
            if windows:
                return windows
    return []


def update_song_ini(path: Path, values: dict[str, object]) -> None:
    """Troca só as chaves pedidas no song.ini (as outras linhas ficam como estão)."""
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines() if path.is_file() else ["[song]"]
    pending = dict(values)
    for i, line in enumerate(lines):
        key = line.partition("=")[0].strip().lower()
        if "=" in line and key in pending:
            lines[i] = f"{key} = {pending.pop(key)}"
    lines += [f"{k} = {v}" for k, v in pending.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def augment(folder: Path, instruments: tuple[str, ...], *, lyrics: Path | None = None,
            auto_lyrics: bool = False, rules_path: Path | None = None, replace_existing: bool = False,
            log=print) -> AugmentResult:
    from ai_charter.analysis.transcribe import transcribe_lyrics
    from ai_charter.charting.common import star_power_windows
    from ai_charter.config import load_rules
    from ai_charter.export.midi_writer import _fret_track, _vocal_track
    from ai_charter.export.song_ini import DIFF_KEYS, estimate_difficulty, estimate_vocal_difficulty
    from ai_charter.pipeline import (
        ANALYSIS_SR, _stem_energy, chart_drums, chart_pitched, chart_vocals, estimate_tuning,
    )
    from ai_charter.separation import AUDIO_EXTENSIONS, stems_from_folder

    folder = Path(folder)
    chart = folder / CHART_NAME
    if not chart.is_file():
        raise RuntimeError(f"a pasta não tem notes.mid: {folder}"
                           + (" (charts .chart ainda não são suportadas)" if (folder / "notes.chart").is_file() else ""))
    invalid = set(instruments) - set(INSTRUMENTS)
    if invalid:
        raise ValueError(f"instrumento(s) inválido(s): {', '.join(sorted(invalid))}")

    original = mido.MidiFile(chart)
    midi = mido.MidiFile(chart)
    tempo = revision._tempo_map(midi)
    rules = load_rules(rules_path)
    # As notas novas usam a resolução da chart (as oficiais costumam ter 480 ticks por batida).
    rules = replace(rules, general=replace(rules.general, resolution=midi.ticks_per_beat))
    result = AugmentResult()

    have = existing_parts(folder)
    todo = []
    for instrument in instruments:
        if instrument in have and not replace_existing:
            result.skipped[instrument] = "já existe na chart (use substituir para refazer)"
        else:
            todo.append(instrument)
    for instrument, reason in result.skipped.items():
        log(f"    {instrument}: {reason}")
    if not todo:
        return result

    stems = stems_from_folder(folder)
    audio_files = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in AUDIO_EXTENSIONS]
    log("Lendo o áudio da pasta...")
    mix = _load_mix(audio_files, ANALYSIS_SR)
    duration = len(mix) / ANALYSIS_SR
    mix_rms = float((mix ** 2).mean() ** 0.5)
    tuning = estimate_tuning(mix, ANALYSIS_SR)
    log(f"    mapa de tempo da chart: {tempo.average_bpm:.1f} BPM médio, {midi.ticks_per_beat} ticks por batida")
    sp_windows = _star_power_windows(midi)
    if not sp_windows:
        sp_windows = star_power_windows(int(tempo.time_to_tick(duration)), midi.ticks_per_beat,
                                        rules.general.starpower_every_measures, rules.general.starpower_phrase_beats)

    new_tracks: dict[str, mido.MidiTrack] = {}
    ini: dict[str, object] = {}
    for instrument in todo:
        stem = stems.get(instrument)
        if not stem:
            result.skipped[instrument] = f"a pasta não tem o stem ({instrument}.ogg)"
            log(f"    {instrument}: {result.skipped[instrument]}")
            continue
        if instrument == "vocals":
            words = _find_lyrics(folder, lyrics)
            if words is None and auto_lyrics:
                words = folder / "letra_automatica.txt"
                log(f"    vocals: transcrevendo a letra com o Whisper (modelo {rules.vocals.whisper_model})...")
                lines = transcribe_lyrics(stem[0], rules.vocals.language, rules.vocals.whisper_model,
                                          gap_seconds=rules.vocals.phrase_gap_beats * 60 / tempo.average_bpm)
                words.write_text("\n".join(lines) + "\n", encoding="utf-8")
                log(f"    vocals: rascunho da letra salvo em {words.name}, revise e rode de novo")
            if words is None:
                result.skipped[instrument] = "sem letra (informe a letra ou use a transcrição automática)"
                log(f"    vocals: {result.skipped[instrument]}")
                continue
            log("    vocals: alinhando a letra à voz...")
            vocals = chart_vocals(stem[0], words, tempo, rules, sp_windows, tuning)
            if not vocals.notes:
                result.skipped[instrument] = "nenhuma nota cantada encontrada"
                log(f"    vocals: {result.skipped[instrument]}")
                continue
            new_tracks[instrument] = _vocal_track(vocals).build()
            ini["diff_vocals"] = estimate_vocal_difficulty(vocals, duration)
            log(f"    vocals: {len(vocals.notes)} notas em {len(vocals.phrases)} frases")
            continue
        energy = _stem_energy(stem, mix_rms)
        if energy < rules.general.min_stem_energy:
            result.skipped[instrument] = f"stem quase silencioso ({energy:.3f})"
            log(f"    {instrument}: {result.skipped[instrument]}")
            continue
        log(f"    {instrument}: transcrevendo...")
        if instrument == "drums":
            track = chart_drums(stem[0], tempo, rules, sp_windows)
        else:
            track, _ = chart_pitched(stem, instrument, tempo, rules, sp_windows, log, tuning)
        new_tracks[instrument] = _fret_track(TRACK_NAMES[instrument], track, midi.ticks_per_beat,
                                             drums=instrument == "drums").build()
        ini[DIFF_KEYS[instrument]] = estimate_difficulty(track, duration)
        if instrument == "drums":
            ini["pro_drums"] = "True"
        log(f"    {instrument}: " + ", ".join(f"{d} {len(n)}" for d, n in track.notes.items()) + " notas")

    if not new_tracks:
        return result
    for instrument, track in new_tracks.items():
        names = {TRACK_NAMES[instrument]} | ({"T1 GEMS"} if instrument == "guitar" else set())
        old = next((t for t in midi.tracks if t.name in names), None)
        if old is not None:
            midi.tracks[midi.tracks.index(old)] = track
        else:
            midi.tracks.append(track)
        result.added.append(instrument)
    merged, result.undone = revision.protect(original, midi, revision.load_locks(folder))
    shutil.copy2(chart, folder / BEFORE_ADD_NAME)
    tmp = folder / (CHART_NAME + ".tmp")
    merged.save(tmp)
    tmp.replace(chart)
    update_song_ini(folder / "song.ini", ini)
    log(f"notes.mid atualizado com: {', '.join(result.added)} (a versão anterior está em {BEFORE_ADD_NAME})")
    if result.undone:
        log(f"{result.undone} nota(s) em compassos travados foram mantidas como estavam")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ai-charter-add",
                                     description="Acrescenta instrumentos a uma chart que já existe.")
    parser.add_argument("folder", type=Path, help="pasta da música (com notes.mid e os stems .ogg)")
    parser.add_argument("--instruments", required=True,
                        help="instrumentos a gerar, separados por vírgula (guitar,bass,drums,keys,vocals)")
    parser.add_argument("--lyrics", type=Path, help="letra para os vocais (padrão: letra.txt da pasta)")
    parser.add_argument("--auto-lyrics", action="store_true", help="sem letra, transcreve a voz com o Whisper")
    parser.add_argument("--rules", type=Path, help="arquivo .toml com regras de charting personalizadas")
    parser.add_argument("--replace", action="store_true", help="refaz também as pistas que já existem")
    args = parser.parse_args(argv)
    instruments = tuple(i.strip() for i in args.instruments.split(",") if i.strip())
    if not args.folder.is_dir():
        parser.error(f"pasta não encontrada: {args.folder}")
    if args.lyrics is not None and not args.lyrics.is_file():
        parser.error(f"arquivo de letra não encontrado: {args.lyrics}")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # no Windows o padrão (cp1252) embaralha os acentos
    try:
        result = augment(args.folder, instruments, lyrics=args.lyrics, auto_lyrics=args.auto_lyrics,
                         rules_path=args.rules, replace_existing=args.replace)
    except (RuntimeError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        sys.exit(1)
    if not result.added:
        print("Nada foi acrescentado.")
        sys.exit(1)


if __name__ == "__main__":
    main()
