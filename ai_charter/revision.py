"""Pedidos de revisão feitos no editor, e a trava dos compassos editados à mão.

Fluxo:
  1. No editor (ai-charter-view), o usuário mexe na chart e clica em "Pedir à IA". A página
     salva o notes.mid, o notes.locks.json (compassos mexidos à mão, por instrumento e
     dificuldade) e o ai_request.json (o texto do pedido e o antes/depois de cada compasso).
  2. Quem atende o pedido (por exemplo o Claude Code) lê a chart com `show`, faz as mudanças
     numa CÓPIA do notes.mid e roda `apply`.
  3. O `apply` devolve aos compassos travados exatamente o que o usuário deixou (o que a IA
     tiver mudado ali é descartado e contado), guarda a versão anterior em notes.before-ai.mid,
     grava o notes.mid e arquiva o pedido como ai_request.done.json.

Uso:
  ai-charter-revise show  "pasta da música" [--instrument guitar] [--difficulty expert]
  ai-charter-revise apply "pasta da música" revisada.mid
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import mido

CHART_NAME = "notes.mid"
LOCKS_NAME = "notes.locks.json"
REQUEST_NAME = "ai_request.json"
DONE_NAME = "ai_request.done.json"
BEFORE_AI_NAME = "notes.before-ai.mid"

FRET_TRACKS = {"PART GUITAR": "guitar", "T1 GEMS": "guitar", "PART BASS": "bass", "PART RHYTHM": "rhythm", "PART KEYS": "keys"}
BASE = {"expert": 96, "hard": 84, "medium": 72, "easy": 60}
# Notas de cada dificuldade: 5 cores (base..base+4), HOPO forçado (+5) e strum forçado (+6)
DIFF_SPAN = 7
TAP, STAR_POWER = 104, 116
SHARED = (TAP, STAR_POWER)      # valem para todas as dificuldades: travadas como "shared"
LANES = "GRYBO"


class Note(NamedTuple):
    start: int
    end: int
    pitch: int
    velocity: int
    channel: int


# ------------------------------------------------------------------ travas
def load_locks(folder: Path) -> dict[str, dict[str, list[tuple[int, int]]]]:
    """{instrumento: {dificuldade | "shared": [(início, fim), ...]}} em ticks; {} sem arquivo."""
    path = folder / LOCKS_NAME
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {instr: {key: [(r[0], r[1]) for r in ranges] for key, ranges in by_key.items()}
            for instr, by_key in data.get("locks", {}).items()}


def _lock_key(pitch: int) -> str | None:
    if pitch in SHARED:
        return "shared"
    for diff, base in BASE.items():
        if base <= pitch < base + DIFF_SPAN:
            return diff
    return None


def _is_locked(ranges: dict[str, list[tuple[int, int]]], pitch: int, tick: int) -> bool:
    key = _lock_key(pitch)
    return key is not None and any(a <= tick < b for a, b in ranges.get(key, ()))


# ------------------------------------------------------------------ trilhas <-> notas
def _split(track: mido.MidiTrack) -> tuple[list[Note], list[tuple[int, mido.Message]]]:
    """Notas (pares on/off) e os outros eventos, em ticks absolutos. O fim da trilha fica de fora."""
    notes, other, open_ = [], [], {}
    tick = 0
    for msg in track:
        tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            open_.setdefault((msg.channel, msg.note), (tick, msg.velocity))
        elif msg.type in ("note_on", "note_off"):
            start = open_.pop((msg.channel, msg.note), None)
            if start is not None:
                notes.append(Note(start[0], tick, msg.note, start[1], msg.channel))
        elif not (msg.is_meta and msg.type == "end_of_track"):
            other.append((tick, msg))
    return notes, other


def _join(other: list[tuple[int, mido.Message]], notes: list[Note]) -> mido.MidiTrack:
    # No mesmo tick: note off antes dos outros eventos, e estes antes de note on.
    events = [(tick, 1, i, msg) for i, (tick, msg) in enumerate(other)]
    for i, n in enumerate(sorted(notes)):
        events.append((n.start, 2, i, mido.Message("note_on", note=n.pitch, velocity=n.velocity, channel=n.channel)))
        events.append((n.end, 0, i, mido.Message("note_off", note=n.pitch, velocity=0, channel=n.channel)))
    events.sort(key=lambda e: e[:3])
    track, now = mido.MidiTrack(), 0
    for tick, _, _, msg in events:
        track.append(msg.copy(time=tick - now))
        now = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


# ------------------------------------------------------------------ apply
def protect(original: mido.MidiFile, revised: mido.MidiFile,
            locks: dict[str, dict[str, list[tuple[int, int]]]]) -> tuple[mido.MidiFile, int]:
    """A chart revisada com os compassos travados de volta como no original.

    Devolve também quantas notas a revisão tinha mudado em trechos travados (e foram desfeitas)."""
    out = mido.MidiFile(type=revised.type, ticks_per_beat=revised.ticks_per_beat)
    originals = {t.name: t for t in original.tracks}
    undone = 0
    for track in revised.tracks:
        ranges = locks.get(FRET_TRACKS.get(track.name, ""), {})
        if not ranges or track.name not in originals:
            out.tracks.append(track)
            continue
        new_notes, other = _split(track)
        old_notes, _ = _split(originals[track.name])
        locked_new = [n for n in new_notes if _is_locked(ranges, n.pitch, n.start)]
        locked_old = [n for n in old_notes if _is_locked(ranges, n.pitch, n.start)]
        key = lambda n: (n.start, n.end, n.pitch)   # noqa: E731
        undone += len({key(n) for n in locked_new} ^ {key(n) for n in locked_old})
        kept = [n for n in new_notes if not _is_locked(ranges, n.pitch, n.start)]
        out.tracks.append(_join(other, kept + locked_old))
    return out, undone


def apply(folder: Path, revised_path: Path) -> int:
    """Grava a revisão na pasta da música respeitando as travas; devolve quantas notas foram desfeitas."""
    chart = folder / CHART_NAME
    merged, undone = protect(mido.MidiFile(chart), mido.MidiFile(revised_path), load_locks(folder))
    shutil.copy2(chart, folder / BEFORE_AI_NAME)
    tmp = folder / (CHART_NAME + ".tmp")
    merged.save(tmp)
    tmp.replace(chart)
    request = folder / REQUEST_NAME
    if request.is_file():
        data = json.loads(request.read_text(encoding="utf-8"))
        data["applied"] = datetime.now().isoformat(timespec="seconds")
        data["undone_in_locked_measures"] = undone
        (folder / DONE_NAME).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        request.unlink()
    return undone


# ------------------------------------------------------------------ show
def measures(midi: mido.MidiFile) -> list[tuple[int, int]]:
    """Compassos [início, fim) em ticks, pelas fórmulas de compasso da trilha de tempo."""
    res = midi.ticks_per_beat
    sigs, tick = [], 0
    for msg in midi.tracks[0]:
        tick += msg.time
        if msg.type == "time_signature":
            sigs.append((tick, msg.numerator, msg.denominator))
    if not sigs or sigs[0][0] > 0:
        sigs.insert(0, (0, 4, 4))
    end = max(sum(m.time for m in t) for t in midi.tracks) + res * 4
    out = []
    for i, (start, num, den) in enumerate(sigs):
        length = res * 4 // den * num
        stop = sigs[i + 1][0] if i + 1 < len(sigs) else end
        out.extend((t, min(t + length, stop)) for t in range(start, stop, length))
    return out


def show(folder: Path, instrument: str = "guitar", difficulty: str = "expert") -> str:
    """A chart em texto, um compasso por linha, com os compassos travados marcados.

    Posição em semicolcheias desde o começo do compasso; cores G R Y B O; ~N = sustain de N
    semicolcheias; h/s = HOPO/strum forçado; t = tap; * = Star Power; [trava] = compasso travado."""
    midi = mido.MidiFile(folder / CHART_NAME)
    track = next((t for t in midi.tracks if FRET_TRACKS.get(t.name) == instrument), None)
    if track is None:
        raise SystemExit(f"a chart não tem {instrument}")
    notes, _ = _split(track)
    ranges = load_locks(folder).get(instrument, {})
    base, step = BASE[difficulty], midi.ticks_per_beat / 4
    in_ = lambda pitch, tick: any(n.start <= tick < n.end for n in notes if n.pitch == pitch)   # noqa: E731
    chords: dict[int, list[Note]] = {}
    for n in notes:
        if base <= n.pitch < base + 5:
            chords.setdefault(n.start, []).append(n)
    lines = [f"{instrument} {difficulty} · {midi.ticks_per_beat} ticks por batida · posições em semicolcheias"]
    for number, (a, b) in enumerate(measures(midi), start=1):
        locked = any(a <= t < b for key in (difficulty, "shared") for t, _ in ranges.get(key, ()))
        items = []
        for tick in sorted(t for t in chords if a <= t < b):
            gems = sorted(chords[tick], key=lambda n: n.pitch)
            text = f"{(tick - a) / step:g}:" + "".join(LANES[n.pitch - base] for n in gems)
            longest = max(n.end - n.start for n in gems)
            if longest > midi.ticks_per_beat / 3:
                text += f"~{longest / step:g}"
            text += "h" * in_(base + 5, tick) + "s" * in_(base + 6, tick) + "t" * in_(TAP, tick) + "*" * in_(STAR_POWER, tick)
            items.append(text)
        if items or locked:
            lines.append(f"{number:4d} {'[trava]' if locked else '       '} | " + "  ".join(items))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ai-charter-revise",
                                     description="Atende pedidos de revisão do editor sem mexer nos compassos editados à mão.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_show = sub.add_parser("show", help="mostra a chart em texto, com os compassos travados")
    p_show.add_argument("folder", type=Path)
    p_show.add_argument("--instrument", default="guitar", choices=sorted(set(FRET_TRACKS.values())))
    p_show.add_argument("--difficulty", default="expert", choices=list(BASE))
    p_apply = sub.add_parser("apply", help="grava uma versão revisada, devolvendo os compassos travados")
    p_apply.add_argument("folder", type=Path)
    p_apply.add_argument("revised", type=Path)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # no Windows o padrão (cp1252) embaralha os acentos
    if args.command == "show":
        print(show(args.folder, args.instrument, args.difficulty))
    else:
        undone = apply(args.folder, args.revised)
        print(f"notes.mid atualizado (a versão anterior está em {BEFORE_AI_NAME}).")
        if undone:
            print(f"{undone} nota(s) mudadas em compassos travados foram desfeitas.")


if __name__ == "__main__":
    main()
