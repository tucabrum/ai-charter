"""Leitor simples de arquivos .chart, usado só nos testes para conferir o resultado."""

from __future__ import annotations

import re
from pathlib import Path

LINE = re.compile(r"^\s*(\S+)\s*=\s*(.*?)\s*$")


def parse_chart(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Retorna {nome_da_seção: [(chave, valor), ...]}."""
    sections: dict[str, list[tuple[str, str]]] = {}
    current = None
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = []
        elif line in ("{", "}") or not line:
            continue
        else:
            match = LINE.match(line)
            assert match and current is not None, f"linha inválida: {raw!r}"
            sections[current].append((match.group(1), match.group(2)))
    return sections


def notes_by_tick(entries: list[tuple[str, str]]) -> dict[int, list[int]]:
    """{tick: [números de nota]} a partir das linhas "tick = N nota duração"."""
    result: dict[int, list[int]] = {}
    for key, value in entries:
        parts = value.split()
        if parts[0] == "N":
            result.setdefault(int(key), []).append(int(parts[1]))
    return result


def parse_midi(path: Path) -> dict[str, list[tuple[int, str, object]]]:
    """Lê um notes.mid: {nome_da_trilha: [(tick, "note"/"lyric"/"text", valor), ...]}.

    Para notas, o valor é (nota_midi, duração).
    """
    import mido

    tracks: dict[str, list] = {}
    for track in mido.MidiFile(path).tracks:
        name, events, now, open_notes = "", [], 0, {}
        for msg in track:
            now += msg.time
            if msg.type == "track_name":
                name = msg.name
            elif msg.type == "note_on" and msg.velocity > 0:
                open_notes[msg.note] = now
            elif msg.type in ("note_off", "note_on"):
                start = open_notes.pop(msg.note)
                events.append((start, "note", (msg.note, now - start)))
            elif msg.type == "lyrics":
                events.append((now, "lyric", msg.text))
            elif msg.type == "text":
                events.append((now, "text", msg.text))
        tracks[name] = sorted(events, key=lambda e: e[0])
    return tracks


def midi_notes(events, low: int, high: int) -> dict[int, list[int]]:
    """{tick: [nota - low]} das notas entre low e high (ex.: Expert da guitarra = 96..100)."""
    result: dict[int, list[int]] = {}
    for tick, kind, value in events:
        if kind == "note" and low <= value[0] <= high:
            result.setdefault(tick, []).append(value[0] - low)
    return result
