"""Escreve o arquivo `notes.chart`, lido pelo Clone Hero e pelo YARG.

Formato (texto), resumidamente:

    [Song]          metadados (nome, artista, resolução...)
    [SyncTrack]     andamento:  "tick = B bpm×1000"  e  "tick = TS 4" (compasso)
    [Events]        seções:     tick = E "section Verse"
    [ExpertSingle]  notas:      "tick = N lane duração"  e  "tick = S 2 duração" (Star Power)
                    (na bateria, "tick = S 64 duração" marca uma virada de ativação)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ai_charter.analysis.tempo import TempoMap
from ai_charter.charting.common import TrackChart

# Nome da seção do .chart para cada instrumento.
TRACK_NAMES = {
    "guitar": "Single",
    "bass": "DoubleBass",
    "keys": "Keyboard",
    "drums": "Drums",
}
DIFFICULTY_NAMES = {"expert": "Expert", "hard": "Hard", "medium": "Medium", "easy": "Easy"}


@dataclass
class SongInfo:
    name: str
    artist: str
    album: str = ""
    genre: str = ""
    year: str = ""
    charter: str = "ai-charter"
    preview_start: float = 0.0
    song_length: float = 0.0
    stems: dict[str, str] = field(default_factory=dict)  # "MusicStream" -> "song.ogg"


def _quote(text: str) -> str:
    return '"' + text.replace('"', "'") + '"'


def write_chart(path: Path, info: SongInfo, tempo: TempoMap, tracks: dict[str, TrackChart],
                sections: list[int]) -> None:
    lines: list[str] = []

    lines += ["[Song]", "{"]
    lines.append(f"  Name = {_quote(info.name)}")
    lines.append(f"  Artist = {_quote(info.artist)}")
    lines.append(f"  Charter = {_quote(info.charter)}")
    if info.album:
        lines.append(f"  Album = {_quote(info.album)}")
    if info.year:
        lines.append(f'  Year = ", {info.year}"')
    lines.append("  Offset = 0")
    lines.append(f"  Resolution = {tempo.resolution}")
    lines.append('  Player2 = bass')
    lines.append("  Difficulty = 0")
    lines.append(f"  PreviewStart = {info.preview_start:.3f}")
    lines.append("  PreviewEnd = 0")
    if info.genre:
        lines.append(f"  Genre = {_quote(info.genre)}")
    lines.append('  MediaType = "cd"')
    for key, filename in info.stems.items():
        lines.append(f"  {key} = {_quote(filename)}")
    lines.append("}")

    lines += ["[SyncTrack]", "{", "  0 = TS 4"]
    lines += [f"  {tick} = B {bpm}" for tick, bpm in tempo.bpm_events()]
    lines.append("}")

    lines += ["[Events]", "{"]
    lines += [f'  {tick} = E "section Section {i + 1}"' for i, tick in enumerate(sections)]
    lines.append("}")

    for instrument, track in tracks.items():
        for difficulty, notes in track.notes.items():
            if not notes:
                continue
            lines += [f"[{DIFFICULTY_NAMES[difficulty]}{TRACK_NAMES[instrument]}]", "{"]
            # Junta notas e Star Power numa lista e ordena pelo tick.
            events: list[tuple[int, int, str]] = []
            for note in notes:
                for lane in note.lanes:
                    events.append((note.tick, 0, f"N {lane} {note.length}"))
                for flag in note.flags:
                    events.append((note.tick, 0, f"N {flag} 0"))
            for tick, length in track.star_power.get(difficulty, []):
                events.append((tick, 1, f"S 2 {length}"))
            for tick, length in track.drum_fills.get(difficulty, []):
                events.append((tick, 1, f"S 64 {length}"))  # virada de ativação (bateria)
            events.sort(key=lambda e: (e[0], e[1]))
            lines += [f"  {tick} = {text}" for tick, _, text in events]
            lines.append("}")

    # UTF-8 com BOM e quebras de linha do Windows, como o editor Moonscraper salva.
    path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8-sig", newline="")
