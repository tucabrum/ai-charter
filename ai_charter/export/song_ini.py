"""Escreve o `song.ini`, que o Clone Hero e o YARG usam para listar a música."""

from __future__ import annotations

from pathlib import Path

from ai_charter.charting.common import TrackChart
from ai_charter.export.chart_writer import SongInfo

# Nome do campo de dificuldade no song.ini para cada instrumento.
DIFF_KEYS = {"guitar": "diff_guitar", "bass": "diff_bass", "drums": "diff_drums", "keys": "diff_keys"}


def estimate_difficulty(track: TrackChart, song_seconds: float) -> int:
    """Nota de 0 a 6 (os "anéis" de dificuldade) pela densidade de notas no Expert."""
    notes = track.notes.get("expert", [])
    if not notes or song_seconds <= 0:
        return -1  # -1 = instrumento não existe
    notes_per_second = len(notes) / song_seconds
    for level, limit in enumerate((1.5, 2.5, 3.5, 4.5, 6.0, 7.5)):
        if notes_per_second < limit:
            return level
    return 6


def write_song_ini(path: Path, info: SongInfo, tracks: dict[str, TrackChart]) -> None:
    values = {
        "name": info.name,
        "artist": info.artist,
        "album": info.album,
        "genre": info.genre,
        "year": info.year,
        "charter": info.charter,
        "song_length": int(info.song_length * 1000),
        "preview_start_time": int(info.preview_start * 1000),
        "delay": 0,
        "loading_phrase": "Chart gerada automaticamente pelo ai-charter.",
    }
    for instrument, key in DIFF_KEYS.items():
        track = tracks.get(instrument)
        values[key] = estimate_difficulty(track, info.song_length) if track else -1
    if "drums" in tracks:
        values["pro_drums"] = "True"

    text = "[song]\n" + "".join(f"{k} = {v}\n" for k, v in values.items())
    path.write_text(text, encoding="utf-8")
