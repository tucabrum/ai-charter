"""Funções de áudio que usam o ffmpeg (conversão, metadados e capa)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg/ffprobe não encontrados. Instale o ffmpeg e coloque-o no PATH.")


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Falha ao rodar {cmd[0]}:\n{result.stderr[-2000:]}")


def to_wav(src: Path, dst: Path, sample_rate: int = 44100) -> None:
    """Converte qualquer áudio (mp3, flac, ...) para WAV estéreo."""
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "2",
          "-ar", str(sample_rate), str(dst)])


def _delay(seconds: float) -> str:
    """Filtro que põe `seconds` de silêncio no começo (o tempo de entrada, ver lead_in.py)."""
    return f"adelay=delays={seconds * 1000:.3f}:all=1"


def to_ogg(src: Path, dst: Path, quality: int = 8, delay: float = 0.0) -> None:
    """Converte para OGG Vorbis, o formato de áudio que o Clone Hero usa (qualidade 0-10).

    `delay`: segundos de silêncio no começo."""
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", *(["-af", _delay(delay)] if delay > 0 else []),
          "-c:a", "libvorbis", "-q:a", str(quality), str(dst)])


def mix_to_ogg(sources: list[Path], dst: Path, quality: int = 8, delay: float = 0.0) -> None:
    """Soma vários áudios (ex.: violão + guitarra) num único .ogg, sem alterar o volume."""
    if len(sources) == 1:
        to_ogg(sources[0], dst, quality, delay)
        return
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for src in sources:
        cmd += ["-i", str(src)]
    mix = f"amix=inputs={len(sources)}:normalize=0:duration=longest" + (f",{_delay(delay)}" if delay > 0 else "")
    cmd += ["-filter_complex", mix, "-c:a", "libvorbis", "-q:a", str(quality), str(dst)]
    _run(cmd)


def read_tags(src: Path) -> dict[str, str]:
    """Lê as tags do arquivo (título, artista, álbum...) em minúsculas.

    No Ogg e no Opus as tags ficam na faixa de áudio, não no arquivo: as dela valem quando o
    arquivo não tem a mesma tag."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams",
         "-select_streams", "a:0", str(src)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {}
    data = json.loads(result.stdout or "{}")
    tags = {}
    for source in [*(s.get("tags", {}) for s in data.get("streams", [])[:1]), data.get("format", {}).get("tags", {})]:
        tags.update({k.lower(): str(v) for k, v in source.items()})
    return tags


def extract_cover(src: Path, dst: Path) -> bool:
    """Salva a capa embutida no arquivo (se houver) como album.png."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-an", "-frames:v", "1",
         "-vf", "scale=512:512", str(dst)],
        capture_output=True,
    )
    return result.returncode == 0 and dst.exists()
