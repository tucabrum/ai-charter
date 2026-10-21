"""Separação da música em stems (guitarra, baixo, bateria, teclado, voz, resto).

Três modos:
  - "demucs": usa o modelo de IA htdemucs_6s (melhor qualidade; precisa do
               pacote `demucs`, instalado com  pip install "ai-charter[separation]");
  - "folder": você já tem os stems separados numa pasta (guitar.wav, bass.wav...);
  - "none":   sem IA - separa só o que dá com filtros (percussivo x harmônico).
               Serve para testar, mas a qualidade das charts é bem menor.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

# Nomes padronizados dos stems usados no resto do programa.
STEMS = ("guitar", "bass", "drums", "keys", "vocals", "other")

# Nomes aceitos para cada stem quando o usuário fornece uma pasta.
ALIASES = {
    "guitar": ("guitar", "guitars", "gtr"),
    "bass": ("bass",),
    "drums": ("drums", "drum", "percussion"),
    "keys": ("keys", "piano", "keyboard", "synth"),
    "vocals": ("vocals", "vocal", "voice", "vox"),
    "other": ("other", "song", "backing", "instrumental", "rest"),
}
AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".opus")


def separate_demucs(mix_wav: Path, work_dir: Path, model: str = "htdemucs_6s") -> dict[str, Path]:
    try:
        import demucs  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            'O pacote "demucs" não está instalado. Rode:  pip install "ai-charter[separation]"\n'
            "ou use --separator none para testar sem separação por IA."
        ) from exc

    out_dir = work_dir / "demucs"
    print(f"Separando stems com o modelo {model} (pode demorar alguns minutos)...")
    subprocess.run([sys.executable, "-m", "demucs", "-n", model, "-o", str(out_dir), str(mix_wav)],
                   check=True)
    stem_dir = out_dir / model / mix_wav.stem
    # O demucs chama o teclado de "piano".
    names = {"guitar": "guitar", "bass": "bass", "drums": "drums", "keys": "piano",
             "vocals": "vocals", "other": "other"}
    return {ours: stem_dir / f"{theirs}.wav" for ours, theirs in names.items()
            if (stem_dir / f"{theirs}.wav").exists()}


def stems_from_folder(folder: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for file in sorted(folder.iterdir()):
        if file.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        for stem, aliases in ALIASES.items():
            if file.stem.lower() in aliases and stem not in found:
                found[stem] = file
    if not found:
        raise RuntimeError(f"Nenhum stem reconhecido em {folder} (use nomes como guitar.wav, bass.wav...)")
    return found


def separate_simple(mix_wav: Path, work_dir: Path) -> dict[str, Path]:
    """Separação sem IA: bateria = parte percussiva, baixo = graves harmônicos,
    guitarra = resto harmônico. Não gera teclado nem voz."""
    y, sr = librosa.load(mix_wav, sr=None, mono=True)
    harmonic, percussive = librosa.effects.hpss(y)
    low = sosfiltfilt(butter(4, 250, btype="lowpass", fs=sr, output="sos"), harmonic)
    high = sosfiltfilt(butter(4, 250, btype="highpass", fs=sr, output="sos"), harmonic)

    out_dir = work_dir / "simple"
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = {}
    for name, signal in (("drums", percussive), ("bass", low), ("guitar", high)):
        stems[name] = out_dir / f"{name}.wav"
        sf.write(stems[name], np.asarray(signal, dtype=np.float32), sr)
    return stems


def separate(mode: str, mix_wav: Path, work_dir: Path, stems_dir: Path | None = None) -> dict[str, Path]:
    if mode == "demucs":
        return separate_demucs(mix_wav, work_dir)
    if mode == "folder":
        if stems_dir is None:
            raise ValueError("Use --stems PASTA junto com --separator folder.")
        return stems_from_folder(stems_dir)
    if mode == "none":
        return separate_simple(mix_wav, work_dir)
    raise ValueError(f"Separador desconhecido: {mode}")
