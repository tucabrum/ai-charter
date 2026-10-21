"""Gera uma música sintética simples (com notas conhecidas) para os testes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 44100
BPM = 120.0
BEAT = 60.0 / BPM
START = 1.0  # a música começa 1 s depois do início do arquivo

# Melodia da "guitarra": uma nota por colcheia, sobe e desce (notas MIDI).
MELODY = [52, 55, 57, 59, 62, 64, 62, 59] * 4 + [57, 57, 60, 60, 64, 64, 67, 67] * 2
BASS = [40, 40, 43, 45] * 12


def tone(midi: float, length: float, harmonics: int = 6) -> np.ndarray:
    t = np.arange(int(length * SR)) / SR
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    wave = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, harmonics + 1) if f0 * h < SR / 2)
    release = np.clip((length - t) / 0.02, 0.0, 1.0)  # fade-out de 20 ms (sem "clique" no fim)
    envelope = np.minimum(1.0, t / 0.005) * np.exp(-t * 3.0) * release
    return wave * envelope


def drum(kind: str, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(0.4 * SR)) / SR
    noise = rng.standard_normal(len(t))
    if kind == "kick":
        freq = 45 + 55 * np.exp(-t * 40)
        return np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t * 12)
    if kind == "snare":
        body = np.sin(2 * np.pi * 200 * t) * np.exp(-t * 25)
        wires = sosfilt(butter(4, [1000, 8000], btype="bandpass", fs=SR, output="sos"), noise)
        return 0.6 * body + 0.8 * wires * np.exp(-t * 20)
    if kind == "crash":
        t = np.arange(int(1.5 * SR)) / SR
        metal = sosfilt(butter(4, 4000, btype="highpass", fs=SR, output="sos"),
                        rng.standard_normal(len(t)))
        return 0.5 * metal * np.exp(-t * 2)
    metal = sosfilt(butter(4, 7000, btype="highpass", fs=SR, output="sos"), noise)
    return 0.4 * metal * np.exp(-t * 60)


def add(track: np.ndarray, sound: np.ndarray, at: float, gain: float = 1.0) -> None:
    i = int(at * SR)
    end = min(len(track), i + len(sound))
    track[i:end] += gain * sound[: end - i]


def render(folder: Path) -> dict[str, Path]:
    """Cria mix.flac e os stems guitar/bass/drums.wav na pasta. Retorna os caminhos."""
    rng = np.random.default_rng(0)
    n_beats = len(MELODY) // 2
    total = START + n_beats * BEAT + 2.0
    guitar, bass, drums = (np.zeros(int(total * SR)) for _ in range(3))

    for i, midi in enumerate(MELODY):
        add(guitar, tone(midi, BEAT * 0.5), START + i * BEAT / 2, 0.25)
    for i, midi in enumerate(BASS[:n_beats]):
        add(bass, tone(midi, BEAT, harmonics=3), START + i * BEAT, 0.4)
    for beat in range(n_beats):
        at = START + beat * BEAT
        add(drums, drum("kick" if beat % 2 == 0 else "snare", rng), at, 0.8)
        # Crash no início de cada bloco de 8 batidas; chimbal em colcheias no resto.
        add(drums, drum("crash" if beat % 8 == 0 else "hihat", rng), at)
        add(drums, drum("hihat", rng), at + BEAT / 2)

    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, signal in (("guitar", guitar), ("bass", bass), ("drums", drums)):
        paths[name] = folder / f"{name}.wav"
        sf.write(paths[name], np.stack([signal, signal], axis=1).astype(np.float32), SR)
    mix = guitar + bass + drums
    mix /= np.abs(mix).max() * 1.1
    paths["mix"] = folder / "mix.flac"
    sf.write(paths["mix"], np.stack([mix, mix], axis=1).astype(np.float32), SR)
    return paths
