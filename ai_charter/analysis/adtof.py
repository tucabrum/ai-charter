"""Transcrição de bateria do ADTOF (Zehren et al., ISMIR 2021), se instalado.

O ADTOF foi treinado com milhares de charts de bateria de jogos de ritmo, sobre o mix completo. Ele
dá, a cada 10 ms, a chance de bumbo, caixa, tom, chimbal e prato (5 classes). O ai-charter usa essas
chances como medidas extras do modelo de bateria (drum_model.py): o ADTOF é forte em achar o golpe e
a família da peça; o retrato acústico do kit decide qual tom e qual prato.

Instalação (opcional; uso não comercial, licença CC BY-NC-SA do ADTOF):
  pip install "git+https://github.com/xavriley/ADTOF-pytorch"
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

FPS = 100
CLASSES = ("bumbo", "caixa", "tom", "chimbal", "prato")   # ordem das saídas do ADTOF (35, 38, 47, 42, 49)
SR = 44100

_MODEL = None


def available() -> bool:
    try:
        import adtof_pytorch  # noqa: F401
    except ImportError:
        return False
    return True


def _model():
    global _MODEL
    if _MODEL is None:
        import torch
        from adtof_pytorch import (calculate_n_bins, create_frame_rnn_model, get_default_weights_path,
                                   load_pytorch_weights)

        weights = get_default_weights_path()
        torch.load(weights, map_location="cpu", weights_only=True)   # confere: só tensores, nada executável
        _MODEL = load_pytorch_weights(create_frame_rnn_model(calculate_n_bins()), weights).eval()
    return _MODEL


def activations(audio: np.ndarray, sr: int) -> np.ndarray | None:
    """(quadros, 5) a 100 quadros/s para o áudio (mono ou (2, n)), ou None sem o ADTOF instalado."""
    if not available():
        return None
    import soundfile as sf
    import torch
    from adtof_pytorch import load_audio_for_model

    data = audio.T if audio.ndim == 2 else audio
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mix.wav"
        sf.write(path, data, sr)
        with torch.no_grad():
            return _model()(load_audio_for_model(str(path))).numpy()[0].astype(np.float32)


def at_times(act: np.ndarray, times: np.ndarray, radius: int = 3) -> np.ndarray:
    """A maior chance de cada classe até `radius` quadros (30 ms) em volta de cada instante."""
    out = np.zeros((len(times), act.shape[1]), dtype=np.float32)
    if not len(act):
        return out
    frames = np.clip(np.round(np.asarray(times) * FPS).astype(int), 0, len(act) - 1)
    for d in range(-radius, radius + 1):
        out = np.maximum(out, act[np.clip(frames + d, 0, len(act) - 1)])
    return out


def hit_times(act: np.ndarray) -> np.ndarray:
    """Os instantes (s) em que o ADTOF ouve um golpe de qualquer peça, com os limiares dele."""
    from adtof_pytorch import FRAME_RNN_THRESHOLDS, LABELS_5, PeakPicker

    picked = PeakPicker(thresholds=FRAME_RNN_THRESHOLDS, fps=FPS).pick(act[None], labels=LABELS_5)[0]
    return np.unique(np.concatenate([np.asarray(t, dtype=float) for t in picked.values()] + [np.zeros(0)]))
