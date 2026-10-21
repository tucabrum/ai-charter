"""Divisão da música em seções (para o modo prática) e trecho de preview."""

from __future__ import annotations

import librosa
import numpy as np

from ai_charter.analysis.tempo import TempoMap


def find_sections(y: np.ndarray, sr: int, tempo: TempoMap, target_seconds: float) -> list[int]:
    """Retorna os ticks onde cada seção começa (sempre no início de um compasso).

    Agrupa compassos com harmonia/timbre parecidos usando clustering.
    """
    duration = len(y) / sr
    n_sections = int(np.clip(round(duration / target_seconds), 2, 16))
    measure_ticks = tempo.resolution * 4
    n_measures = int(tempo.time_to_tick(duration) // measure_ticks)
    if n_measures < n_sections * 2:
        return [0]

    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13)
    features = np.vstack([chroma, librosa.util.normalize(mfcc, axis=1)])

    # Uma coluna de features por compasso (média dos frames do compasso).
    starts = tempo.tick_to_time(np.arange(n_measures + 1) * measure_ticks)
    frames = librosa.time_to_frames(starts, sr=sr, hop_length=hop)
    frames = np.clip(frames, 0, features.shape[1] - 1)
    per_measure = np.stack(
        [features[:, a:max(b, a + 1)].mean(axis=1) for a, b in zip(frames[:-1], frames[1:])], axis=1
    )

    bounds = librosa.segment.agglomerative(per_measure, n_sections)
    return sorted({int(m) * measure_ticks for m in bounds} | {0})


def find_preview_start(y: np.ndarray, sr: int, window_seconds: float = 30.0) -> float:
    """Início (s) do trecho de 30 s mais "cheio" da música, usado como preview."""
    rms = librosa.feature.rms(y=y, hop_length=sr // 4)[0]  # 4 valores por segundo
    window = int(window_seconds * 4)
    if len(rms) <= window:
        return 0.0
    totals = np.convolve(rms, np.ones(window), mode="valid")
    # Evita o comecinho e o final da música.
    lo, hi = len(totals) // 10, len(totals) - len(totals) // 10
    return float((lo + np.argmax(totals[lo:hi])) / 4)
