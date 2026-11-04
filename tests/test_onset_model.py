"""Modelo de ataques do baixo: medidas dos picos candidatos e a escolha dos que são notas."""

import numpy as np

from ai_charter.analysis.tempo import TempoMap
from ai_charter.onset_model import FEATURE_NAMES, OnsetModel, features

SR, HOP = 22050, 256
FPS = SR / HOP


def test_features_mark_the_grid_and_the_neighbours_in_beats():
    tempo = TempoMap(beat_times=np.arange(0, 20, 0.5), resolution=192)   # 120 bpm: 0,5 s por batida
    n = 600
    env = np.full(n, 0.01, dtype=np.float32)
    rms = np.full(n, 0.01, dtype=np.float32)
    cqt = np.full((48, n), 1e-3, dtype=np.float32)
    times = np.array([1.0, 1.125, 2.0])        # na batida, na semicolcheia, na batida
    frames = np.round(times * FPS).astype(int)
    env[frames] = 1.0
    x = features(frames, {int(frames[0])}, env, rms, cqt, SR, HOP, times, tempo)
    f = [dict(zip(FEATURE_NAMES, row)) for row in x]
    assert x.shape == (3, len(FEATURE_NAMES))
    assert f[0]["grade_16"] == 0 and f[0]["limiar_atual"] == 1 and f[1]["limiar_atual"] == 0
    assert abs(f[1]["grade_8"] - 0.25) < 0.02            # semicolcheia: 1/4 de batida da colcheia
    assert abs(f[1]["batidas_seguinte"] - 1.75) < 0.05   # 0,875 s até o próximo pico


def test_select_keeps_the_peaks_the_model_likes():
    w1 = np.zeros((len(FEATURE_NAMES), 1), dtype=np.float32)
    w1[FEATURE_NAMES.index("limiar_atual"), 0] = 40.0
    params = {"mean": [0.0] * len(FEATURE_NAMES), "std": [1.0] * len(FEATURE_NAMES), "w1": w1.tolist(),
              "b1": [0.0], "w2": [[1.0]], "b2": [-20.0], "threshold": 0.5, "features": FEATURE_NAMES}
    x = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32)
    x[[0, 2], FEATURE_NAMES.index("limiar_atual")] = 1.0
    assert list(OnsetModel(params).select(np.array([10, 20, 30]), x)) == [10, 30]
