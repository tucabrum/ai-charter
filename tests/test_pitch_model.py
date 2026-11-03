"""Modelo de altura do baixo: medidas de cada candidato e a troca pela nota mais provável."""

import numpy as np

from ai_charter.analysis.pitched import DetectedNote
from ai_charter.pitch_model import CANDIDATES, FEATURE_NAMES, PitchModel, note_features


def note(t, pitch, loud_bins):
    spectrum = np.full(48, 1e-4, dtype=np.float32)
    for b, e in loud_bins.items():
        spectrum[b] = e
    return DetectedNote(t, t + 0.2, float(28 + pitch), 1, 1.0, spectrum=spectrum, late_spectrum=spectrum,
                        fmin_midi=28)


def test_each_note_gets_one_row_per_candidate_inside_the_spectrum():
    notes = [note(0.0, 24, {12: 0.5, 24: 1.0}), note(0.5, 2, {2: 1.0})]
    x, bins = note_features(notes)
    assert x.shape == (2 * len(CANDIDATES), len(FEATURE_NAMES))
    assert list(bins[0]) == [24 + d for d in CANDIDATES]
    assert bins[1][CANDIDATES.index(-12)] == -1 and bins[1][CANDIDATES.index(0)] == 2   # abaixo do Mi1: fora


def test_rescore_moves_each_note_to_the_best_candidate():
    # Rede de mentira: a nota do candidato é a energia dele logo depois do ataque.
    w1 = np.zeros((len(FEATURE_NAMES), 1), dtype=np.float32)
    w1[FEATURE_NAMES.index("ataque_+0"), 0] = 1.0
    params = {"mean": [0.0] * len(FEATURE_NAMES), "std": [1.0] * len(FEATURE_NAMES), "w1": w1.tolist(),
              "b1": [20.0], "w2": [[1.0]], "b2": [-20.0], "features": FEATURE_NAMES, "candidates": list(CANDIDATES)}
    notes = [note(0.0, 24, {12: 1.0, 24: 0.6}), note(0.5, 24, {24: 1.0, 12: 0.01})]
    PitchModel(params).rescore(notes)
    assert [n.pitch for n in notes] == [28 + 12, 28 + 24]
