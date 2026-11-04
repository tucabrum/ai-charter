"""Modelo de sustain do baixo: medidas da curva de volume em batidas e a decisão nota longa x curta."""

import numpy as np

from ai_charter.analysis.pitched import DetectedNote, estimate_pitches
from ai_charter.analysis.tempo import TempoMap
from ai_charter.sustain_model import FEATURE_NAMES, SustainModel, note_features

RATE = 100.0   # quadros por segundo


def decay(seconds_held):
    """Volume que fica no pico por `seconds_held` e cai a quase nada."""
    env = np.full(200, 0.05, dtype=np.float32)
    env[: int(seconds_held * RATE)] = 1.0
    return env


def note(t, held):
    return DetectedNote(t, t + 0.1, 40.0, 1, 1.0, envelope=decay(held), envelope_rate=RATE)


def test_features_read_the_volume_in_beats():
    f = dict(zip(FEATURE_NAMES, note_features(decay(0.3), RATE, gap=1.0, beat=0.5)))
    assert f["espaco"] == 2.0
    assert f["volume_1_4"] == 1.0 and f["volume_4_4"] == np.float32(0.05)   # cai antes de 1 batida (0,5 s)
    assert abs(f["cai_0.5"] - 0.6) < 0.05                                    # 0,3 s = 0,6 batida


def test_apply_keeps_only_the_long_notes():
    # Rede de mentira: a nota é longa se o volume ainda está alto 1 batida depois do pico.
    w1 = np.zeros((len(FEATURE_NAMES), 1), dtype=np.float32)
    w1[FEATURE_NAMES.index("volume_4_4"), 0] = 40.0
    params = {"mean": [0.0] * len(FEATURE_NAMES), "std": [1.0] * len(FEATURE_NAMES), "w1": w1.tolist(),
              "b1": [0.0], "w2": [[1.0]], "b2": [-20.0], "threshold": 0.5, "features": FEATURE_NAMES}
    tempo = TempoMap(beat_times=np.arange(0, 10, 0.5), resolution=192)
    notes = [note(0.0, held=1.5), note(1.5, held=0.1), note(3.0, held=1.5)]
    SustainModel(params).apply(notes, tempo)
    assert notes[0].end == 1.5            # longa: vai até a próxima nota
    assert notes[1].end == notes[1].time  # curta: sem sustain


def test_max_voices_one_never_makes_a_chord():
    spectrum = np.zeros(48)
    spectrum[[5, 12]] = 1.0
    assert len(estimate_pitches(spectrum, 0.5)) == 2
    assert estimate_pitches(spectrum, 0.5, max_voices=1) == [5]
