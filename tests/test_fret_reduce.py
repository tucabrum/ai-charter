"""Hard/Medium/Easy do baixo pelo modelo de redução: medidas de cada nota do Expert e o uso no build_fret_track."""

import numpy as np

from ai_charter.charting.common import Candidate
from ai_charter.charting.fret_reduce import DIFFICULTIES, FEATURE_NAMES, ReduceModel, features
from ai_charter.charting.frets import build_fret_track
from ai_charter.config import load_rules

RES = 192


def fake_model(feature: str, thresholds=(0.5, 0.5, 0.5)) -> ReduceModel:
    """Rede de mentira: a nota fica (nas três dificuldades) se `feature` vale 1."""
    w1 = np.zeros((len(FEATURE_NAMES), 1), dtype=np.float32)
    w1[FEATURE_NAMES.index(feature), 0] = 40.0
    return ReduceModel({"mean": [0.0] * len(FEATURE_NAMES), "std": [1.0] * len(FEATURE_NAMES), "w1": w1.tolist(),
                        "b1": [0.0], "w2": [[1.0] * 3], "b2": [-20.0] * 3, "thresholds": list(thresholds)})


def test_features_read_the_bar_position_and_the_neighbours():
    notes = [(0, (0,), 0), (RES // 4, (0,), 0), (RES, (2,), RES * 2)]
    f = [dict(zip(FEATURE_NAMES, row)) for row in features(notes, RES, lambda _t: 120.0)]
    assert f[0]["posicao_0"] == 1 and f[1]["posicao_1"] == 1 and f[2]["posicao_4"] == 1
    assert f[1]["mesma_cor_anterior"] == 1 and f[2]["sobe_anterior"] == 1
    assert abs(f[2]["batidas_anterior"] - 0.75) < 1e-6 and f[2]["sustain"] == 0.5
    assert f[0]["notas_no_compasso"] == 3 / 16


def test_an_easier_difficulty_only_keeps_notes_of_the_harder_one():
    model = fake_model("posicao_0")
    p = np.array([[0.9, 0.2, 0.9], [0.9, 0.9, 0.9]], dtype=np.float32)
    model.thresholds = dict(zip(DIFFICULTIES, (0.5, 0.5, 0.5)))
    import ai_charter.drum_model as dm
    real = dm.mlp
    dm.mlp = lambda params, x: p
    try:
        keep = model.keep(np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32))
    finally:
        dm.mlp = real
    assert list(keep["medium"]) == [False, True] and list(keep["easy"]) == [False, True]


def test_build_fret_track_takes_the_lower_difficulties_from_the_model():
    cands = [Candidate(i * RES // 2, 1.0, i * RES // 2 + 10, 40.0 + (i % 3)) for i in range(32)]
    track = build_fret_track(cands, load_rules().frets, RES, 8, [], reducer=fake_model("posicao_0"))
    on_beat = [c.tick for c in cands if c.tick % (4 * RES) == 0]   # só o 1º tempo de cada compasso
    for d in DIFFICULTIES:
        assert [n.tick for n in track.notes[d]] == on_beat
