"""Hard, Medium e Easy da bateria aprendidos das reduções da Harmonix.

Para cada nota do Expert, uma rede pequena dá a chance de ela ficar em cada dificuldade, olhando a
peça, o lugar no compasso, o que toca junto e em volta, quantas notas o compasso tem e o andamento.
O modelo fica em drum_reduce.json, ao lado de drum_model.json (ai-charter-bench train-reduce); sem
ele, valem as regras de [drums.hard/medium/easy].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ai_charter.analysis.drums import CRASH, HIHAT, KICK, RIDE, SNARE, TOM_HIGH, TOM_LOW, TOM_MID

MODEL_PATH = Path(__file__).resolve().parent.parent / "drum_reduce.json"
PIECES = (KICK, SNARE, HIHAT, TOM_HIGH, RIDE, TOM_MID, CRASH, TOM_LOW)
DIFFICULTIES = ("hard", "medium", "easy")
_NEIGHBORS = (("antes_1/16", -1), ("depois_1/16", 1), ("antes_1/8", -2), ("depois_1/8", 2))
FEATURE_NAMES = ([f"peca_{p}" for p in PIECES] + [f"posicao_{i}" for i in range(16)] + ["fora_da_grade"]
                 + [f"junto_{p}" for p in PIECES]
                 + [f"{side}_{p}" for side, _ in _NEIGHBORS for p in PIECES]
                 + ["notas_no_compasso", "notas_da_peca_no_compasso", "bpm"])


def features(notes: list[tuple[int, str]], resolution: int, bpm_at) -> np.ndarray:
    """Uma linha por nota (tick, peça) do Expert de UMA música. bpm_at(tick) -> andamento ali."""
    step = resolution / 4
    at: dict[int, set[str]] = {}
    for tick, piece in notes:
        at.setdefault(tick, set()).add(piece)
    bar_of = lambda t: int(t // (4 * resolution))   # noqa: E731
    per_bar: dict[int, int] = {}
    per_bar_piece: dict[tuple[int, str], int] = {}
    for tick, piece in notes:
        per_bar[bar_of(tick)] = per_bar.get(bar_of(tick), 0) + 1
        per_bar_piece[(bar_of(tick), piece)] = per_bar_piece.get((bar_of(tick), piece), 0) + 1
    out = np.zeros((len(notes), len(FEATURE_NAMES)), dtype=np.float32)
    for i, (tick, piece) in enumerate(notes):
        row = out[i]
        row[PIECES.index(piece)] = 1.0
        q = tick / step
        if abs(q - round(q)) < 0.25:
            row[8 + int(round(q)) % 16] = 1.0
        else:
            row[24] = 1.0
        for p in at[tick]:
            if p != piece:
                row[25 + PIECES.index(p)] = 1.0
        for k, (_, d) in enumerate(_NEIGHBORS):
            for p in at.get(int(round(tick + d * step)), ()):
                row[33 + 8 * k + PIECES.index(p)] = 1.0
        row[65] = per_bar[bar_of(tick)] / 16.0
        row[66] = per_bar_piece[(bar_of(tick), piece)] / 16.0
        row[67] = bpm_at(tick) / 200.0
    return out


class ReduceModel:
    def __init__(self, params: dict):
        from ai_charter.drum_model import _arrays

        self.params = _arrays(params)
        self.thresholds = dict(zip(DIFFICULTIES, params["thresholds"]))

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "ReduceModel | None":
        if not path.is_file():
            return None
        params = json.loads(path.read_text(encoding="utf-8"))
        if params.get("features") != FEATURE_NAMES or params.get("difficulties") != list(DIFFICULTIES):
            return None
        return cls(params)

    def keep(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Para cada dificuldade, quais notas do Expert ficam."""
        from ai_charter.drum_model import mlp

        p = mlp(self.params, x)
        return {d: p[:, k] >= self.thresholds[d] for k, d in enumerate(DIFFICULTIES)}


def bpm_from_tempo(tempo):
    """bpm_at(tick) a partir de um TempoMap."""
    periods = np.diff(np.asarray(tempo.beat_times))
    return lambda tick: 60.0 / float(periods[min(max(int(tick // tempo.resolution), 0), len(periods) - 1)])
