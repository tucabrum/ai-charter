"""Hard, Medium e Easy do baixo aprendidos das reduções da Harmonix.

Para cada nota do Expert, uma rede pequena dá a chance de ela ficar em cada dificuldade, olhando o lugar
no compasso, a distância às notas vizinhas, quantas notas o compasso tem, o andamento, as lanes do Expert
em volta (a mesma cor, quanto sobe ou desce), o sustain e se é acorde. Sem o modelo, o Medium e o Easy
saíam com 25-35% de notas a mais que os da Harmonix. O modelo fica em fret_reduce_bass.json
(ai-charter-bench-bass train-reduce, só os MIDIs oficiais); sem ele, valem as regras de [frets.*].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "fret_reduce_bass.json"
DIFFICULTIES = ("hard", "medium", "easy")
OPEN = 7   # nota aberta no .chart: fica abaixo do verde

FEATURE_NAMES = ([f"posicao_{i}" for i in range(16)] + ["fora_da_grade"]
                 + ["batidas_anterior", "batidas_seguinte", "batidas_anterior_2", "batidas_seguinte_2",
                    "notas_no_compasso", "notas_na_batida", "bpm",
                    "mesma_cor_anterior", "mesma_cor_seguinte", "salto_anterior", "salto_seguinte",
                    "sobe_anterior", "sobe_seguinte", "sustain", "acorde", "depois_de_pausa"])


def _low(lanes) -> int:
    return min(-1 if lane == OPEN else lane for lane in lanes)


def features(notes: list[tuple[int, tuple[int, ...], int]], resolution: int, bpm_at) -> np.ndarray:
    """Uma linha por nota (tick, lanes, duração em ticks) do Expert de UMA música, em ordem.
    bpm_at(tick) -> andamento ali."""
    step = resolution / 4
    ticks = np.array([t for t, _, _ in notes], dtype=float)
    lanes = [_low(l) for _, l, _ in notes]
    bar = (ticks // (4 * resolution)).astype(int)
    beat = (ticks // resolution).astype(int)
    per_bar = {b: int(np.sum(bar == b)) for b in set(bar)}
    per_beat = {b: int(np.sum(beat == b)) for b in set(beat)}
    out = np.zeros((len(notes), len(FEATURE_NAMES)), dtype=np.float32)

    def gap(i, j):
        return min(abs(ticks[j] - ticks[i]) / resolution, 4.0) if 0 <= j < len(notes) else 4.0

    for i, (tick, chord, length) in enumerate(notes):
        row = out[i]
        q = tick / step
        if abs(q - round(q)) < 0.25:
            row[int(round(q)) % 16] = 1.0
        else:
            row[16] = 1.0
        row[17:21] = [gap(i, i - 1), gap(i, i + 1), gap(i, i - 2), gap(i, i + 2)]
        row[21] = per_bar[bar[i]] / 16.0
        row[22] = per_beat[beat[i]] / 4.0
        row[23] = bpm_at(tick) / 200.0
        for k, j in ((0, i - 1), (1, i + 1)):
            if 0 <= j < len(notes):
                d = lanes[i] - lanes[j]
                row[24 + k] = float(d == 0)
                row[26 + k] = abs(d) / 4.0
                row[28 + k] = float(np.sign(d))
        row[30] = min(length / resolution, 4.0) / 4.0
        row[31] = float(len(chord) > 1)
        row[32] = float(gap(i, i - 1) >= 1.0)
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
        """Para cada dificuldade, quais notas do Expert ficam. Uma nota que sai de uma dificuldade sai
        também das mais fáceis (a Harmonix quase nunca põe no Easy uma nota que tirou do Medium)."""
        from ai_charter.drum_model import mlp

        p = mlp(self.params, x)
        out, above = {}, np.ones(len(x), dtype=bool)
        for k, d in enumerate(DIFFICULTIES):
            above = above & (p[:, k] >= self.thresholds[d])
            out[d] = above
        return out
