"""Modelo de sustain do baixo: decide se cada nota é longa na chart.

O detector (analysis/pitched.py) acaba a nota quando o volume cai abaixo de release_ratio × pico, mas o
baixo decai devagar e quase toda nota com espaço até a próxima virava sustain (26% das notas, contra 14%
na Harmonix, e só 44% delas certas). O modelo olha a curva de volume depois do pico, medida em batidas
(como a Harmonix decide: uma nota segurada por pelo menos 1/3 de batida), e o espaço até a próxima nota.
Treinado com as charts oficiais: ai-charter-bench-bass collect-sustain / train-sustain.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).with_name("sustain_model_bass.json")
QUARTERS = (1, 2, 3, 4, 6, 8, 12, 16)   # onde ler o volume: em 1/4 de batida depois do pico
DROPS = (0.2, 0.35, 0.5)                # quando o volume cai abaixo destas frações do pico

FEATURE_NAMES = (["log_espaco", "espaco"] + [f"volume_{q}_4" for q in QUARTERS] + ["volume_final", "volume_min"]
                 + [f"cai_{d}" for d in DROPS])


def note_features(env: np.ndarray, rate: float, gap: float, beat: float) -> list[float]:
    """`env`: volume do pico até a próxima nota, relativo ao pico; `rate`: quadros por segundo;
    `gap`: segundos até a próxima nota; `beat`: segundos de uma batida ali."""
    env = np.asarray(env, dtype=np.float32)
    gap_beats = gap / beat
    out = [float(np.log(gap_beats + 1e-3)), min(gap_beats, 8.0)]
    for q in QUARTERS:
        f = int(q * beat / 4 * rate)
        out.append(float(env[f]) if f < len(env) else -1.0)
    out += [float(env[-4:].mean()) if len(env) else 0.0, float(env.min()) if len(env) else 0.0]
    for d in DROPS:
        below = np.nonzero(env < d)[0]
        out.append(min((below[0] if len(below) else len(env)) / rate / beat, 8.0))
    return out


def song_features(notes, tempo) -> np.ndarray:
    """Medidas de cada nota detectada (DetectedNote com envelope), no tempo da música."""
    rows = []
    for i, n in enumerate(notes):
        nxt = notes[i + 1].time if i + 1 < len(notes) else n.time + len(n.envelope) / n.envelope_rate
        tick = tempo.time_to_tick(n.time)
        beat = float(tempo.tick_to_time(tick + tempo.resolution) - tempo.tick_to_time(tick))
        rows.append(note_features(n.envelope, n.envelope_rate, nxt - n.time, max(beat, 1e-3)))
    return np.array(rows, dtype=np.float32).reshape(len(notes), len(FEATURE_NAMES))


class SustainModel:
    def __init__(self, params: dict):
        from ai_charter.drum_model import _arrays

        self.params = _arrays(params)
        self.info = params.get("info", {})
        self.threshold = float(params.get("threshold", 0.5))

    @classmethod
    def load(cls, path: Path | None = None) -> "SustainModel | None":
        path = path or MODEL_PATH
        if not path.is_file():
            return None
        params = json.loads(path.read_text(encoding="utf-8"))
        if params.get("features") != FEATURE_NAMES:
            return None   # modelo de outra versão das medidas
        return cls(params)

    def probabilities(self, notes, tempo) -> np.ndarray:
        from ai_charter.drum_model import mlp

        return mlp(self.params, song_features(notes, tempo))[:, 0]

    def apply(self, notes, tempo) -> None:
        """Nota curta acaba no ataque (sem sustain); nota longa vai até a próxima (a chart corta antes)."""
        if not notes or any(n.envelope is None for n in notes):
            return
        for i, (n, p) in enumerate(zip(notes, self.probabilities(notes, tempo))):
            if p < self.threshold:
                n.end = n.time
            elif i + 1 < len(notes):
                n.end = max(n.end, notes[i + 1].time)


def save(params: dict, threshold: float, info: dict, path: Path = MODEL_PATH) -> None:
    from ai_charter.drum_model import _jsonable

    out = _jsonable(params)
    out.update(features=FEATURE_NAMES, threshold=threshold, info=info)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
