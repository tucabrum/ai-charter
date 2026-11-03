"""Modelo de altura do baixo: escolhe a nota de cada ataque entre a ouvida e as vizinhas/oitavas.

O detector (analysis/pitched.py) acha a nota pela soma dos harmônicos, mas no baixo a fundamental soa
fraca e ele confunde a oitava ou cai numa nota vizinha. Para cada ataque, o modelo dá uma nota a cada
CANDIDATO (a nota ouvida deslocada por CANDIDATES semitons) e fica com a melhor. As medidas são
relativas ao candidato (a energia nos harmônicos dele, uma oitava abaixo, nas vizinhas), então o mesmo
modelo vale em qualquer altura. Treinado com o Pro Bass das charts oficiais (corda e casa de cada nota):
ai-charter-bench-bass collect / train-pitch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).with_name("pitch_model_bass.json")
CANDIDATES = (-24, -12, -7, -5, -2, -1, 0, 1, 2, 5, 7, 12)
# Onde olhar, em semitons a partir do candidato: abaixo (o candidato pode ser um harmônico), vizinhas
# e os harmônicos dele (2x = +12, 3x = +19, 4x = +24, 5x = +28, 6x = +31, 7x = +34, 8x = +36).
OFFSETS = (-24, -12, -7, -5, -2, -1, 0, 1, 2, 3, 4, 7, 12, 19, 24, 28, 31, 34, 36)
HARMONICS = ((0, 1.0), (12, 0.5), (19, 0.33), (24, 0.25), (28, 0.2))
FLOOR = -14.0   # log de energia onde não há bin (fora do espectro)

FEATURE_NAMES = ([f"ataque_{o:+d}" for o in OFFSETS] + [f"sustenta_{o:+d}" for o in OFFSETS]
                 + ["saliencia", "saliencia_rank", "altura", "vs_mediana", "desvio", "e_a_ouvida", "e_oitava",
                    "vs_anterior", "igual_anterior", "oitava_anterior", "vs_seguinte", "igual_seguinte",
                    "oitava_seguinte", "sem_vizinhas"])


def _log_spectrum(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    peak = float(s.max()) if len(s) and s.max() > 0 else 1.0
    return np.maximum(np.log(s / peak + 1e-6), FLOOR)


def _salience(s: np.ndarray) -> np.ndarray:
    out = np.zeros_like(s, dtype=np.float64)
    for offset, weight in HARMONICS:
        out[: len(s) - offset] += weight * s[offset:]
    return out


def note_features(notes) -> tuple[np.ndarray, np.ndarray]:
    """Medidas de cada candidato de cada nota: (x [notas * candidatos, atributos], bin de cada candidato
    [notas, candidatos], -1 = fora do espectro). As notas precisam do espectro (DetectedNote.spectrum)."""
    heard = np.array([round(n.pitch) - n.fmin_midi for n in notes], dtype=int)
    median = float(np.median(heard)) if len(heard) else 0.0
    rows, bins = [], np.full((len(notes), len(CANDIDATES)), -1, dtype=int)
    for i, n in enumerate(notes):
        early, late = _log_spectrum(n.spectrum), _log_spectrum(n.late_spectrum)
        size = len(early)
        sal = _salience(np.asarray(n.spectrum, dtype=np.float64))
        sal = sal / (sal.max() + 1e-12)
        order = np.argsort(-sal)
        rank = np.empty(size); rank[order] = np.arange(size)
        prev = heard[i - 1] if i > 0 else None
        nxt = heard[i + 1] if i + 1 < len(notes) else None
        for k, d in enumerate(CANDIDATES):
            c = heard[i] + d
            if not 0 <= c < size:
                rows.append(np.zeros(len(FEATURE_NAMES), np.float32))
                continue
            bins[i, k] = c
            look = lambda spec: [spec[c + o] if 0 <= c + o < size else FLOOR for o in OFFSETS]  # noqa: E731

            def near(other):
                if other is None:
                    return [0.0, 0.0, 0.0]
                diff = c - other
                return [diff / 12.0, float(diff == 0), float(diff != 0 and diff % 12 == 0)]

            rows.append(np.array(look(early) + look(late)
                                 + [np.log(sal[c] + 1e-6), min(rank[c], 12) / 12.0, c / 48.0, (c - median) / 12.0,
                                    d / 12.0, float(d == 0), float(abs(d) in (12, 24))]
                                 + near(prev) + near(nxt) + [float(prev is None or nxt is None)],
                                 dtype=np.float32))
    return np.array(rows, dtype=np.float32).reshape(len(notes) * len(CANDIDATES), len(FEATURE_NAMES)), bins


class PitchModel:
    def __init__(self, params: dict):
        from ai_charter.drum_model import _arrays

        self.params = _arrays(params)
        self.info = params.get("info", {})

    @classmethod
    def load(cls, path: Path | None = None) -> "PitchModel | None":
        path = path or MODEL_PATH
        if not path.is_file():
            return None
        params = json.loads(path.read_text(encoding="utf-8"))
        if params.get("features") != FEATURE_NAMES or tuple(params.get("candidates", ())) != CANDIDATES:
            return None   # modelo de outra versão das medidas
        return cls(params)

    def scores(self, notes) -> tuple[np.ndarray, np.ndarray]:
        from ai_charter.drum_model import mlp

        x, bins = note_features(notes)
        p = mlp(self.params, x)[:, 0].reshape(len(notes), len(CANDIDATES))
        return np.where(bins >= 0, p, -1.0), bins

    def rescore(self, notes) -> None:
        """Troca a altura de cada nota pelo candidato mais provável."""
        if not notes or any(n.spectrum is None for n in notes):
            return
        p, bins = self.scores(notes)
        best = np.argmax(p, axis=1)
        for n, k, row in zip(notes, best, bins):
            n.pitch = float(n.fmin_midi + row[k])


def save(params: dict, info: dict, path: Path = MODEL_PATH) -> None:
    from ai_charter.drum_model import _jsonable

    out = _jsonable(params)
    out.update(features=FEATURE_NAMES, candidates=list(CANDIDATES), info=info)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
