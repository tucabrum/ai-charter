"""Modelo de ataques do baixo: escolhe, entre os picos do envelope de ataques, os que são notas.

O detector (analysis/pitched.py) fica com os picos acima de um limiar (onset_delta). No baixo separado
pelo Demucs o ataque sai amaciado: 91% das notas oficiais aparecem como pico do envelope, mas só 71%
passavam do limiar. Baixar o limiar não resolve (para achar 80%, metade das notas seriam falsas). O modelo
olha cada pico sem filtro: a força dele e a das vizinhas, a subida do volume, a mudança do espectro e da
nota, a distância aos picos vizinhos e onde ele cai na grade do tempo (semicolcheia, colcheia, tercina).
Treinado com os stems oficiais e as charts da Harmonix: ai-charter-bench-bass collect-onsets / train-onsets.
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np

MODEL_PATH = Path(__file__).with_name("onset_model_bass.json")
CANDIDATE_DELTA = 0.005   # limiar dos picos candidatos: quase todo pico do envelope
CANDIDATE_WAIT = 0.03     # dois candidatos a pelo menos 30 ms um do outro

FEATURE_NAMES = ["forca", "forca_vs_mediana", "forca_vs_maior", "subida_volume", "volume_db", "volume_vs_mediana",
                 "muda_espectro", "muda_grave", "mesma_nota", "salto_nota", "forca_anterior", "forca_seguinte",
                 "limiar_atual", "batidas_anterior", "batidas_seguinte", "grade_16", "grade_tercina", "grade_8"]


def candidates(onset_env: np.ndarray, sr: int, hop: int, delta: float) -> tuple[np.ndarray, set[int]]:
    """Os picos candidatos e, entre eles, os que o limiar atual (`delta`) já aceitava."""
    frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop, delta=CANDIDATE_DELTA,
                                        wait=int(CANDIDATE_WAIT * sr / hop), units="frames")
    kept = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop, delta=delta,
                                      wait=int(0.06 * sr / hop), units="frames")
    return frames, {int(f) for f in kept}


def features(frames: np.ndarray, kept: set[int], onset_env: np.ndarray, rms: np.ndarray, cqt: np.ndarray,
             sr: int, hop: int, times: np.ndarray, tempo) -> np.ndarray:
    """Medidas de cada pico candidato. `times`: o instante de cada um (s); `tempo`: o mapa de tempo."""
    from ai_charter.analysis.pitched import _harmonic_salience

    n = min(cqt.shape[1], len(rms), len(onset_env))
    fps = sr / hop
    win = int(fps)   # 1 s de contexto para cada lado
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    scale = np.percentile(onset_env[frames], 95) + 1e-9 if len(frames) else 1.0
    rows = []
    for k, f in enumerate(frames):
        prv = frames[k - 1] if k else max(0, f - win)
        nxt = frames[k + 1] if k + 1 < len(frames) else min(n, f + win)
        lo, hi = max(0, f - win), min(n, f + win)
        pk = f + int(np.argmax(rms[f: max(f + 1, min(f + 6, nxt))]))
        before_rms = rms[max(f - 4, 0): max(f - 1, 1)].mean()
        if f - 1 > prv:
            a = cqt[:, max(prv, f - 10): f - 1].mean(axis=1)
        else:
            a = cqt[:, max(0, f - 3): f + 1].mean(axis=1)
        b = cqt[:, min(f + 2, n - 1): max(f + 3, min(f + 12, nxt))].mean(axis=1)
        na, nb = a / (np.linalg.norm(a) + 1e-12), b / (np.linalg.norm(b) + 1e-12)
        pa, pb = int(np.argmax(_harmonic_salience(a))), int(np.argmax(_harmonic_salience(b)))
        tick = tempo.time_to_tick(times[k])
        beat_index = np.floor(tick / tempo.resolution)
        beat = max(float(tempo.tick_to_time((beat_index + 1) * tempo.resolution)
                         - tempo.tick_to_time(beat_index * tempo.resolution)), 1e-3)
        frac = (tick / tempo.resolution) % 1.0
        rows.append([onset_env[f] / scale, onset_env[f] / (np.median(onset_env[lo:hi]) + 1e-9),
                     onset_env[f] / (onset_env[lo:hi].max() + 1e-9),
                     np.log(rms[pk] / (before_rms + 1e-12) + 1e-6), rms_db[pk], rms_db[pk] - np.median(rms_db[lo:hi]),
                     1 - float(na @ nb), np.log((b[:24].sum() + 1e-9) / (a[:24].sum() + 1e-9)), float(pa == pb),
                     min(abs(pa - pb), 24) / 12, onset_env[prv] / scale, onset_env[min(nxt, n - 1)] / scale,
                     float(int(f) in kept), min((f - prv) / fps / beat, 4.0), min((nxt - f) / fps / beat, 4.0),
                     abs(frac * 4 - round(frac * 4)) / 4, abs(frac * 3 - round(frac * 3)) / 3,
                     abs(frac * 2 - round(frac * 2)) / 2])
    return np.array(rows, dtype=np.float32).reshape(len(frames), len(FEATURE_NAMES))


class OnsetModel:
    def __init__(self, params: dict):
        from ai_charter.drum_model import _arrays

        self.params = _arrays(params)
        self.info = params.get("info", {})
        self.threshold = float(params.get("threshold", 0.5))

    @classmethod
    def load(cls, path: Path | None = None) -> "OnsetModel | None":
        path = path or MODEL_PATH
        if not path.is_file():
            return None
        params = json.loads(path.read_text(encoding="utf-8"))
        if params.get("features") != FEATURE_NAMES:
            return None   # modelo de outra versão das medidas
        return cls(params)

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        from ai_charter.drum_model import mlp

        return mlp(self.params, x)[:, 0] if len(x) else np.zeros(0, dtype=np.float32)

    def select(self, frames: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Os picos candidatos que o modelo acha que são notas."""
        return frames[self.probabilities(x) >= self.threshold]


def save(params: dict, threshold: float, info: dict, path: Path = MODEL_PATH) -> None:
    from ai_charter.drum_model import _jsonable

    out = _jsonable(params)
    out.update(features=FEATURE_NAMES, threshold=threshold, info=info)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
