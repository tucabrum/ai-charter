"""Modelo de bateria aprendido com charts oficiais.

Para cada ataque, o transcritor mede um retrato acústico (analysis/drums.py: FEATURE_NAMES). Junto
com o lugar do ataque na levada (posição no compasso e vizinhos), uma rede neural pequena dá a
probabilidade de cada peça. Ela foi treinada com charts oficiais de Rock Band (ai-charter-bench
train-drums): é assim que o charting "sabe" que caixa costuma cair no 2 e no 4, que quase toda
colcheia tem um prato marcando o tempo e que tom no meio de um groove é raro.

Qual tom é qual (agudo, médio, surdo) o modelo não decide: a afinação muda de kit para kit. Ele diz
só "é tom", e o tom vem dos sons do kit da própria música, ordenados pela altura.

O modelo fica em drum_model.json, ao lado deste arquivo. Sem ele, valem só as regras do transcritor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ai_charter.analysis.drums import CRASH, FEATURE_NAMES, HIHAT, KICK, RIDE, SNARE, TOM

MODEL_PATH = Path(__file__).with_name("drum_model.json")
OUTPUTS = (KICK, SNARE, HIHAT, RIDE, CRASH, TOM)
CONTEXT_NAMES = [f"posicao_{i}" for i in range(16)] + ["fora_da_grade", "vizinho_1/16", "vizinho_1/8",
                                                       "batida_forte", "compasso_denso"]


INPUT_NAMES = FEATURE_NAMES + [f"rel_{n}" for n in FEATURE_NAMES] + CONTEXT_NAMES


def song_relative(features: np.ndarray) -> np.ndarray:
    """Cada medida comparada ao resto da mesma música (mediana e faixa entre os 10% e 90%):
    é o que deixa o modelo reconhecer um kit, uma mixagem ou um vazamento que ele nunca viu."""
    if not len(features):
        return features
    lo, mid, hi = np.percentile(features, [10, 50, 90], axis=0)
    return ((features - mid) / (hi - lo + 1e-6)).astype(np.float32)


def model_inputs(features: np.ndarray, ticks: np.ndarray, resolution: int) -> np.ndarray:
    """Entrada do modelo para os ataques de UMA música: retrato, retrato relativo e lugar na levada."""
    return np.hstack([features, song_relative(features), context_features(ticks * 192 / resolution, 192)])


def context_features(ticks: np.ndarray, resolution: int) -> np.ndarray:
    """Onde cada ataque cai na levada (compasso 4/4)."""
    step = resolution / 4
    q = ticks / step
    on_grid = np.abs(q - np.round(q)) < 0.25
    pos = np.round(q).astype(int) % 16
    out = np.zeros((len(ticks), len(CONTEXT_NAMES)), dtype=np.float32)
    out[np.arange(len(ticks))[on_grid], pos[on_grid]] = 1.0
    out[:, 16] = ~on_grid
    # Distância até o ataque vizinho de cada lado
    order = np.sort(ticks)
    left_i, right_i = np.searchsorted(order, ticks, "left"), np.searchsorted(order, ticks, "right")
    prev_gap = np.where(left_i > 0, ticks - order[np.maximum(left_i - 1, 0)], np.inf)
    next_gap = np.where(right_i < len(order), order[np.minimum(right_i, len(order) - 1)] - ticks, np.inf)
    for col, dist in ((17, step), (18, 2 * step)):
        out[:, col] = (np.abs(prev_gap - dist) < 0.3 * step) | (np.abs(next_gap - dist) < 0.3 * step)
    out[:, 19] = on_grid & (pos % 4 == 0)
    bar = (ticks // (resolution * 4)).astype(int)
    counts = np.bincount(bar - bar.min()) if len(bar) else np.zeros(1)
    out[:, 20] = np.minimum(counts[bar - bar.min()] / 16.0, 2.0) if len(bar) else 0.0
    return out


class DrumModel:
    """Rede de uma camada oculta: entrada (retrato + contexto) -> probabilidade de cada peça."""

    def __init__(self, params: dict):
        self.mean = np.array(params["mean"], dtype=np.float32)
        self.std = np.array(params["std"], dtype=np.float32)
        self.w1 = np.array(params["w1"], dtype=np.float32)
        self.b1 = np.array(params["b1"], dtype=np.float32)
        self.w2 = np.array(params["w2"], dtype=np.float32)
        self.b2 = np.array(params["b2"], dtype=np.float32)
        self.thresholds = np.array(params["thresholds"], dtype=np.float32)
        self.info = params.get("info", {})

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "DrumModel | None":
        if not path.is_file():
            return None
        params = json.loads(path.read_text(encoding="utf-8"))
        if params.get("features") != INPUT_NAMES or params.get("outputs") != list(OUTPUTS):
            return None   # modelo de outra versão do retrato: melhor não usar
        return cls(params)

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, ((x - self.mean) / self.std) @ self.w1 + self.b1)
        return 1.0 / (1.0 + np.exp(-np.clip(h @ self.w2 + self.b2, -30.0, 30.0)))


def train(x: np.ndarray, y: np.ndarray, hidden: int = 48, epochs: int = 60, lr: float = 3e-3,
          seed: int = 0, log=print) -> dict:
    """Treina a rede (entropia cruzada, Adam, minilotes). Devolve os parâmetros (sem limiares)."""
    rng = np.random.default_rng(seed)
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-6
    xs = ((x - mean) / std).astype(np.float32)
    params = {"w1": rng.normal(0, np.sqrt(2 / xs.shape[1]), (xs.shape[1], hidden)).astype(np.float32),
              "b1": np.zeros(hidden, np.float32),
              "w2": rng.normal(0, np.sqrt(1 / hidden), (hidden, y.shape[1])).astype(np.float32),
              "b2": np.log(y.mean(axis=0) / (1 - y.mean(axis=0)) + 1e-9).astype(np.float32)}
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(p) for k, p in params.items()}
    step = 0
    for epoch in range(epochs):
        order = rng.permutation(len(xs))
        total = 0.0
        for start in range(0, len(xs), 512):
            idx = order[start: start + 512]
            xb, yb = xs[idx], y[idx]
            z1 = xb @ params["w1"] + params["b1"]
            h = np.maximum(0.0, z1)
            p = 1.0 / (1.0 + np.exp(-(h @ params["w2"] + params["b2"])))
            total += float(-(yb * np.log(p + 1e-7) + (1 - yb) * np.log(1 - p + 1e-7)).sum())
            g2 = (p - yb) / len(xb)
            grads = {"w2": h.T @ g2, "b2": g2.sum(axis=0)}
            gh = (g2 @ params["w2"].T) * (z1 > 0)
            grads["w1"], grads["b1"] = xb.T @ gh, gh.sum(axis=0)
            step += 1
            for k in params:
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                params[k] -= lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
        if epoch % 10 == 9:
            log(f"  época {epoch + 1}: perda {total / len(xs):.4f}")
    return {"mean": mean, "std": std, **params}


def save(params: dict, thresholds: np.ndarray, info: dict, path: Path = MODEL_PATH) -> None:
    out = {k: np.asarray(v).round(5).tolist() for k, v in params.items()}
    out.update(thresholds=np.asarray(thresholds).round(3).tolist(), features=INPUT_NAMES,
               outputs=list(OUTPUTS), info=info)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
