"""Modelo de bateria aprendido com charts oficiais.

Duas etapas, as duas redes neurais pequenas treinadas com charts oficiais de Rock Band
(ai-charter-bench train-drums):

1. **Som.** Para cada ataque, o transcritor mede um retrato acústico (analysis/drums.py:
   FEATURE_NAMES). Junto com o lugar do ataque no compasso, a rede dá a probabilidade de cada peça.
2. **Levada.** A segunda rede olha as probabilidades da primeira nos compassos em volta (o que tocou
   em cada semicolcheia, um compasso para trás e um para frente) e decide de novo. É ela que sabe
   como as levadas são chartadas: chimbal em linha contínua, caixa no 2 e no 4, virada no fim da
   frase. Um golpe ambíguo no meio de oito colcheias de chimbal vira chimbal, não caixa.

Qual tom é qual (agudo, médio, surdo) o modelo não decide: a afinação muda de kit para kit. Ele diz
só "é tom", e o tom vem dos sons do kit da própria música, ordenados pela altura.

O modelo fica em drum_model.json, ao lado deste arquivo. Sem ele, valem só as regras do transcritor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ai_charter.analysis import adtof
from ai_charter.analysis.drums import CRASH, FEATURE_NAMES, HIHAT, KICK, RIDE, RING_NAMES, SNARE, TOM

MODEL_PATH = Path(__file__).with_name("drum_model.json")
# Com o ADTOF instalado: o mesmo modelo, mais as chances que o ADTOF dá para cada golpe
ADTOF_MODEL_PATH = Path(__file__).with_name("drum_model_adtof.json")
OUTPUTS = (KICK, SNARE, HIHAT, RIDE, CRASH, TOM)
CONTEXT_NAMES = [f"posicao_{i}" for i in range(16)] + ["fora_da_grade", "vizinho_1/16", "vizinho_1/8",
                                                       "batida_forte", "compasso_denso"]


INPUT_NAMES = FEATURE_NAMES + [f"rel_{n}" for n in FEATURE_NAMES] + CONTEXT_NAMES
ADTOF_NAMES = [f"adtof_{c}" for c in adtof.CLASSES]
INPUT_NAMES_ADTOF = INPUT_NAMES + ADTOF_NAMES
# Mais as medidas de ressonância (altura do tom, quanto os pratos sustentam), na música e relativas a ela
INPUT_NAMES_RING = INPUT_NAMES_ADTOF + RING_NAMES + [f"rel_{n}" for n in RING_NAMES]

# Etapa da levada: semicolcheias em volta do ataque (meio compasso de cada lado em detalhe, mais o
# mesmo lugar no compasso anterior e no seguinte) e, em cada uma, a chance de cada peça + "houve ataque"
GROOVE_OFFSETS = tuple(range(-8, 9)) + (-16, -12, 12, 16)
GROOVE_NAMES = ([f"levada_{o:+d}_{p}" for o in GROOVE_OFFSETS for p in (*OUTPUTS, "ataque")]
                + [f"proprio_{p}" for p in OUTPUTS] + CONTEXT_NAMES[:17])


def song_relative(features: np.ndarray) -> np.ndarray:
    """Cada medida comparada ao resto da mesma música (mediana e faixa entre os 10% e 90%):
    é o que deixa o modelo reconhecer um kit, uma mixagem ou um vazamento que ele nunca viu."""
    if not len(features):
        return features
    lo, mid, hi = np.percentile(features, [10, 50, 90], axis=0)
    return ((features - mid) / (hi - lo + 1e-6)).astype(np.float32)


def model_inputs(features: np.ndarray, ticks: np.ndarray, resolution: int,
                 adtof_feats: np.ndarray | None = None, ring: np.ndarray | None = None) -> np.ndarray:
    """Entrada do modelo para os ataques de UMA música: retrato, retrato relativo, lugar na levada e,
    se houver, as chances do ADTOF (adtof.at_times) e as medidas de ressonância (ring_features)."""
    parts = [features, song_relative(features), context_features(ticks * 192 / resolution, 192)]
    parts += [adtof_feats] if adtof_feats is not None else []
    parts += [ring, song_relative(ring)] if ring is not None else []
    return np.hstack(parts)


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


def groove_inputs(probs: np.ndarray, ticks: np.ndarray, resolution: int) -> np.ndarray:
    """Entrada da etapa da levada para os ataques de UMA música: o que a etapa do som achou em cada
    semicolcheia em volta (GROOVE_OFFSETS), o que achou do próprio ataque e o lugar no compasso."""
    if not len(ticks):
        return np.zeros((0, len(GROOVE_NAMES)), dtype=np.float32)
    ticks = ticks * 192 / resolution
    slot = np.round(ticks / 48).astype(int)
    pad = max(abs(o) for o in GROOVE_OFFSETS)
    idx = slot - slot.min() + pad
    grid = np.zeros((idx.max() + pad + 1, len(OUTPUTS) + 1), dtype=np.float32)
    for k in range(len(OUTPUTS)):
        np.maximum.at(grid[:, k], idx, probs[:, k])
    grid[idx, -1] = 1.0
    return np.hstack([grid[idx + o] for o in GROOVE_OFFSETS]
                     + [probs.astype(np.float32), context_features(ticks, 192)[:, :17]])


def mlp(params: dict, x: np.ndarray, rows: np.ndarray | None = None, chunk: int = 65536) -> np.ndarray:
    """Probabilidades da rede (uma camada oculta) para as linhas `rows` de x, em blocos."""
    rows = np.arange(len(x)) if rows is None else rows
    out = np.empty((len(rows), len(params["b2"])), dtype=np.float32)
    for s in range(0, len(rows), chunk):
        xb = (x[rows[s: s + chunk]].astype(np.float32) - params["mean"]) / params["std"]
        h = np.maximum(0.0, xb @ params["w1"] + params["b1"])
        out[s: s + chunk] = 1.0 / (1.0 + np.exp(-np.clip(h @ params["w2"] + params["b2"], -30.0, 30.0)))
    return out


def _arrays(params: dict) -> dict:
    out = {k: np.array(params[k], dtype=np.float32) for k in ("mean", "std", "w1", "b1", "w2", "b2")}
    out["std"] = np.where(out["std"] > 0, out["std"], 1.0).astype(np.float32)   # coluna constante
    return out


class DrumModel:
    """Etapa do som (retrato + contexto -> peça) e, se houver, etapa da levada por cima."""

    def __init__(self, params: dict):
        self.sound = _arrays(params)
        self.groove = _arrays(params["groove"]) if params.get("groove") else None
        self.thresholds = np.array(params["thresholds"], dtype=np.float32)
        self.info = params.get("info", {})
        self.uses_adtof = params.get("features") in (INPUT_NAMES_ADTOF, INPUT_NAMES_RING)
        self.uses_ring = params.get("features") == INPUT_NAMES_RING
        self.w1, self.w2 = self.sound["w1"], self.sound["w2"]

    @classmethod
    def load(cls, path: Path | None = None) -> "DrumModel | None":
        """O modelo em `path`; sem `path`, o com ADTOF se ele estiver instalado, senão o básico."""
        if path is None:
            if adtof.available() and ADTOF_MODEL_PATH.is_file():
                model = cls.load(ADTOF_MODEL_PATH)
                if model is not None:
                    return model
            path = MODEL_PATH
        if not path.is_file():
            return None
        params = json.loads(path.read_text(encoding="utf-8"))
        if params.get("features") not in (INPUT_NAMES, INPUT_NAMES_ADTOF, INPUT_NAMES_RING)                 or params.get("outputs") != list(OUTPUTS):
            return None   # modelo de outra versão do retrato: melhor não usar
        if params.get("groove") and params["groove"].get("features") != GROOVE_NAMES:
            return None
        return cls(params)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Só a etapa do som."""
        return mlp(self.sound, x)

    def predict_song(self, features: np.ndarray, ticks: np.ndarray, resolution: int,
                     adtof_feats: np.ndarray | None = None, ring: np.ndarray | None = None) -> np.ndarray:
        """Os ataques de uma música inteira: etapa do som e, por cima, a da levada."""
        if self.uses_adtof and adtof_feats is None:
            raise ValueError("este modelo precisa das chances do ADTOF para cada ataque")
        if self.uses_ring and ring is None:
            raise ValueError("este modelo precisa das medidas de ressonância (ring_features)")
        probs = self.predict(model_inputs(features, ticks, resolution, adtof_feats if self.uses_adtof else None,
                                          ring if self.uses_ring else None))
        if self.groove is None or not len(probs):
            return probs
        return mlp(self.groove, groove_inputs(probs, ticks, resolution))


def _standardize_stats(x: np.ndarray, rows: np.ndarray, chunk: int = 262144) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(x.shape[1]); sq = np.zeros(x.shape[1])
    for s in range(0, len(rows), chunk):
        xb = x[rows[s: s + chunk]].astype(np.float64)
        total += xb.sum(axis=0)
        sq += (xb ** 2).sum(axis=0)
    mean = total / len(rows)
    std = np.sqrt(np.maximum(sq / len(rows) - mean ** 2, 0.0)) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def train(x: np.ndarray, y: np.ndarray, rows: np.ndarray | None = None, hidden: int = 48, epochs: int = 60,
          lr: float = 3e-3, seed: int = 0, log=print) -> dict:
    """Treina a rede (entropia cruzada, Adam, minilotes) nas linhas `rows` de x e y, sem copiar x
    inteiro (cabe com milhões de ataques). Devolve os parâmetros (sem limiares)."""
    rng = np.random.default_rng(seed)
    rows = np.arange(len(x)) if rows is None else rows
    mean, std = _standardize_stats(x, rows)
    prior = y[rows].mean(axis=0)
    params = {"w1": rng.normal(0, np.sqrt(2 / x.shape[1]), (x.shape[1], hidden)).astype(np.float32),
              "b1": np.zeros(hidden, np.float32),
              "w2": rng.normal(0, np.sqrt(1 / hidden), (hidden, y.shape[1])).astype(np.float32),
              "b2": np.log(prior / (1 - prior) + 1e-9).astype(np.float32)}
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(p) for k, p in params.items()}
    step = 0
    for epoch in range(epochs):
        order = rows[rng.permutation(len(rows))]
        total = 0.0
        for start in range(0, len(order), 512):
            idx = np.sort(order[start: start + 512])
            xb, yb = (x[idx].astype(np.float32) - mean) / std, y[idx]
            z1 = xb @ params["w1"] + params["b1"]
            h = np.maximum(0.0, z1)
            p = 1.0 / (1.0 + np.exp(-np.clip(h @ params["w2"] + params["b2"], -30.0, 30.0)))
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
        if epoch % 5 == 4 or epoch == epochs - 1:
            log(f"  época {epoch + 1}/{epochs}: perda {total / len(rows):.4f}")
    return {"mean": mean, "std": std, **params}


def _jsonable(params: dict) -> dict:
    # Pesos com 5 casas; média e desvio com 6 algarismos (um desvio de 1e-6 não pode virar 0)
    return {k: (np.asarray(v).astype(float).round(5) if k not in ("mean", "std")
                else np.array([float(f"{x:.6g}") for x in np.asarray(v, dtype=float)])).tolist()
            for k, v in params.items()}


def save(params: dict, thresholds: np.ndarray, info: dict, path: Path = MODEL_PATH,
         groove: dict | None = None, features: list[str] = INPUT_NAMES) -> None:
    out = _jsonable(params)
    out.update(thresholds=np.asarray(thresholds).round(3).tolist(), features=features,
               outputs=list(OUTPUTS), info=info)
    if groove is not None:
        out["groove"] = {**_jsonable(groove), "features": GROOVE_NAMES}
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


# --- Cor de cada tom (amarelo, azul ou verde), aprendida das charts oficiais ---------------------
# A afinação muda de kit para kit: o que decide a cor é a altura do golpe comparada à dos outros
# tons DA MÚSICA, e à do tom de antes e de depois (viradas costumam descer do agudo ao surdo).
TOM_LANE_PATH = Path(__file__).with_name("drum_tom_lanes.json")
TOM_LANES = ("tom_high", "tom_mid", "tom_low")
TOM_CONTEXT_NAMES = ["tom_altura_rank", "tom_centro_rank", "tom_altura_vs_antes", "tom_altura_vs_depois",
                     "tom_centro_vs_antes", "tom_centro_vs_depois", "tom_gap_antes", "tom_gap_depois"]


def tom_context(features: np.ndarray, adtof_feats: np.ndarray, ticks: np.ndarray, resolution: int,
                min_chance: float = 0.2, ring: np.ndarray | None = None) -> np.ndarray:
    """Para cada ataque de UMA música: onde a altura dele fica entre os golpes de tom da música
    (os que o ADTOF ouve como tom) e quanto ela difere do golpe de tom anterior e do seguinte."""
    # A altura: o pico dos graves medido na ressonância, se houver; senão, a do retrato
    altura = ring[:, RING_NAMES.index("tom_freq")] if ring is not None else features[:, FEATURE_NAMES.index("altura_graves")]
    levels = features[:, [FEATURE_NAMES.index(f"nivel_{i}") for i in range(12)]]
    centro = (levels * np.arange(12)).sum(axis=1) / (levels.sum(axis=1) + 1e-9)
    is_tom = adtof_feats[:, adtof.CLASSES.index("tom")] >= min_chance
    out = np.zeros((len(features), len(TOM_CONTEXT_NAMES)), dtype=np.float32)
    if not is_tom.any():
        return out
    beats = ticks / resolution
    rank = lambda v, ref: np.searchsorted(np.sort(ref), v) / max(1, len(ref))   # noqa: E731
    out[:, 0], out[:, 1] = rank(altura, altura[is_tom]), rank(centro, centro[is_tom])
    idx, at = np.flatnonzero(is_tom), np.arange(len(features))
    before = np.searchsorted(idx, at, "left") - 1     # o golpe de tom anterior a cada ataque
    after = np.searchsorted(idx, at, "right")         # e o seguinte
    for j, (col_a, col_c, col_g) in ((before, (2, 4, 6)), (after, (3, 5, 7))):
        ok = (j >= 0) & (j < len(idx))
        other = idx[np.clip(j, 0, len(idx) - 1)]
        gap = np.abs(beats[other] - beats)
        ok &= gap <= 2.0
        out[ok, col_a] = altura[ok] - altura[other[ok]]
        out[ok, col_c] = centro[ok] - centro[other[ok]]
        out[:, col_g] = np.where(ok, gap, 2.0)
    return out


def tom_lane_inputs(features: np.ndarray, ticks: np.ndarray, resolution: int, adtof_feats: np.ndarray,
                    ring: np.ndarray | None = None) -> np.ndarray:
    return np.hstack([model_inputs(features, ticks, resolution, adtof_feats, ring),
                      tom_context(features, adtof_feats, ticks, resolution, ring=ring)])


class TomLaneModel:
    """Rede pequena: entrada do modelo de bateria + contexto de tons -> chance de cada cor."""

    def __init__(self, params: dict):
        self.params = _arrays(params)
        self.uses_ring = params.get("features") == INPUT_NAMES_RING + TOM_CONTEXT_NAMES

    @classmethod
    def load(cls, path: Path | None = None) -> "TomLaneModel | None":
        path = path or TOM_LANE_PATH
        if not adtof.available() or not path.is_file():
            return None
        params = json.loads(path.read_text(encoding="utf-8"))
        if params.get("features") not in (INPUT_NAMES_ADTOF + TOM_CONTEXT_NAMES, INPUT_NAMES_RING + TOM_CONTEXT_NAMES):
            return None
        return cls(params)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return mlp(self.params, x)


# --- Cor de cada prato grande (azul ou verde), aprendida das charts oficiais ---------------------
# Na chart, o azul nem sempre é uma condução: muitas vezes é um segundo ataque ou um china. Como nos
# tons, o que decide é como o charter distribuiu os pratos do kit: o som do golpe comparado aos
# outros pratos da música, quanto ele sustenta e se vem marcando o tempo em sequência ou sozinho.
CYM_LANE_PATH = Path(__file__).with_name("drum_cym_lanes.json")
CYM_LANES = (RIDE, CRASH)
CYM_CONTEXT_NAMES = ["prato_centro_rank", "prato_sustenta_rank", "prato_gap_antes", "prato_gap_depois",
                     "prato_vizinhos_no_compasso"]


def cym_context(features: np.ndarray, adtof_feats: np.ndarray, ticks: np.ndarray, resolution: int,
                ring: np.ndarray, min_chance: float = 0.3) -> np.ndarray:
    """Para cada ataque de UMA música: onde o som dele fica entre os golpes de prato da música (os que
    o ADTOF ouve como prato), e quão perto estão os golpes de prato vizinhos (marcando o tempo?)."""
    levels = features[:, [FEATURE_NAMES.index(f"nivel_{i}") for i in range(12)]]
    centro = (levels * np.arange(12)).sum(axis=1) / (levels.sum(axis=1) + 1e-9)
    sustenta = ring[:, RING_NAMES.index("sustenta_700ms")]
    is_cym = adtof_feats[:, adtof.CLASSES.index("prato")] >= min_chance
    out = np.zeros((len(features), len(CYM_CONTEXT_NAMES)), dtype=np.float32)
    out[:, 2:4] = 4.0
    if not is_cym.any():
        return out
    rank = lambda v, ref: np.searchsorted(np.sort(ref), v) / max(1, len(ref))   # noqa: E731
    out[:, 0], out[:, 1] = rank(centro, centro[is_cym]), rank(sustenta, sustenta[is_cym])
    beats = ticks / resolution
    idx, at = np.flatnonzero(is_cym), np.arange(len(features))
    before, after = np.searchsorted(idx, at, "left") - 1, np.searchsorted(idx, at, "right")
    for col, j in ((2, before), (3, after)):
        ok = (j >= 0) & (j < len(idx))
        out[ok, col] = np.minimum(np.abs(beats[idx[np.clip(j, 0, len(idx) - 1)]] - beats)[ok], 4.0)
    bar = (beats // 4).astype(int)
    per_bar = np.bincount(bar[is_cym] - bar.min(), minlength=bar.max() - bar.min() + 1)
    out[:, 4] = per_bar[bar - bar.min()] / 8.0
    return out


def cym_lane_inputs(features: np.ndarray, ticks: np.ndarray, resolution: int, adtof_feats: np.ndarray,
                    ring: np.ndarray) -> np.ndarray:
    return np.hstack([model_inputs(features, ticks, resolution, adtof_feats, ring),
                      cym_context(features, adtof_feats, ticks, resolution, ring)])


class CymLaneModel:
    """Rede pequena: entrada do modelo de bateria + contexto de pratos -> condução (azul) ou ataque (verde)."""

    def __init__(self, params: dict):
        self.params = _arrays(params)

    @classmethod
    def load(cls, path: Path | None = None) -> "CymLaneModel | None":
        path = path or CYM_LANE_PATH
        if not adtof.available() or not path.is_file():
            return None
        params = json.loads(path.read_text(encoding="utf-8"))
        if params.get("features") != INPUT_NAMES_RING + CYM_CONTEXT_NAMES:
            return None
        return cls(params)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return mlp(self.params, x)
