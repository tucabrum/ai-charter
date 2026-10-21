"""Escolhe, compasso a compasso, qual faixa de guitarra vira a chart.

Quando a música tem mais de uma guitarra (ex.: violão base + guitarra solo), o
jogo só tem UMA faixa de guitarra. A regra:
  - a guitarra que está "na frente" (mais alta, notas simples, mais aguda) vence;
  - trocar de faixa tem um custo, para não ficar alternando a cada compasso;
  - depois de uma troca, a faixa escolhida fica por pelo menos `min_hold_measures`
    compassos (um "grupo de batidas coeso"), mesmo que pause um pouquinho.

A decisão é feita por programação dinâmica (Viterbi) sobre os compassos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_charter.analysis.tempo import TempoMap
from ai_charter.charting.common import Candidate
from ai_charter.config import GuitarSelectionRules


@dataclass
class GuitarSource:
    name: str
    cands: list[Candidate]
    audio: np.ndarray   # mono
    sr: int


def _measure_scores(sources: list[GuitarSource], tempo: TempoMap, resolution: int, n_measures: int,
                    rules: GuitarSelectionRules) -> np.ndarray:
    """Nota de "liderança" de cada fonte em cada compasso (NaN = fonte inativa)."""
    measure = resolution * 4
    loud = np.zeros((len(sources), n_measures))
    for s, src in enumerate(sources):
        for m in range(n_measures):
            a, b = (int(t * src.sr) for t in tempo.tick_to_time([m * measure, (m + 1) * measure]))
            chunk = src.audio[a:b]
            loud[s, m] = np.sqrt(np.mean(chunk ** 2)) if len(chunk) else 0.0
    loud_db = 20 * np.log10(loud + 1e-9)
    best_db = loud_db.max(axis=0)

    scores = np.full((len(sources), n_measures), np.nan)
    for m in range(n_measures):
        stats = []
        for s, src in enumerate(sources):
            notes = [c for c in src.cands if m * measure <= c.tick < (m + 1) * measure]
            if not notes or loud_db[s, m] - best_db[m] < rules.active_db:
                continue
            single = np.mean([c.voices == 1 for c in notes])
            pitch = float(np.median([c.pitch for c in notes]))
            stats.append((s, loud_db[s, m] - best_db[m], single, pitch))
        if not stats:
            continue
        lowest = min(p for *_, p in stats)
        for s, rel_db, single, pitch in stats:
            scores[s, m] = (rules.weight_loudness * rel_db / 10        # 0 para a mais alta, -1 a cada 10 dB
                            + rules.weight_single_notes * single       # solo = notas simples
                            + rules.weight_pitch * (pitch - lowest) / 12)  # solo = mais agudo
    return scores


def choose_sources(scores: np.ndarray, rules: GuitarSelectionRules) -> list[int]:
    """Qual fonte usar em cada compasso (Viterbi com permanência mínima)."""
    n_sources, n_measures = scores.shape
    hold = max(1, rules.min_hold_measures)
    any_active = ~np.all(np.isnan(scores), axis=0)
    # Custo de usar a fonte s no compasso m: -nota; inativa custa `inactive_cost`
    # (compassos sem nenhuma guitarra não custam nada para ninguém).
    emission = np.where(np.isnan(scores), np.where(any_active, rules.inactive_cost, 0.0), -scores)

    # Estado = (fonte, há quantos compassos estamos nela, até `hold`).
    INF = float("inf")
    cost = np.full((n_sources, hold), INF)
    cost[:, hold - 1] = emission[:, 0]  # no começo, pode trocar livremente
    back = np.zeros((n_measures, n_sources, hold, 2), dtype=int)
    for m in range(1, n_measures):
        new = np.full((n_sources, hold), INF)
        for s in range(n_sources):
            for d in range(hold):
                if cost[s, d] == INF:
                    continue
                # Continuar na mesma fonte.
                nd = min(d + 1, hold - 1)
                total = cost[s, d] + emission[s, m]
                if total < new[s, nd]:
                    new[s, nd], back[m, s, nd] = total, (s, d)
                # Trocar: só depois de ficar `hold` compassos na fonte atual.
                if d == hold - 1:
                    for t in range(n_sources):
                        if t == s:
                            continue
                        total = cost[s, d] + rules.switch_cost + emission[t, m]
                        if total < new[t, 0]:
                            new[t, 0], back[m, t, 0] = total, (s, d)
        cost = new

    s, d = np.unravel_index(int(np.argmin(cost)), cost.shape)
    path = [int(s)]
    for m in range(n_measures - 1, 0, -1):
        s, d = back[m, s, d]
        path.append(int(s))
    return path[::-1]


def merge_guitars(sources: list[GuitarSource], tempo: TempoMap, resolution: int,
                  rules: GuitarSelectionRules) -> tuple[list[Candidate], list[str]]:
    """Junta as guitarras numa faixa só. Retorna as notas e a fonte escolhida por compasso."""
    if len(sources) == 1:
        return sources[0].cands, [sources[0].name]
    measure = resolution * 4
    last_tick = max((c.tick for src in sources for c in src.cands), default=0)
    n_measures = last_tick // measure + 1
    scores = _measure_scores(sources, tempo, resolution, n_measures, rules)
    path = choose_sources(scores, rules)
    merged = [c for s, src in enumerate(sources) for c in src.cands if path[c.tick // measure] == s]
    return sorted(merged, key=lambda c: c.tick), [sources[s].name for s in path]
