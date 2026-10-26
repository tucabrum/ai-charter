"""Compassos que se repetem na música saem iguais na chart, como um charter faria.

As charts oficiais repetem muito: ~70% dos compassos têm um gêmeo idêntico na mesma música (o
refrão que volta, a levada do verso). A transcrição, golpe a golpe, erra um pouco diferente em cada
repetição. Aqui os compassos que soam parecidos (o mesmo desenho de peças e forças nas semicolcheias)
formam um grupo, e cada nota do grupo fica ou sai por maioria, igual em todos eles. De quebra, um
erro isolado numa das repetições é corrigido pelas outras.
"""

from __future__ import annotations

import numpy as np

from ai_charter.charting.common import Candidate


def consolidate_repeats(cands: list[Candidate], bar_starts: np.ndarray, resolution: int,
                        similarity: float = 0.9, majority: float = 0.5) -> list[Candidate]:
    """`bar_starts`: tick do início de cada compasso. Só compassos com todas as notas na grade de
    semicolcheias entram nos grupos (tercinas e afins ficam como estão)."""
    if len(bar_starts) < 2 or not cands:
        return cands
    step = resolution / 4
    pieces = sorted({c.piece for c in cands})
    col = {p: i for i, p in enumerate(pieces)}
    bar_of = np.searchsorted(bar_starts, [c.tick for c in cands], side="right") - 1
    by_bar: dict[int, list[Candidate]] = {}
    for b, c in zip(bar_of, cands):
        by_bar.setdefault(int(b), []).append(c)
    slots = int(round((np.median(np.diff(bar_starts)) if len(bar_starts) > 1 else 4 * resolution) / step))
    vectors, bars = [], []
    for b, members in by_bar.items():
        if b < 0 or b + 1 >= len(bar_starts) or len(members) < 3:
            continue
        pos = [(c.tick - bar_starts[b]) / step for c in members]
        if any(abs(q - round(q)) > 1e-6 or not 0 <= round(q) < slots for q in pos):
            continue
        v = np.zeros((slots, len(pieces)))
        for q, c in zip(pos, members):
            v[int(round(q)), col[c.piece]] = max(v[int(round(q)), col[c.piece]], c.strength)
        vectors.append(v.ravel())
        bars.append(b)
    if len(bars) < 2:
        return cands
    V = np.array(vectors)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    groups: list[list[int]] = []
    centers: list[np.ndarray] = []
    for i in range(len(bars)):            # cada compasso entra no grupo mais parecido (ou abre um)
        if centers:
            sims = np.array([c @ V[i] / (np.linalg.norm(c) + 1e-9) for c in centers])
            j = int(sims.argmax())
            if sims[j] >= similarity:
                groups[j].append(i)
                centers[j] = centers[j] + V[i]
                continue
        groups.append([i])
        centers.append(V[i].copy())
    replaced: dict[int, list[Candidate]] = {}
    for g in groups:
        if len(g) < 2:
            continue
        notes: dict[tuple[int, str], list[float]] = {}
        for i in g:
            b = bars[i]
            for c in by_bar[b]:
                notes.setdefault((int(round((c.tick - bar_starts[b]) / step)), c.piece), []).append(c.strength)
        keep = {k: float(np.mean(v)) for k, v in notes.items() if len(v) >= majority * len(g)}
        for i in g:
            b = bars[i]
            replaced[b] = [Candidate(int(round(bar_starts[b] + q * step)), s, piece=p) for (q, p), s in keep.items()]
    out = [c for b, c in zip(bar_of, cands) if int(b) not in replaced]
    for notes in replaced.values():
        out += notes
    return sorted(out, key=lambda c: (c.tick, c.piece))
