"""Decide o acorde de cada nota e deixa cada acorde SEMPRE com o mesmo formato.

1. Cada nota recebe o acorde (ou nota simples) mais parecido (analysis/chords.py).
2. Correções (só quando o acorde corrigido também pontua quase igual na nota):
   - troca de acorde: nas primeiras palhetadas de um acorde novo, o acorde
     anterior ainda está soando e "puxa" o reconhecimento para trás. Se a
     palhetada cai em cima de um tempo e as duas seguintes já são do acorde
     novo, ela também é (corrigido de trás para frente, em cadeia);
   - palhetada isolada: diferente das duas vizinhas, que concordam entre si,
     segue as vizinhas.
3. Acordes viram notas da chart com um número fixo de notas (tríade = 3,
   power chord = 2) e uma altura fixa por FUNDAMENTAL (a nota grave típica dela),
   para o mesmo acorde cair sempre nas mesmas lanes. Acordes com a mesma
   fundamental (ex.: A e Am, que às vezes se confundem) ficam com a mesma altura.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ai_charter.analysis.chords import CHORDS, chord_voices, is_single_note
from ai_charter.charting.common import Candidate


def label_chords(cands: list[Candidate], resolution: int, margin: float) -> None:
    """Preenche `chord`, `voices` e `pitch` das notas que têm pontuação de acorde."""
    items = [c for c in cands if c.chord_scores is not None]
    if not items:
        return
    labels = [int(np.argmax(c.chord_scores)) for c in items]

    def root(k: int) -> int:
        return CHORDS[labels[k]][0]

    def close_enough(k: int, other: int) -> bool:
        return items[k].chord_scores[other] >= items[k].chord_scores[labels[k]] - margin

    # Troca de acorde atrasada. De trás para frente, para corrigir em cadeia: se
    # as duas palhetadas seguintes já são do acorde novo, esta também passa a ser.
    # (Compara só a fundamental: A e Am, por exemplo, ficam no mesmo lugar da chart.)
    for k in range(len(items) - 3, -1, -1):
        new = labels[k + 1]
        if (root(k) != root(k + 1) == root(k + 2) and items[k].tick % resolution == 0
                and close_enough(k, new)):
            labels[k] = new

    # Palhetada isolada no meio de um acorde (as duas vizinhas concordam): segue as vizinhas.
    for k in range(1, len(items) - 1):
        around = labels[k - 1]
        if root(k + 1) == root(k - 1) != root(k) and close_enough(k, around):
            labels[k] = around

    # Altura fixa de cada fundamental (nos acordes): a mediana da nota grave dela.
    lows: dict[int, list[float]] = defaultdict(list)
    for c, label in zip(items, labels):
        root = CHORDS[label][0]
        low = c.lowest_by_class[root] if c.lowest_by_class is not None else np.nan
        if not is_single_note(label) and not np.isnan(low):
            lows[root].append(low)
    fixed = {root: float(np.median(v)) for root, v in lows.items()}

    for c, label in zip(items, labels):
        root = CHORDS[label][0]
        c.chord = label
        if is_single_note(label):
            c.voices = 1
            if round(c.pitch) % 12 != root and not np.isnan(c.lowest_by_class[root]):
                c.pitch = float(c.lowest_by_class[root])   # a nota estimada não bate com a reconhecida
        else:
            c.voices = chord_voices(label)
            c.pitch = fixed.get(root, c.pitch)
