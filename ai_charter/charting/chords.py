"""Decide o acorde de cada nota e deixa cada acorde SEMPRE com o mesmo formato.

1. Acorde ou nota simples: decidido pelo espectro por semitom (ver chordal_notes).
   Nos acordes, qual acorde é: o modelo mais parecido (analysis/chords.py).
2. Correções (só quando o acorde corrigido também pontua quase igual na nota):
   - troca de acorde: nas primeiras palhetadas de um acorde novo, o acorde
     anterior ainda está soando e "puxa" o reconhecimento para trás. Se a
     palhetada cai em cima de um tempo e as duas seguintes já são do acorde
     novo, ela também é (corrigido de trás para frente, em cadeia). Só vale
     depois de um acorde tocado várias vezes (que continua soando);
   - palhetada isolada: diferente das duas vizinhas, que concordam entre si,
     segue as vizinhas.
3. O tipo do acorde (maior, menor, power chord ou nota simples) é decidido por
   trecho com a mesma fundamental, somando as pontuações de todas as palhetadas.
4. Acordes viram notas da chart com um número fixo de notas (tríade = 3,
   power chord = 2) e uma altura fixa por FUNDAMENTAL (a nota grave típica dela),
   para o mesmo acorde cair sempre nas mesmas lanes. Acordes com a mesma
   fundamental (ex.: A e Am, que às vezes se confundem) ficam com a mesma altura.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ai_charter.analysis.chords import CHORDS, chord_voices, is_single_note

SINGLE_NOTE_MASK = np.array([is_single_note(i) for i in range(len(CHORDS))])
from ai_charter.charting.common import Candidate
from ai_charter.config import PitchedDetection

# Só um acorde tocado várias vezes seguidas "soa por cima" do próximo.
MIN_RINGING_STRUMS = 3


def _decide_kind(items: list[Candidate], labels: list[int], start: int, end: int,
                 third_margin: float) -> None:
    """Decide o tipo (maior, menor ou power chord) de um trecho com a mesma fundamental.

    Somar as pontuações do trecho inteiro é bem mais confiável que olhar uma
    palhetada só. Power chord (fundamental + quinta) não tem terça: se "maior" e
    "menor" pontuam praticamente igual, não há terça de verdade soando. (Guitarras
    distorcidas criam um "fantasma" de terça, mas igual para os dois lados.)
    """
    root = CHORDS[labels[start]][0]
    span = range(start, end + 1)
    major_index, minor_index = CHORDS.index((root, "")), CHORDS.index((root, "m"))
    # Diferença maior x menor de cada palhetada (em valor absoluto, para um lado
    # não "cancelar" o outro quando o acorde anterior ainda soa no começo).
    gaps = [abs(items[k].chord_scores[major_index] - items[k].chord_scores[minor_index]) for k in span]
    total = sum(items[k].chord_scores for k in span)
    if np.mean(gaps) < third_margin:
        kind = "5"
    else:
        kind = "" if total[major_index] > total[minor_index] else "m"
    for k in span:
        labels[k] = CHORDS.index((root, kind))


def chordal_notes(items: list[Candidate], cfg: PitchedDetection) -> list[bool]:
    """Quais notas são acordes (e não notas simples).

    Decidido pelo espectro por semitom, que distingue uma nota distorcida (cheia de
    harmônicos) de um acorde de verdade: o estimador achou 2+ notas, OU soa forte
    uma quinta (+7) ou uma terça logo acima da nota grave (os harmônicos de uma nota
    simples ficam uma oitava ou mais acima). Como acorde x nota vem em trechos,
    a decisão final é a maioria entre as notas vizinhas.
    """
    poly = [c.voices >= 2 or c.fifth_ratio >= cfg.chord_fifth_ratio or c.third_ratio >= cfg.chord_third_ratio
            for c in items]
    half = cfg.chord_smoothing
    return [float(np.mean(poly[max(0, k - half): k + half + 1])) > 0.5 for k in range(len(items))]


def label_chords(cands: list[Candidate], resolution: int, cfg: PitchedDetection) -> None:
    """Preenche `chord`, `voices` e `pitch` das notas que têm pontuação de acorde."""
    scored = [c for c in cands if c.chord_scores is not None]
    if not scored:
        return
    chordal = chordal_notes(scored, cfg)
    for c, is_chord in zip(scored, chordal):
        if not is_chord:   # nota simples: fica com a altura estimada
            c.voices = 1
            c.chord = CHORDS.index((round(c.pitch) % 12, "note"))
    items = [c for c, is_chord in zip(scored, chordal) if is_chord]
    if not items:
        return
    margin = cfg.chord_change_margin
    # Qual acorde: o modelo de acorde mais parecido (sem os modelos de nota simples).
    labels = [int(np.argmax(np.where(SINGLE_NOTE_MASK, -np.inf, c.chord_scores))) for c in items]

    def root(k: int) -> int:
        return CHORDS[labels[k]][0]

    def close_enough(k: int, other: int) -> bool:
        return items[k].chord_scores[other] >= items[k].chord_scores[labels[k]] - margin

    def ringing_before(k: int) -> bool:
        """O acorde antes da nota k foi tocado pelo menos MIN_RINGING_STRUMS vezes seguidas?"""
        if k < MIN_RINGING_STRUMS:
            return False
        previous = root(k - 1)
        return all(root(j) == previous for j in range(k - MIN_RINGING_STRUMS, k))

    # Troca de acorde atrasada. De trás para frente, para corrigir em cadeia: se
    # as duas palhetadas seguintes já são do acorde novo, esta também passa a ser.
    # (Compara só a fundamental: A e Am, por exemplo, ficam no mesmo lugar da chart.)
    # Só depois de um acorde que vinha soando: num riff com trocas rápidas, não há
    # acorde anterior "por cima", e a regra estragaria trocas legítimas.
    for k in range(len(items) - 3, -1, -1):
        new = labels[k + 1]
        if (root(k) != root(k + 1) == root(k + 2) and items[k].tick % resolution == 0
                and close_enough(k, new) and ringing_before(k)):
            labels[k] = new

    # Palhetada isolada no meio de um acorde (as duas vizinhas concordam): segue as vizinhas.
    for k in range(1, len(items) - 1):
        around = labels[k - 1]
        if root(k + 1) == root(k - 1) != root(k) and close_enough(k, around):
            labels[k] = around

    # Tipo do acorde decidido por trecho (palhetadas seguidas com a mesma fundamental).
    start = 0
    while start < len(items):
        end = start
        while end + 1 < len(items) and root(end + 1) == root(start):
            end += 1
        _decide_kind(items, labels, start, end, cfg.power_chord_third_margin)
        start = end + 1

    # Altura fixa de cada fundamental: a mediana da nota grave dela.
    lows: dict[int, list[float]] = defaultdict(list)
    for c, label in zip(items, labels):
        root = CHORDS[label][0]
        low = c.lowest_by_class[root] if c.lowest_by_class is not None else np.nan
        if not np.isnan(low):
            lows[root].append(low)
    fixed = {root: float(np.median(v)) for root, v in lows.items()}

    for c, label in zip(items, labels):
        root = CHORDS[label][0]
        c.chord = label
        power_chord = CHORDS[label][1] == "5"
        c.voices = cfg.power_chord_notes if power_chord else chord_voices(label)
        c.pitch = fixed.get(root, c.pitch)
