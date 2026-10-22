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
     segue as vizinhas;
   - troca relâmpago: um acorde de uma palhetada só entre dois grupos estáveis
     fica com o grupo mais parecido (sem a exigência de "quase igual").
3. O tipo do acorde (maior, menor, power chord ou nota simples) é decidido por
   trecho com a mesma fundamental, somando as pontuações de todas as palhetadas.
4. Acordes viram notas da chart com um número fixo de notas (2 por padrão:
   charts com acordes de 3 notas em toda palhetada ficam "infladas"; 3 só nos
   acordes de acento, segurados por muito tempo) e uma altura fixa por FUNDAMENTAL (a nota grave típica dela),
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
# Acordes fora do vocabulário (cifra) perdem isto na pontuação: na prática, nunca são escolhidos.
OUT_OF_VOCABULARY = 1.0


def vocabulary_mask(names: list[str]) -> np.ndarray:
    """Quais modelos de CHORDS um vocabulário permite ("Am" -> Lá menor e Lá power chord).

    Notas simples sempre valem. O power chord da fundamental também, porque é como a guitarra
    distorcida toca o acorde da cifra."""
    from ai_charter.cifra import parse_chord

    allowed = SINGLE_NOTE_MASK.copy()
    for name in names:
        parsed = parse_chord(name)   # aceita o jeito de escrever das cifras: Bb, F#m7, D/F#...
        if parsed is None:
            raise ValueError(f"acorde inválido em allowed_chords: {name!r}")
        root, kind = parsed
        for k in {kind, "5"}:
            if (root, k) in CHORDS:
                allowed[CHORDS.index((root, k))] = True
    return allowed


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
    if cfg.allowed_chords:   # vocabulário da cifra: os outros acordes praticamente somem
        allowed = vocabulary_mask(cfg.allowed_chords)
        for c in items:
            c.chord_scores = np.where(allowed, c.chord_scores, c.chord_scores - OUT_OF_VOCABULARY)
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

    # Troca relâmpago: um acorde de uma palhetada só, entre dois grupos estáveis
    # (cada um com MIN_RINGING_STRUMS+ palhetadas), quase sempre é o som de uma
    # troca (o acorde anterior ainda soando, ou o próximo chegando antes). Na chart,
    # isso vira um "pulo" sem sentido: a palhetada fica com o grupo mais parecido.
    runs = _runs([root(k) for k in range(len(items))])
    for (a0, a1), (b0, b1), (c0, c1) in zip(runs, runs[1:], runs[2:]):
        stable = a1 - a0 >= MIN_RINGING_STRUMS and c1 - c0 >= MIN_RINGING_STRUMS
        if b1 - b0 == 1 and stable:
            left, right = labels[a1 - 1], labels[c0]
            labels[b0] = left if items[b0].chord_scores[left] >= items[b0].chord_scores[right] else right

    # Ritmo harmônico: os acordes trocam nos tempos fortes (de preferência no compasso).
    _follow_harmonic_rhythm(items, labels, resolution)

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

    accent_ticks = cfg.accent_min_beats * resolution
    for c, label in zip(items, labels):
        root = CHORDS[label][0]
        c.chord = label
        power_chord = CHORDS[label][1] == "5"
        c.voices = min(chord_voices(label), cfg.power_chord_notes if power_chord else cfg.chord_notes)
        # Acorde segurado por muito tempo (fim de frase, "pancada"): acento com mais notas.
        if c.end_tick - c.tick >= accent_ticks:
            c.voices = max(c.voices, min(chord_voices(label), cfg.accent_chord_notes))
        c.pitch = fixed.get(root, c.pitch)


# Custo de trocar de acorde em cada colcheia do compasso 4/4 (0 = "1", 4 = 3º tempo).
# Trocar no 1º tempo é o normal; no 3º é comum; nos outros, só com bastante evidência.
CHANGE_COST = [0.15, 1.2, 1.0, 1.2, 0.4, 1.2, 1.0, 1.2]
SLOTS_PER_BAR = len(CHANGE_COST)
OTHER_ROOT_LOSS = 0.03   # custo mínimo de uma palhetada usar outra fundamental que não a dela
PASSAGE_GAP_BEATS = 2       # pausa maior que isso separa dois trechos de acordes
MIN_PASSAGE_STRUMS = 8      # trechos menores não têm "ritmo harmônico" para seguir
MIN_STRUMS_PER_CHORD = 4    # palhetadas por acorde, em média, para o trecho ser de base (e não riff)

# Pontuação de cada fundamental (0-11): o melhor acorde (não nota simples) dela.
_ROOT_OF = np.array([root for root, _ in CHORDS])


def _root_scores(scores: np.ndarray) -> np.ndarray:
    masked = np.where(SINGLE_NOTE_MASK, -np.inf, scores)
    return np.array([masked[_ROOT_OF == r].max() for r in range(12)])


def _follow_harmonic_rhythm(items: list[Candidate], labels: list[int], resolution: int) -> None:
    """Escolhe o acorde de cada TEMPO do trecho, preferindo trocas no início do compasso.

    Em músicas com acordes palhetados, o acorde quase sempre troca no 1º tempo do
    compasso (às vezes no 3º). O reconhecimento palhetada a palhetada erra perto
    das trocas (o acorde anterior ainda soa) e no meio de acordes longos, e o
    resultado é uma troca de cor no meio do compasso que não existe na música.

    Para cada trecho de acordes seguidos, um Viterbi escolhe uma fundamental por
    colcheia: o custo de cada colcheia é o quanto as palhetadas dela "perdem" ao
    usar aquela fundamental em vez da preferida delas, e trocar custa CHANGE_COST
    conforme a posição no compasso. Como não sabemos onde começa o compasso (o
    detector de andamento acha as batidas, não o "1", e às vezes trava no
    contratempo), as 8 posições possíveis são testadas e fica a de menor custo.
    """
    k = 0
    while k < len(items):
        end = k
        while end + 1 < len(items) and items[end + 1].tick - items[end].tick <= PASSAGE_GAP_BEATS * resolution:
            end += 1
        strums = end - k + 1
        changes = len(_runs([CHORDS[labels[j]][0] for j in range(k, end + 1)]))
        # Riffs (acorde trocando a cada uma ou duas palhetadas) ficam como estão:
        # só trechos de base, com cada acorde palhetado várias vezes, seguem o compasso.
        if strums >= MIN_PASSAGE_STRUMS and strums / changes >= MIN_STRUMS_PER_CHORD:
            _passage_rhythm(items, labels, k, end + 1, resolution)
        k = end + 1


def _passage_rhythm(items: list[Candidate], labels: list[int], start: int, end: int, resolution: int) -> None:
    slot_ticks = resolution // 2   # uma colcheia
    all_slots = [round(items[k].tick / slot_ticks) for k in range(start, end)]
    # Só colcheias com palhetada: trocar numa colcheia vazia seria o mesmo que
    # trocar na próxima palhetada, mas com o custo de outra posição do compasso.
    slots = sorted(set(all_slots))
    index = {slot: i for i, slot in enumerate(slots)}
    roots = sorted({CHORDS[labels[k]][0] for k in range(start, end)})
    # loss[i, r]: quanto as palhetadas da colcheia slots[i] perdem usando a fundamental roots[r].
    loss = np.zeros((len(slots), len(roots)))
    for k, slot in zip(range(start, end), all_slots):
        # Parte do acorde já decidido para a palhetada (que já leva em conta o acorde
        # anterior ainda soando): ele não perde nada; os outros perdem a diferença de
        # pontuação, e pelo menos OTHER_ROOT_LOSS.
        per_root = _root_scores(items[k].chord_scores)[roots]
        here = np.maximum(per_root.max() - per_root, OTHER_ROOT_LOSS)
        here[roots.index(CHORDS[labels[k]][0])] = 0.0
        loss[index[slot]] += here
    best_total, best_path = np.inf, None
    for phase in range(SLOTS_PER_BAR):
        cost = loss[0].copy()
        back = np.zeros((len(slots), len(roots)), dtype=int)
        for i in range(1, len(slots)):
            change = CHANGE_COST[(slots[i] - phase) % SLOTS_PER_BAR]
            move = cost.min() + change
            back[i] = np.where(cost <= move, np.arange(len(roots)), int(np.argmin(cost)))
            cost = np.minimum(cost, move) + loss[i]
        if cost.min() < best_total:
            best_total = cost.min()
            path = [int(np.argmin(cost))]
            for i in range(len(slots) - 1, 0, -1):
                path.append(int(back[i][path[-1]]))
            best_path = path[::-1]
    for k, slot in zip(range(start, end), all_slots):
        root = roots[best_path[index[slot]]]
        if CHORDS[labels[k]][0] != root:
            same_root = np.where((_ROOT_OF == root) & ~SINGLE_NOTE_MASK)[0]
            labels[k] = int(same_root[np.argmax(items[k].chord_scores[same_root])])


def _runs(values: list) -> list[tuple[int, int]]:
    """Trechos de valores iguais seguidos: [(início, fim exclusivo), ...]."""
    runs, start = [], 0
    for k in range(1, len(values) + 1):
        if k == len(values) or values[k] != values[start]:
            runs.append((start, k))
            start = k
    return runs
