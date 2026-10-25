"""Gera as charts de guitarra, baixo e teclado (5 lanes) em cada dificuldade.

Regras de lanes (as mesmas que os charters da comunidade seguem): o jogo não
tem como representar a nota exata, então seguimos o CONTORNO da melodia:
  - verde é a região mais grave, laranja a mais aguda;
  - mesma nota                     -> mesma lane;
  - nota diferente da anterior     -> lane diferente (nunca "repetir" a cor);
  - nota mais aguda que a anterior -> lane mais à direita (e vice-versa).
Quando a melodia tem mais notas do que lanes (ex.: escala de 6 notas subindo),
é impossível cumprir tudo: aí o padrão é REPOSICIONADO (a mão "volta" e
continua subindo, como G R Y B -> R Y B O), em vez de repetir a mesma cor.
"""

from __future__ import annotations

import numpy as np

from collections import Counter, defaultdict

from ai_charter.charting.common import (
    Candidate, ChartNote, Repeats, TrackChart, copy_repeats, enforce_reduction, find_repeats,
    select_rhythm, star_power_for,
)
from ai_charter.config import FretDifficultyRules

FORCE_FLAG = 5  # no .chart, a nota 5 inverte strum <-> HOPO


# "Custos" de cada quebra de regra entre duas notas seguidas. O algoritmo escolhe
# as lanes com o menor custo total, então quanto maior o número, mais grave a quebra.
COST_SAME_PITCH_MOVED = 10.0     # mesma nota em outra lane
COST_REPEATED_LANE = 8.0         # nota diferente na mesma lane (o "laranja duplo")
COST_RESET = 4.0                 # reposicionar: mover a lane no sentido contrário da melodia
COST_RESET_ON_BEAT = 3.0         # ...mais natural no início de uma batida
COST_SMALL_JUMP = 1.0            # salto grande na melodia (≥ 7 semitons) virando só 1 lane
COST_PER_LANE_FROM_IDEAL = 0.1   # distância da lane "ideal" pela região da melodia
# Entre a nota atual e a RETRASADA (ex.: 62 64 62 -> as duas 62 na mesma lane).
COST_SAME_PITCH_MOVED_2_BACK = 4.0


def _ideal_lanes(p: np.ndarray, n_lanes: int, window: int) -> np.ndarray:
    """Lane "ideal" de cada nota, pela posição dela entre as notas vizinhas."""
    top = n_lanes - 1
    ideal = np.empty(len(p))
    for i in range(len(p)):
        nearby = p[max(0, i - window): i + window + 1]
        lo, hi = nearby.min(), nearby.max()
        span = max(hi - lo, top)  # pelo menos 1 semitom por lane
        pad = (span - (hi - lo)) / 2  # centraliza trechos com pouca variação
        ideal[i] = (p[i] - lo + pad) / span * top
    return ideal


def _transition_cost(diff: float, a: int, b: int, on_beat: bool) -> float:
    """Custo de ir da lane `a` para a lane `b` quando a melodia muda `diff` semitons."""
    if abs(diff) < 0.5:
        return 0.0 if a == b else COST_SAME_PITCH_MOVED
    if a == b:
        return COST_REPEATED_LANE
    if (diff > 0) != (b > a):
        return COST_RESET_ON_BEAT if on_beat else COST_RESET
    return COST_SMALL_JUMP if abs(diff) >= 7 and abs(b - a) == 1 else 0.0


def assign_lanes(pitches: list[float], n_lanes: int, window: int,
                 ticks: list[int] | None = None, resolution: int = 192,
                 max_lanes: list[int] | None = None) -> list[int]:
    """Converte alturas (MIDI) em lanes 0..n_lanes-1 seguindo o contorno melódico.

    Usa programação dinâmica (algoritmo de Viterbi): para cada nota, guarda o
    menor custo para chegar a cada par (lane da nota anterior, lane da nota atual);
    no fim, volta pelo caminho mais barato. Assim a decisão considera a frase
    inteira, e cada nota é comparada com a anterior e com a retrasada.
    `max_lanes[i]`: lane mais alta permitida para a nota i (acordes ocupam mais lanes).
    """
    p = np.asarray(pitches, dtype=float)
    n = len(p)
    if n == 0:
        return []
    ideal = _ideal_lanes(p, n_lanes, window)
    unary = COST_PER_LANE_FROM_IDEAL * np.abs(np.arange(n_lanes)[None, :] - ideal[:, None])
    if max_lanes is not None:
        # Acordes ocupam várias lanes: a lane mais grave de um acorde não pode passar
        # de max_lanes[i] (ex.: um acorde de 3 notas em 5 lanes começa no máximo no amarelo).
        unary[np.arange(n_lanes)[None, :] > np.asarray(max_lanes)[:, None]] = np.inf
    if n == 1:
        return [int(np.argmin(unary[0]))]

    def on_beat(i: int) -> bool:
        return ticks is not None and ticks[i] % resolution == 0

    # cost[a, b] = menor custo das notas 0..i, com a nota i-1 na lane a e a nota i na lane b.
    cost = np.full((n_lanes, n_lanes), np.inf)
    for a in range(n_lanes):
        for b in range(n_lanes):
            cost[a, b] = unary[0, a] + unary[1, b] + _transition_cost(p[1] - p[0], a, b, on_beat(1))
    came_from = np.zeros((n, n_lanes, n_lanes), dtype=int)  # came_from[i, b, c] = lane da nota i-2

    for i in range(2, n):
        new = np.full((n_lanes, n_lanes), np.inf)
        same_as_2_back = abs(p[i] - p[i - 2]) < 0.5
        for b in range(n_lanes):
            for c in range(n_lanes):
                step = _transition_cost(p[i] - p[i - 1], b, c, on_beat(i)) + unary[i, c]
                for a in range(n_lanes):
                    total = cost[a, b] + step
                    if same_as_2_back and a != c:
                        total += COST_SAME_PITCH_MOVED_2_BACK
                    if total < new[b, c]:
                        new[b, c] = total
                        came_from[i, b, c] = a
        cost = new

    # Volta do fim para o começo pelo caminho de menor custo.
    b, c = np.unravel_index(int(np.argmin(cost)), cost.shape)
    result = [int(c), int(b)]
    for i in range(n - 1, 1, -1):
        a = came_from[i, b, c]
        result.append(int(a))
        b, c = a, b
    return result[::-1]


PITCH_WINDOW_BEATS = 64  # mesma nota, mesma lane: a maioria entre as vezes que ela toca nesta distância (16 compassos)


def _unify_pitch_lanes(chosen: list[Candidate], lanes: list[int], sizes: list[int], resolution: int,
                       keep: set[int] = frozenset()) -> list[int]:
    """A mesma nota (simples) fica na mesma lane dentro de um trecho: a lane em que ela cai mais vezes
    a até PITCH_WINDOW_BEATS dali (no empate, a mais comum na música toda, e depois a da primeira vez
    que ela toca). Notas iguais seguidas mudam juntas. Não muda se a nota ficaria na lane de uma
    vizinha simples que é outra nota, ou se inverteria o sentido da melodia com ela (um acorde vizinho
    não conta: acorde seguido de nota numa das lanes dele é normal). Repete até nada mudar, porque
    mudar uma nota pode liberar a vizinha. `keep`: notas que não mudam."""
    window = PITCH_WINDOW_BEATS * resolution
    singles = [i for i, s in enumerate(sizes) if s == 1]
    ticks = np.array([chosen[i].tick for i in singles])
    overall: dict[int, Counter] = defaultdict(Counter)   # desempate: a lane da nota na música toda
    first_lane: dict[int, int] = {}                       # ...e, empatando de novo, a da 1ª vez que ela toca
    for i in singles:
        overall[round(chosen[i].pitch)][lanes[i]] += 1
        first_lane.setdefault(round(chosen[i].pitch), lanes[i])
    out = list(lanes)
    for _ in range(4):
        runs: list[list[int]] = []   # notas simples iguais e seguidas, na mesma lane
        for i in singles:
            if (runs and runs[-1][-1] == i - 1 and abs(chosen[i].pitch - chosen[i - 1].pitch) < 0.5
                    and out[i] == out[i - 1]):
                runs[-1].append(i)
            else:
                runs.append([i])
        changed = False
        for run in runs:
            first, last = run[0], run[-1]
            if keep & set(run):
                continue
            pitch = round(chosen[first].pitch)
            lo = np.searchsorted(ticks, chosen[first].tick - window)
            hi = np.searchsorted(ticks, chosen[last].tick + window, "right")
            votes = Counter(lanes[singles[k]] for k in range(lo, hi) if round(chosen[singles[k]].pitch) == pitch)
            want = max(votes, key=lambda lane: (votes[lane], overall[pitch][lane], lane == first_lane[pitch]))
            if want == out[first]:
                continue
            ok = True
            for j, i in ((first - 1, first), (last + 1, last)):
                if not 0 <= j < len(chosen) or sizes[j] >= 2:
                    continue
                if abs(chosen[j].pitch - chosen[i].pitch) < 0.5:   # a mesma nota, mas fora da sequência
                    ok = ok and out[j] == want
                elif out[j] == want or (chosen[j].pitch > chosen[i].pitch) != (out[j] > want):
                    ok = False
            if ok:
                for i in run:
                    out[i] = want
                changed = True
        if not changed:
            break
    return out


def _repeated_lanes(chosen: list[Candidate], lanes: list[int], repeats: Repeats,
                    resolution: int) -> dict[int, int]:
    """Lane obrigatória de cada nota de um bloco que se repete: as repetições usam o mesmo padrão
    de lanes.

    Um trecho que se repete na música é charteado igual. Sem isso, cada repetição ficava com as lanes
    que a janela de notas vizinhas decidisse ali (uma subida de 1 tom virava verde -> laranja num
    compasso e vermelho -> azul no seguinte, com as mesmas notas). O padrão escolhido é, entre os que
    aparecem nas repetições, o que deixa a mesma nota na mesma lane dentro do bloco; depois, o que põe
    as notas na lane de costume delas na música (assim blocos diferentes também concordam); depois, o
    que encaixa em mais repetições sem quebrar regra na emenda com as notas de fora (outra nota na
    mesma lane, contorno ao contrário); depois, o mais comum. Onde ele não encaixa, o bloco fica como está."""
    indexes: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(chosen):
        indexes[repeats.block(c.tick)].append(i)
    groups: dict[int, list[int]] = defaultdict(list)
    for block in sorted(indexes):
        groups[repeats.source.get(block, block)].append(block)

    def fits(idx: list[int], values: tuple[int, ...]) -> bool:
        pairs = []
        if idx[0] > 0:
            pairs.append((idx[0] - 1, idx[0], lanes[idx[0] - 1], values[0]))
        if idx[-1] + 1 < len(chosen):
            pairs.append((idx[-1], idx[-1] + 1, values[-1], lanes[idx[-1] + 1]))
        return all(_transition_cost(chosen[b].pitch - chosen[a].pitch, la, lb,
                                    chosen[b].tick % resolution == 0) < COST_RESET_ON_BEAT
                   for a, b, la, lb in pairs)

    usual: dict[int, Counter] = defaultdict(Counter)   # a lane de cada nota simples na música toda
    for c, lane in zip(chosen, lanes):
        if c.voices == 1:
            usual[round(c.pitch)][lane] += 1
    favorite = {pitch: votes.most_common(1)[0][0] for pitch, votes in usual.items()}

    fixed: dict[int, int] = {}
    for source, blocks in groups.items():
        shape = [repeats.offset(chosen[i].tick) for i in indexes[source]]
        blocks = [b for b in blocks if [repeats.offset(chosen[i].tick) for i in indexes[b]] == shape]
        if len(blocks) < 2:
            continue
        patterns = Counter(tuple(lanes[i] for i in indexes[b]) for b in blocks)
        pitches = [round(chosen[i].pitch) for i in indexes[source]]

        def steady(p: tuple[int, ...]) -> int:   # notas simples que caem sempre na mesma lane no bloco
            by_pitch: dict[int, set] = defaultdict(set)
            for i, pitch, lane in zip(indexes[source], pitches, p):
                if chosen[i].voices == 1:
                    by_pitch[pitch].add(lane)
            return -sum(len(v) - 1 for v in by_pitch.values())

        def usual_lanes(p: tuple[int, ...]) -> int:   # notas simples na lane de costume delas
            return sum(chosen[i].voices == 1 and favorite.get(pitch) == lane
                       for i, pitch, lane in zip(indexes[source], pitches, p))

        best = max(patterns, key=lambda p: (steady(p), usual_lanes(p), sum(fits(indexes[b], p) for b in blocks),
                                             patterns[p]))
        for b in blocks:
            if fits(indexes[b], best):
                fixed.update(zip(indexes[b], best))
    return fixed


def _unify_chord_lanes(chosen: list[Candidate], lanes: list[int]) -> list[int]:
    """Cada acorde (mesma altura e mesmo nº de notas) usa a lane que recebe na maioria das vezes.

    Assim o mesmo acorde tem sempre o mesmo formato na chart. Só não muda uma nota
    se ela ficaria na mesma lane de uma vizinha que é outro acorde/nota (aí o
    jogador veria "o mesmo acorde" onde a música muda).
    """
    def identity(c: Candidate) -> tuple[int, int]:
        return round(c.pitch), c.voices

    votes: dict[tuple[int, int], Counter] = defaultdict(Counter)
    for c, lane in zip(chosen, lanes):
        if c.voices >= 2:
            votes[identity(c)][lane] += 1
    favorite = {key: v.most_common(1)[0][0] for key, v in votes.items()}

    lanes = list(lanes)
    for i, c in enumerate(chosen):
        want = favorite.get(identity(c)) if c.voices >= 2 else None
        if want is None or want == lanes[i]:
            continue
        clash = any(0 <= j < len(chosen) and identity(chosen[j]) != identity(c) and lanes[j] == want
                    for j in (i - 1, i + 1))
        if not clash:
            lanes[i] = want
    return lanes


# Um acorde precisa aparecer pelo menos tantas vezes para ganhar um formato próprio na tabela.
MIN_CHORD_OCCURRENCES = 3
COST_SMALLER_SHAPE = 1.0      # usar um formato com menos notas que o acorde
COST_SHARED_SHAPE = 3.0       # dois acordes diferentes com o mesmo formato
COST_SHARED_NEIGHBORS = 30.0  # ...que tocam um logo depois do outro: o jogador não veria a troca
COST_PER_LANE_FROM_SPREAD = 0.3


def _shape_palette(n_lanes: int, max_chord: int) -> list[tuple[int, ...]]:
    """Todos os formatos de acorde possíveis, do mais grave para o mais agudo.

    Ex. com 5 lanes e até 3 notas: GR, GY, GRY, RY, GB, RB, RYB, YB, RO, YO, YBO, BO. Os de 2 notas
    vão até 2 lanes de intervalo (GB, RO): uma música com muitos power chords precisa de formatos
    para que acordes seguidos não fiquem iguais.
    """
    shapes = set()
    for size in range(2, max_chord + 1):
        for base in range(n_lanes):
            shape = tuple(range(base, base + size))           # vizinhas (GRY, RY...)
            if shape[-1] < n_lanes:
                shapes.add(shape)
        if size == 2:
            shapes.update((b, b + 2) for b in range(n_lanes - 2))  # pulando uma (GY, RB...)
            shapes.update((b, b + 3) for b in range(n_lanes - 3))  # pulando duas (GB, RO)
    return sorted(shapes, key=lambda s: (np.mean(s), -len(s), s))


def _chord_shape_table(chosen: list[Candidate], sizes: list[int], n_lanes: int,
                       max_chord: int) -> dict[tuple[int, int], tuple[int, ...]]:
    """Formato fixo para cada acorde da música (identificado por altura e nº de notas).

    Os acordes são ordenados do mais grave para o mais agudo e distribuídos pelos
    formatos disponíveis na mesma ordem (o contorno continua certo), preferindo
    formatos com o nº de notas do acorde e evitando dois acordes no mesmo formato.
    Com mais acordes que formatos, alguns dividem formato; dois acordes que tocam um logo
    depois do outro na música (ex.: Sol5 -> Fá5) quase nunca: o jogador não veria a troca.
    Acordes de 2 e de 3 notas são distribuídos cada um na sua fila (um acorde de 3 notas no
    meio da fila dos de 2 impediria dois deles de dividir formato).
    """
    counts = Counter((round(c.pitch), size) for c, size in zip(chosen, sizes) if size >= 2)
    neighbors = set()   # pares de acordes diferentes que tocam em seguida
    keys = [(round(c.pitch), size) if size >= 2 else None for c, size in zip(chosen, sizes)]
    for a, b in zip(keys, keys[1:]):
        if a and b and a != b:
            neighbors.add(frozenset((a, b)))
    palette = _shape_palette(n_lanes, max_chord)
    table: dict[tuple[int, int], tuple[int, ...]] = {}
    for size in sorted({key[1] for key in counts}):
        chords = sorted(key for key, n in counts.items() if key[1] == size and n >= MIN_CHORD_OCCURRENCES)
        shapes = [s for s in palette if len(s) <= size]
        if chords and shapes:
            table.update(_spread_shapes(chords, shapes, neighbors))
    return table


def _spread_shapes(chords: list[tuple[int, int]], palette: list[tuple[int, ...]],
                   neighbors: set) -> dict[tuple[int, int], tuple[int, ...]]:
    """Distribui os acordes (do mais grave ao mais agudo) pelos formatos, na mesma ordem."""
    centers = np.array([np.mean(s) for s in palette])
    targets = np.linspace(centers[0], centers[-1], len(chords)) if len(chords) > 1 else [np.mean(centers)]

    def fit(i: int, j: int) -> float:
        want, size = chords[i][1], len(palette[j])
        if size > want:
            return np.inf
        return (COST_SMALLER_SHAPE if size < want else 0.0) + COST_PER_LANE_FROM_SPREAD * abs(centers[j] - targets[i])

    # cost[i][j]: melhor custo dos acordes 0..i, com o acorde i no formato j (ordem crescente).
    n, m = len(chords), len(palette)
    cost = np.full((n, m), np.inf)
    back = np.zeros((n, m), dtype=int)
    cost[0] = [fit(0, j) for j in range(m)]
    for i in range(1, n):
        shared = COST_SHARED_NEIGHBORS if frozenset((chords[i - 1], chords[i])) in neighbors else COST_SHARED_SHAPE
        for j in range(m):
            here = fit(i, j)
            if here == np.inf:
                continue
            options = [cost[i - 1][p] + (shared if p == j else 0.0) for p in range(j + 1)]
            back[i][j] = int(np.argmin(options))
            cost[i][j] = min(options) + here
    j = int(np.argmin(cost[-1]))
    if cost[-1][j] == np.inf:
        return {}
    table = {}
    for i in range(n - 1, -1, -1):
        table[chords[i]] = palette[j]
        j = back[i][j]
    return table


def chord_shape(base: int, size: int, n_lanes: int) -> tuple[int, ...]:
    """Formato do acorde a partir da lane mais grave."""
    if size <= 1:
        return (base,)
    if size == 2:
        # "Power chord": pula uma lane (ex.: verde + amarelo) quando cabe.
        gap = 2 if n_lanes >= 4 else 1
        base = min(base, n_lanes - 1 - gap)
        return (base, base + gap)
    base = min(base, n_lanes - size)
    return tuple(range(base, base + size))


def _apply_sustains(notes: list[ChartNote], cands: list[Candidate], rules: FretDifficultyRules,
                    resolution: int) -> None:
    min_len = rules.sustain_min_beats * resolution
    gap = resolution // 4  # deixa um respiro de 1/16 antes da próxima nota
    for i, (note, cand) in enumerate(zip(notes, cands)):
        length = cand.end_tick - note.tick
        if i + 1 < len(notes):
            length = min(length, notes[i + 1].tick - note.tick - gap)
        note.length = int(length) if length >= min_len else 0


def _apply_forcing(notes: list[ChartNote], resolution: int, allow_hopos: bool) -> None:
    """Nas dificuldades fáceis, transforma HOPOs automáticos em notas palhetadas."""
    if allow_hopos:
        return
    threshold = int(65 / 192 * resolution)  # limite de HOPO automático do Clone Hero
    for prev, note in zip(notes, notes[1:]):
        natural_hopo = (
            len(note.lanes) == 1
            and note.tick - prev.tick <= threshold
            and note.lanes[0] not in prev.lanes
        )
        if natural_hopo:
            note.flags = (FORCE_FLAG,)


def build_fret_track(cands: list[Candidate], rules: dict[str, FretDifficultyRules],
                     resolution: int, lane_window: int,
                     sp_windows: list[tuple[int, int]]) -> TrackChart:
    """Monta as quatro dificuldades a partir das notas candidatas (já quantizadas)."""
    track = TrackChart(notes={})
    # Trechos com o mesmo ritmo e as mesmas notas são charteados igualzinho.
    repeats = find_repeats(cands, resolution, key=lambda c: (round(c.pitch), c.voices))
    previous_count = len(cands)
    for difficulty, r in rules.items():
        chosen = select_rhythm(cands, resolution, r.min_gap_beats, r.grid, r.allow_offgrid)
        chosen = copy_repeats(chosen, cands, repeats, resolution, min_gap_beats=r.min_gap_beats)
        chosen = enforce_reduction(chosen, previous_count, r.min_reduction, resolution,
                                   group=lambda c: (repeats.source_of(c.tick), repeats.offset(c.tick)))
        previous_count = len(chosen)
        sizes = [min(c.voices, r.max_chord) for c in chosen]
        max_lanes = [min(chord_shape(r.lanes - 1, s, r.lanes)) for s in sizes]
        lanes = assign_lanes([c.pitch for c in chosen], r.lanes, lane_window,
                             [c.tick for c in chosen], resolution, max_lanes=max_lanes)
        lanes = _unify_pitch_lanes(chosen, _unify_chord_lanes(chosen, lanes), sizes, resolution)
        # Blocos repetidos com as mesmas lanes (só onde o padrão encaixa com as notas vizinhas).
        fixed = _repeated_lanes(chosen, lanes, repeats, resolution)
        lanes = [fixed.get(i, lane) for i, lane in enumerate(lanes)]
        # Acordes que se repetem usam a tabela de formatos; o resto segue as lanes.
        table = _chord_shape_table(chosen, sizes, r.lanes, r.max_chord)
        notes = [ChartNote(c.tick, table.get((round(c.pitch), size)) or chord_shape(lane, size, r.lanes))
                 for c, lane, size in zip(chosen, lanes, sizes)]
        _apply_sustains(notes, chosen, r, resolution)
        _apply_forcing(notes, resolution, r.allow_hopos)
        track.notes[difficulty] = notes
        track.star_power[difficulty] = star_power_for(notes, sp_windows)
    return track
