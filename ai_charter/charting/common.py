"""Peças compartilhadas entre a chart de guitarra/baixo/teclado e a de bateria."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Hashable, Sequence

import numpy as np


@dataclass
class ChartNote:
    """Uma nota já pronta para ir para o arquivo .chart."""
    tick: int
    lanes: tuple[int, ...]           # números de nota do .chart (0 = verde, 1 = vermelho, ...)
    length: int = 0                  # duração do sustain em ticks (0 = sem sustain)
    flags: tuple[int, ...] = ()      # marcadores extras (5 = forçar strum/HOPO, 66-68 = pratos)


@dataclass
class TrackChart:
    """As quatro dificuldades de um instrumento."""
    notes: dict[str, list[ChartNote]]                     # "expert" -> notas
    star_power: dict[str, list[tuple[int, int]]] = field(default_factory=dict)  # (tick, duração)
    drum_fills: dict[str, list[tuple[int, int]]] = field(default_factory=dict)  # só bateria
    # Só bateria com modelo: quão em dúvida o modelo ficou em cada golpe (tick -> 0 a 1)
    doubt: dict[int, float] = field(default_factory=dict)


@dataclass
class Candidate:
    """Uma nota detectada já posicionada em ticks, antes de virar ChartNote."""
    tick: int
    strength: float
    end_tick: int = 0
    pitch: float = 0.0
    voices: int = 1
    piece: str = ""   # só para bateria
    chord: int = -1   # acorde reconhecido (índice em analysis.chords.CHORDS), -1 = nenhum
    chord_scores: np.ndarray | None = None
    lowest_by_class: np.ndarray | None = None
    fifth_ratio: float = 0.0
    third_ratio: float = 0.0
    upper: tuple[int, ...] = ()   # notas de cima do acorde, quando vêm da partitura (o mesmo baixo pode
                                  # ter acordes diferentes: Fá3+Lá3, Fá3+Si♭3, Fá3+Sol3)


def quantize_tick(tick: float, resolution: int, allow_triplets: bool) -> int:
    """Encaixa um tick no grid mais próximo (semicolcheia, ou tercina se fizer mais sentido)."""
    straight = resolution / 4  # semicolcheia
    q = round(tick / straight) * straight
    if allow_triplets:
        triplet = resolution / 3  # tercina de colcheia
        qt = round(tick / triplet) * triplet
        # Só usa tercina se ela for MUITO mais precisa que a semicolcheia.
        if abs(tick - qt) < 0.5 * abs(tick - q) and abs(tick - q) > straight * 0.25:
            q = qt
    return int(round(q))


def quantize_runs(ticks, resolution: int, allow_triplets: bool) -> list[int]:
    """Como quantize_tick, mas golpes colados (mais perto que 3/4 de semicolcheia, como um pedal
    duplo ou um rufo) vão juntos para a grade mais fina que melhor os encaixa: fusa ou tercina de
    semicolcheia. Na grade comum dois deles cairiam no mesmo tick e virariam uma nota só."""
    ticks = np.asarray(ticks, dtype=float)
    out = [quantize_tick(t, resolution, allow_triplets) for t in ticks]
    order = np.argsort(ticks, kind="stable")
    close = resolution / 4 * 0.75
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and ticks[order[j + 1]] - ticks[order[j]] < close:
            j += 1
        if j > i:
            run = ticks[order[i:j + 1]]
            grids = [resolution / 8] + ([resolution / 6] if allow_triplets else [])
            best = min(grids, key=lambda g: float(np.mean(np.abs(run - np.round(run / g) * g))))
            q = (np.round(run / best) * best).round().astype(int)
            if len(set(q.tolist())) == len(q):   # só se cada golpe ficar com o seu tick
                for k, idx in enumerate(order[i:j + 1]):
                    out[idx] = int(q[k])
        i = j + 1
    return out


def position_weight(tick: int, resolution: int) -> float:
    """Quão "importante" é uma posição rítmica (início do compasso > batida > contratempo)."""
    if tick % (resolution * 4) == 0:
        return 1.0
    if tick % resolution == 0:
        return 0.8
    if tick % (resolution // 2) == 0:
        return 0.55
    if tick % (resolution // 4) == 0:
        return 0.35
    return 0.3  # tercinas e afins


def on_grid(tick: int, resolution: int, grid: int) -> bool:
    """True se o tick cai numa das `grid` subdivisões da batida."""
    return tick % (resolution // grid) == 0


def select_rhythm(cands: Sequence[Candidate], resolution: int, min_gap_beats: float,
                  grid: int, allow_offgrid: bool) -> list[Candidate]:
    """Escolhe quais notas ficam numa dificuldade.

    As notas mais importantes (fortes e em tempos fortes) são escolhidas
    primeiro; uma nota só entra se não estiver perto demais de outra já escolhida.
    """
    min_gap = min_gap_beats * resolution
    pool = [c for c in cands if allow_offgrid or on_grid(c.tick, resolution, grid)]
    pool.sort(key=lambda c: c.strength + position_weight(c.tick, resolution), reverse=True)

    chosen: list[Candidate] = []
    taken: list[int] = []
    for c in pool:
        if all(abs(c.tick - t) >= min_gap - 1e-6 for t in taken):
            chosen.append(c)
            taken.append(c.tick)
    chosen.sort(key=lambda c: c.tick)
    return chosen


def enforce_reduction(chosen: list[Candidate], previous_count: int, min_reduction: float,
                      resolution: int,
                      group: Callable[[Candidate], Hashable] | None = None) -> list[Candidate]:
    """Garante que a dificuldade tenha pelo menos `min_reduction` (0-1) notas a menos
    que a dificuldade acima, removendo primeiro as notas menos importantes.

    `group(c)`: notas com o mesmo grupo (ex.: a mesma nota em cada repetição de
    um riff) saem juntas, para as repetições continuarem iguais.
    """
    limit = int(previous_count * (1.0 - min_reduction))
    if min_reduction <= 0 or len(chosen) <= limit:
        return chosen
    groups: dict[Hashable, list[Candidate]] = defaultdict(list)
    for i, c in enumerate(chosen):
        groups[group(c) if group else i].append(c)
    # Remove os grupos menos importantes até caber no limite.
    ranked = sorted(groups.values(), key=lambda g: np.mean(
        [c.strength + position_weight(c.tick, resolution) for c in g]))
    count = len(chosen)
    removed: set[int] = set()
    for g in ranked:
        if count <= limit:
            break
        removed.update(id(c) for c in g)
        count -= len(g)
    return [c for c in chosen if id(c) not in removed]


@dataclass
class Repeats:
    """Blocos de 4 batidas que se repetem na música.

    Os blocos começam em `origin` (não necessariamente no "1" do compasso:
    o programa não sabe onde o compasso começa de verdade).
    """
    origin: int
    length: int
    source: dict[int, int]   # bloco -> primeiro bloco com conteúdo idêntico

    def block(self, tick: int) -> int:
        return (tick - self.origin) // self.length

    def offset(self, tick: int) -> int:
        return (tick - self.origin) % self.length

    def source_of(self, tick: int) -> int:
        b = self.block(tick)
        return self.source.get(b, b)


def find_repeats(cands: Sequence[Candidate], resolution: int,
                 key: Callable[[Candidate], Hashable]) -> Repeats:
    """Encontra blocos de 4 batidas com conteúdo idêntico (mesmo ritmo e notas).

    `key(c)` diz o que conta como "igual" (ex.: a nota, ou a peça da bateria).
    Testa os 4 pontos de início possíveis e fica com o que acha mais repetições.
    """
    length = resolution * 4
    best = None
    for beat in range(4):
        origin = beat * resolution - length
        content: dict[int, list] = defaultdict(list)
        for c in cands:
            content[(c.tick - origin) // length].append(((c.tick - origin) % length, key(c)))
        first: dict[tuple, int] = {}
        source = {b: first.setdefault(tuple(sorted(content[b], key=repr)), b) for b in sorted(content)}
        repeated = sum(1 for b, s in source.items() if b != s)
        if best is None or repeated > best[0]:
            best = (repeated, Repeats(origin, length, source))
    return best[1] if best else Repeats(0, length, {})


def copy_repeats(chosen: Sequence[Candidate], cands: Sequence[Candidate], repeats: Repeats,
                 resolution: int, ident: Callable[[Candidate], Hashable] = lambda c: None,
                 min_gap_beats: float = 0.0) -> list[Candidate]:
    """Faz cada bloco repetido usar a MESMA escolha de notas do original.

    Sem isso, duas repetições de um riff podem perder notas diferentes ao
    simplificar (por pequenas diferenças de volume na gravação).
    `ident(c)` distingue notas na mesma posição (ex.: peça da bateria).
    """
    kept = {(repeats.block(c.tick), repeats.offset(c.tick), ident(c)) for c in chosen}
    result = [c for c in cands
              if (repeats.source_of(c.tick), repeats.offset(c.tick), ident(c)) in kept]
    if min_gap_beats > 0:
        # Na emenda entre blocos, garante a distância mínima entre notas.
        min_gap = min_gap_beats * resolution
        cleaned: list[Candidate] = []
        for c in sorted(result, key=lambda c: c.tick):
            if not cleaned or c.tick - cleaned[-1].tick >= min_gap - 1e-6:
                cleaned.append(c)
        result = cleaned
    return sorted(result, key=lambda c: c.tick)


def dedupe(cands: list[Candidate]) -> list[Candidate]:
    """Se duas notas caíram no mesmo tick, mantém só a mais forte."""
    best: dict[int, Candidate] = {}
    for c in cands:
        if c.tick not in best or c.strength > best[c.tick].strength:
            best[c.tick] = c
    return [best[t] for t in sorted(best)]


def star_power_windows(last_tick: int, resolution: int, every_measures: int,
                       phrase_beats: int) -> list[tuple[int, int]]:
    """Janelas (início, duração) onde haverá frases de Star Power.

    São as mesmas para todas as dificuldades, como nas charts oficiais.
    """
    measure = resolution * 4
    windows = []
    start = every_measures * measure // 2  # primeira frase no meio do 1º bloco
    while start + phrase_beats * resolution < last_tick:
        windows.append((start, phrase_beats * resolution))
        start += every_measures * measure
    return windows


def star_power_for(notes: list[ChartNote], windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Ajusta cada janela de Star Power às notas que estão dentro dela."""
    phrases = []
    for start, length in windows:
        inside = [n for n in notes if start <= n.tick < start + length]
        if inside:
            first, last = inside[0], inside[-1]
            phrases.append((first.tick, max(1, last.tick + last.length - first.tick + 1)))
    return phrases


def star_power_for_phrases(phrases: list[tuple[int, int]],
                           windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Star Power do vocal: em cada janela, a primeira frase (inteira) que a toca."""
    chosen = []
    for start, length in windows:
        touching = [p for p in phrases if p[0] < start + length and p[0] + p[1] > start]
        if touching and touching[0] not in chosen:
            chosen.append(touching[0])
    return chosen


def drum_fills_for(notes: list[ChartNote], windows: list[tuple[int, int]], resolution: int,
                   every_measures: int) -> list[tuple[int, int]]:
    """Viradas de ativação do Star Power na bateria (onde o jogador pode ativá-lo).

    Cada virada dura 2 batidas e termina numa nota no 1º tempo de um compasso;
    essa última nota é a que ativa o Star Power. Não pode sobrepor uma frase de Star Power.
    """
    if every_measures <= 0 or not notes:
        return []
    measure = resolution * 4
    note_ticks = {n.tick for n in notes}
    fills = []
    downbeat = every_measures * measure
    while downbeat <= notes[-1].tick:
        start = downbeat - 2 * resolution
        overlaps = any(start <= w + length and w <= downbeat for w, length in windows)
        if downbeat in note_ticks and not overlaps:
            fills.append((start, 2 * resolution))
        downbeat += every_measures * measure
    return fills
