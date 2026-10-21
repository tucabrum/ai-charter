"""Peças compartilhadas entre a chart de guitarra/baixo/teclado e a de bateria."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


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


@dataclass
class Candidate:
    """Uma nota detectada já posicionada em ticks, antes de virar ChartNote."""
    tick: int
    strength: float
    end_tick: int = 0
    pitch: float = 0.0
    voices: int = 1
    piece: str = ""   # só para bateria


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
                      resolution: int) -> list[Candidate]:
    """Garante que a dificuldade tenha pelo menos `min_reduction` (0-1) notas a menos
    que a dificuldade acima, removendo primeiro as notas menos importantes."""
    limit = int(previous_count * (1.0 - min_reduction))
    if min_reduction <= 0 or len(chosen) <= limit:
        return chosen
    ranked = sorted(chosen, key=lambda c: c.strength + position_weight(c.tick, resolution), reverse=True)
    return sorted(ranked[:limit], key=lambda c: c.tick)


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
