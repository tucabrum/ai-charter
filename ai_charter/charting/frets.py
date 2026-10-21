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

from ai_charter.charting.common import (
    Candidate, ChartNote, TrackChart, enforce_reduction, select_rhythm, star_power_for,
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
                 ticks: list[int] | None = None, resolution: int = 192) -> list[int]:
    """Converte alturas (MIDI) em lanes 0..n_lanes-1 seguindo o contorno melódico.

    Usa programação dinâmica (algoritmo de Viterbi): para cada nota, guarda o
    menor custo para chegar a cada par (lane da nota anterior, lane da nota atual);
    no fim, volta pelo caminho mais barato. Assim a decisão considera a frase
    inteira, e cada nota é comparada com a anterior e com a retrasada.
    """
    p = np.asarray(pitches, dtype=float)
    n = len(p)
    if n == 0:
        return []
    ideal = _ideal_lanes(p, n_lanes, window)
    unary = COST_PER_LANE_FROM_IDEAL * np.abs(np.arange(n_lanes)[None, :] - ideal[:, None])
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
    previous_count = len(cands)
    for difficulty, r in rules.items():
        chosen = select_rhythm(cands, resolution, r.min_gap_beats, r.grid, r.allow_offgrid)
        chosen = enforce_reduction(chosen, previous_count, r.min_reduction, resolution)
        previous_count = len(chosen)
        lanes = assign_lanes([c.pitch for c in chosen], r.lanes, lane_window,
                             [c.tick for c in chosen], resolution)
        notes = [
            ChartNote(c.tick, chord_shape(lane, min(c.voices, r.max_chord), r.lanes))
            for c, lane in zip(chosen, lanes)
        ]
        _apply_sustains(notes, chosen, r, resolution)
        _apply_forcing(notes, resolution, r.allow_hopos)
        track.notes[difficulty] = notes
        track.star_power[difficulty] = star_power_for(notes, sp_windows)
    return track
