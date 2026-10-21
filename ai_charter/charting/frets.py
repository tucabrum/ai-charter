"""Gera as charts de guitarra, baixo e teclado (5 lanes) em cada dificuldade.

Regra principal de lanes (como fazem os charters humanos): o jogo não tem
como representar a nota exata, então seguimos o CONTORNO da melodia:
  - nota mais aguda que a anterior -> lane mais à direita;
  - nota mais grave                -> lane mais à esquerda;
  - mesma nota                     -> mesma lane.
"""

from __future__ import annotations

import numpy as np

from ai_charter.charting.common import (
    Candidate, ChartNote, TrackChart, enforce_reduction, select_rhythm, star_power_for,
)
from ai_charter.config import FretDifficultyRules

FORCE_FLAG = 5  # no .chart, a nota 5 inverte strum <-> HOPO


def assign_lanes(pitches: list[float], n_lanes: int, window: int) -> list[int]:
    """Converte alturas (MIDI) em lanes 0..n_lanes-1 seguindo o contorno melódico."""
    p = np.asarray(pitches, dtype=float)
    lanes: list[int] = []
    top = n_lanes - 1
    for i in range(len(p)):
        # Olha as notas vizinhas para saber a "região" atual da melodia.
        nearby = p[max(0, i - window): i + window + 1]
        lo, hi = nearby.min(), nearby.max()
        span = max(hi - lo, top)  # pelo menos 1 semitom por lane
        pad = (span - (hi - lo)) / 2  # centraliza trechos com pouca variação
        lanes.append(int(round((p[i] - lo + pad) / span * top)))

    # Garante que o contorno seja respeitado nota a nota.
    for i in range(1, len(p)):
        diff, prev = p[i] - p[i - 1], lanes[i - 1]
        if abs(diff) < 0.5:
            lanes[i] = prev
        elif diff > 0 and lanes[i] <= prev:
            lanes[i] = min(prev + 1, top)
        elif diff < 0 and lanes[i] >= prev:
            lanes[i] = max(prev - 1, 0)
    return lanes


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
        lanes = assign_lanes([c.pitch for c in chosen], r.lanes, lane_window) if chosen else []
        notes = [
            ChartNote(c.tick, chord_shape(lane, min(c.voices, r.max_chord), r.lanes))
            for c, lane in zip(chosen, lanes)
        ]
        _apply_sustains(notes, chosen, r, resolution)
        _apply_forcing(notes, resolution, r.allow_hopos)
        track.notes[difficulty] = notes
        track.star_power[difficulty] = star_power_for(notes, sp_windows)
    return track
