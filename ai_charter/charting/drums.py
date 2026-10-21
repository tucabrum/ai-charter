"""Gera a chart de bateria (Pro Drums: bumbo + 4 lanes, com pratos marcados)."""

from __future__ import annotations

from ai_charter.analysis.drums import CRASH, HIHAT, KICK, SNARE
from ai_charter.charting.common import (
    Candidate, ChartNote, TrackChart, enforce_reduction, select_rhythm, star_power_for,
)
from ai_charter.config import DrumDifficultyRules

# Lane do .chart para cada peça, e o marcador de prato (Pro Drums) quando houver.
#   0 = bumbo, 1 = vermelho (caixa), 2 = amarelo, 3 = azul, 4 = verde
#   66 = amarelo é prato, 67 = azul é prato, 68 = verde é prato
LANES = {KICK: 0, SNARE: 1, HIHAT: 2, CRASH: 4}
CYMBAL_FLAGS = {HIHAT: 66, CRASH: 68}
# Se houver mais mãos que o permitido, quem fica primeiro.
HAND_PRIORITY = (CRASH, SNARE, HIHAT)


def build_drum_track(cands: list[Candidate], rules: dict[str, DrumDifficultyRules],
                     resolution: int, sp_windows: list[tuple[int, int]]) -> TrackChart:
    track = TrackChart(notes={})
    previous_count = len(cands)
    for difficulty, r in rules.items():
        # Sem crash nesta dificuldade: o crash vira um prato comum (amarelo).
        hits = [c if r.crash or c.piece != CRASH else
                Candidate(c.tick, c.strength, piece=HIHAT) for c in cands]

        # Escolhe o ritmo separadamente para pé, caixa e pratos.
        groups = {
            "kick": ([h for h in hits if h.piece == KICK], r.kick_min_gap_beats),
            "snare": ([h for h in hits if h.piece == SNARE], r.snare_min_gap_beats),
            "cymbal": ([h for h in hits if h.piece in (HIHAT, CRASH)], r.hihat_min_gap_beats),
        }
        chosen = [c for group, gap in groups.values()
                  for c in select_rhythm(group, resolution, gap, r.grid, r.allow_offgrid)]
        chosen = enforce_reduction(chosen, previous_count, r.min_reduction, resolution)
        previous_count = len(chosen)

        by_tick: dict[int, set[str]] = {}
        for c in chosen:
            by_tick.setdefault(c.tick, set()).add(c.piece)

        notes = []
        for tick in sorted(by_tick):
            pieces = by_tick[tick]
            hands = [p for p in HAND_PRIORITY if p in pieces][: r.max_hands]
            kept = ([KICK] if KICK in pieces else []) + hands
            notes.append(ChartNote(
                tick,
                lanes=tuple(sorted(LANES[p] for p in kept)),
                flags=tuple(CYMBAL_FLAGS[p] for p in kept if p in CYMBAL_FLAGS),
            ))
        track.notes[difficulty] = notes
        track.star_power[difficulty] = star_power_for(notes, sp_windows)
    return track
