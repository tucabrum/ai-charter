"""Gera a chart de bateria (Pro Drums: bumbo + 4 lanes, com pratos marcados)."""

from __future__ import annotations

from ai_charter.analysis.drums import CRASH, HIHAT, KICK, RIDE, SNARE, TOM_HIGH, TOM_LOW, TOM_MID, TOMS
from ai_charter.charting.common import (
    Candidate, ChartNote, TrackChart, copy_repeats, drum_fills_for, enforce_reduction,
    find_repeats, select_rhythm, star_power_for,
)
from ai_charter.charting.drum_reduce import ReduceModel
from ai_charter.charting.drum_reduce import features as reduce_features
from ai_charter.config import DrumDifficultyRules

# Lane do .chart para cada peça, e o marcador de prato (Pro Drums) quando houver.
#   0 = bumbo, 1 = vermelho (caixa), 2 = amarelo, 3 = azul, 4 = verde
#   Pratos: chimbal = amarelo, condução = azul, ataque = verde (66/67/68 = "é prato")
#   Tons:   agudo = amarelo, médio = azul, surdo = verde (sem marcador = tom)
LANES = {KICK: 0, SNARE: 1, HIHAT: 2, RIDE: 3, CRASH: 4, TOM_HIGH: 2, TOM_MID: 3, TOM_LOW: 4}
CYMBAL_FLAGS = {HIHAT: 66, RIDE: 67, CRASH: 68}
# Se houver mais mãos que o permitido (ou duas peças na mesma lane), quem fica primeiro.
HAND_PRIORITY = (CRASH, SNARE, TOM_HIGH, TOM_MID, TOM_LOW, RIDE, HIHAT)


def build_drum_track(cands: list[Candidate], rules: dict[str, DrumDifficultyRules],
                     resolution: int, sp_windows: list[tuple[int, int]],
                     fill_every_measures: int, bpm_at=None) -> TrackChart:
    """As quatro dificuldades. Com o modelo de redução (drum_reduce.json), Hard, Medium e Easy
    são as notas do Expert que ele mantém, como a Harmonix faria; sem ele, as regras de cada uma."""
    track = TrackChart(notes={})
    reducer = ReduceModel.load()
    expert_chosen, keep = None, {}
    # Trechos com as mesmas batidas são charteados igualzinho.
    repeats = find_repeats(cands, resolution, key=lambda c: c.piece)
    previous_count = len(cands)
    for difficulty, r in rules.items():
        # Sem crash nesta dificuldade: o crash vira um prato comum (amarelo).
        hits = [c if r.crash or c.piece != CRASH else
                Candidate(c.tick, c.strength, piece=HIHAT) for c in cands]

        if reducer is not None and expert_chosen is not None and difficulty in keep:
            chosen = [c for c, k in zip(expert_chosen, keep[difficulty]) if k]
            if not r.crash:
                chosen = [c if c.piece != CRASH else Candidate(c.tick, c.strength, piece=HIHAT) for c in chosen]
            previous_count = len(chosen)
            track.notes[difficulty] = _notes_for(chosen, r)
            track.star_power[difficulty] = star_power_for(track.notes[difficulty], sp_windows)
            track.drum_fills[difficulty] = drum_fills_for(track.notes[difficulty], sp_windows, resolution,
                                                          fill_every_measures)
            continue
        # Escolhe o ritmo separadamente para pé, caixa, tons e pratos.
        groups = {
            "kick": ([h for h in hits if h.piece == KICK], r.kick_min_gap_beats),
            "snare": ([h for h in hits if h.piece == SNARE], r.snare_min_gap_beats),
            "toms": ([h for h in hits if h.piece in TOMS], r.snare_min_gap_beats),
            "cymbal": ([h for h in hits if h.piece in (HIHAT, RIDE, CRASH)], r.hihat_min_gap_beats),
        }
        chosen = [c for group, gap in groups.values()
                  for c in select_rhythm(group, resolution, gap, r.grid, r.allow_offgrid)]
        chosen = copy_repeats(chosen, hits, repeats, resolution, ident=lambda c: c.piece)
        chosen = enforce_reduction(
            chosen, previous_count, r.min_reduction, resolution,
            group=lambda c: (repeats.source_of(c.tick), repeats.offset(c.tick), c.piece))
        previous_count = len(chosen)
        if difficulty == "expert" and reducer is not None:
            expert_chosen = sorted(chosen, key=lambda c: (c.tick, c.piece))
            keep = reducer.keep(reduce_features([(c.tick, c.piece) for c in expert_chosen], resolution,
                                                bpm_at or (lambda _t: 120.0)))
        notes = _notes_for(chosen, r)
        track.notes[difficulty] = notes
        track.star_power[difficulty] = star_power_for(notes, sp_windows)
        track.drum_fills[difficulty] = drum_fills_for(notes, sp_windows, resolution,
                                                      fill_every_measures)
    return track


def _notes_for(chosen: list[Candidate], r: DrumDifficultyRules) -> list[ChartNote]:
    """As notas de uma dificuldade: uma peça por lane e no máximo r.max_hands mãos (mais o bumbo)."""
    by_tick: dict[int, set[str]] = {}
    for c in chosen:
        by_tick.setdefault(c.tick, set()).add(c.piece)
    notes = []
    for tick in sorted(by_tick):
        pieces = by_tick[tick]
        hands, lanes = [], set()
        for p in HAND_PRIORITY:   # uma peça por lane (ex.: chimbal e tom agudo são o amarelo)
            if p in pieces and LANES[p] not in lanes and len(hands) < r.max_hands:
                hands.append(p)
                lanes.add(LANES[p])
        kept = ([KICK] if KICK in pieces else []) + hands
        notes.append(ChartNote(
            tick,
            lanes=tuple(sorted(LANES[p] for p in kept)),
            flags=tuple(CYMBAL_FLAGS[p] for p in kept if p in CYMBAL_FLAGS),
        ))
    return notes
