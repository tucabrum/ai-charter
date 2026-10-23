"""Testes rápidos das peças individuais (sem processar áudio)."""

import numpy as np
import pytest

from ai_charter.analysis.tempo import TempoMap
from ai_charter.charting.common import (
    Candidate, ChartNote, drum_fills_for, quantize_tick, select_rhythm, star_power_windows,
)
from ai_charter.charting.frets import FORCE_FLAG, assign_lanes, build_fret_track, chord_shape
from ai_charter.config import load_rules
from ai_charter.export.chart_writer import SongInfo, write_chart
from chart_parser import notes_by_tick, parse_chart

RES = 192


def constant_map(bpm=120.0, beats=64):
    return TempoMap(np.arange(beats) * 60.0 / bpm, RES)


def test_tempo_map_round_trip():
    tempo = constant_map()
    assert tempo.time_to_tick(1.0) == pytest.approx(2 * RES)
    assert tempo.tick_to_time(3 * RES) == pytest.approx(1.5)
    assert tempo.bpm_events() == [(0, 120000)]
    # Depois da última batida continua no mesmo andamento.
    assert tempo.time_to_tick(100 * 0.5) == pytest.approx(100 * RES)


def test_quantize_prefers_sixteenths_but_accepts_triplets():
    assert quantize_tick(50, RES, allow_triplets=True) == 48
    assert quantize_tick(65, RES, allow_triplets=True) == 64      # tercina de colcheia
    assert quantize_tick(65, RES, allow_triplets=False) == 48


def test_lanes_follow_melodic_contour():
    lanes = assign_lanes([40, 43, 45, 45, 47, 43, 40], n_lanes=5, window=8)
    assert lanes[0] == 0
    assert lanes[1] > lanes[0] and lanes[2] > lanes[1]
    assert lanes[3] == lanes[2]                       # mesma nota -> mesma lane
    assert lanes[5] < lanes[4] and lanes[6] < lanes[5]
    assert all(0 <= lane <= 4 for lane in lanes)


def test_six_note_run_is_repositioned_instead_of_repeating_a_lane():
    # Escala de 6 notas subindo e voltando: não cabe em 5 lanes.
    run = [52, 55, 57, 59, 62, 64, 62, 59]
    ticks = [i * RES // 2 for i in range(len(run))]
    lanes = assign_lanes(run, n_lanes=5, window=8, ticks=ticks, resolution=RES)
    for i in range(1, len(run)):
        assert lanes[i] != lanes[i - 1], f"nota diferente na mesma lane: {lanes}"
    resets = [i for i in range(1, len(run))
              if (run[i] > run[i - 1]) != (lanes[i] > lanes[i - 1])]
    assert len(resets) == 1 and ticks[resets[0]] % RES == 0   # um reposicionamento, no início da batida
    after = range(resets[0], len(run))
    for i in after:   # depois do reposicionamento, cada nota fica sempre na mesma lane
        for j in after:
            assert (run[i] == run[j]) == (lanes[i] == lanes[j])


def test_chord_shapes_fit_in_lanes():
    assert chord_shape(4, 2, 5) == (2, 4)
    assert chord_shape(0, 2, 5) == (0, 2)
    assert chord_shape(4, 3, 5) == (2, 3, 4)
    assert chord_shape(2, 2, 3) == (1, 2)


def test_select_rhythm_respects_gap_and_grid():
    cands = [Candidate(tick=t, strength=0.5) for t in range(0, RES * 4, RES // 4)]  # semicolcheias
    eighths = select_rhythm(cands, RES, min_gap_beats=0.5, grid=2, allow_offgrid=False)
    assert [c.tick for c in eighths] == list(range(0, RES * 4, RES // 2))
    quarters = select_rhythm(cands, RES, min_gap_beats=1.0, grid=1, allow_offgrid=False)
    assert [c.tick for c in quarters] == [0, 192, 384, 576]


def test_difficulties_get_simpler():
    rules = load_rules()
    pitches = [52, 55, 57, 59, 62, 64, 62, 59] * 4
    cands = [Candidate(tick=i * RES // 4, strength=1.0, end_tick=i * RES // 4 + 40, pitch=p)
             for i, p in enumerate(pitches)]
    track = build_fret_track(cands, rules.frets, RES, lane_window=8, sp_windows=[])
    counts = [len(track.notes[d]) for d in ("expert", "hard", "medium", "easy")]
    assert counts == sorted(counts, reverse=True) and counts[0] > counts[-1]
    assert max(lane for n in track.notes["medium"] for lane in n.lanes) <= 3
    assert max(lane for n in track.notes["easy"] for lane in n.lanes) <= 2


def test_repeated_riffs_are_charted_the_same_in_every_difficulty():
    rng = np.random.default_rng(1)
    riff = [52, 55, 57, 59, 62, 64, 62, 59]          # 8 colcheias = 4 batidas
    offset = RES * 2                                 # começa no 3º tempo do compasso
    cands = [
        Candidate(tick=offset + i * RES // 2, pitch=riff[i % 8], end_tick=offset + i * RES // 2 + 40,
                  strength=float(rng.uniform(0.5, 1.0)))   # volume varia a cada repetição
        for i in range(8 * 6)
    ]
    track = build_fret_track(cands, load_rules().frets, RES, lane_window=8, sp_windows=[])
    for difficulty, notes in track.notes.items():
        blocks: dict[int, list] = {}
        for n in notes:
            k, pos = divmod(n.tick - offset, RES * 4)
            blocks.setdefault(k, []).append((pos, n.lanes))
        first = blocks[0]
        assert all(b == first for b in blocks.values()), difficulty


def test_easy_forces_natural_hopos_to_strum():
    rules = load_rules()
    rules.frets["easy"].min_gap_beats = 0.25
    rules.frets["easy"].grid = 4
    rules.frets["easy"].min_reduction = 0.0
    cands = [Candidate(tick=i * RES // 4, strength=1.0, pitch=40 + 3 * i) for i in range(3)]
    track = build_fret_track(cands, {"easy": rules.frets["easy"]}, RES, 8, [])
    assert all(FORCE_FLAG in n.flags for n in track.notes["easy"][1:])


def test_star_power_windows_are_regular():
    windows = star_power_windows(last_tick=RES * 4 * 40, resolution=RES, every_measures=8, phrase_beats=4)
    starts = [w[0] for w in windows]
    assert np.all(np.diff(starts) == RES * 4 * 8)


def test_drum_fills_end_on_downbeat_and_avoid_star_power():
    measure = RES * 4
    notes = [ChartNote(t, (1,)) for t in range(0, measure * 16 + 1, RES)]   # uma nota por batida
    windows = [(measure * 4, RES * 4)]                                        # Star Power no compasso 4
    fills = drum_fills_for(notes, windows, RES, every_measures=4)
    assert fills == [(measure * 8 - 2 * RES, 2 * RES), (measure * 12 - 2 * RES, 2 * RES),
                     (measure * 16 - 2 * RES, 2 * RES)]


def test_user_rules_override_defaults(tmp_path):
    custom = tmp_path / "rules.toml"
    custom.write_text("[frets.easy]\nlanes = 4\n")
    rules = load_rules(custom)
    assert rules.frets["easy"].lanes == 4
    assert rules.frets["easy"].max_chord == 1    # o resto continua igual ao padrão

    custom.write_text("[frets.easy]\nlanez = 4\n")
    with pytest.raises(ValueError, match="lanez"):
        load_rules(custom)


def test_chart_file_round_trip(tmp_path):
    rules = load_rules()
    cands = [Candidate(tick=i * RES, strength=1.0, end_tick=i * RES + 150, pitch=40 + i) for i in range(8)]
    track = build_fret_track(cands, rules.frets, RES, 8, [(0, RES * 4)])
    path = tmp_path / "notes.chart"
    write_chart(path, SongInfo("Título \"x\"", "Artista"), constant_map(), {"guitar": track}, [0])

    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")         # UTF-8 com BOM
    chart = parse_chart(path)
    assert ("Resolution", "192") in chart["Song"]
    assert ("0", "B 120000") in chart["SyncTrack"]
    for name in ("ExpertSingle", "HardSingle", "MediumSingle", "EasySingle"):
        assert name in chart
    expert = notes_by_tick(chart["ExpertSingle"])
    assert sorted(expert) == [i * RES for i in range(8)]
    assert any(v.startswith("S 2 ") for _, v in chart["ExpertSingle"])


def test_detects_single_notes_and_chords():
    import librosa
    import synth
    from ai_charter.analysis.pitched import detect_pitched_notes

    shapes = [[52], [52, 59], [45, 52], [60, 64, 67], [57]]   # nota, power chords, tríade, nota
    y = np.zeros(synth.SR * 5)
    for i, notes in enumerate(shapes):
        synth.add(y, sum(synth.tone(n, 0.9) for n in notes), 0.2 + i * 0.95, 0.3)
    y = librosa.resample(y, orig_sr=synth.SR, target_sr=22050)

    detected = detect_pitched_notes(y, 22050, load_rules().detection["guitar"])
    assert [(n.pitch, n.voices) for n in detected] == [(52, 1), (52, 2), (45, 2), (60, 3), (57, 1)]


def test_bars_start_on_the_detected_downbeats():
    import numpy as np

    from ai_charter.analysis.tempo import _anchor_at_zero, _bars_on_downbeats

    period = 0.5
    beats = _anchor_at_zero(np.arange(0.3, 30, period), period, 30)   # 0 s, depois 0,3, 0,8...
    downbeats = np.arange(0.3 + 2 * period, 30, 4 * period)            # compasso começa na 3ª batida
    fixed = _bars_on_downbeats(beats, downbeats)
    assert fixed[0] == 0.0
    bars = fixed[::4][1:]
    assert all(np.min(np.abs(downbeats - b)) < 1e-9 for b in bars[(bars > 2) & (bars < 28)])
    assert np.allclose(np.diff(fixed[2:]), period)   # só o primeiro intervalo muda
    assert _bars_on_downbeats(beats, np.array([])) is beats


def test_bar_fix_ends_before_the_first_attack():
    import numpy as np

    from ai_charter.analysis.tempo import _anchor_at_zero, _bars_on_downbeats

    period = 0.5
    beats = _anchor_at_zero(np.arange(0.5, 30, period), period, 30)   # 0 s, 0,5 s, 1 s...
    downbeats = np.arange(1.0, 30, 4 * period)                         # a música começa num compasso, em 1 s
    fixed = _bars_on_downbeats(beats, downbeats, first_onset=1.0)
    assert np.allclose(fixed[:5], [0.0, 0.25, 0.5, 0.75, 1.0])         # o silêncio é que muda
    assert np.allclose(np.diff(fixed[4:]), period)
    # Com introdução longa, tira-se uma batida do silêncio em vez de pôr três.
    downbeats = np.arange(4.5, 30, 4 * period)
    fixed = _bars_on_downbeats(beats, downbeats, first_onset=4.5)
    assert np.isclose(fixed[8], 4.5) and np.allclose(np.diff(fixed[:9]), 4.5 / 8)


def test_skipped_and_doubled_beats_are_regularized():
    import numpy as np

    from ai_charter.analysis.tempo import _regularize_beats

    beats = np.array([0.5, 1.0, 1.5, 2.5, 3.0, 3.02, 3.5, 4.0])   # faltou 2,0; 3,02 é duplicada
    assert np.allclose(_regularize_beats(beats), [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
