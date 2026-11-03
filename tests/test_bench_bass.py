"""Benchmark do baixo: casamento das notas e contorno."""

import mido

from ai_charter.bench_bass import compare, official_notes, pair_notes, pitch_score


def test_pairs_each_official_note_once_with_the_closest():
    theirs = [(1.0, 0.03), (1.05, 0.03), (2.0, 0.03)]
    assert pair_notes([1.04, 1.06, 3.0], theirs) == [(0, 1)]
    assert pair_notes([1.01, 1.04], theirs) == [(0, 0), (1, 1)]


def test_contour_counts_up_down_and_repeat_not_the_exact_color():
    theirs = [(t, (lane,), False, 0.03) for t, lane in ((0.0, 0), (0.5, 2), (1.0, 2), (1.5, 1))]
    shifted = [(t, (lane + 1,), False) for t, (lane,), _, _ in theirs]   # mesmo desenho, uma cor acima
    row = compare(shifted, theirs)
    assert row["acertos"] == 4 and row["contorno"] == [3, 3] and row["cor_exata"] == [0, 4]
    flat = [(t, (0,), True) for t, *_ in theirs]
    row = compare(flat, theirs)
    assert row["contorno"] == [1, 3] and row["sustains"] == [4, 0]


def test_official_notes_reads_the_expert_bass():
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack([mido.MetaMessage("track_name", name="PART BASS")])
    events = []
    for start, pitch, length in ((0, 96, 120), (480, 98, 480), (480, 99, 480), (960, 84, 120)):
        events += [(start, 1, pitch), (start + length, 0, pitch)]
    now = 0
    for tick, on, pitch in sorted(events):
        track.append(mido.Message("note_on", note=pitch, velocity=100 if on else 0, time=tick - now))
        now = tick
    midi.tracks.append(track)
    assert official_notes(midi) == [(0, (0,), 120), (480, (2, 3), 480)]   # o Hard (84) fica de fora
    assert official_notes(mido.MidiFile()) is None


def test_pitch_score_learns_drop_d_and_counts_octave_errors():
    # Mi grave em Drop D (Ré1 = 26): o Pro Bass escreve corda 0 casa 0, que no padrão seria Mi1 (28).
    truth = [(i * 0.5, 0 if i % 2 else 1, i % 5, 0.03) for i in range(40)]
    real = [(26 if s == 0 else 33) + f for _, s, f, _ in truth]
    heard = [(t, float(p + (12 if i == 3 else 0))) for i, ((t, *_), p) in enumerate(zip(truth, real))]
    row = pitch_score(heard, truth)
    assert row["afinação"] == {"0": -2, "1": 0}
    assert (row["exata"], row["oitava"], row["outra"]) == (39, 1, 0)
