"""Regras de jogabilidade: acordes de 2 notas, trocas relâmpago, picos de oitava e afinação."""

import librosa
import numpy as np

from ai_charter.analysis.chords import CHORDS
from ai_charter.analysis.pitched import DetectedNote, _fix_octave_spikes, detect_pitched_notes
from ai_charter.charting.chords import label_chords
from ai_charter.charting.common import Candidate
from ai_charter.compare import Event
from ai_charter.config import load_rules
from ai_charter.critic import critique

RES = 192
C, G, A = CHORDS.index((0, "")), CHORDS.index((7, "")), CHORDS.index((9, ""))


def strum(tick, best, scores=None, length=RES // 2):
    """Palhetada de acorde: `best` pontua mais alto; `scores` ajusta outros acordes."""
    s = np.full(len(CHORDS), 0.2)
    s[best] = 0.9
    for index, value in (scores or {}).items():
        s[index] = value
    lowest = np.full(12, np.nan)
    lowest[[0, 7, 9]] = [48, 43, 45]
    return Candidate(tick, 1.0, end_tick=tick + length, pitch=48, voices=3, chord_scores=s,
                     lowest_by_class=lowest, fifth_ratio=1.0, third_ratio=1.0)


def test_chords_use_two_notes_and_long_chords_are_accents():
    cfg = load_rules().detection["guitar"]
    cands = [strum(i * RES // 2, C) for i in range(6)] + [strum(3 * RES, C, length=4 * RES)]
    label_chords(cands, RES, cfg)
    assert [c.voices for c in cands] == [2] * 6 + [3]   # só o acorde segurado vira acento de 3 notas


def test_one_strum_chord_between_stable_groups_follows_the_closest_group():
    cfg = load_rules().detection["guitar"]
    cands = ([strum(i * RES // 2, C) for i in range(4)]
             + [strum(2 * RES, G, {C: 0.85, A: 0.6})]    # som de troca: parece G, mas C ainda soa
             + [strum(4 * RES + i * RES // 2, A) for i in range(4)])
    label_chords(cands, RES, cfg)
    roots = [CHORDS[c.chord][0] for c in cands]
    assert roots == [0] * 5 + [9] * 4


def test_isolated_octave_spikes_are_fixed_but_octave_riffs_stay():
    def notes(pitches):
        return [DetectedNote(i * 0.2, i * 0.2 + 0.1, float(p), 1, 1.0) for i, p in enumerate(pitches)]

    spiky = notes([40, 40, 40, 52, 40, 40, 28, 40, 40])
    _fix_octave_spikes(spiky)
    assert [n.pitch for n in spiky] == [40] * 9

    # Nota repetida com o começo ouvido uma oitava acima (o harmônico da anterior).
    late = notes([66, 66, 66, 54, 54, 54, 54, 54])
    _fix_octave_spikes(late)
    assert [n.pitch for n in late] == [54] * 8

    riff = notes([40, 52, 40, 52, 40, 52, 40, 52])
    _fix_octave_spikes(riff)
    assert [n.pitch for n in riff] == [40, 52, 40, 52, 40, 52, 40, 52]


def test_detuned_recording_keeps_the_same_note():
    """Gravação ~45 centésimos abaixo do padrão: a nota fica no meio de dois semitons."""
    sr = 22050
    t = np.arange(int(0.25 * sr)) / sr
    hz = librosa.midi_to_hz(40 - 0.45)
    tone = sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, 6)) * np.exp(-t * 6)
    silence = np.zeros(int(0.15 * sr))
    y = np.concatenate([np.concatenate([tone, silence]) for _ in range(8)])
    tuning = float(librosa.estimate_tuning(y=y, sr=sr))
    assert abs(tuning + 0.45) < 0.1
    pitches = {round(n.pitch) for n in detect_pitched_notes(y, sr, load_rules().detection["bass"], tuning)}
    assert pitches == {40}


def test_critic_finds_triples_quick_changes_and_spikes():
    def events(shapes):
        return [Event(i * 0.25, tuple(s)) for i, s in enumerate(shapes)]

    clean = events([(0, 2)] * 4 + [(1, 3)] * 4 + [(0,), (1,), (2,), (1,)])
    c = critique(clean)
    assert (c.triples, c.quick_changes, c.spikes) == (0.0, 0, 0)

    messy = events([(0, 1, 2)] * 3 + [(3, 4)] + [(0, 1, 2)] * 3 + [(0,), (4,), (0,)])
    c = critique(messy)
    # O acorde relâmpago (B+O entre G+R+Y) também é um pico, além do G-O-G no fim.
    assert c.quick_changes == 1 and c.spikes == 2 and c.triples == 0.6
