"""Benchmark dos vocais: leitura da chart oficial, comparação sílaba a sílaba e gafes."""

import mido

from ai_charter.bench_vocals import (Syllable, clean_lyric, compare, gaffes, lyric_lines,
                                     official_vocals, our_vocals, summary)
from ai_charter.charting.vocals import VocalNote, VocalTrack


def _official_midi() -> mido.MidiFile:
    """Duas frases: "Where would you go?" e "think-ing out", com um "+" e uma sílaba falada."""
    events = []   # (tick, mensagem)

    def note(tick, length, pitch, lyric=None):
        if lyric is not None:
            events.append((tick, 0, mido.MetaMessage("lyrics", text=lyric)))
        events.append((tick, 1, mido.Message("note_on", note=pitch, velocity=100)))
        events.append((tick + length, 0, mido.Message("note_off", note=pitch, velocity=0)))

    note(0, 1900, 105)
    note(2000, 1900, 105)
    for k, (lyric, pitch) in enumerate([("Where", 60), ("would", 62), ("+", 64), ("you", 72), ("go?#", 50)]):
        note(k * 300, 200, pitch, lyric)
    for k, (lyric, pitch) in enumerate([("think-", 60), ("ing", 59), ("out", 67)]):
        note(2000 + k * 300, 200, pitch, lyric)
    events.append((100, 0, mido.MetaMessage("text", text="[idle]")))
    track, now = mido.MidiTrack([mido.MetaMessage("track_name", name="PART VOCALS")]), 0
    for tick, _, msg in sorted(events, key=lambda e: e[:2]):
        track.append(msg.copy(time=tick - now))
        now = tick
    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks += [mido.MidiTrack(), track]
    return midi


def test_official_vocals_reads_syllables_phrases_and_spoken():
    syllables, phrases = official_vocals(_official_midi())
    assert [s.text for s in syllables] == ["Where", "would", "you", "go?", "think-", "ing", "out"]
    assert [s.phrase for s in syllables] == [0, 0, 0, 0, 1, 1, 1]
    assert syllables[3].pitch is None and syllables[2].pitch == 72   # "#" = falada
    assert phrases == [(0, 1900), (2000, 3900)]
    assert lyric_lines(syllables) == [["Where", "would", "you", "go?"], ["think-", "ing", "out"]]


def test_clean_lyric_marks():
    assert clean_lyric("A=") == "A-"
    assert clean_lyric("go?#") == "go?"
    assert clean_lyric("$oh^") == "oh"


def _syl(tick, text, phrase, pitch=60, time=None):
    s = Syllable(tick, text, pitch, phrase, tick + 100)
    s.time = tick / 1000 if time is None else time
    return s


def test_compare_counts_timing_pitch_and_phrases():
    theirs = [_syl(0, "a", 0, 60), _syl(500, "b-", 0, 62), _syl(1000, "c", 0, 64), _syl(2000, "d", 1, 65)]
    ours = [_syl(0, "a", 0, 72), _syl(550, "b-", 0, 63), _syl(1300, "c", 1, 64), _syl(2000, "d", 1, 65)]
    r = compare(ours, theirs, [(0.0, 1.1), (1.3, 2.1)], [(0.0, 1.1), (2.0, 2.1)])
    assert r["colocadas"] == 4 and r["no_tempo"] == 3          # "c" errou 300 ms
    assert r["nota_certa"] == 3                                 # 72 é a mesma nota (oitava acima); 63 não
    assert r["frases_oficiais"] == 2 and r["frases_nossas"] == 2 and r["inicio_certo"] == 1
    assert r["frase_certa"] == 1                                # a segunda nossa é longa demais
    s = summary([{**r, "gafes": gaffes(ours, [])}])
    assert s["colocadas"] == 1.0 and s["no_tempo"] == 0.75 and s["frases_f1"] == 0.5


def test_gaffes_stacked_split_and_overlap():
    syl = [_syl(0, "Sua", 0), _syl(0, "a-", 0), _syl(100, "vó", 0), _syl(200, "ven-", 0), _syl(300, "to", 1),
           _syl(400, "ra do", 1)]
    g = gaffes(syl, [(0, 120), (100, 150), (200, 300)])
    assert g == {"mesmo_tick": 2, "palavra_partida": 1, "sobrepostas": 1}


def test_our_vocals_reads_notes_and_spoken_text():
    track = VocalTrack(notes=[VocalNote(0, 100, 60, "ca-"), VocalNote(100, 100, 62, "+"), VocalNote(480, 100, 64, "sa")],
                       phrases=[(0, 300), (480, 200)], spoken=[(800, "oi")])
    ours = our_vocals(track)
    assert [(s.text, s.phrase) for s in ours] == [("ca-", 0), ("sa", 1), ("oi", 1)]
    assert ours[2].pitch is None
