"""Partitura de referência (MIDI / Guitar Pro) guiando a transcrição (reference.py)."""

import shutil
from types import SimpleNamespace

import mido
import numpy as np
import pytest

import synth
from ai_charter.charting.common import Candidate
from ai_charter.config import load_rules
from ai_charter.reference import (
    Alignment, Guide, RefEvent, default_mapping, guess_instrument, parse_mapping, play_order, read_score,
)


def write_midi(path, parts, ticks_per_beat=480):
    """parts: [(nome, programa, canal, [(nota, batidas)])] -> arquivo MIDI, uma pista por parte."""
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    for name, program, channel, notes in parts:
        track = mido.MidiTrack([mido.MetaMessage("track_name", name=name),
                                mido.Message("program_change", program=program, channel=channel)])
        for pitch, beats in notes:
            track.append(mido.Message("note_on", note=pitch, velocity=90, channel=channel, time=0))
            track.append(mido.Message("note_off", note=pitch, velocity=0, channel=channel,
                                      time=int(beats * ticks_per_beat)))
        midi.tracks.append(track)
    midi.save(path)
    return path


def test_reads_midi_tracks_with_guesses(tmp_path):
    path = write_midi(tmp_path / "ref.mid", [
        ("Lead", 30, 0, [(64, 0.5), (67, 0.5)]),
        ("", 33, 1, [(40, 1.0)]),
        ("Kit", 0, 9, [(36, 0.25), (38, 0.25)]),
        ("Pad", 89, 2, [(60, 2.0)]),
    ])
    score = read_score(path)
    assert [(t.index, t.guess, len(t.notes)) for t in score.tracks] == [
        (1, "guitar", 2), (2, "bass", 1), (3, "drums", 2), (4, "keys", 1)]
    assert [(n.beat, n.length, n.pitch) for n in score.tracks[0].notes] == [(0.0, 0.5, 64), (0.5, 0.5, 67)]
    assert score.length_beats == 2.0
    assert default_mapping(score) == {1: "guitar", 2: "bass", 3: "drums", 4: "keys"}


def test_guesses():
    assert guess_instrument("Violão", None, False) == "guitar"
    assert guess_instrument("Baixo elétrico", 25, False) == "bass"        # o nome vence o programa
    assert guess_instrument("Track 3", None, False, [43, 38, 33, 28]) == "bass"
    assert guess_instrument("Track 2", None, False, [64, 59, 55, 50, 45, 40]) == "guitar"
    assert guess_instrument("Voz", None, False) == "vocals"
    assert guess_instrument("Sanfona", None, False) == "keys"
    assert guess_instrument("Track 9", 120, False) == "other"


def test_play_order_unrolls_repeats_and_endings():
    def h(open_=False, close=-1, alt=0):
        return SimpleNamespace(isRepeatOpen=open_, repeatClose=close, repeatAlternative=alt)

    # |: A | B (1ª vez) :| C (2ª vez) | D
    headers = [h(open_=True), h(close=1, alt=1), h(alt=2), h()]
    assert play_order(headers) == [0, 1, 0, 2, 3]
    # A |: B :| x3 C
    assert play_order([h(), h(open_=True, close=2), h()]) == [0, 1, 1, 1, 2]


def test_reads_guitar_pro_with_repeats_and_ties(tmp_path):
    gp = pytest.importorskip("guitarpro")
    from guitarpro import models as m

    song = gp.Song()
    track = song.tracks[0]
    track.name = "Guitarra Base"
    headers = song.measureHeaders
    while len(headers) < 3:
        header = m.MeasureHeader(number=len(headers) + 1, start=headers[-1].start + headers[-1].length)
        headers.append(header)
        track.measures.append(m.Measure(track, header))
    headers[0].isRepeatOpen = True
    headers[1].repeatClose = 1       # |: 1 | 2 :| 3
    for k, measure in enumerate(track.measures):
        voice = measure.voices[0]
        voice.beats.clear()
        for q in range(4):
            beat = m.Beat(voice, duration=m.Duration(value=4), status=m.BeatStatus.normal)
            kind = m.NoteType.tie if (k == 2 and q == 1) else m.NoteType.normal
            beat.notes.append(m.Note(beat, value=k * 2 + (q % 2), string=6, type=kind))
            voice.beats.append(beat)
    gp.write(song, str(tmp_path / "ref.gp5"))

    score = read_score(tmp_path / "ref.gp5")
    (t,) = score.tracks
    assert t.name == "Guitarra Base" and t.guess == "guitar"
    starts = [n.beat for n in t.notes]
    # compassos 1, 2, 1, 2, 3; no 3º a 2ª nota é ligada à 1ª (que fica com 2 batidas)
    assert starts == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19]
    assert [n.pitch for n in t.notes[:8]] == [40, 41, 40, 41, 42, 43, 42, 43]
    assert t.notes[16].length == 2.0
    assert score.length_beats == 20.0


def test_new_guitar_pro_formats_explain_what_to_do(tmp_path):
    (tmp_path / "x.gpx").write_bytes(b"BCFS")
    with pytest.raises(ValueError, match="exporte"):
        read_score(tmp_path / "x.gpx")


def test_parse_mapping():
    assert parse_mapping("2=guitar, 4=bass") == {2: "guitar", 4: "bass"}
    with pytest.raises(ValueError):
        parse_mapping("guitar=2")
    with pytest.raises(ValueError):
        parse_mapping("1=flauta")


def make_guide(events, resolution=192):
    return Guide("guitar", events, resolution, load_rules().detection["guitar"])


def test_guide_corrects_pitch_and_chords():
    res = 192
    events = [RefEvent(0, 96, (52,)), RefEvent(96, 192, (45, 52)),            # nota; power chord (Lá5)
              RefEvent(192, 288, (48, 52, 55)), RefEvent(384, 1200, (45, 48, 52))]   # Dó; Lá menor longo
    cands = [Candidate(0, 1, pitch=50), Candidate(100, 1, pitch=47, voices=1), Candidate(190, 1, pitch=48, voices=2),
             Candidate(390, 1, pitch=45, voices=2), Candidate(600, 1, pitch=60)]   # a última não bate com nada
    lines = []
    out = make_guide(events, res)(cands, log=lines.append)
    assert [(c.pitch, c.voices) for c in out] == [(52, 1), (45, 2), (48, 2), (45, 3), (60, 1)]
    from ai_charter.analysis.chords import chord_name
    assert [chord_name(c.chord) for c in out[:4]] == ["E", "A5", "C", "Am"]
    assert "80%" in lines[0] and "3 corrigidas" in lines[0]   # o Dó já estava certo


def test_guide_is_ignored_when_reference_does_not_match():
    cands = [Candidate(t, 1, pitch=50) for t in range(0, 1920, 192)]
    lines = []
    out = make_guide([RefEvent(5000, 5100, (40,))])(cands, log=lines.append)
    assert all(c.pitch == 50 for c in out)
    assert "ignorada" in lines[0]


def test_alignment_maps_score_beats():
    a = Alignment(np.array([0.0, 1, 2, 3]), np.array([2.0, 3, 4, 6]), 0, 1.0, 0.1)
    assert a.to_audio_beat(1.5) == 3.5
    assert a.to_audio_beat(3.5) == 6.5       # depois do fim: segue no mesmo passo
    assert a.to_audio_beat(-2) is None and a.to_audio_beat(10) is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="precisa do ffmpeg")
def test_pipeline_guided_by_score(tmp_path):
    """A partitura começa sem o silêncio do áudio, está meio tom acima e tem 4 notas diferentes:
    o alinhamento acha o começo e a transposição, e o guia corrige exatamente as 4."""
    from ai_charter.pipeline import Options, run

    paths = synth.render(tmp_path)
    stems = tmp_path / "stems"
    stems.mkdir()
    for name in ("guitar", "bass"):
        shutil.move(paths[name], stems / f"{name}.wav")
    melody = [m + 1 for m in synth.MELODY]
    melody[8:12] = [m + 5 for m in melody[8:12]]
    bass = [b + 1 for b in synth.BASS[:len(synth.MELODY) // 2]]
    ref = write_midi(tmp_path / "ref.mid", [("Guitarra", 29, 0, [(n, 0.5) for n in melody]),
                                            ("Baixo", 33, 1, [(n, 1.0) for n in bass])])
    lines = []
    run(Options(input=paths["mix"], output_dir=tmp_path / "out", separator="folder", stems_dir=stems,
                instruments=("guitar", "bass"), reference=ref), log=lines.append)
    text = "\n".join(lines)
    assert "transposta -1 semitons" in text
    assert "guitar: referência confere com 100% das notas detectadas; 4 corrigidas" in text
    assert "bass: referência confere com 100% das notas detectadas; 0 corrigidas" in text
