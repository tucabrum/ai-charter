"""Pedido de revisão: a IA pode mudar a chart, mas nunca os compassos editados à mão."""

import json

import mido

from ai_charter.revision import apply, protect, show

RES = 192
BAR = RES * 4
GREEN, RED, YELLOW = 96, 97, 98     # Expert
HARD_GREEN = 84
STAR_POWER = 116


def make_midi(notes):
    """notes: [(tick, len, pitch)] na PART GUITAR, com trilha de tempo 4/4."""
    midi = mido.MidiFile(type=1, ticks_per_beat=RES)
    tempo = mido.MidiTrack([mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0),
                            mido.MetaMessage("set_tempo", tempo=500000, time=0)])
    events = []
    for tick, length, pitch in notes:
        events.append((tick, 1, mido.Message("note_on", note=pitch, velocity=100)))
        events.append((tick + length, 0, mido.Message("note_off", note=pitch, velocity=0)))
    guitar, now = mido.MidiTrack([mido.MetaMessage("track_name", name="PART GUITAR", time=0)]), 0
    for tick, _, msg in sorted(events, key=lambda e: e[:2]):
        guitar.append(msg.copy(time=tick - now))
        now = tick
    midi.tracks += [tempo, guitar]
    return midi


def notes_of(midi):
    out, open_, tick = set(), {}, 0
    for msg in midi.tracks[1]:
        tick += msg.time
        if msg.type == "note_on" and msg.velocity:
            open_[msg.note] = tick
        elif msg.type in ("note_on", "note_off"):
            out.add((open_.pop(msg.note), tick - 0, msg.note))
    return {(a, b - a, p) for a, b, p in out}


ORIGINAL = [(0, 48, GREEN), (BAR, 48, RED), (BAR, 48, HARD_GREEN), (BAR, BAR, STAR_POWER)]
# A "IA" mexeu em tudo: compasso 1 e 2 do Expert, o Hard do compasso 2 e o Star Power
REVISED = [(0, 48, YELLOW), (BAR, 48, YELLOW), (BAR + 96, 48, GREEN), (BAR, 48, HARD_GREEN + 1), (BAR, 96, STAR_POWER)]


def test_locked_measures_keep_the_manual_version():
    locks = {"guitar": {"expert": [(BAR, 2 * BAR)]}}   # só o compasso 2 do Expert foi editado à mão
    merged, undone = protect(make_midi(ORIGINAL), make_midi(REVISED), locks)
    got = notes_of(merged)
    assert (0, 48, YELLOW) in got                        # compasso 1: vale a revisão
    assert (BAR, 48, RED) in got                         # compasso 2 do Expert: volta o original
    assert (BAR, 48, YELLOW) not in got and (BAR + 96, 48, GREEN) not in got
    assert (BAR, 48, HARD_GREEN + 1) in got              # o Hard não estava travado
    assert (BAR, 96, STAR_POWER) in got                  # nem o Star Power
    assert undone == 3                                   # amarelo e verde novos saem, vermelho volta


def test_shared_lock_protects_star_power_and_tap_for_all_difficulties():
    locks = {"guitar": {"shared": [(BAR, 2 * BAR)], "hard": [(BAR, 2 * BAR)]}}
    merged, _ = protect(make_midi(ORIGINAL), make_midi(REVISED), locks)
    got = notes_of(merged)
    assert (BAR, BAR, STAR_POWER) in got and (BAR, 96, STAR_POWER) not in got
    assert (BAR, 48, HARD_GREEN) in got
    assert (BAR, 48, YELLOW) in got                      # o Expert não estava travado


def test_apply_writes_backup_and_archives_the_request(tmp_path):
    make_midi(ORIGINAL).save(tmp_path / "notes.mid")
    make_midi(REVISED).save(tmp_path / "revisada.mid")
    (tmp_path / "notes.locks.json").write_text(json.dumps({"locks": {"guitar": {"expert": [[BAR, 2 * BAR, 2]]}}}))
    (tmp_path / "ai_request.json").write_text(json.dumps({"request": "mais amarelo"}))
    assert apply(tmp_path, tmp_path / "revisada.mid") == 3
    assert (BAR, 48, RED) in notes_of(mido.MidiFile(tmp_path / "notes.mid"))
    assert notes_of(mido.MidiFile(tmp_path / "notes.before-ai.mid")) == notes_of(make_midi(ORIGINAL))
    assert not (tmp_path / "ai_request.json").exists()
    done = json.loads((tmp_path / "ai_request.done.json").read_text(encoding="utf-8"))
    assert done["request"] == "mais amarelo" and done["undone_in_locked_measures"] == 3


def test_show_lists_measures_and_marks_locks(tmp_path):
    make_midi(ORIGINAL).save(tmp_path / "notes.mid")
    (tmp_path / "notes.locks.json").write_text(json.dumps({"locks": {"guitar": {"expert": [[BAR, 2 * BAR, 2]]}}}))
    lines = show(tmp_path).splitlines()
    assert lines[1].split() == ["1", "|", "0:G"]
    assert lines[2].split() == ["2", "[trava]", "|", "0:R*"]


def test_drum_locks_protect_notes_and_tom_markers():
    def drums(notes):
        midi = make_midi([])
        events, now = [], 0
        for tick, length, pitch in notes:
            events += [(tick, 1, mido.Message("note_on", note=pitch, velocity=100)),
                       (tick + length, 0, mido.Message("note_off", note=pitch, velocity=0))]
        track = mido.MidiTrack([mido.MetaMessage("track_name", name="PART DRUMS", time=0)])
        for tick, _, msg in sorted(events, key=lambda e: e[:2]):
            track.append(msg.copy(time=tick - now))
            now = tick
        midi.tracks.append(track)
        return midi
    original = drums([(BAR, 12, 98), (BAR, 12, 110)])       # compasso 2: tom amarelo
    revised = drums([(BAR, 12, 99), (2 * BAR, 12, 96)])      # a "IA" trocou por prato azul e pôs um bumbo
    locks = {"drums": {"expert": [(BAR, 2 * BAR)], "shared": [(BAR, 2 * BAR)]}}
    merged, undone = protect(original, revised, locks)
    track = next(t for t in merged.tracks if t.name == "PART DRUMS")
    got, open_, tick = set(), {}, 0
    for msg in track:
        tick += msg.time
        if msg.type == "note_on" and msg.velocity:
            open_[msg.note] = tick
        elif msg.type in ("note_on", "note_off") and msg.note in open_:
            got.add((open_.pop(msg.note), msg.note))
    assert got == {(BAR, 98), (BAR, 110), (2 * BAR, 96)}     # travado volta; fora da trava fica a revisão
    assert undone == 3


def test_lock_protects_open_notes():
    open_expert = GREEN - 1                              # nota aberta do Expert (com [ENHANCED_OPENS])
    original = [(BAR, 48, open_expert)]
    revised = [(BAR, 48, GREEN)]                         # a "IA" trocou a aberta por verde
    merged, undone = protect(make_midi(original), make_midi(revised), {"guitar": {"expert": [(BAR, 2 * BAR)]}})
    assert notes_of(merged) == {(BAR, 48, open_expert)}
    assert undone == 2
