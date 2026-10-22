"""Acrescentar instrumentos a uma chart existente (augment.py), sem mexer nas outras pistas."""

import json
import shutil

import mido
import pytest

import synth
from ai_charter.augment import BEFORE_ADD_NAME, augment, existing_parts, update_song_ini
from ai_charter.pipeline import Options, run

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="precisa do ffmpeg")


def track_bytes(midi, name):
    track = next(t for t in midi.tracks if t.name == name)
    return [m.bytes() if not m.is_meta else repr(m) for m in track]


@pytest.fixture(scope="module")
def guitar_only(tmp_path_factory):
    """Pasta de música gerada só com a guitarra (mas com os stems de baixo e bateria)."""
    folder = tmp_path_factory.mktemp("augment")
    paths = synth.render(folder)
    stems = folder / "stems"
    stems.mkdir()
    for name in ("guitar", "bass", "drums"):
        shutil.move(paths[name], stems / f"{name}.wav")
    result = run(Options(input=paths["mix"], output_dir=folder / "out", separator="folder", stems_dir=stems,
                         instruments=("guitar",), title="Synth", artist="Test"), log=lambda *_: None)
    return result.song_dir


@pytest.fixture
def song(guitar_only, tmp_path):
    copy = tmp_path / "song"
    shutil.copytree(guitar_only, copy)
    return copy


def test_adds_only_the_missing_part(song):
    before = mido.MidiFile(song / "notes.mid")
    assert existing_parts(song) == ["guitar"]
    lines = []
    result = augment(song, ("guitar", "bass"), log=lines.append)
    assert result.added == ["bass"]
    assert "guitar" in result.skipped and "já existe" in result.skipped["guitar"]
    after = mido.MidiFile(song / "notes.mid")
    assert existing_parts(song) == ["guitar", "bass"]
    for name in ("Synth", "EVENTS", "PART GUITAR"):   # tempo, seções e guitarra intactos
        assert track_bytes(after, name) == track_bytes(before, name)
    bass_notes = [m for m in next(t for t in after.tracks if t.name == "PART BASS") if m.type == "note_on"]
    assert len(bass_notes) > 20
    assert (song / BEFORE_ADD_NAME).read_bytes() != (song / "notes.mid").read_bytes()
    ini = (song / "song.ini").read_text("utf-8")
    assert "diff_bass = -1" not in ini and "diff_bass = " in ini
    assert "name = Synth" in ini   # o resto do song.ini continua


def test_new_part_uses_star_power_of_existing_parts(song):
    augment(song, ("bass",), log=lambda *_: None)
    midi = mido.MidiFile(song / "notes.mid")

    def sp(name):
        tick, starts = 0, set()
        for m in next(t for t in midi.tracks if t.name == name):
            tick += m.time
            if m.type == "note_on" and m.velocity and m.note == 116:
                starts.add(tick)
        return starts

    assert sp("PART BASS") and sp("PART BASS") <= sp("PART GUITAR")


def expert_guitar(midi, lo, hi):
    tick, out = 0, []
    for m in next(t for t in midi.tracks if t.name == "PART GUITAR"):
        tick += m.time
        if m.type == "note_on" and m.velocity and 96 <= m.note <= 100 and lo <= tick < hi:
            out.append((tick, m.note))
    return out


def test_replace_keeps_locked_measures(song):
    generated = mido.MidiFile(song / "notes.mid")
    measure = 4 * generated.ticks_per_beat
    # O usuário mexeu à mão nos compassos 1-4 do Expert (troca as cores) e eles ficaram travados.
    edited = mido.MidiFile(song / "notes.mid")
    tick = 0
    for m in next(t for t in edited.tracks if t.name == "PART GUITAR"):
        tick += m.time
        if m.type in ("note_on", "note_off") and 96 <= m.note <= 100 and tick < 4 * measure:
            m.note = 96 + (m.note - 95) % 5
    edited.save(song / "notes.mid")
    (song / "notes.locks.json").write_text(json.dumps({"locks": {"guitar": {"expert": [[0, 4 * measure]]}}}))

    result = augment(song, ("guitar",), replace_existing=True, log=lambda *_: None)
    assert result.added == ["guitar"] and result.undone > 0
    after = mido.MidiFile(song / "notes.mid")
    # Travados: como o usuário deixou. O resto: refeito pela transcrição.
    assert expert_guitar(after, 0, 4 * measure) == expert_guitar(edited, 0, 4 * measure) != []
    assert expert_guitar(after, 0, 4 * measure) != expert_guitar(generated, 0, 4 * measure)
    assert expert_guitar(after, 4 * measure, 10 ** 9) == expert_guitar(generated, 4 * measure, 10 ** 9) != []


def test_missing_stem_and_lyrics_are_reported(song):
    (song / "bass.ogg").unlink()
    result = augment(song, ("bass", "vocals", "keys"), log=lambda *_: None)
    assert result.added == []
    assert set(result.skipped) == {"bass", "vocals", "keys"}
    assert not (song / BEFORE_ADD_NAME).exists()   # nada gravado


def test_folder_without_notes_mid(tmp_path):
    (tmp_path / "notes.chart").write_text("[Song]")
    with pytest.raises(RuntimeError, match=r"\.chart ainda não"):
        augment(tmp_path, ("bass",))


def test_update_song_ini_changes_only_given_keys(tmp_path):
    ini = tmp_path / "song.ini"
    ini.write_text("[Song]\nName = Música\nDiff_Bass = -1\ndelay = 0\n", encoding="utf-8")
    update_song_ini(ini, {"diff_bass": 3, "diff_vocals": 2})
    assert ini.read_text("utf-8") == "[Song]\nName = Música\ndiff_bass = 3\ndelay = 0\ndiff_vocals = 2\n"
