"""Outro stem como fonte de uma pista (stem_sources.py): ex.: o sintetizador vira a guitarra."""

import shutil
from pathlib import Path

import mido
import numpy as np
import pytest
import soundfile as sf

import synth
from ai_charter import stem_sources
from ai_charter.app import BadRequest, parse_add_request, parse_request, song_info
from ai_charter.augment import augment, existing_parts
from ai_charter.cli import parse_args
from ai_charter.pipeline import Options, run

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="precisa do ffmpeg")


def test_parse_accepts_portuguese_names_and_drops_identity():
    assert stem_sources.parse("guitar=other, keys=vocals") == {"guitar": "other", "keys": "vocals"}
    assert stem_sources.parse("guitarra=sintetizador,teclado=voz") == {"guitar": "other", "keys": "vocals"}
    assert stem_sources.parse("guitar=guitar") == {}
    assert stem_sources.parse("") == {}


@pytest.mark.parametrize("text", ["drums=other", "guitar=drums", "vocals=other", "guitar=other,keys=resto",
                                  "guitar", "guitar=nada"])
def test_parse_rejects_bad_swaps(text):
    with pytest.raises(ValueError):
        stem_sources.parse(text)


def test_layout_swaps_the_chosen_stem_with_the_instrument():
    assert stem_sources.layout({"guitar": "other"}) == {"guitar": "other", "other": "guitar"}
    assert stem_sources.layout({"guitar": "keys", "keys": "guitar"}) == {"guitar": "keys", "keys": "guitar"}
    # Em cadeia: a guitarra vai para o lugar que sobrou (o do resto)
    assert stem_sources.layout({"guitar": "keys", "keys": "other"}) == {
        "guitar": "keys", "keys": "other", "other": "guitar"}


def test_arrange_keeps_every_file_and_original_undoes_it():
    stems = {name: [Path(f"{name}.wav")] for name in ("guitar", "bass", "drums", "keys", "vocals", "other")}
    for mapping in ({"guitar": "other"}, {"guitar": "other", "keys": "vocals"}, {"guitar": "keys", "keys": "other"}):
        after = stem_sources.arrange(stems, mapping)
        assert sorted(sum(after.values(), [])) == sorted(sum(stems.values(), []))   # a mixagem total não muda
        assert after["guitar"] == stems[mapping["guitar"]] and after["drums"] == stems["drums"]
        assert stem_sources.original(after, mapping) == stems
    # Sem guitarra na música: o resto vira a guitarra e o lugar do resto fica sem som
    assert stem_sources.arrange({"other": [Path("other.wav")]}, {"guitar": "other"}) == {
        "guitar": [Path("other.wav")]}


def test_cli_app_and_augment_forms(tmp_path):
    music = tmp_path / "m.mp3"
    music.write_bytes(b"x")
    assert parse_args([str(music), "--stem-source", "guitarra=resto"]).stem_sources == {"guitar": "other"}
    assert parse_args([str(music)]).stem_sources is None
    with pytest.raises(SystemExit):
        parse_args([str(music), "--stem-source", "drums=other"])
    form = {"input": str(music), "instruments": ["guitar"]}
    assert parse_request({**form, "stem_sources": {"guitar": "other", "keys": ""}}, {}).stem_sources == {
        "guitar": "other"}
    assert parse_request({**form, "stem_sources": {"guitar": "guitar"}}, {}).stem_sources is None
    with pytest.raises(BadRequest):
        parse_request({**form, "stem_sources": {"drums": "other"}}, {})
    song = tmp_path / "song"
    song.mkdir()
    mido.MidiFile().save(song / "notes.mid")
    assert parse_add_request({"folder": str(song), "instruments": ["keys"],
                              "stem_sources": {"keys": "vocals"}}).stem_sources == {"keys": "vocals"}


@pytest.fixture(scope="module")
def no_guitar(tmp_path_factory):
    """Música sem guitarra (o stem dela é silêncio): a melodia está no "resto", como um sintetizador."""
    folder = tmp_path_factory.mktemp("no_guitar")
    paths = synth.render(folder)
    stems = folder / "stems"
    stems.mkdir()
    shutil.move(paths["guitar"], stems / "other.wav")
    shutil.move(paths["bass"], stems / "bass.wav")
    shutil.move(paths["drums"], stems / "drums.wav")
    y, sr = sf.read(stems / "other.wav")
    sf.write(stems / "guitar.wav", np.zeros_like(y, dtype=np.float32), sr)
    return paths["mix"], stems


def guitar_notes(folder):
    track = next((t for t in mido.MidiFile(folder / "notes.mid").tracks if t.name == "PART GUITAR"), [])
    return [m for m in track if m.type == "note_on" and m.velocity]


@needs_ffmpeg
def test_pipeline_charts_guitar_from_other_and_swaps_the_audio(no_guitar, tmp_path):
    mix, stems = no_guitar
    common = dict(input=mix, separator="folder", stems_dir=stems, instruments=("guitar",), title="S", artist="T")
    plain = run(Options(output_dir=tmp_path / "plain", **common), log=lambda *_: None)
    assert "guitar" not in plain.tracks   # o stem da guitarra é silêncio

    lines = []
    result = run(Options(output_dir=tmp_path / "swap", stem_sources={"guitar": "other"}, **common), log=lines.append)
    assert "guitar" in result.tracks and len(guitar_notes(result.song_dir)) > 20
    assert any("usando o stem other" in line for line in lines)
    # O sintetizador foi para o guitar.ogg (silencia quando o jogador erra) e o silêncio para o song.ogg
    def level(name):
        y, _ = sf.read(result.song_dir / name)
        return float(np.sqrt(np.mean(y ** 2)))
    assert level("guitar.ogg") > 0.01 and level("song.ogg") < 0.001
    assert stem_sources.recorded(result.song_dir) == {"guitar": "other"}


@needs_ffmpeg
def test_augment_swaps_audio_and_can_undo_it(no_guitar, tmp_path):
    """Acrescentar: o teclado sai da guitarra; depois, refeito sem troca, os áudios voltam ao lugar."""
    mix, stems = no_guitar
    song = run(Options(input=mix, output_dir=tmp_path, separator="folder", stems_dir=stems, instruments=("bass",),
                       title="S", artist="T"), log=lambda *_: None).song_dir
    (song / "guitar.ogg").unlink()               # a guitarra silenciosa sai
    shutil.move(song / "song.ogg", song / "guitar.ogg")   # e a melodia fica no guitar.ogg
    original = (song / "guitar.ogg").read_bytes()

    result = augment(song, ("keys",), stem_sources={"keys": "guitar"}, log=lambda *_: None)
    assert result.added == ["keys"] and "keys" in existing_parts(song)
    assert (song / "keys.ogg").read_bytes() == original and not (song / "guitar.ogg").exists()
    assert stem_sources.recorded(song) == {"keys": "guitar"}
    assert song_info(song)["stem_sources"] == {"keys": "guitar"}

    # Refazer a guitarra: ela ouve o stem original da guitarra (que agora está no keys.ogg)
    result = augment(song, ("guitar",), log=lambda *_: None)
    assert result.added == ["guitar"] and len(guitar_notes(song)) > 20
    assert stem_sources.recorded(song) == {"keys": "guitar"}

    result = augment(song, ("keys",), replace_existing=True, log=lambda *_: None)   # sem troca
    assert "keys" in result.skipped                 # o teclado de verdade não existe
    assert stem_sources.recorded(song) == {"keys": "guitar"}   # pista pulada: a troca dela continua

    # O teclado passa a sair do baixo: a melodia volta para o guitar.ogg
    result = augment(song, ("keys",), replace_existing=True, stem_sources={"keys": "bass"}, log=lambda *_: None)
    assert result.added == ["keys"]
    assert (song / "guitar.ogg").read_bytes() == original and (song / "keys.ogg").exists()
    assert not (song / "bass.ogg").exists() and stem_sources.recorded(song) == {"keys": "bass"}

    with pytest.raises(ValueError):   # o baixo já virou o teclado
        augment(song, ("guitar",), replace_existing=True, stem_sources={"guitar": "bass"}, log=lambda *_: None)
