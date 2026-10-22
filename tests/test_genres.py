"""Predefinições de regras por gênero musical (ai_charter/genres/*.toml)."""

import shutil
import tomllib
from importlib import resources

import pytest

from ai_charter.config import genre_for_tag, get_genre, list_genres, load_rules

DEFAULT = tomllib.loads(resources.files("ai_charter").joinpath("default_rules.toml").read_text("utf-8"))


def paths(data, prefix=()):
    for key, value in data.items():
        if isinstance(value, dict):
            yield from paths(value, (*prefix, key))
        else:
            yield (*prefix, key)


def test_every_genre_loads_and_only_overrides_known_keys():
    genres = list_genres()
    assert len(genres) == 10
    known = set(paths(DEFAULT))
    for genre in genres:
        assert genre.name and genre.description and genre.aliases, genre.id
        unknown = set(paths(genre.rules)) - known   # uma tabela com nome errado seria ignorada em silêncio
        assert not unknown, (genre.id, unknown)
        load_rules(genre=genre.id)
        assert genre.tempo in (None, "constant", "variable")


def test_rock_is_the_default_rules():
    assert get_genre("rock").rules == {}
    assert load_rules(genre="rock") == load_rules()


def test_aliases_are_unique_and_match_tags():
    seen = {}
    for genre in list_genres():
        for alias in genre.aliases:
            assert alias not in seen, f"{alias} em {seen.get(alias)} e {genre.id}"
            seen[alias] = genre.id
    assert genre_for_tag("Thrash Metal") == "metal"
    assert genre_for_tag("  bossa nova ") == "mpb"
    assert genre_for_tag("Pop/Rock") == "pop"
    assert genre_for_tag("classic rock") == "rock"
    assert genre_for_tag("polca") is None and genre_for_tag("") is None and genre_for_tag(None) is None


def test_layers_default_then_genre_then_user_file(tmp_path):
    default = load_rules()
    metal = load_rules(genre="metal")
    assert metal.detection["guitar"].onset_delta == 0.03 != default.detection["guitar"].onset_delta
    assert metal.detection["guitar"].fmin == default.detection["guitar"].fmin   # o resto continua
    mine = tmp_path / "minhas.toml"
    mine.write_text("[detection.guitar]\nonset_delta = 0.05\n")
    both = load_rules(mine, genre="metal")
    assert both.detection["guitar"].onset_delta == 0.05             # o arquivo do usuário vence
    assert both.sanitize.noise_ratio == metal.sanitize.noise_ratio  # e o resto do gênero continua


def test_unknown_genre():
    with pytest.raises(ValueError, match="Gênero desconhecido"):
        load_rules(genre="polca")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="precisa do ffmpeg")
def test_pipeline_uses_genre_rules_and_tempo(tmp_path, monkeypatch):
    import synth
    from ai_charter import pipeline

    paths = synth.render(tmp_path)
    stems = tmp_path / "stems"
    stems.mkdir()
    shutil.move(paths["bass"], stems / "bass.wav")
    modes = []
    real = pipeline.estimate_tempo_map
    monkeypatch.setattr(pipeline, "estimate_tempo_map", lambda *a: modes.append(a[3]) or real(*a))
    lines = []
    pipeline.run(pipeline.Options(input=paths["mix"], output_dir=tmp_path / "out", separator="folder",
                                  stems_dir=stems, instruments=("bass",), genre="eletronica"), log=lines.append)
    assert "Regras do gênero: Eletrônica / Dance" in lines
    assert modes == ["constant"]   # o gênero sugere BPM constante quando o tempo está em "auto"


def test_cli_accepts_genre(tmp_path):
    from ai_charter.cli import parse_args

    music = tmp_path / "m.flac"
    music.write_bytes(b"fLaC")
    assert parse_args([str(music), "--genre", "mpb"]).genre == "mpb"
    assert parse_args([str(music)]).genre is None
    with pytest.raises(SystemExit):
        parse_args([str(music), "--genre", "polca"])
