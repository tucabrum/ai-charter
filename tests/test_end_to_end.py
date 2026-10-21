"""Teste completo: música sintética -> pasta pronta para o Clone Hero / YARG."""

import shutil

import pytest

import synth
from ai_charter.pipeline import Options, run
from chart_parser import notes_by_tick, parse_chart

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="precisa do ffmpeg")

RES = 192
EIGHTH = RES // 2


@pytest.fixture(scope="module")
def song(tmp_path_factory):
    folder = tmp_path_factory.mktemp("synth")
    paths = synth.render(folder)
    stems = folder / "stems"
    stems.mkdir()
    for name in ("guitar", "bass", "drums"):
        shutil.move(paths[name], stems / f"{name}.wav")
    out = folder / "out"
    result = run(Options(input=paths["mix"], output_dir=out, separator="folder", stems_dir=stems,
                         title="Synth", artist="Test", make_zip=True), log=lambda *_: None)
    return result


def test_output_folder_has_everything(song):
    files = {p.name for p in song.song_dir.iterdir()}
    assert {"notes.chart", "song.ini", "guitar.ogg", "bass.ogg", "drums.ogg"} <= files
    assert song.zip_path is not None and song.zip_path.exists()
    assert song.bpm == pytest.approx(120, abs=0.5)
    ini = (song.song_dir / "song.ini").read_text()
    assert "pro_drums = True" in ini and "diff_keys = -1" in ini


def test_guitar_expert_matches_melody(song):
    chart = parse_chart(song.song_dir / "notes.chart")
    expert = notes_by_tick(chart["ExpertSingle"])
    ticks = sorted(expert)
    # A melodia tem uma nota por colcheia, começando 1 s (= 2 batidas) depois do início.
    assert ticks == [2 * RES + i * EIGHTH for i in range(len(synth.MELODY))]
    # As lanes sobem e descem junto com a melodia.
    lanes = [expert[t][0] for t in ticks]
    for i in range(1, len(lanes)):
        step = synth.MELODY[i] - synth.MELODY[i - 1]
        if step > 0:
            assert lanes[i] >= lanes[i - 1]
        elif step < 0:
            assert lanes[i] <= lanes[i - 1]
        else:
            assert lanes[i] == lanes[i - 1]


def test_all_difficulties_exist_and_get_easier(song):
    chart = parse_chart(song.song_dir / "notes.chart")
    for instrument in ("Single", "DoubleBass", "Drums"):
        counts = [len(notes_by_tick(chart[f"{d}{instrument}"])) for d in ("Expert", "Hard", "Medium", "Easy")]
        assert counts == sorted(counts, reverse=True), (instrument, counts)
    easy = notes_by_tick(chart["EasySingle"])
    assert max(max(v) for v in easy.values()) <= 2   # só verde, vermelho e amarelo


def test_drums_detect_kick_snare_and_cymbals(song):
    chart = parse_chart(song.song_dir / "notes.chart")
    expert = notes_by_tick(chart["ExpertDrums"])
    beats = range(2 * RES, 2 * RES + 24 * RES, RES)
    kick_beats = [t for t in beats if 0 in expert.get(t, [])]
    snare_beats = [t for t in beats if 1 in expert.get(t, [])]
    # Bumbo nas batidas pares, caixa nas ímpares (pelo menos 90% de acerto).
    assert sum((t - 2 * RES) // RES % 2 == 0 for t in kick_beats) >= 0.9 * 12
    assert sum((t - 2 * RES) // RES % 2 == 1 for t in snare_beats) >= 0.9 * 12
    # Pratos marcados como cymbal (Pro Drums).
    assert any(66 in v for v in expert.values()) and any(68 in v for v in expert.values())
