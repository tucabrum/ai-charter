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
    # A bateria sintética não soa como um kit de verdade: confere as regras do transcritor, não o
    # modelo (como o conftest faz nos outros testes; esta fixture é por módulo e roda fora dele).
    import ai_charter.pipeline as pipeline

    class _NoModel:
        @staticmethod
        def load(*_args, **_kwargs):
            return None

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pipeline, "DrumModel", _NoModel)
        result = run(Options(input=paths["mix"], output_dir=out, separator="folder", stems_dir=stems,
                             title="Synth", artist="Test", make_zip=True, output_format="chart", lead_in=0),
                     log=lambda *_: None)
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
    # A melodia tem uma nota por colcheia, começando 1 s (= 2 batidas) depois do início. Com o Beat This!,
    # 1 s é início de compasso e vai para a batida 4 (o silêncio antes vira 4 batidas de 0,25 s).
    assert ticks[0] in (2 * RES, 4 * RES)
    assert ticks == [ticks[0] + i * EIGHTH for i in range(len(synth.MELODY))]
    # Mesma nota -> mesma lane; nota diferente -> lane diferente.
    lanes = [expert[t][0] for t in ticks]
    for i in range(1, len(lanes)):
        same_pitch = synth.MELODY[i] == synth.MELODY[i - 1]
        assert (lanes[i] == lanes[i - 1]) == same_pitch, (i, lanes)
    # As lanes acompanham a direção da melodia, exceto nos reposicionamentos
    # (escala de 6 notas em 5 lanes): no máximo um por frase de 8 notas.
    against = sum(
        1 for i in range(1, len(lanes))
        if synth.MELODY[i] != synth.MELODY[i - 1]
        and (synth.MELODY[i] > synth.MELODY[i - 1]) != (lanes[i] > lanes[i - 1])
    )
    assert against <= len(lanes) // 8


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
    start = min(notes_by_tick(chart["ExpertSingle"]))   # a música começa junto com a guitarra
    beats = range(start, start + 24 * RES, RES)
    kick_beats = [t for t in beats if 0 in expert.get(t, [])]
    snare_beats = [t for t in beats if 1 in expert.get(t, [])]
    # Bumbo nas batidas pares, caixa nas ímpares (pelo menos 90% de acerto).
    assert sum((t - start) // RES % 2 == 0 for t in kick_beats) >= 0.9 * 12
    assert sum((t - start) // RES % 2 == 1 for t in snare_beats) >= 0.9 * 12
    # Pratos marcados como cymbal (Pro Drums): o chimbal e o prato grande. O "ataque" sintético
    # não soa como um de verdade, e o modelo (feito para kits reais) pode chamá-lo de condução.
    assert any(66 in v for v in expert.values())
    assert any(67 in v or 68 in v for v in expert.values())


def test_star_power_phrases(song):
    chart = parse_chart(song.song_dir / "notes.chart")
    for track in ("ExpertSingle", "ExpertDoubleBass", "ExpertDrums"):
        assert any(v.startswith("S 2 ") for _, v in chart[track]), track
