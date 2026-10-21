"""Testes rápidos: sílabas, alinhamento da letra, escolha de guitarra e o arquivo MIDI."""

import numpy as np
import pytest

from ai_charter.analysis.tempo import TempoMap
from ai_charter.charting.common import ChartNote, TrackChart
from ai_charter.charting.guitar_select import choose_sources
from ai_charter.charting.lyrics import line_syllables, syllabify_en, syllabify_pt
from ai_charter.charting.vocals import VocalNote, VocalTrack, align_lyrics
from ai_charter.config import load_rules
from ai_charter.export.midi_writer import write_midi
from chart_parser import midi_notes, parse_midi

RES = 192


@pytest.mark.parametrize("word, expected", [
    ("caminho", "ca-mi-nho"), ("coração", "co-ra-ção"), ("carro", "car-ro"),
    ("pressa", "pres-sa"), ("chegar", "che-gar"), ("estrada", "es-tra-da"),
    ("queijo", "quei-jo"), ("água", "á-gua"), ("aguentar", "a-guen-tar"),
    ("saúde", "sa-ú-de"), ("dia", "di-a"), ("pão", "pão"), ("sa-u-de", "sa-u-de"),
])
def test_portuguese_syllables(word, expected):
    assert "-".join(syllabify_pt(word)) == expected


def test_english_syllables():
    assert syllabify_en("little") == ["lit", "tle"]
    assert syllabify_en("remember") == ["re", "mem", "ber"]


def test_line_syllables_use_game_format():
    assert line_syllables("Olá, mundo!") == ["O-", "lá,", "mun-", "do!"]


def _notes(ticks):
    """Notas de 1/8 nas posições dadas (em colcheias)."""
    return [VocalNote(t * RES // 2, RES // 2 - 10, 60) for t in ticks]


def test_alignment_one_syllable_per_note():
    notes = _notes([0, 1, 2, 10, 11])
    starts = [True, False, False, True, False]
    texts, lines = align_lyrics(notes, [["a", "b", "c"], ["d", "e"]], starts)
    assert texts == ["a", "b", "c", "d", "e"] and lines == [0, 0, 0, 1, 1]


def test_alignment_extra_note_becomes_melisma():
    notes = _notes([0, 1, 2, 3])
    texts, _ = align_lyrics(notes, [["can-", "tar"]], [True, False, False, False])
    assert texts.count("+") == 2 and texts[0] == "can-"


def test_alignment_missing_note_merges_syllables():
    notes = _notes([0, 1])
    texts, _ = align_lyrics(notes, [["ca-", "mi-", "nho"]], [True, False])
    assert texts in (["cami-", "nho"], ["ca-", "minho"])


def test_alignment_drops_noise_between_lines():
    notes = _notes([0, 1, 6, 12, 13])     # a nota da colcheia 6 está sozinha (ruído)
    starts = [True, False, True, True, False]
    texts, _ = align_lyrics(notes, [["a", "b"], ["c", "d"]], starts)
    assert texts == ["a", "b", None, "c", "d"]


def test_guitar_choice_holds_through_short_rest():
    rules = load_rules().guitar_selection
    nan = np.nan
    # Fonte 0 = violão (sempre tocando), fonte 1 = solo, com 1 compasso de pausa no meio.
    scores = np.array([
        [0.5, 0.5, -0.7, -0.7, 0.3, -0.7, -0.7, 0.5, 0.5],
        [nan, nan, 2.0, 2.0, nan, 2.0, 2.0, nan, nan],
    ])
    assert choose_sources(scores, rules) == [0, 0, 1, 1, 1, 1, 1, 0, 0]


def test_guitar_choice_switches_for_long_breaks():
    rules = load_rules().guitar_selection
    nan = np.nan
    scores = np.array([
        [0.5, -0.7, -0.7, 0.5, 0.5, 0.5, 0.5, -0.7, -0.7],
        [nan, 2.0, 2.0, nan, nan, nan, nan, 2.0, 2.0],
    ])
    assert choose_sources(scores, rules) == [0, 1, 1, 0, 0, 0, 0, 1, 1]


def test_midi_file_layout(tmp_path):
    tempo = TempoMap(np.arange(32) * 0.5, RES)
    guitar = TrackChart(notes={"expert": [ChartNote(0, (0,)), ChartNote(RES, (1, 3), length=RES)],
                               "easy": [ChartNote(0, (2,), flags=(5,))]},
                        star_power={"expert": [(0, RES * 2)]})
    drums = TrackChart(notes={"expert": [ChartNote(0, (0, 2), flags=(66,)), ChartNote(RES, (1, 3))]})
    vocals = VocalTrack(notes=[VocalNote(0, RES, 60, "can-"), VocalNote(RES, RES, 62, "ção")],
                        phrases=[(0, RES * 2)])
    path = tmp_path / "notes.mid"
    write_midi(path, "Teste", tempo, {"guitar": guitar, "drums": drums}, [0], vocals, RES * 8)

    midi = parse_midi(path)
    assert midi_notes(midi["PART GUITAR"], 96, 100) == {0: [0], RES: [1, 3]}
    assert midi_notes(midi["PART GUITAR"], 60, 66) == {0: [2, 6]}          # Easy amarelo + forçar strum
    assert midi_notes(midi["PART GUITAR"], 116, 116) == {0: [0]}           # Star Power
    assert midi_notes(midi["PART DRUMS"], 96, 100) == {0: [0, 2], RES: [1, 3]}
    assert midi_notes(midi["PART DRUMS"], 110, 112) == {RES: [1]}          # azul sem prato = tom
    assert [v for _, k, v in midi["PART VOCALS"] if k == "lyric"] == ["can-", "ção"]
    assert midi_notes(midi["PART VOCALS"], 105, 105) == {0: [0]}           # frase
    assert ("text", "[section Section 1]") in [(k, v) for _, k, v in midi["EVENTS"]]
