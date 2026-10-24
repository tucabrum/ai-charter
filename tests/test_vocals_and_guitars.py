"""Testes rápidos: sílabas, alinhamento da letra, escolha de guitarra e o arquivo MIDI."""

import numpy as np
import pytest

from ai_charter.analysis.tempo import TempoMap
from ai_charter.analysis.transcribe import TimedWord, transcribe_lyrics, words_to_lines
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


def test_transcribed_words_are_grouped_in_lines_by_pauses():
    words = [TimedWord(" No", 0.0, 0.2), TimedWord(" caminho", 0.25, 0.8), TimedWord(" da", 0.85, 1.0),
             TimedWord(" manhã,", 1.05, 2.0), TimedWord(" o", 2.6, 2.8), TimedWord(" sol", 2.85, 3.2)]
    assert words_to_lines(words, gap_seconds=0.45) == ["No caminho da manhã,", "o sol"]


def test_auto_lyrics_explains_how_to_install(monkeypatch, tmp_path):
    import sys
    monkeypatch.setitem(sys.modules, "faster_whisper", None)   # simula pacote não instalado
    with pytest.raises(RuntimeError, match=r"ai-charter\[lyrics\]"):
        transcribe_lyrics(tmp_path / "vocals.wav", "pt", "small", 0.45)


def test_auto_lyrics_hands_whisper_decoded_audio(monkeypatch, tmp_path):
    """O Whisper recebe o áudio já decodificado (16 kHz), e não o arquivo: o decodificador dele
    quebra com a PyAV 15+."""
    import sys
    import types

    import soundfile as sf

    sf.write(tmp_path / "vocals.wav", np.zeros(44100, dtype=np.float32), 44100)
    seen = {}

    class FakeModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def transcribe(self, audio, **_kwargs):
            seen["audio"] = audio
            word = types.SimpleNamespace(word=" oi", start=0.1, end=0.3)
            return [types.SimpleNamespace(words=[word])], None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeModel))
    assert transcribe_lyrics(tmp_path / "vocals.wav", "pt", "small", 0.45) == ["oi"]
    assert isinstance(seen["audio"], np.ndarray) and len(seen["audio"]) == 16000


def test_chord_recognition_from_spectrum():
    from ai_charter.analysis.chords import chord_name, chord_scores, is_single_note

    def spectrum(midis, lowest=40, n=60):
        s = np.zeros(n)
        for m in midis:
            for h, w in ((0, 1.0), (12, 0.5), (19, 0.33), (24, 0.25)):   # nota + harmônicos
                if m + h - lowest < n:
                    s[m + h - lowest] += w
        return s

    assert chord_name(int(np.argmax(chord_scores(spectrum([48, 52, 55, 60, 64]), 40)))) == "C"
    assert chord_name(int(np.argmax(chord_scores(spectrum([45, 52, 57, 60, 64]), 40)))) == "Am"
    single = int(np.argmax(chord_scores(spectrum([64]), 40)))
    assert chord_name(single) == "E" and is_single_note(single)    # nota simples, não acorde de Mi


def test_sanitize_leaves_clean_sparse_measures_alone():
    from ai_charter.charting.common import Candidate
    from ai_charter.charting.sanitize import sanitize

    sr = 22050
    tempo = TempoMap(np.arange(40) * 0.5, RES)                 # 120 BPM
    rng = np.random.default_rng(0)
    audio = np.zeros(sr * 18)
    groove = [0, RES, RES * 3 // 2, RES * 5 // 2, RES * 3, RES * 7 // 2]
    cands = []
    for m in range(8):
        positions = groove if m < 7 else [0]                    # último compasso: um acorde só (de propósito)
        for p in positions:
            tick = m * RES * 4 + p
            t = int(tempo.tick_to_time(tick) * sr)
            audio[t:t + 2000] += rng.standard_normal(2000) * np.exp(-np.arange(2000) / 300)
            cands.append(Candidate(tick, 1.0, end_tick=tick + 50, pitch=48, voices=3, chord=0,
                                   chord_scores=np.ones(48)))
    rules = load_rules().sanitize
    result, simplified = sanitize(cands, audio, sr, tempo, RES, rules)
    assert simplified == [] and result == cands


def test_compare_chart_with_itself_and_finds_time_offset(tmp_path):
    from ai_charter.compare import best_offset, read_chart, score_track

    tempo = TempoMap(np.arange(64) * 0.5, RES)
    rng = np.random.default_rng(3)
    ticks = sorted(set(int(t) for t in rng.integers(0, 50, 40) * (RES // 4)))   # ritmo irregular
    notes = [ChartNote(t, (i % 5,), length=RES if i % 4 == 0 else 0) for i, t in enumerate(ticks)]
    path = tmp_path / "a.mid"
    write_midi(path, "A", tempo, {"guitar": TrackChart(notes={"expert": notes})}, [0], None, RES * 30)

    events = read_chart(path).tracks["PART GUITAR"]["expert"]
    same = score_track(events, events, 0.05, 20.0)
    assert same.recall == same.precision == 1.0 and same.contour_agree == 100.0

    times = [e.time for e in events]
    late = [t + 0.3 for t in times]                       # a mesma chart, 0,3 s atrasada
    assert abs(best_offset(times, late) + 0.3) < 0.01      # precisa voltar 0,3 s
