"""Cifra (Cifra Club): letra, acordes, capotraste e o vocabulário no reconhecimento de acordes."""

import shutil
from dataclasses import replace

import numpy as np
import pytest

from ai_charter.analysis.chords import CHORDS, chord_name
from ai_charter.charting.chords import label_chords, vocabulary_mask
from ai_charter.cifra import best_transposition, parse_cifra, simplify, sounding_vocabulary
from ai_charter.config import load_rules
from test_playability import RES, strum

CIFRA = """Alpargatos - A Viagem
Composição de Fulano
Tom: G
Capotraste na 2ª casa

[Intro] G  D  Em  C  (2x)

[Primeira Parte]

G              D
  Quando eu te vi
Em7          C7M
  Fiquei sem chão

[Solo]
E|-----3-----|
B|---3---3---|

[Refrão]
C    D/F#   G
Ô ô viagem
Parte 1 de 2
"""


def test_reads_lyrics_chords_capo_and_key():
    cifra = parse_cifra(CIFRA)
    assert cifra.lyrics == ["Quando eu te vi", "Fiquei sem chão", "", "Ô ô viagem"]
    assert cifra.capo == 2 and cifra.key == "G"
    assert cifra.vocabulary == ["G", "D", "Em", "C"]
    assert sounding_vocabulary(cifra) == ["A", "E", "F#m", "D"]   # formatos na casa 2


@pytest.mark.parametrize("token, expected", [
    ("Am7", "Am"), ("F#m7(b5)", "F#m"), ("Bb/D", "A#"), ("G5", "G5"), ("C7M", "C"), ("D4", "D"),
    ("E7(9)", "E"), ("A°", "Am"), ("Cadd9", "C"), ("Dsus4", "D"), ("Bm7(11)", "Bm"), ("Ebm", "D#m"),
    ("Quando", None), ("x2", None), ("a", None), ("Em-", "Em"),
])
def test_chord_names(token, expected):
    assert simplify(token) == expected


def test_text_without_chords_is_all_lyrics():
    assert parse_cifra("primeira linha\nsegunda\n\nterceira\n").lyrics == ["primeira linha", "segunda", "",
                                                                            "terceira"]


def test_best_transposition_finds_the_key_of_the_recording():
    chroma = np.zeros(12)
    for pitch_class in (9, 1, 4, 2, 6):   # A, C#, E, D, F#: a música soa em Lá
        chroma[pitch_class] = 1.0
    shift, margin = best_transposition(["G", "D", "Em", "C"], chroma)
    assert shift == 2 and margin > 0


def test_vocabulary_mask_allows_chord_power_chord_and_single_notes():
    mask = vocabulary_mask(["Am", "Gb7"])
    allowed = {chord_name(i) for i in np.flatnonzero(mask) if CHORDS[i][1] != "note"}
    assert allowed == {"Am", "A5", "F#", "F#5"}
    assert all(mask[i] for i, (_, kind) in enumerate(CHORDS) if kind == "note")


def test_vocabulary_overrides_a_wrong_root_and_major_minor():
    cfg = load_rules().detection["guitar"]
    C, Em, E = CHORDS.index((0, "")), CHORDS.index((4, "m")), CHORDS.index((4, ""))
    Am, A = CHORDS.index((9, "m")), CHORDS.index((9, ""))
    # o áudio "acha" Em (o C7M soa Mi-Sol-Si) e Lá maior (a terça é ambígua)
    first = [strum(i * RES // 2, Em, {C: 0.8}) for i in range(4)]
    second = [strum(2 * RES + i * RES // 2, A, {Am: 0.85}) for i in range(4)]
    label_chords(first + second, RES, cfg)
    assert {chord_name(c.chord) for c in first} == {"Em"} and {chord_name(c.chord) for c in second} == {"A"}

    cfg = replace(cfg, allowed_chords=["C", "Am", "G"])
    first = [strum(i * RES // 2, Em, {C: 0.8}) for i in range(4)]
    second = [strum(2 * RES + i * RES // 2, A, {Am: 0.85}) for i in range(4)]
    label_chords(first + second, RES, cfg)
    assert {chord_name(c.chord) for c in first} == {"C"} and {chord_name(c.chord) for c in second} == {"Am"}
    assert E not in {c.chord for c in first}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="precisa do ffmpeg")
def test_pipeline_uses_cifra_lyrics_and_keeps_a_copy(tmp_path):
    import synth
    from ai_charter.pipeline import Options, run

    paths = synth.render(tmp_path)
    stems = tmp_path / "stems"
    stems.mkdir()
    shutil.move(paths["guitar"], stems / "guitar.wav")
    shutil.copy(stems / "guitar.wav", stems / "vocals.wav")   # uma "voz" para a faixa de vocal
    cifra = tmp_path / "cifra.txt"
    cifra.write_text("[Intro] Em G\n\nEm        G\nLa la la la\n", encoding="utf-8")
    lines = []
    result = run(Options(input=paths["mix"], output_dir=tmp_path / "out", separator="folder", stems_dir=stems,
                         instruments=("guitar",), cifra=cifra), log=lines.append)
    assert any(line.startswith("Cifra cifra.txt: 1 linhas de letra, acordes ") for line in lines)
    assert (result.song_dir / "letra.txt").read_text("utf-8") == "La la la la\n"
    assert (result.song_dir / "cifra.txt").read_text("utf-8") == cifra.read_text("utf-8")
    assert "vocals: letra tirada da cifra (salva em letra.txt)" in "\n".join(lines)
