"""Alinhamento forçado da letra (analysis/align.py) e a faixa de vocal feita a partir dele."""

import numpy as np

from ai_charter.analysis.align import FRAME, SyllableSpan, align_emissions, realign_gaps, romanize
from ai_charter.analysis.tempo import TempoMap
from ai_charter.analysis.vocals import SungNote
from ai_charter.charting.vocals import build_aligned_track
from ai_charter.config import load_rules


def _emissions(script: list[tuple[str, int]], n_frames: int) -> np.ndarray:
    """Emissões falsas: em cada trecho (letra, quadro inicial) a rede "ouve" aquela letra por 3 quadros;
    no resto, silêncio (o símbolo vazio)."""
    from torchaudio.pipelines import MMS_FA

    vocab = MMS_FA.get_dict(star=None)
    em = np.full((n_frames, len(vocab)), -20.0, dtype=np.float32)
    em[:, 0] = 0.0
    for ch, frame in script:
        em[frame:frame + 3, :] = -20.0
        em[frame:frame + 3, vocab[ch]] = 0.0
    return em


def test_romanize_keeps_only_model_letters():
    assert romanize("Lá,") == "la"
    assert romanize("Ação!") == "acao"
    assert romanize("don't") == "don't"


def test_align_emissions_finds_each_syllable_and_its_vowel():
    # "ca-" "sa" "oi": c em 10, a em 14 | s em 40, a em 44 | o em 80, i em 84
    script = [("c", 10), ("a", 14), ("s", 40), ("a", 44), ("o", 80), ("i", 84)]
    spans = align_emissions(_emissions(script, 120), [["ca-", "sa"], ["oi"]])
    (ca, sa), (oi,) = spans
    assert abs(ca.start - 10 * FRAME) < 2 * FRAME and abs(ca.vowel - 14 * FRAME) < 2 * FRAME
    assert abs(sa.vowel - 44 * FRAME) < 2 * FRAME
    assert abs(oi.start - 80 * FRAME) < 2 * FRAME and oi.end <= 90 * FRAME


def test_align_emissions_skips_syllables_without_letters():
    spans = align_emissions(_emissions([("o", 10)], 40), [["...", "oh"]])
    assert spans[0][0] is None and spans[0][1] is not None


def test_realign_gaps_redoes_only_the_badly_heard_stretch_between_anchors():
    lines = [["oi"], ["ca-", "sa"], ["sol"]]
    spans = [[SyllableSpan(1.0, 1.2, 0.9, 1.1)],                                     # âncora
             [SyllableSpan(1.3, 1.32, 0.05, 1.31), SyllableSpan(1.34, 1.36, 0.05, 1.35)],   # mal ouvida
             [SyllableSpan(5.0, 5.2, 0.8, 5.1)]]                                     # âncora
    asked = []

    def emit(a, b):   # a rede, ouvindo só o trecho entre as âncoras, entende "ca-sa" em +1 s e +2 s
        asked.append((a, b))
        return _emissions([("c", 50), ("a", 54), ("s", 100), ("a", 104)], int((b - a) / FRAME))

    out = realign_gaps(spans, lines, emit, 10.0)
    assert asked == [(1.2, 5.0)]
    assert out[0] == spans[0] and out[2] == spans[2]
    (ca, sa) = out[1]
    assert abs(ca.start - (1.2 + 50 * FRAME)) < 2 * FRAME and abs(sa.vowel - (1.2 + 104 * FRAME)) < 2 * FRAME
    assert ca.score > 0.5


def _span(t, vowel=None):
    return SyllableSpan(t, t + 0.1, 0.9, t if vowel is None else vowel)


def test_aligned_track_snaps_to_sung_notes_and_marks_melisma():
    rules = load_rules(None)
    tempo = TempoMap(np.arange(0, 64) * 0.5, 192)     # 120 BPM: 1 batida = 0,5 s
    # Sílaba "ca-" ouvida em 1,02 s; a nota cantada começa em 1,00 s e muda de altura em 1,25 s ("+").
    sung = [SungNote(1.0, 1.25, 60.2), SungNote(1.25, 1.5, 62.1), SungNote(1.5, 1.9, 64.0)]
    spans = [[_span(0.98, 1.02), _span(1.48, 1.52)]]
    track = build_aligned_track(spans, sung, [["ca-", "sa"]], tempo, 192, rules.vocals)
    assert [(n.tick, n.pitch, n.lyric) for n in track.notes] == [(384, 60, "ca-"), (480, 62, "+"), (576, 64, "sa")]
    assert track.phrases == [(384, track.notes[-1].tick + track.notes[-1].length - 384)]


def test_aligned_track_never_stacks_or_overlaps():
    rules = load_rules(None)
    tempo = TempoMap(np.arange(0, 64) * 0.5, 192)
    sung = [SungNote(1.0, 3.0, 60.0)]
    # Três sílabas quase juntas (o alinhamento as pôs a 10 ms uma da outra) e uma linha seguinte.
    spans = [[_span(1.0), _span(1.01), _span(1.02)], [_span(1.03)]]
    track = build_aligned_track(spans, sung, [["Su-", "a", "a-"], ["vó"]], tempo, 192, rules.vocals)
    ticks = [n.tick for n in track.notes if n.lyric != "+"]
    assert len(set(ticks)) == 4 and ticks == sorted(ticks)
    assert all(a.tick + a.length <= b.tick for a, b in zip(track.notes, track.notes[1:]))
    assert all(a[0] + a[1] <= b[0] for a, b in zip(track.phrases, track.phrases[1:]))


def test_aligned_track_line_without_voice_is_spoken():
    rules = load_rules(None)
    tempo = TempoMap(np.arange(0, 64) * 0.5, 192)
    track = build_aligned_track([[_span(2.0), _span(2.3)]], [], [["o-", "lá"]], tempo, 192, rules.vocals)
    assert not track.notes and [t for t, _ in track.spoken] == [768, 888]   # 2,3 s = 883 ticks -> fusa mais perto
