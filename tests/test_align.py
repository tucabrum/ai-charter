"""Alinhamento forçado da letra (analysis/align.py) e a faixa de vocal feita a partir dele."""

import numpy as np

from ai_charter.analysis.align import (FRAME, SAMPLE_RATE, SyllableSpan, align_emissions, mute_silence,
                                      realign_gaps, romanize)
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


def test_mute_silence_keeps_letters_out_of_a_silent_intro():
    # A rede não reconheceu "oi" em lugar nenhum (emissões planas): sem a voz, ele cairia em qualquer
    # ponto; com o silêncio proibido, só pode cair onde a voz soa (a partir de 1 s).
    em = np.full((100, 28), -3.0, dtype=np.float32)
    y = np.zeros(2 * SAMPLE_RATE, dtype=np.float32)
    y[SAMPLE_RATE:] = 0.5 * np.sin(np.arange(SAMPLE_RATE) * 2 * np.pi * 220 / SAMPLE_RATE)
    (oi,), = align_emissions(mute_silence(em, y), [["oi"]])
    assert oi.start >= 1.0 - FRAME


def test_aligned_track_ignores_a_short_wobble_in_a_sustain():
    rules = load_rules(None)
    tempo = TempoMap(np.arange(0, 64) * 0.5, 192)
    # "lá" sustentado em Dó, com a voz caindo meio tom por 0,15 s e voltando; depois sobe de verdade.
    sung = [SungNote(1.0, 1.5, 60.0), SungNote(1.5, 1.65, 59.0), SungNote(1.65, 2.2, 60.0), SungNote(2.2, 2.8, 64.0)]
    track = build_aligned_track([[_span(1.0)]], sung, [["lá"]], tempo, 192, rules.vocals)
    assert [(n.pitch, n.lyric) for n in track.notes] == [(60, "lá"), (64, "+")]


def test_scale_is_estimated_from_the_voice_and_borderline_notes_snap_into_it():
    from ai_charter.charting.vocals import _snap, estimate_scale

    # Melodia em Dó maior (Dó Ré Mi Fá Sol Lá Si), Dó e Sol mais longos.
    sung = [SungNote(i, i + (2.0 if p in (60, 67) else 1.0), p) for i, p in enumerate([60, 62, 64, 65, 67, 69, 71, 60, 67])]
    scale = estimate_scale(sung)
    assert scale == {0, 2, 4, 5, 7, 9, 11}
    assert _snap(61.3, scale) == 62.0     # Dó# quase Ré: vai para o Ré
    assert _snap(61.0, scale) == 61.0     # Dó# bem afinado (nota de passagem): fica
    assert _snap(63.6, scale) == 63.6     # já arredonda para Mi, que está na escala: não mexe


def test_mute_silence_uses_the_whole_song_peak_for_a_quiet_stretch():
    # Um solo instrumental: a voz separada só tem ruído baixo (-50 dB do pico da música). Medido
    # contra o próprio trecho, o ruído pareceria voz; contra o pico da música, é silêncio.
    em = np.full((50, 28), -3.0, dtype=np.float32)
    noise = 0.003 * np.ones(SAMPLE_RATE, dtype=np.float32)
    assert (mute_silence(em, noise)[:, 1:] > -30).all()
    assert (mute_silence(em, noise, peak=1.0)[:, 1:] == -30).all()


def test_short_sound_in_a_long_silence_counts_as_silence():
    # 10 s de silêncio com 0,6 s de som no meio (um resto de guitarra no stem da voz, num solo)
    # e depois a voz de verdade, longa: a ponta vira silêncio, a voz não.
    y = np.zeros(14 * SAMPLE_RATE, dtype=np.float32)
    tone = lambda n: 0.5 * np.sin(np.arange(n) * 2 * np.pi * 220 / SAMPLE_RATE)
    y[5 * SAMPLE_RATE:int(5.6 * SAMPLE_RATE)] = tone(int(0.6 * SAMPLE_RATE))
    y[10 * SAMPLE_RATE:] = tone(4 * SAMPLE_RATE)
    em = mute_silence(np.full((700, 28), -3.0, dtype=np.float32), y)
    assert (em[255:275, 1:] == -30).all()         # a ponta (5,1-5,5 s)
    assert (em[510:690, 1:] == -3).all()          # a voz (10,2-13,8 s)


def test_lyric_words_and_matching_with_what_whisper_heard():
    from ai_charter.analysis.align import lyric_words, match_words

    lines = [["Hey", "the-", "re,"], ["lit-", "tle", "girl"]]
    assert lyric_words(lines) == [(0, "Hey"), (0, "there,"), (1, "little"), (1, "girl")]
    # O Whisper ouviu "a there little curl": "there" e "little" batem; "girl" x "curl" não.
    assert match_words(["Hey", "there,", "little", "girl"], ["a", "there", "Little", "curl"]) == [(1, 1), (2, 2)]


def test_line_windows_come_from_the_heard_words():
    from ai_charter.analysis.align import EDGE_WORD_SECONDS, WINDOW_MARGIN, line_windows

    lines = [["But", "give", "me", "mi-", "ne"], ["Hey", "the-", "re,"], ["lit-", "tle", "girl"]]
    heard = [("but", 74.4, 74.6), ("give", 74.6, 75.0), ("me", 75.0, 75.3), ("little", 98.0, 98.3),
             ("girl", 98.4, 98.9)]
    w = line_windows(lines, heard, 300.0)
    assert w[1] is None                                    # "Hey there" não foi ouvido
    assert w[0][0] == 74.4 - WINDOW_MARGIN
    assert abs(w[0][1] - (75.3 + WINDOW_MARGIN + EDGE_WORD_SECONDS)) < 1e-9   # "mine" ficou de fora
    assert w[2] == (98.0 - WINDOW_MARGIN, 98.9 + WINDOW_MARGIN)


def test_lines_without_window_fall_between_the_windows_of_their_neighbours():
    from ai_charter.analysis.align import align_in_windows

    # A rede "ouve" ruído com cara de vogal em 2 s (um solo com vazamento) e a linha de verdade em 6 s.
    em = _emissions([("a", 100), ("o", 300), ("i", 304), ("a", 320)], 400)
    lines = [["a"], ["oi"], ["a"]]
    out = align_in_windows(em, lines, [(1.5, 2.5), None, (6.2, 7.0)])
    assert 1.9 <= out[0][0].start <= 2.1
    assert 5.9 <= out[1][0].start <= 6.1                   # a do meio cai entre as janelas, onde está o "oi"
    assert 6.3 <= out[2][0].start <= 6.5
