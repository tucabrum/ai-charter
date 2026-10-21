"""Teste completo da 2ª música: violão x guitarra solo, e vocais com letra (formato .mid)."""

import shutil

import pytest

import song2
from ai_charter.pipeline import Options, run
from chart_parser import midi_notes, parse_midi

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="precisa do ffmpeg")

RES = 192
MEASURE = RES * 4
FIRST_BAR = int(round(song2.START / song2.BEAT)) // 4  # compasso da chart onde a música começa


@pytest.fixture(scope="module")
def song(tmp_path_factory):
    folder = tmp_path_factory.mktemp("song2")
    paths = song2.render(folder)
    stems = folder / "stems"
    stems.mkdir()
    for name in ("violao", "guitarra", "bass", "drums", "vocals"):
        shutil.move(paths[name], stems / f"{name}.wav")
    result = run(Options(input=paths["mix"], output_dir=folder / "out", separator="folder",
                         stems_dir=stems, lyrics=paths["lyrics"], title="Estrada", artist="Teste"),
                 log=lambda *_: None)
    return result, parse_midi(result.song_dir / "notes.mid")


def bars_of(section: str) -> range:
    """Compassos da chart (0 = primeiro) ocupados por uma seção da música."""
    start = FIRST_BAR
    for name, bars, *_ in song2.SECTIONS:
        if name == section:
            return range(start, start + bars)
        start += bars
    raise KeyError(section)


def test_output_has_midi_and_all_audio(song):
    result, midi = song
    files = {p.name for p in result.song_dir.iterdir()}
    assert {"notes.mid", "song.ini", "guitar.ogg", "bass.ogg", "drums.ogg", "vocals.ogg"} <= files
    assert {"PART GUITAR", "PART BASS", "PART DRUMS", "PART VOCALS", "EVENTS"} <= set(midi)
    assert "diff_vocals = -1" not in (result.song_dir / "song.ini").read_text()


def test_guitar_follows_violao_riff_and_solo(song):
    result, _ = song
    chosen = result.guitar_sources
    for section, expected in (("intro", "violao"), ("verso1", "violao"), ("riff", "guitarra"),
                              ("solo", "guitarra"), ("verso2", "violao")):
        for bar in bars_of(section):
            assert chosen[bar] == expected, (section, bar, chosen[bar])


def test_guitar_does_not_bounce_between_tracks(song):
    result, _ = song
    chosen = result.guitar_sources
    switches = sum(1 for a, b in zip(chosen, chosen[1:]) if a != b)
    assert switches == 2   # violão -> guitarra (riff) -> violão (verso 2)


def test_solo_chart_follows_the_lead_melody(song):
    _, midi = song
    expert = midi_notes(midi["PART GUITAR"], 96, 100)
    for i, bar in enumerate(bars_of("solo")):
        expected = {int(bar * MEASURE + beat * RES) for beat, _, _ in song2.SOLO[i]}
        got = {t for t in expert if bar * MEASURE <= t < (bar + 1) * MEASURE}
        assert got == expected, (i, sorted(got), sorted(expected))   # inclui o compasso de pausa (vazio)


def test_vocals_have_lyrics_split_in_syllables(song):
    _, midi = song
    events = midi["PART VOCALS"]
    lyrics = [value for _, kind, value in events if kind == "lyric"]
    expected = []
    for line in song2.SYLLABLES:
        expected += [s + "-" for s in line[:-1]]   # sílaba no meio da palavra termina com "-"
        expected.append(line[-1])
    # Sílabas que são palavras inteiras (sem "-"): compara sem o hífen.
    assert [s.rstrip("-") for s in lyrics] == [s.rstrip("-") for s in expected]
    assert lyrics[:7] == ["No", "ca-", "mi-", "nho", "da", "ma-", "nhã"]


def test_vocal_notes_have_right_timing_and_pitch(song):
    _, midi = song
    notes = [(t, v) for t, kind, v in midi["PART VOCALS"] if kind == "note" and 36 <= v[0] <= 84]
    expected = []
    for section, first_line in (("verso1", 0), ("verso2", 4)):
        start = bars_of(section)[0] * MEASURE
        for line in range(4):
            tick = start + line * 2 * MEASURE
            syllables = song2.SYLLABLES[first_line + line]
            for midi_note, eighths in zip(song2.MELODIES[first_line + line], song2.RHYTHMS[len(syllables)]):
                expected.append((tick, midi_note))
                tick += eighths * RES // 2
    assert [(t, v[0]) for t, v in notes] == expected


def test_vocal_phrases_match_lyric_lines(song):
    _, midi = song
    phrases = [(t, v[1]) for t, kind, v in midi["PART VOCALS"] if kind == "note" and v[0] == 105]
    assert len(phrases) == len(song2.LYRICS)
    for section, first in (("verso1", 0), ("verso2", 4)):
        start = bars_of(section)[0] * MEASURE
        for line in range(4):
            assert phrases[first + line][0] == start + line * 2 * MEASURE


def test_violao_strums_are_all_charted_as_consistent_chords(song):
    """Toda palhetada do violão vira um acorde, e cada acorde tem (quase) sempre o mesmo formato."""
    from collections import Counter, defaultdict

    _, midi = song
    expert = midi_notes(midi["PART GUITAR"], 96, 100)
    shapes: dict[str, Counter] = defaultdict(Counter)
    total = 0
    for section in ("intro", "verso1", "verso2"):
        for i, bar in enumerate(bars_of(section)):
            for beat in (0, 1, 1.5, 2.5, 3, 3.5):           # ritmo do violão (baixo e cima)
                tick = int(bar * MEASURE + beat * RES)
                assert tick in expert, (section, i, beat)    # nenhuma palhetada some
                lanes = tuple(sorted(expert[tick]))
                assert len(lanes) >= 2, (section, i, beat)    # sempre acorde, nunca nota simples
                shapes[song2.PROGRESSION[i % 4]][lanes] += 1
                total += 1
    # Cada acorde usa o mesmo formato em pelo menos 90% das palhetadas.
    for chord, counts in shapes.items():
        assert counts.most_common(1)[0][1] >= 0.9 * sum(counts.values()), (chord, counts)
    # Acordes vizinhos na progressão (C G Am F) têm formatos diferentes.
    main = {chord: counts.most_common(1)[0][0] for chord, counts in shapes.items()}
    for a, b in zip(song2.PROGRESSION, song2.PROGRESSION[1:] + song2.PROGRESSION[:1]):
        assert main[a] != main[b], (a, b, main)
