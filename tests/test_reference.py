"""Partitura de referência (MIDI / Guitar Pro) guiando a transcrição (reference.py)."""

import shutil
from types import SimpleNamespace

import mido
import numpy as np
import pytest

import synth
from ai_charter.charting.common import Candidate
from ai_charter.config import load_rules
from ai_charter.reference import (
    Alignment, Guide, RefEvent, default_mapping, guess_instrument, parse_mapping, play_order, read_score,
)


def write_midi(path, parts, ticks_per_beat=480):
    """parts: [(nome, programa, canal, [(nota, batidas)])] -> arquivo MIDI, uma pista por parte."""
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    for name, program, channel, notes in parts:
        track = mido.MidiTrack([mido.MetaMessage("track_name", name=name),
                                mido.Message("program_change", program=program, channel=channel)])
        for pitch, beats in notes:
            track.append(mido.Message("note_on", note=pitch, velocity=90, channel=channel, time=0))
            track.append(mido.Message("note_off", note=pitch, velocity=0, channel=channel,
                                      time=int(beats * ticks_per_beat)))
        midi.tracks.append(track)
    midi.save(path)
    return path


def test_reads_midi_tracks_with_guesses(tmp_path):
    path = write_midi(tmp_path / "ref.mid", [
        ("Lead", 30, 0, [(64, 0.5), (67, 0.5)]),
        ("", 33, 1, [(40, 1.0)]),
        ("Kit", 0, 9, [(36, 0.25), (38, 0.25)]),
        ("Pad", 89, 2, [(60, 2.0)]),
    ])
    score = read_score(path)
    assert [(t.index, t.guess, len(t.notes)) for t in score.tracks] == [
        (1, "guitar", 2), (2, "bass", 1), (3, "drums", 2), (4, "keys", 1)]
    assert [(n.beat, n.length, n.pitch) for n in score.tracks[0].notes] == [(0.0, 0.5, 64), (0.5, 0.5, 67)]
    assert score.length_beats == 2.0
    assert default_mapping(score) == {1: "guitar", 2: "bass", 3: "drums", 4: "keys"}


def test_guesses():
    assert guess_instrument("Violão", None, False) == "guitar"
    assert guess_instrument("Baixo elétrico", 25, False) == "bass"        # o nome vence o programa
    assert guess_instrument("Track 3", None, False, [43, 38, 33, 28]) == "bass"
    assert guess_instrument("Track 2", None, False, [64, 59, 55, 50, 45, 40]) == "guitar"
    assert guess_instrument("Voz", None, False) == "vocals"
    assert guess_instrument("Sanfona", None, False) == "keys"
    assert guess_instrument("Track 9", 120, False) == "other"


def test_play_order_unrolls_repeats_and_endings():
    def h(open_=False, close=-1, alt=0):
        return SimpleNamespace(isRepeatOpen=open_, repeatClose=close, repeatAlternative=alt)

    # |: A | B (1ª vez) :| C (2ª vez) | D
    headers = [h(open_=True), h(close=1, alt=1), h(alt=2), h()]
    assert play_order(headers) == [0, 1, 0, 2, 3]
    # A |: B :| x3 C
    assert play_order([h(), h(open_=True, close=2), h()]) == [0, 1, 1, 1, 2]


def test_reads_guitar_pro_with_repeats_and_ties(tmp_path):
    gp = pytest.importorskip("guitarpro")
    from guitarpro import models as m

    song = gp.Song()
    track = song.tracks[0]
    track.name = "Guitarra Base"
    headers = song.measureHeaders
    while len(headers) < 3:
        header = m.MeasureHeader(number=len(headers) + 1, start=headers[-1].start + headers[-1].length)
        headers.append(header)
        track.measures.append(m.Measure(track, header))
    headers[0].isRepeatOpen = True
    headers[1].repeatClose = 1       # |: 1 | 2 :| 3
    for k, measure in enumerate(track.measures):
        voice = measure.voices[0]
        voice.beats.clear()
        for q in range(4):
            beat = m.Beat(voice, duration=m.Duration(value=4), status=m.BeatStatus.normal)
            kind = m.NoteType.tie if (k == 2 and q == 1) else m.NoteType.normal
            beat.notes.append(m.Note(beat, value=k * 2 + (q % 2), string=6, type=kind))
            voice.beats.append(beat)
    gp.write(song, str(tmp_path / "ref.gp5"))

    score = read_score(tmp_path / "ref.gp5")
    (t,) = score.tracks
    assert t.name == "Guitarra Base" and t.guess == "guitar"
    starts = [n.beat for n in t.notes]
    # compassos 1, 2, 1, 2, 3; no 3º a 2ª nota é ligada à 1ª (que fica com 2 batidas)
    assert starts == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19]
    assert [n.pitch for n in t.notes[:8]] == [40, 41, 40, 41, 42, 43, 42, 43]
    assert t.notes[16].length == 2.0
    assert score.length_beats == 20.0


def test_broken_guitar_pro_files_are_explained(tmp_path):
    (tmp_path / "vazio.gpx").write_bytes(b"BCFS")
    with pytest.raises(ValueError, match="score.gpif"):
        read_score(tmp_path / "vazio.gpx")
    (tmp_path / "outro.gp").write_bytes(b"isso nao e guitar pro")
    with pytest.raises(ValueError, match="não parece"):
        read_score(tmp_path / "outro.gp")
    (tmp_path / "x.pdf").write_bytes(b"%PDF")
    with pytest.raises(ValueError, match="não reconhecido"):
        read_score(tmp_path / "x.pdf")


def test_parse_mapping():
    assert parse_mapping("2=guitar, 4=bass") == {2: "guitar", 4: "bass"}
    with pytest.raises(ValueError):
        parse_mapping("guitar=2")
    with pytest.raises(ValueError):
        parse_mapping("1=flauta")


def make_guide(events, resolution=192):
    return Guide("guitar", events, resolution, load_rules().detection["guitar"])


def test_guide_corrects_pitch_and_chords():
    res = 192
    events = [RefEvent(0, 96, (52,)), RefEvent(96, 192, (45, 52)),            # nota; power chord (Lá5)
              RefEvent(192, 288, (48, 52, 55)), RefEvent(384, 1200, (45, 48, 52))]   # Dó; Lá menor longo
    cands = [Candidate(0, 1, pitch=50), Candidate(100, 1, pitch=47, voices=1), Candidate(190, 1, pitch=48, voices=2),
             Candidate(390, 1, pitch=45, voices=2), Candidate(600, 1, pitch=60)]   # a última não bate com nada
    lines = []
    out = make_guide(events, res)(cands, log=lines.append)
    assert [(c.pitch, c.voices) for c in out] == [(52, 1), (45, 2), (48, 2), (45, 3), (60, 1)]
    from ai_charter.analysis.chords import chord_name
    assert [chord_name(c.chord) for c in out[:4]] == ["E", "A5", "C", "Am"]
    assert "80%" in lines[0] and "3 corrigidas" in lines[0]   # o Dó já estava certo


def test_guide_is_ignored_when_reference_does_not_match():
    cands = [Candidate(t, 1, pitch=50) for t in range(0, 1920, 192)]
    lines = []
    out = make_guide([RefEvent(5000, 5100, (40,))])(cands, log=lines.append)
    assert all(c.pitch == 50 for c in out)
    assert "ignorada" in lines[0]


def test_alignment_maps_score_beats():
    a = Alignment(np.array([0.0, 1, 2, 3]), np.array([2.0, 3, 4, 6]), 0, 1.0, 0.1)
    assert a.to_audio_beat(1.5) == 3.5
    assert a.to_audio_beat(3.5) == 6.5       # depois do fim: segue no mesmo passo
    assert a.to_audio_beat(-2) is None and a.to_audio_beat(10) is None


def beat_this_grid(total: float) -> np.ndarray:
    """Mapa de tempo como o do Beat This! na música sintética: começa em 0 s e o compasso começa
    em 1,5 s, então a primeira batida (0 a 1,5 s) é esticada e a música começa no meio dela (1 s)."""
    return np.concatenate([[0.0], np.arange(synth.START + synth.BEAT, total + 1, synth.BEAT)])


@pytest.mark.parametrize("outro", [0, 8])
def test_alignment_on_beat_this_grid(tmp_path, outro):
    """Com `outro` batidas a mais no fim, a partitura fica mais longa que a gravação: antes o DTW do
    librosa invertia os papéis (procurava a gravação inteira num pedaço da partitura) e a referência
    deixava de conferir em tudo."""
    import soundfile as sf
    from ai_charter.reference import align

    mix, sr = sf.read(synth.render(tmp_path)["mix"])
    y = mix.mean(axis=1) if mix.ndim > 1 else mix
    beats = len(synth.MELODY) // 2
    score = read_score(write_midi(tmp_path / "ref.mid", [
        ("Guitarra", 29, 0, [(n + 1, 0.5) for n in synth.MELODY] + [(49, 0.5), (53, 0.5)] * outro),
        ("Baixo", 33, 1, [(n + 1, 1.0) for n in synth.BASS[:beats]] + [(37, 1.0)] * outro)]))
    a = align(score, score.tracks, y, sr, beat_this_grid(len(y) / sr))
    assert (a.factor, a.transpose) == (1.0, -1)
    # a partitura começa em 1 s: 2/3 da primeira batida do mapa; daí em diante, batida a batida
    assert a.to_audio_beat(0) == pytest.approx(2 / 3, abs=0.05)
    for beat in (1, 4, 10, 20):
        assert a.to_audio_beat(beat) == pytest.approx(beat, abs=0.05)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="precisa do ffmpeg")
def test_pipeline_guided_by_score(tmp_path):
    """A partitura começa sem o silêncio do áudio, está meio tom acima e tem 4 notas diferentes:
    o alinhamento acha o começo e a transposição, e o guia corrige exatamente as 4."""
    from ai_charter.pipeline import Options, run

    paths = synth.render(tmp_path)
    stems = tmp_path / "stems"
    stems.mkdir()
    for name in ("guitar", "bass", "drums"):
        shutil.move(paths[name], stems / f"{name}.wav")
    melody = [m + 1 for m in synth.MELODY]
    melody[8:12] = [m + 5 for m in melody[8:12]]
    beats = len(synth.MELODY) // 2
    bass = [b + 1 for b in synth.BASS[:beats]]
    ref = write_midi(tmp_path / "ref.mid", [
        ("Guitarra", 29, 0, [(n, 0.5) for n in melody]),
        ("Baixo", 33, 1, [(n, 1.0) for n in bass]),
        # o áudio tem bumbo e caixa alternados; nos golpes da caixa a partitura pede tom de chão
        ("Bateria", 0, 9, [(36 if b % 2 == 0 else 41, 1.0) for b in range(beats)]),
        ("Chimbal", 0, 9, [(42, 0.5)] * (2 * beats)),
    ])
    lines = []
    result = run(Options(input=paths["mix"], output_dir=tmp_path / "out", separator="folder", stems_dir=stems,
                         instruments=("guitar", "bass", "drums"), reference=ref), log=lines.append)
    text = "\n".join(lines)
    assert "transposta -1 semitons" in text
    guitar_line = next(line for line in lines if line.startswith("    guitar: referência confere com"))
    assert "100% das notas detectadas" in guitar_line
    # As 4 notas trocadas na partitura são corrigidas. Com o Beat This!, o mapa de tempo estica a primeira
    # batida (0 a 1,5 s; a música começa em 1 s) e a chart grava a 2ª nota (1,25 s) em 1,125 s, a meio
    # caminho entre duas notas da partitura: ela pode ganhar a altura da vizinha.
    assert int(guitar_line.split("; ")[1].split()[0]) in (4, 5)
    assert "bass: referência confere com 100% das notas detectadas; 0 corrigidas" in text
    drum_line = next(line for line in lines if line.startswith("    drums: referência confere com"))
    assert int(drum_line.split("com ")[1].split("%")[0]) >= 80
    # onde a pista tinha caixa (vermelho) nos golpes da partitura com tom de chão, agora é o verde sem prato
    expert = result.tracks["drums"].notes["expert"]
    assert not any(1 in n.lanes for n in expert)
    assert any(4 in n.lanes and 68 not in n.flags for n in expert)


# ---------------------------------------------------------------------------
# Guitar Pro 6+ (.gp / .gpx)
# ---------------------------------------------------------------------------
GPIF = """<?xml version="1.0" encoding="utf-8"?>
<GPIF>
  <Tracks>
    <Track id="0"><Name>Guitarra Solo</Name>
      <Staves><Staff><Properties><Property name="Tuning"><Pitches>40 45 50 55 59 64</Pitches></Property>
      </Properties></Staff></Staves></Track>
    <Track id="1"><Name>Drums</Name><InstrumentSet><Type>drumKit</Type><Elements>
      <Element><Articulations><Articulation><OutputMidiNumber>36</OutputMidiNumber></Articulation></Articulations></Element>
      <Element><Articulations><Articulation><OutputMidiNumber>38</OutputMidiNumber></Articulation>
        <Articulation><OutputMidiNumber>42</OutputMidiNumber></Articulation></Articulations></Element>
    </Elements></InstrumentSet></Track>
  </Tracks>
  <MasterBars>
    <MasterBar><Time>4/4</Time><Repeat start="true" end="true" count="2"/><Bars>0 1</Bars></MasterBar>
    <MasterBar><Time>2/4</Time><Bars>2 3</Bars></MasterBar>
  </MasterBars>
  <Bars>
    <Bar id="0"><Voices>0 -1 -1 -1</Voices></Bar>
    <Bar id="1"><Voices>1 -1 -1 -1</Voices></Bar>
    <Bar id="2"><Voices>2 -1 -1 -1</Voices></Bar>
    <Bar id="3"><Voices>3 -1 -1 -1</Voices></Bar>
  </Bars>
  <Voices>
    <Voice id="0"><Beats>0 1 2 3</Beats></Voice>
    <Voice id="1"><Beats>4 5</Beats></Voice>
    <Voice id="2"><Beats>6 7</Beats></Voice>
    <Voice id="3"><Beats>8</Beats></Voice>
  </Voices>
  <Beats>
    <Beat id="0"><Rhythm ref="1"/><Notes>0 1</Notes></Beat>
    <Beat id="1"><Rhythm ref="0"/><Notes>2</Notes></Beat>
    <Beat id="2"><GraceNotes>BeforeBeat</GraceNotes><Rhythm ref="0"/><Notes>3</Notes></Beat>
    <Beat id="3"><Rhythm ref="2"/><Notes>4</Notes></Beat>
    <Beat id="4"><Rhythm ref="3"/><Notes>10 11</Notes></Beat>
    <Beat id="5"><Rhythm ref="3"/><Notes>12</Notes></Beat>
    <Beat id="6"><Rhythm ref="0"/><Notes>5</Notes></Beat>
    <Beat id="7"><Rhythm ref="0"/><Notes>6</Notes></Beat>
    <Beat id="8"><Rhythm ref="3"/></Beat>
  </Beats>
  <Notes>
    <Note id="0"><Properties><Property name="String"><String>0</String></Property><Property name="Fret"><Fret>3</Fret></Property></Properties></Note>
    <Note id="1"><Properties><Property name="String"><String>1</String></Property><Property name="Fret"><Fret>5</Fret></Property></Properties></Note>
    <Note id="2"><Properties><Property name="String"><String>5</String></Property><Property name="Fret"><Fret>0</Fret></Property><Property name="Muted"><Enable/></Property></Properties></Note>
    <Note id="3"><Properties><Property name="String"><String>2</String></Property><Property name="Fret"><Fret>9</Fret></Property></Properties></Note>
    <Note id="4"><Properties><Property name="String"><String>2</String></Property><Property name="Fret"><Fret>2</Fret></Property></Properties></Note>
    <Note id="5"><Properties><Property name="String"><String>3</String></Property><Property name="Fret"><Fret>4</Fret></Property></Properties></Note>
    <Note id="6"><Tie origin="false" destination="true"/><Properties><Property name="String"><String>3</String></Property><Property name="Fret"><Fret>4</Fret></Property></Properties></Note>
    <Note id="10"><InstrumentArticulation>0</InstrumentArticulation></Note>
    <Note id="11"><InstrumentArticulation>2</InstrumentArticulation></Note>
    <Note id="12"><InstrumentArticulation>1</InstrumentArticulation></Note>
  </Notes>
  <Rhythms>
    <Rhythm id="0"><NoteValue>Quarter</NoteValue></Rhythm>
    <Rhythm id="1"><NoteValue>Quarter</NoteValue><AugmentationDot count="1"/></Rhythm>
    <Rhythm id="2"><NoteValue>Quarter</NoteValue><PrimaryTuplet num="3" den="2"/></Rhythm>
    <Rhythm id="3"><NoteValue>Half</NoteValue></Rhythm>
  </Rhythms>
</GPIF>
"""


def check_gpif_tracks(score):
    guitar, drums = score.tracks
    assert (guitar.name, guitar.guess, drums.guess, drums.drums) == ("Guitarra Solo", "guitar", "drums", True)
    # compasso 1 (4/4) tocado duas vezes, depois o 2 (2/4) com uma nota ligada; a nota abafada e a de
    # enfeite ficam de fora; a semínima pontuada vale 1,5 e a tercina, 2/3
    notes = [(n.beat, round(n.length, 3), n.pitch) for n in guitar.notes]
    assert notes == [(0, 1.5, 43), (0, 1.5, 50), (2.5, 0.667, 52),
                     (4, 1.5, 43), (4, 1.5, 50), (6.5, 0.667, 52), (8, 2.0, 59)]
    assert [(n.beat, n.pitch) for n in drums.notes] == [(0, 36), (0, 42), (2, 38), (4, 36), (4, 42), (6, 38)]
    assert score.length_beats == 10


def test_reads_guitar_pro_7_gp_file(tmp_path):
    import zipfile

    path = tmp_path / "musica.gp"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("VERSION", "7.0")
        archive.writestr("Content/score.gpif", GPIF)
    check_gpif_tracks(read_score(path))


class BitWriter:
    def __init__(self):
        self.bits = []

    def write(self, value, count, reversed_=False):
        order = range(count) if reversed_ else range(count - 1, -1, -1)
        self.bits += [(value >> i) & 1 for i in order]

    def bytes(self):
        bits = self.bits + [0] * (-len(self.bits) % 8)
        return bytes(int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8))


def bcfz(data: bytes) -> bytes:
    """Compacta como o Guitar Pro 6 (só com trechos literais, que o formato também aceita)."""
    w = BitWriter()
    for i in range(0, len(data), 3):
        chunk = data[i:i + 3]
        w.write(0, 1)
        w.write(len(chunk), 2, reversed_=True)
        for byte in chunk:
            w.write(byte, 8)
    return b"BCFZ" + len(data).to_bytes(4, "little") + w.bytes()


def bcfs(files: dict) -> bytes:
    """Sistema de arquivos do .gpx: setor 0 vazio, uma entrada por arquivo e os dados em setores de 4 KB."""
    sector = 0x1000
    entries = list(files.items())
    sectors = [bytes(sector)] + [b""] * len(entries)   # setor 0 + uma entrada por arquivo
    for k, (name, content) in enumerate(entries):
        chunks = [content[i:i + sector].ljust(sector, b"\0") for i in range(0, len(content), sector)]
        pointers = list(range(len(sectors), len(sectors) + len(chunks)))
        entry = bytearray(sector)
        entry[0:4] = (2).to_bytes(4, "little")
        entry[4:4 + len(name)] = name.encode()
        entry[0x8C:0x90] = len(content).to_bytes(4, "little")
        for j, p in enumerate(pointers):
            entry[0x94 + 4 * j:0x98 + 4 * j] = p.to_bytes(4, "little")
        sectors[1 + k] = bytes(entry)
        sectors += chunks
    return b"BCFS" + b"".join(sectors)


def test_reads_guitar_pro_6_gpx_file(tmp_path):
    container = bcfs({"misc.xml": b"<x/>", "score.gpif": GPIF.encode()})
    (tmp_path / "plain.gpx").write_bytes(container)
    check_gpif_tracks(read_score(tmp_path / "plain.gpx"))
    (tmp_path / "packed.gpx").write_bytes(bcfz(container))
    check_gpif_tracks(read_score(tmp_path / "packed.gpx"))


def test_bcfz_back_references():
    from ai_charter.reference import _bcfz_decompress

    w = BitWriter()
    w.write(0, 1)
    w.write(3, 2, True)
    for byte in b"abc":
        w.write(byte, 8)
    w.write(1, 1)   # cópia de 5 bytes a partir de 3 atrás: só os 3 que existem ("abc")
    w.write(3, 4)
    w.write(3, 3, True)
    w.write(5, 3, True)
    w.write(1, 1)   # repete o último byte
    w.write(3, 4)
    w.write(1, 3, True)
    w.write(1, 3, True)
    data = (7).to_bytes(4, "little") + w.bytes()
    assert _bcfz_decompress(data) == b"abcabcc"


def test_new_guitar_pro_without_score_is_explained(tmp_path):
    import zipfile

    with zipfile.ZipFile(tmp_path / "x.gp", "w") as archive:
        archive.writestr("VERSION", "7.0")
    with pytest.raises(ValueError, match="score.gpif"):
        read_score(tmp_path / "x.gp")


# ---------------------------------------------------------------------------
# Bateria
# ---------------------------------------------------------------------------
def test_drum_guide_fixes_pieces_but_keeps_rhythm_kick_and_hands():
    from ai_charter.charting.common import ChartNote, TrackChart
    from ai_charter.reference import DrumEvent, DrumGuide

    notes = [ChartNote(0, (0, 1)),               # bumbo + caixa; a partitura diz bumbo + tom agudo
             ChartNote(96, (2,), flags=(66,)),   # chimbal; a partitura diz condução
             ChartNote(192, (0, 1)),             # confere: nada muda
             ChartNote(288, (1,)),               # uma mão; a partitura tem ataque + caixa: fica o ataque
             ChartNote(400, (4,), flags=(68,))]  # nenhum golpe da partitura perto: nada muda
    track = TrackChart(notes={"expert": notes, "easy": [ChartNote(0, (1,))]})
    events = [DrumEvent(0, frozenset({"kick", "tom_high"})), DrumEvent(96, frozenset({"ride"})),
              DrumEvent(192, frozenset({"kick", "snare"})), DrumEvent(290, frozenset({"crash", "snare"}))]
    lines = []
    DrumGuide(events, 192)(track, log=lines.append)
    assert [(n.lanes, n.flags) for n in track.notes["expert"]] == [
        ((0, 2), ()), ((3,), (67,)), ((0, 1), ()), ((4,), (68,)), ((4,), (68,))]
    assert (track.notes["easy"][0].lanes, track.notes["easy"][0].flags) == ((2,), ())
    assert "80% das batidas; 3 corrigidas no Expert" in lines[0]


def test_drum_guide_ignored_when_it_does_not_match():
    from ai_charter.charting.common import ChartNote, TrackChart
    from ai_charter.reference import DrumEvent, DrumGuide

    track = TrackChart(notes={"expert": [ChartNote(t, (1,)) for t in range(0, 1920, 192)]})
    lines = []
    DrumGuide([DrumEvent(5000, frozenset({"tom_low"}))], 192)(track, log=lines.append)
    assert all(n.lanes == (1,) for n in track.notes["expert"]) and "ignorada" in lines[0]
