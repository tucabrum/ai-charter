"""Partitura de referência (MIDI ou Guitar Pro) para GUIAR a transcrição.

Uso:
  ai-charter musica.flac --reference musica.gp5                         # pistas escolhidas por palpite
  ai-charter musica.flac --reference musica.mid --reference-tracks 2=guitar,4=bass
  ai-charter-ref musica.gp5                                             # lista as pistas e os palpites

As notas da chart continuam vindo do áudio (o que a gravação realmente toca). A
partitura só corrige cada nota detectada que cai perto de uma nota dela: a ALTURA
(e, com ela, a cor) e o ACORDE (quantas notas, power chord ou não). Na bateria, corrige
a PEÇA de cada golpe já charteado (caixa ou tom, chimbal, condução ou ataque, qual tom).
Notas que a partitura tem e o áudio não, ou vice-versa, ficam como a transcrição decidiu.

Formatos: MIDI (.mid, .kar), Guitar Pro 3-5 (.gp3/.gp4/.gp5, com o pyguitarpro) e Guitar Pro 6
ou mais novo (.gpx e .gp, lidos aqui mesmo: são um pacote com a partitura em XML, o score.gpif).

Passos:
  1. Leitura: as pistas do arquivo, cada nota em batidas (semínimas) desde o início
     (repetições do Guitar Pro já desdobradas). Cada pista ganha um palpite de
     instrumento (nome, programa MIDI, canal 10 = bateria, número de cordas).
  2. Alinhamento com a gravação: compara, batida a batida, as notas que soam na
     partitura com o "chroma" da mixagem (quanto soa de cada uma das 12 notas), por
     DTW. Aguenta introdução diferente, trechos a mais ou a menos e andamento que
     oscila. Testa também a partitura transposta (afinação meio tom abaixo,
     capotraste) e com o dobro ou a metade das batidas do detector.
  3. Guia: cada nota detectada recebe a altura e o formato de acorde da nota da
     partitura mais próxima (até uma semicolcheia de distância). Se pouca coisa
     confere, a referência é ignorada e o log avisa. Com várias pistas no mesmo
     instrumento (solo, base...), cada compasso usa a que mais confere com o áudio.
     A fidelidade (--reference-fidelity) diz se a partitura também dá o RITMO: só
     corrige (corrigir), substitui os compassos em que confere (completar) ou vale
     onde tiver notas (seguir, o padrão). A bateria é corrigida depois de
     pronta (a transcrição dela não muda): cada nota da pista fica com as peças que a
     partitura toca ali (mapa General MIDI), mantendo o bumbo e o número de mãos.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np

MIDI_EXTENSIONS = (".mid", ".midi", ".kar")
GP_EXTENSIONS = (".gp3", ".gp4", ".gp5")
NEW_GP_EXTENSIONS = (".gpx", ".gp", ".gp6", ".gp7", ".gp8")
INSTRUMENTS = ("guitar", "bass", "keys", "drums", "vocals")
PITCHED = ("guitar", "bass", "keys")
GUIDED = (*PITCHED, "drums")              # instrumentos que a referência guia (por enquanto)
DRUM_MATCH_BEATS = 0.125                  # bateria: golpes mais densos, tolerância menor (fusa)
MATCH_BEATS = 0.25                        # distância máxima entre a nota detectada e a da partitura
MIN_MATCHED = 0.3                         # abaixo disso a referência não confere: é ignorada
FIDELITY = ("corrigir", "completar", "seguir")   # quanto a partitura manda nas notas (ver Guide)
DEFAULT_FIDELITY = "seguir"
BAR_MATCH = 0.5                           # completar: parte das notas detectadas do compasso que confere
CHORD_ONSET_BEATS = 1 / 32                # notas que começam até aqui juntas são um acorde
WARP_COST = 0.2                           # alinhamento: custo de cada passo fora da diagonal
UNMATCHED_COST = 0.5                      # alinhamento: passo da partitura depois do fim da gravação

NAME_HINTS = (   # (instrumento, padrão no nome da pista), na ordem em que são testados
    ("drums", r"drum|bateria|percuss|kit"),
    ("bass", r"bass|baixo|contrabaixo"),
    ("vocals", r"vocal|voice|voz|vox|cant|melod|lead vocal|singer"),
    ("keys", r"piano|key|teclad|organ|[óo]rg[ãa]o|synth|sintet|sanfona|acorde[ãa]o|accordion|pad"),
    ("guitar", r"guit|gtr|viol[ãa]o|violao|lead|rhythm|base|solo|dist|clean|acoust|nylon|steel"),
)


@dataclass
class RefNote:
    beat: float      # início, em semínimas desde o começo da partitura
    length: float    # duração em semínimas
    pitch: int       # nota MIDI (na bateria: a peça no mapa General MIDI)


@dataclass
class RefTrack:
    index: int
    name: str
    guess: str                 # palpite: guitar, bass, keys, drums, vocals ou other
    notes: list[RefNote] = field(default_factory=list)
    program: int | None = None
    drums: bool = False

    def summary(self) -> dict:
        return {"index": self.index, "name": self.name, "guess": self.guess, "notes": len(self.notes),
                "program": self.program, "drums": self.drums}


@dataclass
class Score:
    path: Path
    tracks: list[RefTrack]

    @property
    def length_beats(self) -> float:
        return max((n.beat + n.length for t in self.tracks for n in t.notes), default=0.0)


def guess_instrument(name: str, program: int | None, drums: bool, strings: list[int] | None = None) -> str:
    """Palpite do instrumento de uma pista."""
    if drums:
        return "drums"
    lower = name.lower()
    for instrument, pattern in NAME_HINTS:
        if re.search(pattern, lower):
            return instrument
    if strings:   # Guitar Pro: baixo tem 4-6 cordas graves (a mais grave abaixo do Mi da guitarra)
        return "bass" if min(strings) < 40 and len(strings) <= 6 and max(strings) <= 55 else "guitar"
    if program is not None:
        if 24 <= program <= 31:
            return "guitar"
        if 32 <= program <= 39:
            return "bass"
        if program <= 23 or 80 <= program <= 103:
            return "keys"
        if 52 <= program <= 54:
            return "vocals"
    return "other"


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------
def read_score(path: Path) -> Score:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in MIDI_EXTENSIONS:
        return Score(path, _read_midi(path))
    if suffix in GP_EXTENSIONS:
        return Score(path, _read_guitar_pro(path))
    if suffix in NEW_GP_EXTENSIONS:
        return Score(path, _read_new_guitar_pro(path))
    raise ValueError(f"formato de partitura não reconhecido: {path.name} (use .mid, .gp3, .gp4, .gp5, .gpx ou .gp)")


def _read_midi(path: Path) -> list[RefTrack]:
    import mido

    midi = mido.MidiFile(path, clip=True)
    res = midi.ticks_per_beat
    tracks: list[RefTrack] = []
    for track in midi.tracks:
        name, program = track.name.strip(), {}
        by_channel: dict[int, list[RefNote]] = {}
        open_notes: dict[tuple[int, int], int] = {}
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "program_change":
                program.setdefault(msg.channel, msg.program)
            elif msg.type == "note_on" and msg.velocity > 0:
                open_notes.setdefault((msg.channel, msg.note), tick)
            elif msg.type in ("note_on", "note_off"):
                start = open_notes.pop((msg.channel, msg.note), None)
                if start is not None:
                    by_channel.setdefault(msg.channel, []).append(
                        RefNote(start / res, max(tick - start, 1) / res, msg.note))
        for channel, notes in sorted(by_channel.items()):
            label = name or f"Pista {len(tracks) + 1}"
            if len(by_channel) > 1:
                label += f" (canal {channel + 1})"
            drums = channel == 9
            prog = program.get(channel)
            tracks.append(RefTrack(len(tracks) + 1, label, guess_instrument(label, prog, drums),
                                   sorted(notes, key=lambda n: (n.beat, n.pitch)), prog, drums))
    return tracks


def play_order(headers) -> list[int]:
    """Ordem em que os compassos tocam, com as repetições e as casas de 1ª/2ª vez desdobradas."""
    order, i, start, pass_no = [], 0, 0, 1
    while i < len(headers) and len(order) < 20 * len(headers):
        h = headers[i]
        if h.isRepeatOpen and i != start:   # começa um novo trecho com repetição
            start, pass_no = i, 1
        if h.repeatAlternative and not (h.repeatAlternative >> (pass_no - 1)) & 1:
            i += 1   # casa de outra vez: pula nesta passada
            continue
        order.append(i)
        if h.repeatClose > 0 and pass_no <= h.repeatClose:
            pass_no += 1
            i = start
            continue
        if h.repeatClose > 0:   # acabou a repetição
            start, pass_no = i + 1, 1
        i += 1
    return order


def _read_guitar_pro(path: Path) -> list[RefTrack]:
    try:
        import guitarpro
    except ImportError as exc:
        raise RuntimeError('para ler Guitar Pro instale: pip install "ai-charter[refs]"') from exc
    from guitarpro.models import NoteType

    song = guitarpro.parse(str(path))
    quarter = 960   # Duration.quarterTime do pyguitarpro
    headers = song.measureHeaders
    order = play_order(headers)
    offsets, now = [], 0   # início (em ticks) de cada compasso tocado, na ordem desdobrada
    for i in order:
        offsets.append(now)
        now += headers[i].length
    tracks = []
    for t in song.tracks:
        strings = [s.value for s in t.strings]
        drums = bool(t.isPercussionTrack)
        notes: list[RefNote] = []
        last_on_string: dict[int, RefNote] = {}
        for played, i in zip(offsets, order):
            measure = t.measures[i]
            for voice in measure.voices:
                for beat in voice.beats:
                    at = played + beat.start - measure.header.start
                    length = beat.duration.time
                    for note in beat.notes:
                        if note.type == NoteType.tie and note.string in last_on_string:
                            last = last_on_string[note.string]   # ligadura: a nota anterior continua
                            last.length = (at + length) / quarter - last.beat
                            continue
                        if note.type in (NoteType.rest, NoteType.dead):
                            continue
                        pitch = note.value if drums else note.realValue
                        ref = RefNote(at / quarter, length / quarter, pitch)
                        notes.append(ref)
                        last_on_string[note.string] = ref
        program = t.channel.instrument if t.channel else None
        tracks.append(RefTrack(len(tracks) + 1, t.name.strip() or f"Pista {len(tracks) + 1}",
                               guess_instrument(t.name, program, drums, None if drums else strings),
                               sorted(notes, key=lambda n: (n.beat, n.pitch)), program, drums))
    return tracks


# ---------------------------------------------------------------------------
# Guitar Pro 6 ou mais novo: .gp (zip) e .gpx (pacote BCFZ/BCFS) com o score.gpif (XML)
# ---------------------------------------------------------------------------
class _Bits:
    """Leitor bit a bit (do bit mais alto para o mais baixo de cada byte)."""

    def __init__(self, data: bytes, start: int = 0):
        self.data, self.pos = data, start * 8

    def bit(self) -> int:
        byte = self.data[self.pos >> 3]   # IndexError no fim dos dados
        value = (byte >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return value

    def bits(self, count: int) -> int:
        value = 0
        for i in range(count - 1, -1, -1):
            value |= self.bit() << i
        return value

    def bits_reversed(self, count: int) -> int:
        value = 0
        for i in range(count):
            value |= self.bit() << i
        return value


def _bcfz_decompress(data: bytes) -> bytes:
    """Descompacta um BCFZ (o conteúdo depois da assinatura): trechos literais e cópias de trás."""
    expected = int.from_bytes(data[:4], "little")
    bits, out = _Bits(data, 4), bytearray()
    try:
        while len(out) < expected:
            if bits.bit():   # cópia de um trecho já descompactado
                word = bits.bits(4)
                offset, size = bits.bits_reversed(word), bits.bits_reversed(word)
                start = len(out) - offset
                for i in range(min(offset, size)):
                    out.append(out[start + i])
            else:            # até 3 bytes literais
                for _ in range(bits.bits_reversed(2)):
                    out.append(bits.bits(8))
    except IndexError:
        pass
    return bytes(out)


def gpx_files(data: bytes) -> dict[str, bytes]:
    """Os arquivos de um pacote .gpx (Guitar Pro 6): um sistema de arquivos em setores de 4 KB."""
    if data[:4] == b"BCFZ":
        data = _bcfz_decompress(data[4:])
    if data[:4] != b"BCFS":
        raise ValueError("arquivo .gpx inválido (pacote BCFS não encontrado)")
    data = data[4:]
    sector, files, used = 0x1000, {}, set()

    def int32(at: int) -> int:
        return int.from_bytes(data[at:at + 4], "little") if at + 4 <= len(data) else 0

    for offset in range(sector, len(data) - 3, sector):   # o setor 0 é o cabeçalho
        if offset // sector in used or int32(offset) != 2:   # só entradas de arquivo (tipo 2)
            continue
        name = data[offset + 4:offset + 4 + 127].split(b"\0")[0].decode("latin-1")
        size, pointers, chunks, k = int32(offset + 0x8C), offset + 0x94, [], 0
        while (block := int32(pointers + 4 * k)) != 0 and k < 1000:
            k += 1
            used.add(block)   # setor de dados: não é lido como entrada
            chunks.append(data[block * sector:(block + 1) * sector])
        files[name] = b"".join(chunks)[:size]
    return files


def _read_new_guitar_pro(path: Path) -> list[RefTrack]:
    data = Path(path).read_bytes()
    if data[:2] == b"PK":   # .gp (Guitar Pro 7+): zip com Content/score.gpif
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            name = next((n for n in archive.namelist() if n.lower().endswith("score.gpif")), None)
            if name is None:
                raise ValueError(f"{Path(path).name}: o arquivo não tem a partitura (score.gpif)")
            return parse_gpif(archive.read(name))
    if data[:4] in (b"BCFZ", b"BCFS"):   # .gpx (Guitar Pro 6)
        files = gpx_files(data)
        name = next((n for n in files if n.lower().endswith("score.gpif")), None)
        if name is None:
            raise ValueError(f"{Path(path).name}: o pacote não tem a partitura (score.gpif)")
        return parse_gpif(files[name])
    raise ValueError(f"{Path(path).name}: não parece um arquivo do Guitar Pro 6 ou mais novo")


NOTE_VALUES = {"Whole": 4.0, "Half": 2.0, "Quarter": 1.0, "Eighth": 0.5, "16th": 0.25, "32nd": 0.125,
               "64th": 0.0625, "128th": 0.03125, "256th": 0.015625}
STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ACCIDENTALS = {"": 0, "#": 1, "b": -1, "##": 2, "bb": -2, "x": 2}


def _ids(text: str | None) -> list[int]:
    return [int(x) for x in (text or "").split()]


def _rhythm_beats(rhythm) -> float:
    beats = NOTE_VALUES.get((rhythm.findtext("NoteValue") or "Quarter").strip(), 1.0)
    dot = rhythm.find("AugmentationDot")
    if dot is not None:
        beats *= 2 - 0.5 ** int(dot.get("count", "1"))
    tuplet = rhythm.find("PrimaryTuplet")
    if tuplet is not None and int(tuplet.get("num", "1")) > 0:
        beats *= int(tuplet.get("den", "1")) / int(tuplet.get("num", "1"))
    return beats


def _property(note, name: str, child: str) -> str | None:
    return note.findtext(f"Properties/Property[@name='{name}']/{child}")


def _note_pitch(note, tuning: list[int], drums: bool, articulations: list[int]) -> int | None:
    """Nota MIDI de uma nota do GPIF (na bateria, a peça no mapa General MIDI)."""
    if drums:
        articulation = note.findtext("InstrumentArticulation")
        if articulation is not None and articulation.strip().isdigit() and int(articulation) < len(articulations):
            return articulations[int(articulation)]
    midi = _property(note, "Midi", "Number")
    if midi is not None and midi.strip().lstrip("-").isdigit():
        return int(midi)
    string, fret = _property(note, "String", "String"), _property(note, "Fret", "Fret")
    if string is not None and fret is not None and tuning and int(string) < len(tuning):
        return tuning[int(string)] + int(fret)   # cordas da mais grave (0) para a mais aguda
    for kind in ("ConcertPitch", "TransposedPitch"):
        step = _property(note, kind, "Pitch/Step")
        if step:
            accidental = (_property(note, kind, "Pitch/Accidental") or "").strip()
            octave = int(_property(note, kind, "Pitch/Octave") or 4)
            return 12 * octave + STEPS.get(step.strip().upper(), 0) + ACCIDENTALS.get(accidental, 0)
    return None


def parse_gpif(xml: bytes) -> list[RefTrack]:
    """As pistas de um score.gpif (Guitar Pro 6+), com repetições e casas de 1ª/2ª vez desdobradas."""
    root = ElementTree.fromstring(xml)
    by_id = {kind: {int(el.get("id")): el for el in root.findall(f"./{kind}s/{kind}")}
             for kind in ("Bar", "Voice", "Beat", "Note", "Rhythm")}
    master_bars = root.findall("./MasterBars/MasterBar")
    headers = []
    for mb in master_bars:
        num, den = (mb.findtext("Time") or "4/4").split("/")
        repeat = mb.find("Repeat")
        endings = _ids(mb.findtext("AlternateEndings"))
        headers.append(SimpleNamespace(
            isRepeatOpen=repeat is not None and repeat.get("start") == "true",
            repeatClose=int(repeat.get("count", "2")) - 1 if repeat is not None and repeat.get("end") == "true" else -1,
            repeatAlternative=sum(1 << (e - 1) for e in endings if e > 0),
            length=int(num) * 4 / int(den)))
    order = play_order(headers)
    offsets, now = [], 0.0
    for i in order:
        offsets.append(now)
        now += headers[i].length

    tracks = []
    for ti, track in enumerate(root.findall("./Tracks/Track")):
        name = (track.findtext("Name") or "").strip() or f"Pista {ti + 1}"
        tuning = _ids(track.findtext(".//Property[@name='Tuning']/Pitches"))
        program_text = track.findtext("GeneralMidi/Program") or track.findtext(".//Sounds/Sound/MIDI/Program")
        program = int(program_text) if program_text and program_text.strip().isdigit() else None
        channel = track.findtext("GeneralMidi/PrimaryChannel") or track.findtext(".//MidiConnection/PrimaryChannel")
        kit = (track.findtext("InstrumentSet/Type") or "").strip().lower()
        instrument = track.find("Instrument")
        drums = (kit == "drumkit" or (channel or "").strip() == "9"
                 or (instrument is not None and "drum" in (instrument.get("ref") or "").lower()))
        articulations = [int(a.findtext("OutputMidiNumber") or 0)
                         for a in track.findall("InstrumentSet/Elements/Element/Articulations/Articulation")]
        notes: list[RefNote] = []
        last_on_string: dict[str, RefNote] = {}
        for start, index in zip(offsets, order):
            bar_ids = _ids(master_bars[index].findtext("Bars"))
            bar = by_id["Bar"].get(bar_ids[ti]) if ti < len(bar_ids) else None
            if bar is None:
                continue
            for voice_id in _ids(bar.findtext("Voices")):
                voice = by_id["Voice"].get(voice_id)
                if voice is None:
                    continue
                at = start
                for beat_id in _ids(voice.findtext("Beats")):
                    beat = by_id["Beat"].get(beat_id)
                    if beat is None or beat.find("GraceNotes") is not None:   # nota de enfeite não ocupa tempo
                        continue
                    rhythm = beat.find("Rhythm")
                    length = _rhythm_beats(by_id["Rhythm"][int(rhythm.get("ref"))]) if rhythm is not None else 1.0
                    for note_id in _ids(beat.findtext("Notes")):
                        note = by_id["Note"].get(note_id)
                        if note is None:
                            continue
                        key = _property(note, "String", "String") or str(note_id)
                        tie = note.find("Tie")
                        if tie is not None and tie.get("destination") == "true" and key in last_on_string:
                            last = last_on_string[key]   # ligadura: a nota anterior continua
                            last.length = at + length - last.beat
                            continue
                        if note.find("Properties/Property[@name='Muted']") is not None:
                            continue   # nota abafada (x): não tem altura
                        pitch = _note_pitch(note, tuning, drums, articulations)
                        if pitch is None:
                            continue
                        ref = RefNote(at, length, pitch)
                        notes.append(ref)
                        last_on_string[key] = ref
                    at += length
        tracks.append(RefTrack(len(tracks) + 1, name,
                               guess_instrument(name, program, drums, None if drums or not tuning else tuning),
                               sorted(notes, key=lambda n: (n.beat, n.pitch)), program, drums))
    return tracks


def default_mapping(score: Score) -> dict[int, str]:
    """Pistas -> instrumento pelo palpite (todas as pistas com notas e palpite conhecido)."""
    return {t.index: t.guess for t in score.tracks if t.notes and t.guess in INSTRUMENTS}


def parse_mapping(text: str) -> dict[int, str]:
    """ "2=guitar,4=bass" -> {2: "guitar", 4: "bass"} """
    mapping = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        index, sep, instrument = part.partition("=")
        if not sep or not index.strip().isdigit() or instrument.strip() not in INSTRUMENTS:
            raise ValueError(f"pista inválida em --reference-tracks: {part!r} (use número=instrumento, "
                             f"com instrumento em {', '.join(INSTRUMENTS)})")
        mapping[int(index)] = instrument.strip()
    return mapping


# ---------------------------------------------------------------------------
# Alinhamento com a gravação
# ---------------------------------------------------------------------------
@dataclass
class Alignment:
    ref_beats: np.ndarray     # batidas da partitura...
    audio_beats: np.ndarray   # ...e as batidas da gravação onde elas caem (crescentes)
    transpose: int            # semitons a somar à partitura para soar como a gravação
    factor: float             # batidas da gravação por batida da partitura (0.5, 1 ou 2)
    cost: float               # distância média no caminho (0 = igual, 1 = nada a ver)

    def to_audio_beat(self, beat: float) -> float | None:
        """Batida da gravação para uma batida da partitura (None fora do trecho alinhado)."""
        if beat < self.ref_beats[0] - 0.5 or beat > self.ref_beats[-1] + 1.5:
            return None
        return float(np.interp(beat, self.ref_beats, self.audio_beats,
                               right=self.audio_beats[-1] + (beat - self.ref_beats[-1]) * self.factor))


def score_chroma(tracks: list[RefTrack], step: float, n_steps: int) -> np.ndarray:
    """Quanto soa de cada nota (12 linhas) em cada passo de `step` semínimas da partitura."""
    chroma = np.zeros((12, n_steps))
    for track in tracks:
        for n in track.notes:
            first, last = int(n.beat / step), int((n.beat + max(n.length, step / 4)) / step - 1e-9)
            for k in range(max(first, 0), min(last, n_steps - 1) + 1):
                overlap = min(n.beat + n.length, (k + 1) * step) - max(n.beat, k * step)
                chroma[n.pitch % 12, k] += max(overlap, step / 4)
    return chroma


def audio_chroma(y: np.ndarray, sr: int, beat_times: np.ndarray) -> np.ndarray:
    """Chroma da gravação, média em cada batida do mapa de tempo."""
    import librosa

    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop)
    frames = np.clip(frames, 0, chroma.shape[1])
    out = np.zeros((12, len(beat_times) - 1))
    for j in range(len(beat_times) - 1):
        a, b = frames[j], max(frames[j + 1], frames[j] + 1)
        out[:, j] = chroma[:, a:b].mean(axis=1) if a < chroma.shape[1] else 0
    return out


def _normalize(chroma: np.ndarray) -> np.ndarray:
    chroma = chroma + 1e-3   # trechos em silêncio não ficam com vetor nulo (sem direção)
    return chroma / np.linalg.norm(chroma, axis=0, keepdims=True)


def align_chroma(ref: np.ndarray, audio: np.ndarray) -> tuple[np.ndarray, float]:
    """Caminho (passo da partitura, batida da gravação) por DTW e o custo médio por passo.

    A partitura pode cobrir só um trecho da gravação (introdução ou final a mais no áudio). As duas
    já estão em batidas, então o caminho esperado é a diagonal: sair dela (esticar ou encolher um
    trecho) custa WARP_COST, e o custo devolvido conta isso. Sem esse custo, o chroma (que muda
    devagar) deixa o DTW espremer a música inteira em poucos compassos quase sem pagar nada."""
    import librosa

    C = 1 - _normalize(ref).T @ _normalize(audio)   # distância de cosseno
    # Com subseq, o librosa procura a sequência MENOR dentro da maior: se a partitura fosse mais longa,
    # ele procuraria a gravação inteira num pedaço dela. A gravação ganha batidas neutras no fim
    # (partitura que passa do fim do áudio) para continuar sendo a maior.
    n, m = C.shape
    C = np.hstack([C, np.full((n, n - m + 1 if n >= m else 1), UNMATCHED_COST)])
    D, path = librosa.sequence.dtw(C=C, subseq=True, weights_add=np.array([0.0, WARP_COST, WARP_COST]))
    path = path[::-1]
    cost = float(D[path[-1, 0], path[-1, 1]] / len(path))
    return path[path[:, 1] < m], cost


def regular_grid(beat_times: np.ndarray) -> np.ndarray:
    """As batidas do mapa de tempo com o começo refeito no período normal.

    O mapa começa em 0 s e põe os compassos nos inícios detectados (analysis/tempo.py), então a
    primeira batida costuma ser mais longa (ou mais curta) que as outras. O alinhamento anda de
    batida em batida: com ela esticada, a partitura não poderia começar no meio dela."""
    if len(beat_times) < 4:
        return beat_times
    period = float(np.median(np.diff(beat_times[1:])))
    before = beat_times[1] - period * np.arange(int(beat_times[1] / period + 0.25), 0, -1)
    return np.concatenate([np.maximum(before, 0.0), beat_times[1:]])


def align(score: Score, tracks: list[RefTrack], y: np.ndarray, sr: int, beat_times: np.ndarray) -> Alignment:
    """Melhor alinhamento entre as pistas (com altura) da partitura e a gravação."""
    pitched = [t for t in tracks if not t.drums and t.notes]
    if not pitched:
        raise ValueError("a partitura não tem pistas com notas (além da bateria) para alinhar")
    grid = regular_grid(beat_times)
    audio = audio_chroma(y, sr, grid)
    best = None
    for factor in (1.0, 2.0, 0.5):
        step = 1 / factor   # passos de partitura por batida da gravação
        n_steps = int(np.ceil(score.length_beats / step)) + 1
        ref = score_chroma(pitched, step, n_steps)
        for transpose in range(12):
            path, cost = align_chroma(np.roll(ref, transpose, axis=0), audio)
            if len(path) and (best is None or cost < best[0]):
                best = (cost, factor, transpose, path, step)
    if best is None:
        raise ValueError("a partitura não se alinha com nenhum trecho da gravação")
    cost, factor, transpose, path, step = best
    steps, beats = path[:, 0], path[:, 1].astype(float)
    ref_beats = np.unique(steps)
    audio_beats = np.array([beats[steps == k].mean() for k in ref_beats])
    # batida da grade -> instante -> batida (fracionária) do mapa de tempo
    audio_beats = np.interp(np.interp(audio_beats, np.arange(len(grid)), grid),
                            beat_times, np.arange(len(beat_times)))
    audio_beats = np.maximum.accumulate(audio_beats)
    return Alignment(ref_beats * step, audio_beats, transpose if transpose <= 6 else transpose - 12, factor, cost)


# ---------------------------------------------------------------------------
# Guia
# ---------------------------------------------------------------------------
@dataclass
class RefEvent:
    tick: int
    end_tick: int
    pitches: tuple[int, ...]   # notas que começam juntas (acorde), da mais grave para a mais aguda


def events_in_ticks(tracks: list[RefTrack], alignment: Alignment, resolution: int) -> list[RefEvent]:
    """Notas da partitura agrupadas em acordes e levadas para os ticks da gravação."""
    notes = sorted((n for t in tracks for n in t.notes), key=lambda n: (n.beat, n.pitch))
    groups: list[list[RefNote]] = []
    for n in notes:
        if groups and n.beat - groups[-1][0].beat <= CHORD_ONSET_BEATS:
            groups[-1].append(n)
        else:
            groups.append([n])
    events = []
    for group in groups:
        start = alignment.to_audio_beat(group[0].beat)
        end = alignment.to_audio_beat(max(n.beat + n.length for n in group))
        if start is None:
            continue
        end = start + 0.1 if end is None or end <= start else end
        pitches = tuple(sorted({n.pitch + alignment.transpose for n in group}))
        events.append(RefEvent(int(round(start * resolution)), int(round(end * resolution)), pitches))
    return events


def chord_voices(pitches: tuple[int, ...], length_beats: float, cfg) -> int:
    """Quantas notas o acorde da partitura tem na chart (as mesmas regras da transcrição)."""
    classes = {p % 12 for p in pitches}
    if len(classes) <= 1:
        return 1
    root = pitches[0] % 12
    if classes <= {root, (root + 7) % 12}:
        voices = cfg.power_chord_notes
    elif length_beats >= cfg.accent_min_beats:
        voices = cfg.accent_chord_notes
    else:
        voices = cfg.chord_notes
    return max(1, min(voices, len(pitches)))


def _nearest(events: list, ticks: np.ndarray, tick: int, tolerance: float):
    """O evento mais próximo de `tick` (até `tolerance` ticks), ou None."""
    if not len(ticks):
        return None
    i = int(np.searchsorted(ticks, tick))
    best = min((j for j in (i - 1, i) if 0 <= j < len(ticks)), key=lambda j: abs(ticks[j] - tick))
    return events[best] if abs(ticks[best] - tick) <= tolerance else None


def _by_bar(items: list, bar: int) -> dict[int, list]:
    out: dict[int, list] = {}
    for item in items:
        out.setdefault(item.tick // bar, []).append(item)
    return out


@dataclass
class Guide:
    """Corrige as notas detectadas de um instrumento com as da partitura.

    `events` são as notas da partitura (uma lista) ou, com várias pistas no mesmo instrumento (solo,
    base...), as notas de cada pista ({nome: lista}). Aí cada compasso usa a pista que mais confere
    com o que foi detectado ali, em vez de somar todas.

    `fidelity` diz quanto a partitura manda (FIDELITY):
      corrigir  - as notas são as detectadas; a partitura só corrige altura e acorde;
      completar - nos compassos em que a detecção confere com a partitura (BAR_MATCH), as notas
                  passam a ser as da partitura (o ritmo dela, sem as falhas e sobras da detecção);
                  nos outros, fica a detecção corrigida;
      seguir    - onde a partitura tem notas, valem as dela; onde ela pausa, a chart também pausa.
                  A detecção só fica antes e depois do trecho que a partitura cobre."""
    instrument: str
    events: list[RefEvent] | dict[str, list[RefEvent]]
    resolution: int
    cfg: object                       # PitchedDetection do instrumento
    fidelity: str = "corrigir"
    stats: dict = field(default_factory=dict)

    def __post_init__(self):
        self.tracks = self.events if isinstance(self.events, dict) else {"": self.events}
        self.tracks = {name: events for name, events in self.tracks.items() if events} or {"": []}

    def choose_tracks(self, cands: list) -> tuple[list[RefEvent], dict[str, int]]:
        """As notas da partitura, compasso a compasso, da pista que mais confere com a detecção ali.
        Conta o ataque (até MATCH_BEATS) e, mais ainda, a mesma nota (sem olhar a oitava); notas a
        mais na pista pesam contra (um solo por cima da base não ganha da base)."""
        if len(self.tracks) == 1:
            (events,) = self.tracks.values()
            return events, {}
        bar, tolerance = 4 * self.resolution, MATCH_BEATS * self.resolution
        detected = _by_bar(cands, bar)
        per_track = {name: _by_bar(events, bar) for name, events in self.tracks.items()}
        last_bar = max((e.tick // bar for events in self.tracks.values() for e in events), default=-1)
        chosen, previous, count = [], None, {}
        for b in range(last_bar + 1):
            options = [name for name in self.tracks if per_track[name].get(b)]
            if not options:
                continue
            found = detected.get(b, [])

            def score(name: str) -> float:
                events = per_track[name][b]
                ticks = np.array([e.tick for e in events])
                total = 0.0
                for c in found:
                    event = _nearest(events, ticks, c.tick, tolerance)
                    if event is not None:
                        total += 1 + (round(c.pitch) % 12 in {p % 12 for p in event.pitches})
                return total / (len(found) + len(events)) + (0.05 if name == previous else 0.0)

            if found:
                name = max(options, key=score)
            else:   # nada detectado no compasso: continua na mesma pista, se ela toca ali
                name = previous if previous in options else options[0]
            chosen.extend(per_track[name][b])
            count[name] = count.get(name, 0) + 1
            previous = name
        return chosen, count

    def _apply(self, c, event: RefEvent) -> bool:
        """Dá a `c` a altura e o acorde do evento da partitura. Retorna se algo mudou."""
        from ai_charter.analysis.chords import CHORDS

        voices = chord_voices(event.pitches, (event.end_tick - event.tick) / self.resolution, self.cfg)
        pitch = float(event.pitches[0])
        changed = round(c.pitch) != round(pitch) or c.voices != voices
        c.pitch, c.voices = pitch, voices
        root = event.pitches[0] % 12
        classes = {p % 12 for p in event.pitches}
        kind = ("note" if voices == 1 else "5" if classes <= {root, (root + 7) % 12}
                else "m" if (root + 3) % 12 in classes and (root + 4) % 12 not in classes else "")
        c.chord = CHORDS.index((root, kind))
        return changed

    def _from_score(self, events: list[RefEvent], strength: float) -> list:
        from ai_charter.charting.common import Candidate

        out = []
        for event in events:
            c = Candidate(event.tick, strength, end_tick=event.end_tick)
            self._apply(c, event)
            out.append(c)
        return out

    def __call__(self, cands: list, source: str = "", log=print) -> list:
        events, track_count = self.choose_tracks(cands)
        ticks = np.array([e.tick for e in events])
        tolerance = MATCH_BEATS * self.resolution
        matches = [_nearest(events, ticks, c.tick, tolerance) for c in cands]
        matched = sum(m is not None for m in matches)
        share = matched / len(cands) if cands else 0.0
        label = f"    {self.instrument}" + (f" ({source})" if source else "")
        if share < MIN_MATCHED:
            log(f"{label}: a referência confere com só {share:.0%} das notas detectadas, ignorada")
            self.stats = {"matched": matched, "notes": len(cands), "changed": 0, "applied": False}
            return cands
        changed = sum(self._apply(c, event) for c, event in zip(cands, matches) if event is not None)
        line = f"{label}: referência confere com {share:.0%} das notas detectadas; {changed} corrigidas"
        if track_count:
            line += "; pistas por compasso: " + ", ".join(
                f"{name} ({n})" for name, n in sorted(track_count.items(), key=lambda kv: -kv[1]))
        self.stats = {"matched": matched, "notes": len(cands), "changed": changed, "applied": True,
                      "tracks": track_count, "bars_from_score": 0}
        if self.fidelity != "corrigir" and events:
            cands, bars = self._replace_bars(cands, matches, events)
            self.stats["bars_from_score"] = bars
            line += f"; {bars} compasso(s) com o ritmo da partitura ({self.fidelity})"
        log(line)
        return cands

    def _replace_bars(self, cands: list, matches: list, events: list[RefEvent]) -> tuple[list, int]:
        """Troca as notas detectadas pelas da partitura nos compassos que a fidelidade pede."""
        bar = 4 * self.resolution
        detected, from_score = _by_bar(cands, bar), _by_bar(events, bar)
        matched_in = {}
        for c, event in zip(cands, matches):
            matched_in[c.tick // bar] = matched_in.get(c.tick // bar, 0) + (event is not None)
        strength = float(np.median([c.strength for c in cands])) if cands else 1.0
        first, last = events[0].tick // bar, events[-1].tick // bar
        out, replaced = [], 0
        for b in sorted(set(detected) | set(from_score)):
            found, score = detected.get(b, []), from_score.get(b, [])
            if self.fidelity == "seguir":
                use_score = first <= b <= last
            else:   # completar
                use_score = bool(found and score) and matched_in.get(b, 0) >= BAR_MATCH * len(found)
            if use_score:
                out.extend(self._from_score(score, strength))
                replaced += bool(score)
            else:
                out.extend(found)
        return sorted(out, key=lambda c: c.tick), replaced


# Bateria: peça de cada nota do mapa General MIDI (canal 10).
GM_DRUMS = {35: "kick", 36: "kick", 37: "snare", 38: "snare", 39: "snare", 40: "snare",
            42: "hihat", 44: "hihat", 46: "hihat", 49: "crash", 52: "crash", 55: "crash", 57: "crash",
            51: "ride", 53: "ride", 59: "ride", 41: "tom_low", 43: "tom_low", 45: "tom_mid", 47: "tom_mid",
            48: "tom_high", 50: "tom_high"}


@dataclass
class DrumEvent:
    tick: int
    pieces: frozenset   # peças que a partitura toca juntas ali (nomes de analysis.drums)


def drum_events_in_ticks(tracks: list[RefTrack], alignment: Alignment, resolution: int) -> list[DrumEvent]:
    groups: list[tuple[float, set]] = []
    for n in sorted((n for t in tracks for n in t.notes), key=lambda n: n.beat):
        piece = GM_DRUMS.get(n.pitch)
        if piece is None:
            continue
        if groups and n.beat - groups[-1][0] <= CHORD_ONSET_BEATS:
            groups[-1][1].add(piece)
        else:
            groups.append((n.beat, {piece}))
    events = []
    for beat, pieces in groups:
        at = alignment.to_audio_beat(beat)
        if at is not None:
            events.append(DrumEvent(int(round(at * resolution)), frozenset(pieces)))
    return events


@dataclass
class DrumGuide:
    """Corrige as PEÇAS da pista de bateria pronta com as da partitura.

    Não muda a transcrição (nem o ritmo): cada nota da pista que cai num golpe da partitura
    fica com as peças das mãos que a partitura toca ali, com o mesmo número de mãos (o limite
    de cada dificuldade) e o mesmo bumbo da pista."""
    events: list[DrumEvent]
    resolution: int
    stats: dict = field(default_factory=dict)

    def _nearest(self, tick: int, ticks: np.ndarray) -> DrumEvent | None:
        if not len(ticks):
            return None
        i = int(np.searchsorted(ticks, tick))
        best = min((j for j in (i - 1, i) if 0 <= j < len(ticks)), key=lambda j: abs(ticks[j] - tick))
        return self.events[best] if abs(ticks[best] - tick) <= DRUM_MATCH_BEATS * self.resolution else None

    def __call__(self, track, log=print):
        from ai_charter.charting.drums import CYMBAL_FLAGS, HAND_PRIORITY, LANES

        ticks = np.array([e.tick for e in self.events])
        expert = track.notes.get("expert", [])
        matched = sum(self._nearest(n.tick, ticks) is not None for n in expert)
        share = matched / len(expert) if expert else 0.0
        if share < MIN_MATCHED:
            log(f"    drums: a referência confere com só {share:.0%} das batidas, ignorada")
            self.stats = {"matched": matched, "notes": len(expert), "changed": 0, "applied": False}
            return track
        cymbal_of = {flag: piece for piece, flag in CYMBAL_FLAGS.items()}
        tom_of = {2: "tom_high", 3: "tom_mid", 4: "tom_low"}
        changed = 0
        for difficulty, notes in track.notes.items():
            for note in notes:
                event = self._nearest(note.tick, ticks)
                if event is None:
                    continue
                hands_wanted = [lane for lane in note.lanes if lane != 0]
                reference = [p for p in HAND_PRIORITY if p in event.pieces]
                if not hands_wanted or not reference:
                    continue
                chosen, lanes = [], set()
                for piece in reference:   # uma peça por lane, até o número de mãos da nota
                    if LANES[piece] not in lanes and len(chosen) < len(hands_wanted):
                        chosen.append(piece)
                        lanes.add(LANES[piece])
                # A partitura tem menos mãos ali: as outras ficam como a transcrição decidiu.
                for lane in hands_wanted:
                    if len(chosen) >= len(hands_wanted):
                        break
                    if lane not in lanes:
                        flag = next((f for f in note.flags if f in cymbal_of and LANES[cymbal_of[f]] == lane), None)
                        chosen.append(cymbal_of[flag] if flag else ("snare" if lane == 1 else tom_of[lane]))
                        lanes.add(lane)
                new_lanes = tuple(sorted(([0] if 0 in note.lanes else []) + [LANES[p] for p in chosen]))
                new_flags = tuple(sorted(CYMBAL_FLAGS[p] for p in chosen if p in CYMBAL_FLAGS))
                if (new_lanes, new_flags) != (tuple(sorted(note.lanes)), tuple(sorted(note.flags))):
                    if difficulty == "expert":
                        changed += 1
                    note.lanes, note.flags = new_lanes, new_flags
        log(f"    drums: referência confere com {share:.0%} das batidas; {changed} corrigidas no Expert")
        self.stats = {"matched": matched, "notes": len(expert), "changed": changed, "applied": True}
        return track


def short_name(track: RefTrack) -> str:
    """Nome curto da pista para o log: "Monte Money | Lead Guitar | Gibson" -> "1 Lead Guitar"."""
    parts = [p.strip() for p in track.name.split("|") if p.strip()]
    return f"{track.index} {parts[1] if len(parts) > 1 else (parts[0] if parts else '')}".strip()


def load_guides(path: Path, mapping: dict[int, str] | None, y: np.ndarray, sr: int, tempo, rules,
                log=print, fidelity: str = DEFAULT_FIDELITY) -> dict[str, "Guide | DrumGuide"]:
    """Lê a partitura, alinha com a gravação e devolve um guia por instrumento (guitar, bass, keys, drums)."""
    if fidelity not in FIDELITY:
        raise ValueError(f"fidelidade inválida: {fidelity!r} (use {', '.join(FIDELITY)})")
    score = read_score(path)
    mapping = default_mapping(score) if mapping is None else mapping
    by_index = {t.index: t for t in score.tracks}
    unknown = sorted(set(mapping) - set(by_index))
    if unknown:
        raise ValueError(f"a partitura não tem a(s) pista(s) {', '.join(map(str, unknown))} "
                         f"(tem {len(score.tracks)})")
    log(f"Referência {score.path.name}: " + "; ".join(
        f"{t.index} {t.name} -> {mapping.get(t.index, '(não usada)')}" for t in score.tracks if t.notes))
    # Todas as pistas com altura ajudam a alinhar, mesmo as que não guiam nenhum instrumento.
    alignment = align(score, score.tracks, y, sr, tempo.beat_times)
    log(f"    alinhada com a gravação (distância {alignment.cost:.2f}"
        + (f", transposta {alignment.transpose:+d} semitons" if alignment.transpose else "")
        + (f", {alignment.factor:g} batida(s) da gravação por batida da partitura" if alignment.factor != 1 else "")
        + ")")
    guides = {}
    for instrument in PITCHED:
        tracks = [by_index[i] for i, inst in mapping.items() if inst == instrument]
        if tracks:   # com várias pistas no instrumento, o guia escolhe uma por compasso
            events = {short_name(t): events_in_ticks([t], alignment, tempo.resolution) for t in tracks}
            guides[instrument] = Guide(instrument, events, tempo.resolution, rules.detection[instrument], fidelity)
    drum_tracks = [by_index[i] for i, inst in mapping.items() if inst == "drums"]
    if drum_tracks:
        guides["drums"] = DrumGuide(drum_events_in_ticks(drum_tracks, alignment, tempo.resolution),
                                    tempo.resolution)
    others = sorted({inst for inst in mapping.values() if inst not in GUIDED})
    if others:
        log(f"    {', '.join(others)}: a referência ainda não guia esse(s) instrumento(s)")
    return guides


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ai-charter-ref",
                                     description="Lista as pistas de uma partitura (MIDI ou Guitar Pro).")
    parser.add_argument("score", type=Path, help="arquivo .mid, .gp3, .gp4, .gp5, .gpx ou .gp")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        score = read_score(args.score)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"{score.path.name}: {len(score.tracks)} pista(s), {score.length_beats:.0f} batidas")
    for t in score.tracks:
        print(f"  {t.index:>2}  {t.name:<30} {len(t.notes):>5} notas   palpite: {t.guess}")
    mapping = default_mapping(score)
    print("\nPadrão do --reference-tracks: " + (",".join(f"{i}={inst}" for i, inst in mapping.items()) or "(nenhuma)"))


if __name__ == "__main__":
    main()
