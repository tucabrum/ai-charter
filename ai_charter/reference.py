"""Partitura de referência (MIDI ou Guitar Pro) para GUIAR a transcrição.

Uso:
  ai-charter musica.flac --reference musica.gp5                         # pistas escolhidas por palpite
  ai-charter musica.flac --reference musica.mid --reference-tracks 2=guitar,4=bass
  ai-charter-ref musica.gp5                                             # lista as pistas e os palpites

As notas da chart continuam vindo do áudio (o que a gravação realmente toca). A
partitura só corrige cada nota detectada que cai perto de uma nota dela: a ALTURA
(e, com ela, a cor) e o ACORDE (quantas notas, power chord ou não). Notas que a
partitura tem e o áudio não, ou vice-versa, ficam como a transcrição decidiu.

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
     confere, a referência é ignorada e o log avisa.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MIDI_EXTENSIONS = (".mid", ".midi", ".kar")
GP_EXTENSIONS = (".gp3", ".gp4", ".gp5")
NEW_GP_EXTENSIONS = (".gpx", ".gp", ".gp6", ".gp7")
INSTRUMENTS = ("guitar", "bass", "keys", "drums", "vocals")
GUIDED = ("guitar", "bass", "keys")       # instrumentos que a referência guia (por enquanto)
MATCH_BEATS = 0.25                        # distância máxima entre a nota detectada e a da partitura
MIN_MATCHED = 0.3                         # abaixo disso a referência não confere: é ignorada
CHORD_ONSET_BEATS = 1 / 32                # notas que começam até aqui juntas são um acorde

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
        raise ValueError(f"o formato {suffix} (Guitar Pro 6 ou mais novo) ainda não é lido: exporte a partitura "
                         "como .gp5 ou MIDI (no Guitar Pro, TuxGuitar ou MuseScore)")
    raise ValueError(f"formato de partitura não reconhecido: {path.name} (use .mid, .gp3, .gp4 ou .gp5)")


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

    A partitura pode cobrir só um trecho da gravação (introdução ou final a mais no áudio)."""
    import librosa

    D, path = librosa.sequence.dtw(X=_normalize(ref), Y=_normalize(audio), metric="cosine", subseq=True)
    path = path[::-1]
    return path, float(D[-1, path[-1, 1]] / len(path))


def align(score: Score, tracks: list[RefTrack], y: np.ndarray, sr: int, beat_times: np.ndarray) -> Alignment:
    """Melhor alinhamento entre as pistas (com altura) da partitura e a gravação."""
    pitched = [t for t in tracks if not t.drums and t.notes]
    if not pitched:
        raise ValueError("a partitura não tem pistas com notas (além da bateria) para alinhar")
    audio = audio_chroma(y, sr, beat_times)
    best = None
    for factor in (1.0, 2.0, 0.5):
        step = 1 / factor   # passos de partitura por batida da gravação
        n_steps = int(np.ceil(score.length_beats / step)) + 1
        ref = score_chroma(pitched, step, n_steps)
        for transpose in range(12):
            path, cost = align_chroma(np.roll(ref, transpose, axis=0), audio)
            if best is None or cost < best[0]:
                best = (cost, factor, transpose, path, step)
    cost, factor, transpose, path, step = best
    steps, beats = path[:, 0], path[:, 1].astype(float)
    ref_beats = np.unique(steps)
    audio_beats = np.array([beats[steps == k].mean() for k in ref_beats])
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


@dataclass
class Guide:
    """Corrige as notas detectadas de um instrumento com as da partitura."""
    instrument: str
    events: list[RefEvent]
    resolution: int
    cfg: object                       # PitchedDetection do instrumento
    stats: dict = field(default_factory=dict)

    def nearest(self, tick: int, ticks: np.ndarray) -> RefEvent | None:
        if not len(ticks):
            return None
        i = int(np.searchsorted(ticks, tick))
        best = min((j for j in (i - 1, i) if 0 <= j < len(ticks)), key=lambda j: abs(ticks[j] - tick))
        return self.events[best] if abs(ticks[best] - tick) <= MATCH_BEATS * self.resolution else None

    def __call__(self, cands: list, source: str = "", log=print) -> list:
        from ai_charter.analysis.chords import CHORDS

        ticks = np.array([e.tick for e in self.events])
        matches = [self.nearest(c.tick, ticks) for c in cands]
        matched = sum(m is not None for m in matches)
        share = matched / len(cands) if cands else 0.0
        label = f"    {self.instrument}" + (f" ({source})" if source else "")
        if share < MIN_MATCHED:
            log(f"{label}: a referência confere com só {share:.0%} das notas detectadas, ignorada")
            self.stats = {"matched": matched, "notes": len(cands), "changed": 0, "applied": False}
            return cands
        changed = 0
        for c, event in zip(cands, matches):
            if event is None:
                continue
            voices = chord_voices(event.pitches, (event.end_tick - event.tick) / self.resolution, self.cfg)
            pitch = float(event.pitches[0])
            if round(c.pitch) != round(pitch) or c.voices != voices:
                changed += 1
            c.pitch, c.voices = pitch, voices
            root = event.pitches[0] % 12
            classes = {p % 12 for p in event.pitches}
            kind = ("note" if voices == 1 else "5" if classes <= {root, (root + 7) % 12}
                    else "m" if (root + 3) % 12 in classes and (root + 4) % 12 not in classes else "")
            c.chord = CHORDS.index((root, kind))
        log(f"{label}: referência confere com {share:.0%} das notas detectadas; {changed} corrigidas")
        self.stats = {"matched": matched, "notes": len(cands), "changed": changed, "applied": True}
        return cands


def load_guides(path: Path, mapping: dict[int, str] | None, y: np.ndarray, sr: int, tempo, rules,
                log=print) -> dict[str, Guide]:
    """Lê a partitura, alinha com a gravação e devolve um guia por instrumento (guitar, bass, keys)."""
    score = read_score(path)
    mapping = default_mapping(score) if mapping is None else mapping
    by_index = {t.index: t for t in score.tracks}
    unknown = sorted(set(mapping) - set(by_index))
    if unknown:
        raise ValueError(f"a partitura não tem a(s) pista(s) {', '.join(map(str, unknown))} "
                         f"(tem {len(score.tracks)})")
    log(f"Referência {score.path.name}: " + "; ".join(
        f"{t.index} {t.name} -> {mapping.get(t.index, '(não usada)')}" for t in score.tracks if t.notes))
    used = [by_index[i] for i in mapping]
    alignment = align(score, used or score.tracks, y, sr, tempo.beat_times)
    log(f"    alinhada com a gravação (distância {alignment.cost:.2f}"
        + (f", transposta {alignment.transpose:+d} semitons" if alignment.transpose else "")
        + (f", {alignment.factor:g} batida(s) da gravação por batida da partitura" if alignment.factor != 1 else "")
        + ")")
    guides = {}
    for instrument in GUIDED:
        tracks = [by_index[i] for i, inst in mapping.items() if inst == instrument]
        if tracks:
            guides[instrument] = Guide(instrument, events_in_ticks(tracks, alignment, tempo.resolution),
                                       tempo.resolution, rules.detection[instrument])
    others = sorted({inst for inst in mapping.values() if inst not in GUIDED})
    if others:
        log(f"    {', '.join(others)}: a referência ainda não guia esse(s) instrumento(s)")
    return guides


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ai-charter-ref",
                                     description="Lista as pistas de uma partitura (MIDI ou Guitar Pro).")
    parser.add_argument("score", type=Path, help="arquivo .mid, .gp3, .gp4 ou .gp5")
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
