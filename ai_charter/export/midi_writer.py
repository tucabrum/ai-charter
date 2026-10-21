"""Escreve o arquivo `notes.mid` no padrão Rock Band, lido pelo Clone Hero e pelo YARG.

Diferente do `.chart`, o MIDI suporta a faixa de VOCAL jogável (com afinação e letra).

Estrutura:
  trilha 0        andamento (BPM) e compasso
  EVENTS          seções ("[section Section 1]") e fim da música ("[end]")
  PART GUITAR / PART BASS / PART KEYS
                  cada dificuldade usa uma faixa de notas MIDI:
                  Easy 60-64, Medium 72-76, Hard 84-88, Expert 96-100 (verde..laranja);
                  +6 = forçar strum; 116 = Star Power
  PART DRUMS      mesmas faixas (bumbo, vermelho, amarelo, azul, verde);
                  110-112 = amarelo/azul/verde são tons (sem isso, são pratos);
                  120-124 = virada de ativação do Star Power
  PART VOCALS     notas 36-84 = afinação; evento de letra no início de cada nota;
                  105 = frase; 116 = Star Power
"""

from __future__ import annotations

from pathlib import Path

import mido

from ai_charter.analysis.tempo import TempoMap
from ai_charter.charting.common import TrackChart
from ai_charter.charting.frets import FORCE_FLAG
from ai_charter.charting.vocals import VocalTrack

TRACK_NAMES = {"guitar": "PART GUITAR", "bass": "PART BASS", "keys": "PART KEYS", "drums": "PART DRUMS"}
BASE_NOTE = {"easy": 60, "medium": 72, "hard": 84, "expert": 96}
FORCE_STRUM = 6
STAR_POWER = 116
PHRASE = 105
DRUM_FILL = range(120, 125)
TOM_MARKERS = {2: 110, 3: 111, 4: 112}     # lane -> marcador de tom
CYMBAL_FLAGS = {2: 66, 3: 67, 4: 68}       # lane -> marcador de prato que usamos internamente


def _safe_text(text: str) -> str:
    """O texto do MIDI é Latin-1 (padrão do Rock Band); cobre todos os acentos do português."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


class _Track:
    """Ajuda a montar uma trilha com tempos absolutos (o MIDI usa tempos relativos)."""

    def __init__(self, name: str):
        self.name = name
        self.events: list[tuple[int, int, mido.Message | mido.MetaMessage]] = []

    def note(self, tick: int, length: int, pitch: int) -> None:
        # Na mesma posição, o "note_off" vem antes do "note_on" (ordem 0 e 2).
        self.events.append((tick, 2, mido.Message("note_on", note=pitch, velocity=100, time=0)))
        self.events.append((tick + max(1, length), 0, mido.Message("note_off", note=pitch, velocity=0, time=0)))

    def meta(self, tick: int, kind: str, **kwargs) -> None:
        self.events.append((tick, 1, mido.MetaMessage(kind, time=0, **kwargs)))

    def build(self) -> mido.MidiTrack:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=_safe_text(self.name), time=0))
        now = 0
        for tick, _, msg in sorted(self.events, key=lambda e: (e[0], e[1])):
            track.append(msg.copy(time=tick - now))
            now = tick
        track.append(mido.MetaMessage("end_of_track", time=0))
        return track


def _fret_track(name: str, track: TrackChart, resolution: int, drums: bool) -> _Track:
    out = _Track(name)
    short = max(1, resolution // 16)
    for difficulty, notes in track.notes.items():
        base = BASE_NOTE[difficulty]
        for note in notes:
            length = note.length or short
            for lane in note.lanes:
                out.note(note.tick, length, base + lane)
                if drums and lane in TOM_MARKERS and CYMBAL_FLAGS[lane] not in note.flags:
                    out.note(note.tick, short, TOM_MARKERS[lane])
            if not drums and FORCE_FLAG in note.flags:
                out.note(note.tick, short, base + FORCE_STRUM)
    # No MIDI, Star Power e viradas valem para todas as dificuldades: usamos as do Expert.
    for tick, length in track.star_power.get("expert", []):
        out.note(tick, length, STAR_POWER)
    for tick, length in track.drum_fills.get("expert", []):
        for pitch in DRUM_FILL:
            out.note(tick, length, pitch)
    return out


def _vocal_track(vocals: VocalTrack) -> _Track:
    out = _Track("PART VOCALS")
    for note in vocals.notes:
        out.meta(note.tick, "lyrics", text=_safe_text(note.lyric))
        out.note(note.tick, note.length, note.pitch)
    for tick, text in vocals.spoken:   # trechos falados: letra na tela, sem nota para cantar
        out.meta(tick, "lyrics", text=_safe_text(text))
    for tick, length in vocals.phrases:
        out.note(tick, length, PHRASE)
    for tick, length in vocals.star_power:
        out.note(tick, length, STAR_POWER)
    return out


def write_midi(path: Path, song_name: str, tempo: TempoMap, tracks: dict[str, TrackChart],
               sections: list[int], vocals: VocalTrack | None, end_tick: int) -> None:
    res = tempo.resolution
    midi = mido.MidiFile(type=1, ticks_per_beat=res)

    sync = _Track(song_name)
    sync.meta(0, "time_signature", numerator=4, denominator=4)
    last = None
    for i, period in enumerate(tempo.beat_times[1:] - tempo.beat_times[:-1]):
        microseconds = int(round(period * 1_000_000))  # duração de uma batida
        if microseconds != last:
            sync.meta(i * res, "set_tempo", tempo=microseconds)
            last = microseconds
    midi.tracks.append(sync.build())

    events = _Track("EVENTS")
    for i, tick in enumerate(sections):
        events.meta(tick, "text", text=f"[section Section {i + 1}]")
    events.meta(end_tick, "text", text="[end]")
    midi.tracks.append(events.build())

    for instrument, track in tracks.items():
        midi.tracks.append(_fret_track(TRACK_NAMES[instrument], track, res, instrument == "drums").build())
    if vocals is not None and vocals.notes:
        midi.tracks.append(_vocal_track(vocals).build())
    midi.save(path)
