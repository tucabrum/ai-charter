"""Tempo de entrada: compassos de silêncio antes da primeira nota, como nas charts do RB e do GH.

Sem ele, uma música que começa tocando já começa com notas, e o jogador é pego de surpresa.
A entrada é posta na exportação, depois de toda a análise (mapa de tempo, partitura, letra...),
que continua sendo feita no áudio original:

- entram compassos inteiros no começo, no andamento do primeiro compasso, então a grade de
  compassos e batidas não muda: tudo só anda para frente;
- cada .ogg recebe o mesmo tempo de silêncio no começo (na mesma codificação de sempre);
- notes.mid ou notes.chart: todos os eventos andam o mesmo número de ticks, menos o nome da
  pista, o andamento e o compasso do tick 0, que passam a valer também para a entrada.

O editor e o Acrescentar leem o notes.mid e os .ogg da pasta, que continuam combinando.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

MIN_SECONDS = 3.0      # tempo mínimo antes da primeira nota
BEATS_PER_BAR = 4      # os arquivos do charter são sempre 4/4
KEEP_AT_START = {"track_name", "set_tempo", "time_signature", "key_signature", "instrument_name", "copyright"}


@dataclass
class LeadIn:
    bars: int = 0
    ticks: int = 0
    seconds: float = 0.0


def first_tick(tracks: dict, vocals=None) -> int | None:
    """O tick da primeira nota de qualquer instrumento ou dos vocais (None sem notas)."""
    ticks = [notes[0].tick for track in tracks.values() for notes in track.notes.values() if notes]
    if vocals is not None and vocals.notes:
        ticks.append(vocals.notes[0].tick)
    return min(ticks) if ticks else None


def plan(tempo, first: int | None, min_seconds: float = MIN_SECONDS) -> LeadIn:
    """Quantos compassos pôr antes da música para a primeira nota ficar a min_seconds do começo."""
    if first is None or min_seconds <= 0:
        return LeadIn()
    missing = min_seconds - float(tempo.tick_to_time(first))
    if missing <= 0:
        return LeadIn()
    beat = float(tempo.beat_times[1] - tempo.beat_times[0])   # o 1º compasso dá o andamento da entrada
    bar = BEATS_PER_BAR * beat
    bars = math.ceil(missing / bar - 1e-9)
    return LeadIn(bars, bars * BEATS_PER_BAR * tempo.resolution, bars * bar)


def shift_midi(path: Path, ticks: int) -> None:
    """Adia todos os eventos do notes.mid em `ticks` (os de configuração do tick 0 ficam)."""
    import mido

    if ticks <= 0:
        return
    midi = mido.MidiFile(path)
    for i, track in enumerate(midi.tracks):
        now, events = 0, []
        for n, msg in enumerate(track):
            now += msg.time
            if msg.type == "end_of_track":
                continue
            keep = now == 0 and msg.is_meta and msg.type in KEEP_AT_START
            events.append((now if keep else now + ticks, n, msg))
        events.sort(key=lambda e: (e[0], e[1]))
        out, last = mido.MidiTrack(), 0
        for tick, _, msg in events:
            out.append(msg.copy(time=tick - last))
            last = tick
        out.append(mido.MetaMessage("end_of_track", time=0))
        midi.tracks[i] = out
    midi.save(path)


EVENT_LINE = re.compile(r"^(\s*)(\d+)(\s*=\s*)(\S+)(.*)$")


def shift_chart(path: Path, ticks: int) -> None:
    """Adia todos os eventos do notes.chart em `ticks` (o andamento e o compasso do tick 0 ficam)."""
    if ticks <= 0:
        return
    lines, section = [], ""
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip().startswith("["):
            section = line.strip()
        m = EVENT_LINE.match(line)
        if m and section != "[Song]":
            tick = int(m.group(2))
            if not (section == "[SyncTrack]" and tick == 0):
                line = f"{m.group(1)}{tick + ticks}{m.group(3)}{m.group(4)}{m.group(5)}"
        lines.append(line)
    # Como o chart_writer grava: UTF-8 com BOM e fim de linha do Windows.
    Path(path).write_text("\r\n".join(lines) + "\r\n", encoding="utf-8-sig", newline="")


def shift_file(path: Path, ticks: int) -> None:
    (shift_midi if Path(path).suffix.lower() == ".mid" else shift_chart)(path, ticks)
