"""Monta a faixa de vocal: notas no grid, frases e a letra sílaba por sílaba."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ai_charter.analysis.tempo import TempoMap
from ai_charter.analysis.vocals import SungNote
from ai_charter.charting.common import quantize_tick
from ai_charter.config import VocalRules

# Faixa de notas aceita pelos jogos na trilha de vocal (MIDI).
MIN_PITCH, MAX_PITCH = 36, 84

# Custos do alinhamento letra x notas (quanto maior, mais o algoritmo evita).
COST_LINE_MISMATCH = 2.0   # começo de linha da letra fora de uma pausa da voz (ou vice-versa)
COST_MELISMA = 1.5         # nota extra vira "+" (a sílaba anterior continua)
COST_MERGE = 2.5           # duas sílabas numa nota só
COST_SKIP_NOTE = 3.0       # nota descartada (provavelmente ruído)


@dataclass
class VocalNote:
    tick: int
    length: int
    pitch: int
    lyric: str = ""


@dataclass
class VocalTrack:
    notes: list[VocalNote]
    phrases: list[tuple[int, int]]                       # (tick, duração)
    star_power: list[tuple[int, int]] = field(default_factory=list)


def _to_grid(sung: list[SungNote], tempo: TempoMap, resolution: int,
             allow_triplets: bool) -> list[VocalNote]:
    """Encaixa as notas cantadas no grid (início e fim em semicolcheias)."""
    sixteenth = resolution // 4
    notes: list[VocalNote] = []
    for s in sung:
        start = quantize_tick(tempo.time_to_tick(s.time), resolution, allow_triplets)
        end = int(round(tempo.time_to_tick(s.end) / sixteenth)) * sixteenth
        pitch = int(round(s.pitch))
        while pitch < MIN_PITCH:
            pitch += 12
        while pitch > MAX_PITCH:
            pitch -= 12
        note = VocalNote(start, max(sixteenth, end - start), pitch)
        if notes and notes[-1].tick == start:
            # Duas notas no mesmo lugar (ex.: uma consoante "m"/"n" curta + a vogal):
            # fica a mais longa, que é a nota cantada de verdade.
            if note.length > notes[-1].length:
                notes[-1] = note
            continue
        notes.append(note)
    # Uma nota não pode invadir a seguinte.
    for note, nxt in zip(notes, notes[1:]):
        note.length = max(1, min(note.length, nxt.tick - note.tick))
    return notes


def _audio_phrase_starts(notes: list[VocalNote], gap_ticks: float) -> list[bool]:
    """True para cada nota que começa depois de uma pausa (início de frase pela voz)."""
    return [i == 0 or n.tick - (notes[i - 1].tick + notes[i - 1].length) >= gap_ticks
            for i, n in enumerate(notes)]


def _join(first: str, second: str) -> str:
    """Junta duas sílabas numa nota só ("ca-" + "mi-" -> "cami-"; "da" + "ma-" -> "da ma-")."""
    return first[:-1] + second if first.endswith("-") else f"{first} {second}"


def align_lyrics(notes: list[VocalNote], lines: list[list[str]],
                 phrase_start: list[bool]) -> tuple[list[str | None], list[int]]:
    """Distribui as sílabas pelas notas (programação dinâmica).

    Retorna, para cada nota, o texto (ou None se a nota for descartada) e o
    número da linha da letra a que ela pertence.
    """
    syllables = [s for line in lines for s in line]
    line_of = [i for i, line in enumerate(lines) for _ in line]
    line_start = [j == 0 or line_of[j] != line_of[j - 1] for j in range(len(syllables))]
    n, m = len(notes), len(syllables)
    INF = float("inf")
    cost = np.full((n + 1, m + 1), INF)
    move = np.zeros((n + 1, m + 1), dtype=np.int8)  # 1 = sílaba, 2 = "+", 3 = duas sílabas, 4 = descarta
    cost[0, 0] = 0.0

    def boundary(i: int, j: int) -> float:
        return COST_LINE_MISMATCH if line_start[j] != phrase_start[i] else 0.0

    for i in range(n):
        for j in range(m + 1):
            here = cost[i, j]
            if here == INF:
                continue
            options = []
            if j < m:
                options.append((here + boundary(i, j), j + 1, 1))
            if j > 0:
                options.append((here + COST_MELISMA + (COST_LINE_MISMATCH if phrase_start[i] else 0.0), j, 2))
            if j + 1 < m:
                extra = COST_LINE_MISMATCH if line_start[j + 1] else 0.0
                options.append((here + COST_MERGE + boundary(i, j) + extra, j + 2, 3))
            options.append((here + COST_SKIP_NOTE, j, 4))
            for total, nj, kind in options:
                if total < cost[i + 1, nj]:
                    cost[i + 1, nj] = total
                    move[i + 1, nj] = kind

    # Volta pelo caminho mais barato.
    texts: list[str | None] = [None] * n
    lines_of_notes = [0] * n
    i, j = n, m
    if cost[n, m] == INF:  # letra maior do que o possível: usa o máximo de sílabas que couber
        j = int(np.argmin(cost[n]))
    while i > 0:
        kind = move[i, j]
        i -= 1
        if kind == 1:
            j -= 1
            texts[i], lines_of_notes[i] = syllables[j], line_of[j]
        elif kind == 2:
            texts[i], lines_of_notes[i] = "+", line_of[j - 1]
        elif kind == 3:
            j -= 2
            texts[i], lines_of_notes[i] = _join(syllables[j], syllables[j + 1]), line_of[j]
        else:
            lines_of_notes[i] = line_of[j - 1] if j > 0 else 0
    return texts, lines_of_notes


def build_vocal_track(sung: list[SungNote], lines: list[list[str]], tempo: TempoMap,
                      resolution: int, allow_triplets: bool, cfg: VocalRules) -> VocalTrack:
    notes = _to_grid(sung, tempo, resolution, allow_triplets)
    starts = _audio_phrase_starts(notes, cfg.phrase_gap_beats * resolution)
    texts, line_numbers = align_lyrics(notes, lines, starts)

    kept = [(note, text, line) for note, text, line in zip(notes, texts, line_numbers) if text is not None]
    for note, text, _ in kept:
        note.lyric = text

    # Cada linha da letra vira uma frase (do início da 1ª nota ao fim da última).
    phrases = []
    for line in sorted({line for _, _, line in kept}):
        members = [note for note, _, ln in kept if ln == line]
        start = members[0].tick
        phrases.append((start, members[-1].tick + members[-1].length - start))
    return VocalTrack(notes=[note for note, _, _ in kept], phrases=phrases)
