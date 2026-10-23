"""Monta a faixa de vocal: notas no grid, frases e a letra sílaba por sílaba.

O alinhamento da letra tem dois níveis:
  1. LINHAS -> TRECHOS DE VOZ: a voz é dividida em trechos pela energia (canto e
     fala), e cada linha da letra vai para o trecho cuja duração combina com a
     quantidade de sílabas, sempre em ordem. Assim a letra não "escorrega".
  2. Dentro de cada linha: se há notas cantadas suficientes, a linha é CANTADA e
     cada sílaba vai para uma nota; senão, ela é FALADA: a letra aparece na tela
     sem nota para cantar (como o Rock Band/Guitar Hero fazem com trechos falados).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np

from ai_charter.analysis.tempo import TempoMap
from ai_charter.analysis.vocals import SungNote, VoiceSegment
from ai_charter.charting.common import quantize_tick
from ai_charter.config import VocalRules

# Faixa de notas aceita pelos jogos na trilha de vocal (MIDI).
MIN_PITCH, MAX_PITCH = 36, 84

# Custos do alinhamento letra x notas (quanto maior, mais o algoritmo evita).
COST_LINE_MISMATCH = 2.0   # começo de linha da letra fora de uma pausa da voz (ou vice-versa)
COST_MELISMA = 1.5         # nota extra vira "+" (a sílaba anterior continua)
COST_MERGE = 2.5           # duas sílabas numa nota só
COST_SKIP_NOTE = 3.0       # nota descartada (provavelmente ruído)

# Custos do alinhamento linhas x trechos de voz.
COST_EXTRA_SEGMENT = 0.6   # cada trecho a mais juntado numa linha
COST_EXTRA_LINE = 0.5      # cada linha a mais no mesmo trecho (refrões sem pausa têm muitas)
COST_SKIP_SEGMENT = 1.0    # trecho de voz sem letra (grito, eco, voz de apoio)
COST_SKIP_LINE = 3.0       # linha da letra sem trecho de voz
MAX_SEGMENTS_PER_LINE = 4
MAX_LINES_PER_SEGMENT = 40


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
    spoken: list[tuple[int, str]] = field(default_factory=list)   # letra sem nota (trechos falados)


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


def align_lines_to_segments(lines: list[list[str]],
                            segments: list[VoiceSegment]) -> list[tuple[float, float] | None]:
    """Para cada linha da letra, o intervalo de tempo (s) em que ela é cantada/falada.

    Programação dinâmica sobre (linhas, trechos): uma linha pode ocupar vários
    trechos seguidos, ou várias linhas um trecho só; o custo mede o quanto o número
    de sílabas combina com o número de ATAQUES de voz no trecho (cada sílaba,
    cantada ou falada, costuma gerar um ataque; a duração não serve, porque fala
    é bem mais rápida que canto).
    """
    n_lines, n_segs = len(lines), len(segments)
    if not n_lines or not n_segs:
        return [None] * n_lines
    syllables = [len(line) for line in lines]
    attacks = [len(s.onsets) for s in segments]
    # Proporção sílabas/ataques da música (às vezes um ataque cobre 2 sílabas, ou vice-versa).
    ratio = sum(syllables) / max(sum(attacks), 1)

    def fit(n_syll: int, n_attacks: int) -> float:
        return abs(np.log((n_syll + 1) / (n_attacks * ratio + 1)))

    INF = float("inf")
    cost = np.full((n_lines + 1, n_segs + 1), INF)
    back: dict[tuple[int, int], tuple[int, int]] = {}
    cost[0, 0] = 0.0
    for j in range(n_lines + 1):
        for i in range(n_segs + 1):
            here = cost[j, i]
            if here == INF:
                continue
            moves = []
            if i < n_segs:
                moves.append((j, i + 1, COST_SKIP_SEGMENT))
            if j < n_lines:
                moves.append((j + 1, i, COST_SKIP_LINE))
            for q in range(1, MAX_SEGMENTS_PER_LINE + 1):          # 1 linha, q trechos
                if j < n_lines and i + q <= n_segs:
                    moves.append((j + 1, i + q, fit(syllables[j], sum(attacks[i:i + q]))
                                  + COST_EXTRA_SEGMENT * (q - 1)))
            for p in range(2, MAX_LINES_PER_SEGMENT + 1):           # p linhas, 1 trecho
                if j + p <= n_lines and i < n_segs:
                    moves.append((j + p, i + 1, fit(sum(syllables[j:j + p]), attacks[i])
                                  + COST_EXTRA_LINE * (p - 1)))
            for nj, ni, step in moves:
                if here + step < cost[nj, ni]:
                    cost[nj, ni] = here + step
                    back[(nj, ni)] = (j, i)

    spans: list[tuple[float, float] | None] = [None] * n_lines
    j, i = n_lines, n_segs
    while (j, i) != (0, 0):
        pj, pi = back[(j, i)]
        if j > pj and i > pi:
            start, end = segments[pi].start, segments[i - 1].end
            if j - pj == 1:
                spans[pj] = (start, end)
            else:   # várias linhas num trecho: corta nas pausas internas
                bounds = [start] + _cut_points(segments[pi], syllables[pj:j]) + [end]
                for k, (a, b) in zip(range(pj, j), zip(bounds, bounds[1:])):
                    spans[k] = (a, b)
        j, i = pj, pi
    return spans


def _cut_points(segment: VoiceSegment, syllables: list[int]) -> list[float]:
    """Onde cortar um trecho com várias linhas: nas pausas internas que melhor
    combinam com a posição esperada de cada corte (pela proporção de sílabas),
    preferindo as pausas mais longas. Sem pausa por perto, corta na posição esperada.
    """
    duration = segment.end - segment.start
    total, targets, t = sum(syllables), [], segment.start
    for n in syllables[:-1]:
        t += duration * n / total
        targets.append(t)
    pauses = [((a + b) / 2, b - a) for a, b in segment.pauses]
    if not pauses:
        return targets
    # Programação dinâmica: cada corte usa uma pausa diferente, em ordem (ou fica no alvo).
    n_t, n_p = len(targets), len(pauses)
    far = 0.15 * duration                      # pausa longe demais do alvo não serve
    INF = float("inf")
    cost = np.full((n_t + 1, n_p + 1), INF)
    choice: dict[tuple[int, int], tuple[int, int, float]] = {}
    cost[0, :] = 0.0
    for k in range(1, n_t + 1):
        for q in range(n_p + 1):
            # corte k sem pausa (fica no alvo)
            if cost[k - 1, q] + 1.0 < cost[k, q]:
                cost[k, q] = cost[k - 1, q] + 1.0
                choice[(k, q)] = (k - 1, q, targets[k - 1])
            # corte k na pausa q-1
            if q >= 1:
                mid, length = pauses[q - 1]
                miss = abs(mid - targets[k - 1])
                if miss <= far:
                    step = miss / far - min(length, 0.5)
                    best_prev = min(range(q), key=lambda r: cost[k - 1, r])
                    total_cost = cost[k - 1, best_prev] + step
                    if total_cost < cost[k, q]:
                        cost[k, q] = total_cost
                        choice[(k, q)] = (k - 1, best_prev, mid)
    q = int(np.argmin(cost[n_t]))
    cuts, k = [], n_t
    while k > 0:
        k, q, at = choice[(k, q)]
        cuts.append(at)
    return sorted(cuts)


def _spread(times: list[float], start: float, end: float, count: int) -> list[float]:
    """`count` instantes para as sílabas faladas: os ataques do trecho, ou espaçados por igual."""
    inside = [t for t in times if start <= t < end]
    if len(inside) >= count:
        picks = np.linspace(0, len(inside) - 1, count).round().astype(int)
        return [inside[k] for k in picks]
    return list(np.linspace(start, start + (end - start) * (count - 1) / max(count, 1), count))


VOCAL_GRID = 8          # divisões da batida para a voz (fusas): a voz não é tão presa ao grid
MELISMA_SECONDS = 0.1   # nota nova dentro da sílaba, depois disso, vira "+" (melisma)
SNAP_BEFORE = 0.1       # o início da sílaba procura uma nota cantada desde um pouco antes da consoante...
SNAP_AFTER = 0.12       # ...até um pouco depois da vogal


def _pitch_in(sung: list[SungNote], start: float, end: float) -> float | None:
    """Afinação da voz entre start e end: mediana das notas cantadas, pesada pela duração."""
    parts = [(n.pitch, min(n.end, end) - max(n.time, start)) for n in sung if n.time < end and n.end > start]
    parts = [(p, d) for p, d in parts if d > 0]
    if not parts or sum(d for _, d in parts) < 0.3 * (end - start):
        return None
    parts.sort()
    half, acc = sum(d for _, d in parts) / 2, 0.0
    for pitch, d in parts:
        acc += d
        if acc >= half:
            return pitch
    return parts[-1][0]


def _fit_pitch(pitch: float) -> int:
    pitch = int(round(pitch))
    while pitch < MIN_PITCH:
        pitch += 12
    while pitch > MAX_PITCH:
        pitch -= 12
    return pitch


def build_aligned_track(spans: list[list], sung: list[SungNote], lines: list[list[str]],
                        tempo: TempoMap, resolution: int, cfg: VocalRules) -> VocalTrack:
    """Faixa de vocal a partir do alinhamento forçado (analysis/align.py): cada sílaba começa onde
    o alinhamento a ouviu, e a afinação vem das notas cantadas (pYIN) dentro dela. Cada linha
    da letra é uma frase. Notas cantadas que mudam de altura no meio da sílaba viram "+".

    Garante o que o jogo exige: uma sílaba por tick, notas sem sobreposição, frases sem sobreposição.
    """
    step = max(1, resolution // VOCAL_GRID)
    flat = [(li, si) for li, line in enumerate(lines) for si in range(len(line))]
    # Sílabas sem tempo (sem letra que o modelo conheça): logo depois da anterior.
    times: list[tuple[float, float]] = []
    onsets = [n.time for n in sung]
    for li, si in flat:
        span = spans[li][si]
        if span is None:
            t = times[-1][1] if times else 0.0
            times.append((t, t + 0.05))
            continue
        # A nota começa onde a voz começa a soar afinada: o primeiro início de nota cantada entre a
        # consoante e um pouco depois da vogal. Sem nenhum, a vogal.
        vowel = span.vowel if span.vowel is not None else span.start
        lo = max(span.start - SNAP_BEFORE, times[-1][0] + 0.02 if times else 0.0)
        k = bisect.bisect_left(onsets, lo)
        start = onsets[k] if k < len(onsets) and onsets[k] <= vowel + SNAP_AFTER else vowel
        times.append((start, max(span.end, start)))
    next_start = [times[k + 1][0] for k in range(len(times) - 1)] + [float("inf")]

    def to_tick(t: float) -> int:
        return int(round(tempo.time_to_tick(t) / step)) * step

    track = VocalTrack(notes=[], phrases=[])
    last_tick = -step
    k = 0
    for li, line in enumerate(lines):
        first_k, k = k, k + len(line)
        # Altura e fim de cada sílaba: as notas cantadas até a próxima sílaba.
        sylls = []
        for j in range(first_k, k):
            start, end = times[j]
            limit = min(next_start[j], start + 4.0)
            inside = [n for n in sung if n.time < limit and n.end > start]
            end = max(end, min(limit, max((n.end for n in inside), default=end)))
            sylls.append((start, end, _pitch_in(sung, start, end), inside))
        voiced = [p for _, _, p, _ in sylls if p is not None]
        if not line:
            continue
        if len(voiced) >= cfg.sung_min_ratio * len(line):
            notes: list[VocalNote] = []
            for si, (start, end, pitch, inside) in enumerate(sylls):
                if pitch is None:   # consoante/sussurro no meio do canto: altura da vizinha
                    near = [p for _, _, p, _ in sylls[si + 1:] + sylls[:si][::-1] if p is not None]
                    pitch = near[0] if near else 60.0
                tick = max(to_tick(start), last_tick + step)
                pieces = [(tick, _fit_pitch(pitch), line[si])]
                # Melisma: nota de outra altura que começa depois do começo da sílaba.
                for n in inside:
                    if n.time >= start + MELISMA_SECONDS and n.time < end \
                            and abs(_fit_pitch(n.pitch) - pieces[-1][1]) >= 1:
                        t = max(to_tick(n.time), pieces[-1][0] + step)
                        pieces.append((t, _fit_pitch(n.pitch), "+"))
                end_tick = max(to_tick(end), pieces[-1][0] + step)
                for (t, p, text), nxt in zip(pieces, pieces[1:] + [(end_tick, 0, "")]):
                    notes.append(VocalNote(t, nxt[0] - t, p, text))
                last_tick = pieces[-1][0]
            track.notes += notes
            track.phrases.append((notes[0].tick, notes[-1].tick + notes[-1].length - notes[0].tick))
        else:   # linha falada: só a letra, sem nota para cantar
            ticks = []
            for start, *_ in sylls:
                ticks.append(max(to_tick(start), last_tick + step))
                last_tick = ticks[-1]
            track.spoken += list(zip(ticks, line))
            track.phrases.append((ticks[0], ticks[-1] + step - ticks[0]))
    _no_overlaps(track)
    return track


def _no_overlaps(track: VocalTrack) -> None:
    """Uma nota não pode invadir a seguinte, nem uma frase a seguinte."""
    for note, nxt in zip(track.notes, track.notes[1:]):
        note.length = max(1, min(note.length, nxt.tick - note.tick))
    track.phrases.sort()
    fixed = []
    for (tick, length), nxt in zip(track.phrases, track.phrases[1:] + [(None, None)]):
        if nxt[0] is not None:
            length = max(1, min(length, nxt[0] - tick))
        fixed.append((tick, length))
    track.phrases = fixed


def build_vocal_track(sung: list[SungNote], segments: list[VoiceSegment], lines: list[list[str]],
                      tempo: TempoMap, resolution: int, allow_triplets: bool, cfg: VocalRules) -> VocalTrack:
    spans = align_lines_to_segments(lines, segments)
    onsets = sorted(t for s in segments for t in s.onsets)
    track = VocalTrack(notes=[], phrases=[])
    sixteenth = resolution // 4
    for line, span in zip(lines, spans):
        if span is None:
            continue
        start, end = span
        inside = [s for s in sung if start - 0.05 <= s.time < end]
        if len(inside) >= cfg.sung_min_ratio * len(line):
            # Linha cantada: cada sílaba numa nota.
            notes = _to_grid(inside, tempo, resolution, allow_triplets)
            starts = [True] + [False] * (len(notes) - 1)
            texts, _ = align_lyrics(notes, [line], starts)
            kept = [n for n, text in zip(notes, texts) if text is not None]
            for note, text in zip(notes, texts):
                if text is not None:
                    note.lyric = text
            if kept:
                track.notes += kept
                track.phrases.append((kept[0].tick, kept[-1].tick + kept[-1].length - kept[0].tick))
                continue
        # Linha falada: só a letra, sem nota para cantar.
        ticks = []
        for t in _spread(onsets, start, end, len(line)):
            tick = quantize_tick(tempo.time_to_tick(t), resolution, allow_triplets)
            ticks.append(max(tick, ticks[-1] + sixteenth) if ticks else tick)
        track.spoken += list(zip(ticks, line))
        track.phrases.append((ticks[0], ticks[-1] + sixteenth - ticks[0]))
    _no_overlaps(track)   # linhas vizinhas podem se encostar
    return track
