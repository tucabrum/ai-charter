"""Compara uma chart gerada com uma chart de referência (ex.: a oficial do jogo).

    python -m ai_charter.compare oficial/notes.mid gerada/notes.mid

Charts são subjetivas: dois charters podem usar cores diferentes para a mesma
música e as duas estarem certas. Por isso a comparação NÃO olha se a nota é
verde ou vermelha. Ela mede:

  - tempo: quantas notas da referência aparecem na chart gerada no mesmo
    momento (recall) e quantas notas geradas existem na referência (precisão);
  - acorde x nota simples: nas notas em comum, se as duas usam acorde juntas;
  - contorno: entre duas notas seguidas, se as duas charts sobem, descem ou
    repetem juntas (sem olhar a cor exata);
  - sustains: se as notas longas coincidem;
  - densidade: notas por segundo;
  - vocais: tempo, afinação (também ignorando a oitava), sílaba e frases.

Antes de comparar, as duas charts são alinhadas no tempo (as versões podem ter
introduções diferentes).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import mido
import numpy as np

DIFFICULTIES = {"expert": 96, "hard": 84, "medium": 72, "easy": 60}
INSTRUMENTS = ("PART GUITAR", "PART BASS", "PART DRUMS")
VOCAL_RANGE = (36, 84)
PHRASE = 105


# ------------------------------------------------------------------ leitura
@dataclass
class Event:
    time: float                 # segundos
    lanes: tuple[int, ...]      # 0 = verde/bumbo ... 4 = laranja/verde
    length: float = 0.0         # segundos


@dataclass
class VocalNote:
    time: float
    length: float
    pitch: int
    lyric: str = ""


@dataclass
class Chart:
    tracks: dict[str, dict[str, list[Event]]] = field(default_factory=dict)  # instrumento -> dif -> eventos
    vocals: list[VocalNote] = field(default_factory=list)
    phrases: list[tuple[float, float]] = field(default_factory=list)
    sections: list[tuple[float, str]] = field(default_factory=list)


def _tempo_function(midi: mido.MidiFile):
    """Função tick -> segundos, a partir dos eventos de andamento."""
    changes = [(0, 500000)]
    for track in midi.tracks:
        now = 0
        for msg in track:
            now += msg.time
            if msg.type == "set_tempo":
                changes.append((now, msg.tempo))
    changes.sort()
    ticks, seconds, tempos = [0], [0.0], [changes[0][1]]
    for tick, tempo in changes[1:]:
        seconds.append(seconds[-1] + (tick - ticks[-1]) * tempos[-1] / 1e6 / midi.ticks_per_beat)
        ticks.append(tick)
        tempos.append(tempo)
    ticks_a, seconds_a, tempos_a = np.array(ticks), np.array(seconds), np.array(tempos)

    def to_seconds(tick: float) -> float:
        i = int(np.searchsorted(ticks_a, tick, side="right") - 1)
        return float(seconds_a[i] + (tick - ticks_a[i]) * tempos_a[i] / 1e6 / midi.ticks_per_beat)
    return to_seconds


def _notes(track) -> list[tuple[int, int, int]]:
    """(tick, nota, duração em ticks) de uma trilha."""
    now, open_notes, out = 0, {}, []
    for msg in track:
        now += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            open_notes.setdefault(msg.note, []).append(now)
        elif msg.type in ("note_off", "note_on") and open_notes.get(msg.note):
            start = open_notes[msg.note].pop(0)
            out.append((start, msg.note, now - start))
    return out


def read_chart(path: Path) -> Chart:
    midi = mido.MidiFile(path)
    seconds = _tempo_function(midi)
    chart = Chart()
    for track in midi.tracks:
        name = next((m.name for m in track if m.type == "track_name"), "")
        if name in INSTRUMENTS:
            width = 6 if name == "PART DRUMS" else 5   # bateria de 5 lanes usa uma nota a mais
            by_diff = {}
            for diff, base in DIFFICULTIES.items():
                grouped: dict[int, list] = {}
                for tick, note, length in _notes(track):
                    if base <= note < base + width:
                        grouped.setdefault(tick, []).append((note - base, length))
                by_diff[diff] = [
                    Event(seconds(t), tuple(sorted(l for l, _ in v)), seconds(t + max(n for _, n in v)) - seconds(t))
                    for t, v in sorted(grouped.items())
                ]
            chart.tracks[name] = by_diff
        elif name == "PART VOCALS":
            lyrics, now = {}, 0
            for msg in track:
                now += msg.time
                if msg.type in ("lyrics", "text") and not msg.text.startswith("["):
                    lyrics[now] = msg.text
            for tick, note, length in _notes(track):
                if VOCAL_RANGE[0] <= note <= VOCAL_RANGE[1]:
                    start = seconds(tick)
                    chart.vocals.append(VocalNote(start, seconds(tick + length) - start, note, lyrics.get(tick, "")))
                elif note == PHRASE:
                    chart.phrases.append((seconds(tick), seconds(tick + length)))
            chart.vocals.sort(key=lambda v: v.time)
            chart.phrases.sort()
        elif name == "EVENTS":
            now = 0
            for msg in track:
                now += msg.time
                if msg.type == "text" and msg.text.startswith("[section"):
                    chart.sections.append((seconds(now), msg.text[9:-1].strip()))
    return chart


# ------------------------------------------------------------------ comparação
def match(reference: list[float], generated: list[float], tolerance: float) -> list[tuple[int, int]]:
    """Pares (i, j) de notas no mesmo momento (cada nota usada uma vez só)."""
    pairs, j = [], 0
    used = set()
    for i, t in enumerate(reference):
        while j < len(generated) and generated[j] < t - tolerance:
            j += 1
        best, k = None, j
        while k < len(generated) and generated[k] <= t + tolerance:
            if k not in used and (best is None or abs(generated[k] - t) < abs(generated[best] - t)):
                best = k
            k += 1
        if best is not None:
            used.add(best)
            pairs.append((i, best))
    return pairs


def best_offset(reference: list[float], generated: list[float], search: float = 3.0,
                tolerance: float = 0.05) -> float:
    """Deslocamento (s) a somar na chart gerada para alinhar com a referência."""
    if not reference or not generated:
        return 0.0
    ref = np.array(reference)
    gen = np.array(generated)

    def score(offset: float) -> tuple[int, float]:
        """(notas que encaixam, -erro médio delas): mais notas, e no empate, o menor erro."""
        idx = np.clip(np.searchsorted(ref, gen + offset), 1, len(ref) - 1)
        nearest = np.minimum(np.abs(ref[idx] - gen - offset), np.abs(ref[idx - 1] - gen - offset))
        close = nearest <= tolerance
        return int(np.sum(close)), -float(np.mean(nearest[close])) if close.any() else 0.0

    coarse = max(np.arange(-search, search, 0.01), key=score)
    return float(max(np.arange(coarse - 0.02, coarse + 0.02, 0.002), key=score))


def _direction(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return int(np.sign(np.mean(b) - np.mean(a)))


@dataclass
class TrackScore:
    reference: int
    generated: int
    matched: int
    timing_ms: float
    chord_agree: float           # % das notas em comum com a mesma decisão acorde/nota
    ref_chords_as_single: int    # acorde na referência, nota simples na gerada
    single_as_chord: int
    contour_agree: float         # % de pares seguidos com a mesma direção (sobe/desce/repete)
    sustain_agree: float
    ref_density: float
    gen_density: float

    @property
    def recall(self) -> float:
        return self.matched / self.reference if self.reference else 0.0

    @property
    def precision(self) -> float:
        return self.matched / self.generated if self.generated else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def score_track(ref: list[Event], gen: list[Event], tolerance: float, duration: float,
                sustain_seconds: float = 0.3) -> TrackScore:
    pairs = match([e.time for e in ref], [e.time for e in gen], tolerance)
    timing = [abs(ref[i].time - gen[j].time) * 1000 for i, j in pairs]
    chord_same = [(len(ref[i].lanes) > 1) == (len(gen[j].lanes) > 1) for i, j in pairs]
    chords_as_single = sum(len(ref[i].lanes) > 1 and len(gen[j].lanes) == 1 for i, j in pairs)
    single_as_chord = sum(len(ref[i].lanes) == 1 and len(gen[j].lanes) > 1 for i, j in pairs)
    sustain_same = [(ref[i].length >= sustain_seconds) == (gen[j].length >= sustain_seconds) for i, j in pairs]
    contour = []
    for (i1, j1), (i2, j2) in zip(pairs, pairs[1:]):
        if i2 == i1 + 1 and j2 == j1 + 1:   # notas seguidas nas duas charts
            contour.append(_direction(ref[i1].lanes, ref[i2].lanes) == _direction(gen[j1].lanes, gen[j2].lanes))
    pct = lambda v: 100.0 * float(np.mean(v)) if v else 0.0  # noqa: E731
    return TrackScore(
        reference=len(ref), generated=len(gen), matched=len(pairs),
        timing_ms=float(np.median(timing)) if timing else 0.0,
        chord_agree=pct(chord_same), ref_chords_as_single=chords_as_single, single_as_chord=single_as_chord,
        contour_agree=pct(contour), sustain_agree=pct(sustain_same),
        ref_density=len(ref) / duration if duration else 0.0, gen_density=len(gen) / duration if duration else 0.0,
    )


def _clean_lyric(text: str) -> str:
    return re.sub(r"[^a-z0-9à-ü']", "", text.lower())


@dataclass
class VocalScore:
    reference: int
    generated: int
    matched: int
    pitch_exact: float
    pitch_class: float
    pitch_shift: int             # deslocamento constante mais comum (gerada - referência, semitons)
    pitch_after_shift: float     # afinação exata depois de descontar esse deslocamento
    lyric_agree: float
    ref_phrases: int
    gen_phrases: int
    phrase_starts_matched: int
    generated_in_unsung: int     # notas geradas em frases que a referência não pontua (faladas)


def score_vocals(ref: Chart, gen: Chart, offset: float, tolerance: float) -> VocalScore:
    gen_notes = [VocalNote(v.time + offset, v.length, v.pitch, v.lyric) for v in gen.vocals]
    pairs = match([v.time for v in ref.vocals], [v.time for v in gen_notes], tolerance)
    exact = [ref.vocals[i].pitch == gen_notes[j].pitch for i, j in pairs]
    # Gravações fora do padrão A=440 deixam a voz "entre" dois semitons: as duas charts
    # podem arredondar para lados diferentes. Desconta o deslocamento mais comum.
    diffs = [gen_notes[j].pitch - ref.vocals[i].pitch for i, j in pairs]
    shift = max(set(diffs), key=diffs.count) if diffs else 0
    after_shift = [d == shift for d in diffs]
    pclass = [ref.vocals[i].pitch % 12 == gen_notes[j].pitch % 12 for i, j in pairs]
    lyric = [_clean_lyric(ref.vocals[i].lyric) == _clean_lyric(gen_notes[j].lyric) for i, j in pairs]
    gen_starts = [a + offset for a, _ in gen.phrases]
    starts = sum(any(abs(a - g) <= 0.3 for g in gen_starts) for a, _ in ref.phrases)
    sung = [(a, b) for a, b in ref.phrases if any(a <= v.time < b for v in ref.vocals)]
    unsung = [(a, b) for a, b in ref.phrases if (a, b) not in sung]
    in_unsung = sum(any(a <= v.time < b for a, b in unsung) for v in gen_notes)
    pct = lambda v: 100.0 * float(np.mean(v)) if v else 0.0  # noqa: E731
    return VocalScore(len(ref.vocals), len(gen_notes), len(pairs), pct(exact), pct(pclass),
                      shift, pct(after_shift), pct(lyric),
                      len(ref.phrases), len(gen.phrases), starts, in_unsung)


# ------------------------------------------------------------------ relatório
NAMES = {"PART GUITAR": "Guitarra", "PART BASS": "Baixo", "PART DRUMS": "Bateria"}


def compare(reference: Path, generated: Path, tolerance: float = 0.05) -> str:
    ref, gen = read_chart(reference), read_chart(generated)
    key = next((name for name in INSTRUMENTS if name in ref.tracks and name in gen.tracks), None)
    offset = best_offset([e.time for e in ref.tracks[key]["expert"]],
                         [e.time for e in gen.tracks[key]["expert"]]) if key else 0.0
    for track in gen.tracks.values():
        for events in track.values():
            for e in events:
                e.time += offset
    duration = max((e.time for t in ref.tracks.values() for evs in t.values() for e in evs), default=0.0)

    lines = [f"Alinhamento: chart gerada deslocada {offset * 1000:+.0f} ms", ""]
    for name in INSTRUMENTS:
        if name not in ref.tracks or name not in gen.tracks:
            continue
        lines.append(f"## {NAMES[name]}")
        lines.append("| Dificuldade | Ref. | Gerada | Em comum | Recall | Precisão | F1 | Erro (ms) "
                     "| Acorde x nota | Contorno | Sustain | Notas/s (ref → gerada) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for diff in DIFFICULTIES:
            s = score_track(ref.tracks[name][diff], gen.tracks[name][diff], tolerance, duration)
            lines.append(
                f"| {diff} | {s.reference} | {s.generated} | {s.matched} | {s.recall:.0%} | {s.precision:.0%} "
                f"| {s.f1:.0%} | {s.timing_ms:.0f} | {s.chord_agree:.0f}% (acorde→nota {s.ref_chords_as_single}, "
                f"nota→acorde {s.single_as_chord}) | {s.contour_agree:.0f}% | {s.sustain_agree:.0f}% "
                f"| {s.ref_density:.1f} → {s.gen_density:.1f} |")
        if ref.sections:
            lines.append("")
            lines.append(f"Expert por seção ({NAMES[name]}):")
            lines.append("| Seção | Ref. | Gerada | Recall | Precisão |")
            lines.append("|---|---|---|---|---|")
            bounds = ref.sections + [(float("inf"), "")]
            for (start, title), (end, _) in zip(bounds, bounds[1:]):
                r = [e for e in ref.tracks[name]["expert"] if start <= e.time < end]
                g = [e for e in gen.tracks[name]["expert"] if start <= e.time < end]
                s = score_track(r, g, tolerance, max(end - start, 1e-9) if end != float("inf") else 1.0)
                lines.append(f"| {title} | {s.reference} | {s.generated} | {s.recall:.0%} | {s.precision:.0%} |")
        lines.append("")
    if ref.vocals and gen.vocals:
        v = score_vocals(ref, gen, offset, tolerance=0.1)
        lines += [
            "## Vocais",
            f"- Notas cantadas: referência {v.reference}, gerada {v.generated}, em comum {v.matched} "
            f"(recall {v.matched / max(v.reference, 1):.0%}, precisão {v.matched / max(v.generated, 1):.0%})",
            f"- Afinação nas notas em comum: exata {v.pitch_exact:.0f}%, mesma nota ignorando a oitava {v.pitch_class:.0f}%; "
            f"deslocamento constante mais comum {v.pitch_shift:+d} semitom(ns) -> descontando-o, {v.pitch_after_shift:.0f}% iguais",
            f"- Sílaba certa nas notas em comum: {v.lyric_agree:.0f}%",
            f"- Frases: referência {v.ref_phrases}, gerada {v.gen_phrases}, "
            f"inícios de frase coincidentes {v.phrase_starts_matched}",
            f"- Notas geradas em frases que a referência não pontua (faladas): {v.generated_in_unsung}",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compara uma chart gerada (notes.mid) com uma de referência.")
    parser.add_argument("reference", type=Path, help="notes.mid de referência (ex.: oficial)")
    parser.add_argument("generated", type=Path, help="notes.mid gerado pelo ai-charter")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="diferença máxima (s) para duas notas contarem como a mesma (padrão 0.05)")
    args = parser.parse_args(argv)
    print(compare(args.reference, args.generated, args.tolerance))


if __name__ == "__main__":
    main()
