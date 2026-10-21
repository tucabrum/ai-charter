"""Sanitização: simplifica compassos em que o programa teve dificuldade.

Gravações reais têm chiado, cordas soltas soando sem querer, raspadas... Nesses
trechos a transcrição fica "suja": palhetadas faltando, notas fora do ritmo e
acordes trocados. Em vez de passar isso para a chart, o programa decide:

  1. Aprende o RITMO TÍPICO da faixa (as posições das notas no compasso que
     mais se repetem nos compassos limpos) e quais ACORDES aparecem nela.
  2. Mede, em cada compasso, o RUÍDO de fundo (energia aguda que fica entre as
     notas, comparada com o resto da faixa) e quanto o ritmo detectado foge
     do típico (palhetadas faltando ou notas fora do lugar).
  3. Compasso com ruído alto E ritmo bagunçado é "confuso" e vira um caminho
     regular: o ritmo típico, com um acorde só (o mais provável entre os acordes
     conhecidos). Em trechos de notas simples (solo), sem ritmo típico, só tira
     as notas fora do grid de colcheias - nunca inventa notas.

Um compasso limpo nunca é mexido, mesmo que seja diferente do resto (ex.: uma
pausa ou um acorde só no final da música).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace

import librosa
import numpy as np

from ai_charter.analysis.chords import is_single_note
from ai_charter.analysis.tempo import TempoMap
from ai_charter.charting.common import Candidate
from ai_charter.config import SanitizeRules


def _noise_by_measure(audio: np.ndarray, sr: int, tempo: TempoMap, measure: int,
                      measures: list[int]) -> dict[int, float]:
    """Ruído de fundo agudo de cada compasso, relativo à mediana da faixa (1.0 = normal)."""
    spec = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    high, total = spec[freqs >= 3000].mean(axis=0), spec.mean(axis=0)
    level = {}
    for m in measures:
        a, b = (int(t * sr / 256) for t in tempo.tick_to_time([m * measure, (m + 1) * measure]))
        if b - a < 4:
            continue
        # Piso agudo (o que fica entre as notas) em relação aos picos do compasso.
        level[m] = np.percentile(high[a:b], 10) / (np.percentile(total[a:b], 90) + 1e-12)
    reference = np.median(list(level.values())) if level else 0.0
    return {m: v / (reference + 1e-12) for m, v in level.items()}


def sanitize(cands: list[Candidate], audio: np.ndarray, sr: int, tempo: TempoMap,
             resolution: int, rules: SanitizeRules) -> tuple[list[Candidate], list[int]]:
    """Devolve as notas com os compassos confusos simplificados, e quais compassos foram."""
    if not rules.enabled or not cands:
        return cands, []
    measure = resolution * 4
    by_measure: dict[int, list[Candidate]] = defaultdict(list)
    for c in cands:
        by_measure[c.tick // measure].append(c)

    # 1. Ritmo típico: o conjunto de posições que mais se repete.
    patterns = Counter(tuple(sorted({c.tick % measure for c in notes}))
                       for notes in by_measure.values() if len(notes) >= 2)
    groove = None
    if patterns:
        pattern, count = patterns.most_common(1)[0]
        if count >= rules.min_groove_measures:
            groove = set(pattern)

    # 2. Ruído e "bagunça" de cada compasso.
    noise = _noise_by_measure(audio, sr, tempo, measure, sorted(by_measure))
    confused = []
    for m, notes in by_measure.items():
        if noise.get(m, 0.0) < rules.noise_ratio:
            continue
        positions = {c.tick % measure for c in notes}
        if groove:
            found = len(positions & groove) / len(groove)
            extra = len(positions - groove) / len(positions)
        else:
            found, extra = 1.0, sum(p % (resolution // 2) != 0 for p in positions) / len(positions)
        if found < rules.min_found or extra > rules.max_extra:
            confused.append(m)
    if not confused:
        return cands, []

    # Acordes conhecidos: os que aparecem nos compassos limpos.
    clean = [c for m, notes in by_measure.items() if m not in confused for c in notes]
    known = Counter(c.chord for c in clean if c.chord >= 0 and not is_single_note(c.chord))
    example = {c.chord: c for c in clean if c.chord in known}
    known_chords = [chord for chord, n in known.items() if n >= rules.min_groove_measures]
    default_strength = float(np.median([c.strength for c in clean])) if clean else 0.5

    # 3. Caminho regular para cada compasso confuso.
    previous_chord = None
    result = [c for m, notes in by_measure.items() if m not in confused for c in notes]
    for m in sorted(by_measure):
        notes = by_measure[m]
        chords_here = [c for c in notes if c.chord_scores is not None]
        mostly_chords = chords_here and sum(not is_single_note(c.chord) for c in chords_here) * 2 > len(chords_here)
        if m not in confused:
            if mostly_chords:
                previous_chord = Counter(c.chord for c in chords_here).most_common(1)[0][0]
            continue
        if groove and known_chords and (mostly_chords or not chords_here):
            if chords_here:
                total = sum(c.chord_scores for c in chords_here)
                chord = max(known_chords, key=lambda k: total[k])
            else:
                chord = previous_chord if previous_chord in example else known_chords[0]
            model = example[chord]
            existing = {c.tick % measure: c for c in notes}
            for position in sorted(groove):
                tick = m * measure + position
                base = existing.get(position) or Candidate(tick, default_strength, end_tick=tick)
                result.append(replace(base, pitch=model.pitch, voices=model.voices, chord=chord,
                                      chord_scores=model.chord_scores))
            previous_chord = chord
        else:
            # Notas simples: só tira o que está fora do grid de colcheias.
            result += [c for c in notes if c.tick % (resolution // 2) == 0]
    return sorted(result, key=lambda c: c.tick), sorted(confused)
