"""Detecção de batidas e mapa de tempo.

No formato .chart, o tempo é medido em "ticks": cada batida (semínima) tem
`resolution` ticks (192 por padrão). O `TempoMap` guarda o instante (em
segundos) de cada batida, e assim consegue converter segundos <-> ticks.
A batida número i fica sempre no tick i * resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.signal import savgol_filter

from ai_charter.analysis.pitched import ONSET_LATENCY

# Se as batidas detectadas se desviam menos que isso (segundos) de uma reta,
# consideramos que a música tem tempo constante (gravada com metrônomo).
CONSTANT_TEMPO_TOLERANCE = 0.015


@dataclass
class TempoMap:
    beat_times: np.ndarray  # instante (s) de cada batida; beat_times[0] é sempre 0.0
    resolution: int = 192

    @property
    def beat_ticks(self) -> np.ndarray:
        return np.arange(len(self.beat_times)) * self.resolution

    def time_to_tick(self, t):
        """Converte segundos em ticks (aceita um número ou um array)."""
        t = np.asarray(t, dtype=float)
        ticks = np.interp(t, self.beat_times, self.beat_ticks)
        # Depois da última batida, continua com o último intervalo.
        last_period = self.beat_times[-1] - self.beat_times[-2]
        after = t > self.beat_times[-1]
        ticks = np.where(
            after,
            self.beat_ticks[-1] + (t - self.beat_times[-1]) / last_period * self.resolution,
            ticks,
        )
        return ticks if ticks.ndim else float(ticks)

    def tick_to_time(self, tick):
        """Converte ticks em segundos (operação inversa de time_to_tick)."""
        return np.interp(tick, self.beat_ticks, self.beat_times)

    def bpm_events(self) -> list[tuple[int, int]]:
        """Lista (tick, BPM×1000) para a seção [SyncTrack] do .chart.

        Só emite um evento quando o BPM realmente muda.
        """
        events = []
        last = None
        periods = np.diff(self.beat_times)
        for i, period in enumerate(periods):
            bpm_milli = int(round(60_000 / period))  # (60 / período) × 1000
            if bpm_milli != last:
                events.append((i * self.resolution, bpm_milli))
                last = bpm_milli
        return events

    @property
    def average_bpm(self) -> float:
        return 60.0 / float(np.median(np.diff(self.beat_times)))


def _anchor_at_zero(beats: np.ndarray, period: float, duration: float) -> np.ndarray:
    """Estende as batidas para cobrir a música toda e começar exatamente em 0 s."""
    beats = list(beats)
    # Estende para trás, até perto do início do áudio.
    while beats[0] - period > 0:
        beats.insert(0, beats[0] - period)
    # A primeira batida precisa estar no tempo 0. Se a próxima batida ficar
    # perto demais de 0, a removemos para não criar um BPM absurdo no começo.
    if beats[0] < 0.5 * period:
        beats.pop(0)
    beats.insert(0, 0.0)
    # Estende para frente, até depois do fim do áudio.
    while beats[-1] < duration + period:
        beats.append(beats[-1] + period)
    return np.array(beats)


def _align_phase(beats: np.ndarray, onsets: np.ndarray, period: float) -> np.ndarray:
    """Desloca o grid de batidas para coincidir com os ataques reais da música.

    Para cada ataque, mede a distância até a colcheia mais próxima do grid;
    a mediana dessas distâncias é o quanto o grid está adiantado/atrasado.
    """
    grid = np.sort(np.concatenate([beats, beats[:-1] + np.diff(beats) / 2]))
    idx = np.clip(np.searchsorted(grid, onsets), 1, len(grid) - 1)
    nearest = np.where(onsets - grid[idx - 1] < grid[idx] - onsets, grid[idx - 1], grid[idx])
    residual = onsets - nearest
    residual = residual[np.abs(residual) < period / 8]
    return beats + (np.median(residual) if len(residual) else 0.0)


def estimate_tempo_map(y: np.ndarray, sr: int, resolution: int = 192,
                       mode: str = "auto") -> TempoMap:
    """Detecta as batidas da música e monta o mapa de tempo.

    mode:
      "auto"     - escolhe sozinho entre constante e variável;
      "constant" - um único BPM para a música toda;
      "variable" - acompanha pequenas variações de andamento (bateria ao vivo).
    """
    duration = len(y) / sr
    hop = 256
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop)
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)

    if len(beats) < 4:
        # Pouquíssimas batidas detectadas: usa 120 BPM fixo.
        period = 0.5
        return TempoMap(_anchor_at_zero(np.array([period]), period, duration), resolution)

    # Ajusta uma reta "tempo = início + período × número_da_batida".
    index = np.arange(len(beats))
    period, offset = np.polyfit(index, beats, 1)
    residual = np.std(beats - (offset + period * index))

    if mode == "constant" or (mode == "auto" and residual < CONSTANT_TEMPO_TOLERANCE):
        smoothed = offset + period * index
    else:
        # Suaviza as batidas (remove tremidas da detecção), mas segue o andamento real.
        window = min(9, len(beats) - (1 - len(beats) % 2))
        smoothed = savgol_filter(beats, window_length=window, polyorder=1) if window >= 3 else beats

    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop, units="time")
    smoothed = _align_phase(smoothed, onsets - ONSET_LATENCY, float(period))
    return TempoMap(_anchor_at_zero(smoothed, float(period), duration), resolution)
