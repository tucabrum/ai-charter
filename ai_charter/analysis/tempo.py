"""Detecção de batidas e mapa de tempo.

No formato .chart, o tempo é medido em "ticks": cada batida (semínima) tem
`resolution` ticks (192 por padrão). O `TempoMap` guarda o instante (em
segundos) de cada batida, e assim consegue converter segundos <-> ticks.
A batida número i fica sempre no tick i * resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
    # Fórmula de compasso: (batida onde começa, batidas por compasso); por padrão 4/4 a música toda
    meters: list = field(default_factory=lambda: [(0, 4)])

    def bar_starts(self) -> np.ndarray:
        """Índices das batidas em que cada compasso começa, seguindo as fórmulas de compasso."""
        starts, changes = [], sorted(self.meters)
        for k, (beat, n) in enumerate(changes):
            end = changes[k + 1][0] if k + 1 < len(changes) else len(self.beat_times)
            starts += list(range(beat, end, max(1, n)))
        return np.array(starts, dtype=int)

    def bar_times(self) -> np.ndarray:
        return np.asarray(self.beat_times)[self.bar_starts()]

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


def _regularize_beats(beats: np.ndarray) -> np.ndarray:
    """Preenche batidas que o detector pulou (numa pausa, num corte) e tira as duplicadas, para que a
    batida i seja mesmo a i-ésima: o ajuste do andamento conta as batidas pela posição na lista, e uma
    batida faltando faria a grade escorregar uma batida dali em diante."""
    if len(beats) < 4:
        return beats
    period = float(np.median(np.diff(beats)))
    out = [beats[0]]
    for b in beats[1:]:
        gap = b - out[-1]
        if gap < 0.5 * period:      # batida duplicada
            continue
        missing = int(round(gap / period)) - 1
        if missing > 0:
            out.extend(out[-1] + gap * np.arange(1, missing + 1) / (missing + 1))
        out.append(b)
    return np.asarray(out, dtype=float)


def _meter(beats: np.ndarray, downbeats: np.ndarray, min_share: float = 0.6) -> int:
    """Quantas batidas tem o compasso da música: a distância mais comum entre inícios de compasso
    seguidos (3 numa valsa, 4 no rock). Sem maioria clara, ou 2 (quase sempre um 4/4 visto pela
    metade), fica 4."""
    if len(downbeats) < 4 or len(beats) < 6:
        return 4
    period = float(np.median(np.diff(beats[1:])))
    idx = np.clip(np.searchsorted(beats, downbeats), 1, len(beats) - 1)
    nearest = np.where(downbeats - beats[idx - 1] < beats[idx] - downbeats, idx - 1, idx)
    close = nearest[np.abs(beats[nearest] - downbeats) < period / 4]
    gaps = np.diff(np.unique(close))
    gaps = gaps[(gaps >= 2) & (gaps <= 7)]
    if not len(gaps):
        return 4
    counts = np.bincount(gaps, minlength=8)
    best = int(counts.argmax())
    return best if best in (3, 5, 6, 7) and counts[best] >= min_share * len(gaps) else 4


def _bars_on_downbeats(beats: np.ndarray, downbeats: np.ndarray,
                       first_onset: float | None = None, meter: int = 4) -> np.ndarray:
    """Faz os compassos (batidas 0, meter, 2*meter...) começarem nos inícios de compasso detectados.

    `beats` já começa em 0 s (_anchor_at_zero). A fase que vale é a mais comum entre todos os inícios
    de compasso detectados. Para chegar nela, o trecho antes do primeiro ataque (`first_onset`) é
    redividido em batidas iguais, uma ou mais a menos ou a mais, o que ficar mais perto do andamento:
    assim a música começa já na grade certa. Se a música começa na primeira batida (ou sem
    `first_onset`), tiram-se batidas logo depois do 0 ou, se não houver de onde tirar, a primeira é
    dividida."""
    if not len(downbeats) or len(beats) < 6:
        return beats
    period = float(np.median(np.diff(beats[1:])))
    idx = np.clip(np.searchsorted(beats, downbeats), 1, len(beats) - 1)
    nearest = np.where(downbeats - beats[idx - 1] < beats[idx] - downbeats, idx - 1, idx)
    close = np.abs(beats[nearest] - downbeats) < period / 4
    if not close.any():
        return beats
    phase = int(np.bincount(nearest[close] % meter, minlength=meter).argmax())
    if phase == 0:
        return beats
    # Última batida antes do primeiro ataque (ou junto dele): dali para trás, só silêncio.
    last = int(np.searchsorted(beats, first_onset + period / 4)) - 1 if first_onset is not None else 0
    if 0 < last < len(beats) - 1:
        counts = [n for n in (last - phase, last + meter - phase) if n >= 1]
        n = min(counts, key=lambda n: abs(np.log(beats[last] / n / period)))
        return np.concatenate([np.linspace(0.0, beats[last], n + 1), beats[last + 1:]])
    if len(beats) > phase + 1 and beats[phase + 1] - beats[0] > 0:
        return np.delete(beats, np.arange(1, phase + 1))
    extra = meter - phase
    fillers = np.linspace(beats[0], beats[1], extra + 2)[1:-1]
    return np.insert(beats, 1, fillers)


_BEAT_THIS = None


def _beat_this(y: np.ndarray, sr: int, download: bool):
    """Batidas e inícios de compasso pelo Beat This! (CPJKU, ISMIR 2024), se instalado. Sem
    `download`, só usa o modelo se ele já estiver baixado (os testes e o uso offline não travam)."""
    global _BEAT_THIS
    try:
        import torch
        from beat_this.inference import Audio2Beats
    except ImportError:
        return None
    if _BEAT_THIS is None:
        ckpt = Path(torch.hub.get_dir()) / "checkpoints" / "beat_this-final0.ckpt"
        if not download and not ckpt.is_file():
            return None
        try:
            _BEAT_THIS = Audio2Beats(checkpoint_path=str(ckpt) if ckpt.is_file() else "final0",
                                     device="cpu", dbn=False)
        except Exception:
            return None
    beats, downbeats = _BEAT_THIS(y.astype(np.float32), sr)
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


def estimate_tempo_map(y: np.ndarray, sr: int, resolution: int = 192,
                       mode: str = "auto", tracker: str = "auto") -> TempoMap:
    """Detecta as batidas da música e monta o mapa de tempo.

    mode:
      "auto"     - escolhe sozinho entre constante e variável;
      "constant" - um único BPM para a música toda;
      "variable" - acompanha pequenas variações de andamento (bateria ao vivo).

    tracker:
      "auto"      - Beat This! se instalado e já baixado; senão, o detector do librosa;
      "beat_this" - Beat This! (baixa o modelo, ~78 MB, na primeira vez);
      "librosa"   - o detector simples do librosa (não acha o início do compasso).
    """
    duration = len(y) / sr
    hop = 256
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    found = _beat_this(y, sr, download=tracker == "beat_this") if tracker != "librosa" else None
    if found is not None:
        beats, downbeats = found
        beats = _regularize_beats(beats)
    else:
        _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop)
        beats, downbeats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop), np.array([])

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
    onsets = onsets - ONSET_LATENCY
    smoothed = _align_phase(smoothed, onsets, float(period))
    anchored = _anchor_at_zero(smoothed, float(period), duration)
    first_onset = float(onsets[0]) if len(onsets) else None
    meter = _meter(anchored, downbeats)
    return TempoMap(_bars_on_downbeats(anchored, downbeats, first_onset, meter), resolution, [(0, meter)])
