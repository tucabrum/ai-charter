"""Transcrição de bateria.

Cada peça da bateria ocupa uma faixa de frequência diferente:
  - bumbo (kick): graves, abaixo de ~120 Hz;
  - caixa (snare): corpo em 150-400 Hz + "estalo" em 1,5-5 kHz;
  - chimbal/pratos: agudos, acima de ~7 kHz.
Medimos o "fluxo espectral" (aumento repentino de energia) em cada faixa e
procuramos picos. Um prato que continua soando logo após o ataque é um crash.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from ai_charter.config import DrumDetection

HOP = 256
N_FFT = 2048
# Este detector marca as batidas ~13 ms ANTES do início real (a janela da
# STFT "enxerga" o ataque antes do seu centro); compensamos esse adiantamento.
ONSET_LATENCY = -0.013

KICK, SNARE, HIHAT, CRASH = "kick", "snare", "hihat", "crash"


@dataclass
class DrumHit:
    time: float
    piece: str       # KICK, SNARE, HIHAT ou CRASH
    strength: float  # ~0 a 1


def _band_flux(S_log: np.ndarray, freqs: np.ndarray, band) -> np.ndarray:
    """Fluxo espectral (quanto a energia subiu) dentro de uma faixa de frequência.

    Normalizado para que as batidas mais fortes da música fiquem perto de 1.
    """
    rows = (freqs >= band[0]) & (freqs < band[1])
    if not rows.any():
        return np.zeros(S_log.shape[1])
    flux = np.maximum(0.0, np.diff(S_log[rows], axis=1, prepend=S_log[rows][:, :1])).mean(axis=0)
    return flux / (np.percentile(flux, 99.5) + 1e-9)


def _band_energy(S: np.ndarray, freqs: np.ndarray, band) -> np.ndarray:
    rows = (freqs >= band[0]) & (freqs < band[1])
    return S[rows].mean(axis=0) if rows.any() else np.zeros(S.shape[1])


def _peaks(env: np.ndarray, threshold: float) -> np.ndarray:
    """Encontra picos isolados (pelo menos ~35 ms entre eles) acima do limiar."""
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=8, post_avg=3,
                                   delta=0.05, wait=3)
    return peaks[env[peaks] >= threshold]


def detect_drum_hits(y: np.ndarray, sr: int, cfg: DrumDetection) -> list[DrumHit]:
    if not np.any(y):
        return []
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    # Compressão logarítmica relativa ao máximo: sons muito baixos quase não contam
    # (em decibéis puros, um ruído quase inaudível viraria um "salto" enorme).
    S_log = np.log1p(1000.0 * S / (S.max() + 1e-12))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    to_time = lambda f: float(librosa.frames_to_time(f, sr=sr, hop_length=HOP)) - ONSET_LATENCY  # noqa: E731

    hits: list[DrumHit] = []
    kick = _band_flux(S_log, freqs, cfg.kick_band)
    body = _band_flux(S_log, freqs, cfg.snare_band)
    snap = _band_flux(S_log, freqs, cfg.snare_snap_band)
    near = lambda env, f: env[max(f - 2, 0): f + 3].max()  # noqa: E731  (maior valor perto do frame f)

    # Bumbo: pico nos graves, e os graves precisam "ganhar" do corpo da caixa
    # (a caixa também tem um pouco de grave, mas proporcionalmente menos).
    for f in _peaks(kick, cfg.threshold):
        if near(kick, f) >= cfg.dominance * near(body, f):
            hits.append(DrumHit(to_time(f), KICK, float(min(kick[f], 1.0))))

    # Caixa: precisa de corpo E de estalo, e o corpo precisa ganhar dos graves.
    snare = np.sqrt(body * snap)
    for f in _peaks(snare, cfg.threshold):
        if near(body, f) >= cfg.dominance * near(kick, f):
            hits.append(DrumHit(to_time(f), SNARE, float(min(snare[f], 1.0))))

    # Chimbal / crash
    high = _band_flux(S_log, freqs, cfg.hihat_band)
    high_energy = _band_energy(S, freqs, cfg.hihat_band)
    # Chimbal fechado some em poucos milissegundos; crash continua soando.
    # Medimos entre 60 e 100 ms depois do ataque (antes da próxima batida).
    a, b = int(0.06 * sr / HOP), int(0.1 * sr / HOP) + 1
    for f in _peaks(high, cfg.threshold):
        peak = high_energy[f: f + 3].max() + 1e-12
        tail = high_energy[f + a: f + b]
        ratio = tail.min() / peak if len(tail) else 0.0
        is_crash = high[f] >= cfg.crash_min_strength and ratio >= cfg.crash_sustain_ratio
        hits.append(DrumHit(to_time(f), CRASH if is_crash else HIHAT, float(min(high[f], 1.0))))

    hits.sort(key=lambda h: h.time)
    return hits
