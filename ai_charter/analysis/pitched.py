"""Transcrição de instrumentos com altura (guitarra, baixo, teclado).

Para cada stem, encontramos:
  1. os ATAQUES (onsets): os momentos em que uma nota começa;
  2. a ALTURA (pitch) de cada nota, usando o espectro CQT (que tem um "bin"
     por semitom), e se ela parece um acorde (várias notas ao mesmo tempo);
  3. a DURAÇÃO, olhando quando o volume cai (para gerar sustains).
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from ai_charter.config import PitchedDetection

HOP = 256
# O librosa marca os ataques ~12 ms depois do início real da nota
# (medido com áudio sintético nos testes); descontamos esse atraso.
ONSET_LATENCY = 0.012
# Harmônicos de uma nota, em semitons acima da fundamental, e seus pesos.
# (2× freq = +12 semitons, 3× = +19, 4× = +24, 5× = +28)
HARMONICS = ((0, 1.0), (12, 0.5), (19, 0.33), (24, 0.25), (28, 0.2))


@dataclass
class DetectedNote:
    time: float        # início da nota (s)
    end: float         # fim da nota (s)
    pitch: float       # nota MIDI mais grave detectada (60 = Dó central)
    voices: int        # quantas notas soam juntas (1 = nota simples, 2+ = acorde)
    strength: float    # força do ataque, normalizada (~0 a 1)


def _harmonic_salience(spectrum: np.ndarray) -> np.ndarray:
    """Para cada semitom, soma a energia dele e dos seus harmônicos.

    Isso ajuda a achar a nota fundamental em vez de um harmônico dela.
    """
    salience = np.zeros_like(spectrum)
    n = len(spectrum)
    for offset, weight in HARMONICS:
        salience[: n - offset] += weight * spectrum[offset:]
    return salience


def estimate_pitches(spectrum: np.ndarray, chord_ratio: float, max_voices: int = 3) -> list[int]:
    """Estima quais notas (índices de bin) estão soando num trecho do espectro.

    Pega a nota mais forte, "apaga" ela e seus harmônicos, e procura de novo.
    Uma nova nota só conta se for pelo menos `chord_ratio` × a primeira.
    """
    spectrum = spectrum.astype(float).copy()
    found: list[int] = []
    first = None
    for _ in range(max_voices):
        salience = _harmonic_salience(spectrum)
        best = int(np.argmax(salience))
        if first is None:
            first = salience[best]
            if first <= 0:
                break
        elif salience[best] < chord_ratio * first:
            break
        found.append(best)
        for offset, _ in HARMONICS:
            lo, hi = max(best + offset - 1, 0), min(best + offset + 2, len(spectrum))
            spectrum[lo:hi] *= 0.1
    return found


def detect_pitched_notes(y: np.ndarray, sr: int, cfg: PitchedDetection) -> list[DetectedNote]:
    """Detecta as notas de um stem mono `y` com taxa de amostragem `sr`."""
    if not np.any(y):
        return []

    # --- 1. Ataques ---------------------------------------------------------
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=HOP, delta=cfg.onset_delta, units="frames"
    )
    if len(onset_frames) == 0:
        return []
    env_scale = np.percentile(onset_env[onset_frames], 95) + 1e-9

    # --- 2. Espectro CQT (um bin por semitom) ------------------------------
    fmin_midi = librosa.note_to_midi(cfg.fmin)
    cqt = np.abs(librosa.cqt(
        y, sr=sr, hop_length=HOP, fmin=librosa.midi_to_hz(fmin_midi),
        n_bins=12 * cfg.octaves, bins_per_octave=12,
    ))

    # --- 3. Volume ao longo do tempo (para gate e duração) -----------------
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    n_frames = min(cqt.shape[1], len(rms))
    notes: list[DetectedNote] = []
    for i, frame in enumerate(onset_frames):
        if frame >= n_frames:
            break
        next_frame = onset_frames[i + 1] if i + 1 < len(onset_frames) else n_frames
        next_frame = min(next_frame, n_frames)

        # Ignora ataques muito baixos (ruído ou vazamento de outro instrumento).
        peak_frame = frame + int(np.argmax(rms[frame: min(frame + 6, next_frame)])) if next_frame > frame else frame
        if rms_db[peak_frame] < cfg.min_rms_db:
            continue
        # Se o volume CAI no "ataque", é o fim de uma nota (um corte/clique), não um início.
        before = rms[max(frame - 4, 0): max(frame - 1, 1)].mean()
        if rms[peak_frame] < 0.8 * before:
            continue

        # Analisa a altura um pouco depois do ataque (o ataque em si é ruidoso).
        a = min(frame + 2, next_frame - 1)
        b = max(a + 1, min(frame + 12, next_frame))
        spectrum = cqt[:, a:b].mean(axis=1)
        bins = estimate_pitches(spectrum, cfg.chord_ratio)
        if not bins:
            continue

        # A nota termina quando o volume cai abaixo de release_ratio × pico.
        peak = rms[peak_frame]
        end_frame = next_frame
        below = np.nonzero(rms[peak_frame:next_frame] < peak * cfg.release_ratio)[0]
        if len(below):
            end_frame = peak_frame + int(below[0])

        notes.append(DetectedNote(
            time=float(librosa.frames_to_time(frame, sr=sr, hop_length=HOP)) - ONSET_LATENCY,
            end=float(librosa.frames_to_time(end_frame, sr=sr, hop_length=HOP)) - ONSET_LATENCY,
            pitch=float(fmin_midi + min(bins)),
            voices=len(bins),
            strength=float(min(onset_env[frame] / env_scale, 1.5)),
        ))
    return notes
