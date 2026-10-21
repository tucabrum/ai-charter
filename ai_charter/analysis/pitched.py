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
from scipy.ndimage import percentile_filter

from ai_charter.analysis.chords import chord_scores
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
    # Só com recognize_chords: semelhança com cada acorde de analysis.chords.CHORDS,
    # e a nota (MIDI) mais grave que soa forte em cada uma das 12 classes (Dó, Dó#...).
    chord_scores: np.ndarray | None = None
    lowest_by_class: np.ndarray | None = None
    # Energia uma quinta (+7) e uma terça (+3/+4, +15/+16) acima da nota mais grave,
    # relativa a ela. Uma nota simples (mesmo distorcida) quase não tem; acordes têm.
    fifth_ratio: float = 0.0
    third_ratio: float = 0.0


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


def _energy_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    """Ataques pela subida de ENERGIA (bom para palhetadas, notas com "estalo").

    1. Subtração espectral: em cada faixa de frequência, estima o ruído de fundo
       (o nível que está sempre presente nos últimos ~2 s) e o remove. Chiado
       constante some; palhetadas, que são rápidas, continuam.
    2. SuperFlux (lag=2, max_size=3): pega ataques fracos (palhetada para cima)
       sem confundir vibrato de nota longa com ataque novo.
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP, n_mels=128)
    window = max(3, int(2.0 * sr / HOP))
    floor = percentile_filter(mel, percentile=10, size=(1, window), mode="nearest")
    clean = np.maximum(mel - floor, 0) + 1e-10 * mel.max()
    return librosa.onset.onset_strength(S=librosa.power_to_db(clean, ref=np.max), sr=sr,
                                        hop_length=HOP, lag=2, max_size=3)


def _pitch_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    """Ataques pela TROCA DE NOTA: subida de energia em cada semitom (espectro CQT).

    Numa guitarra distorcida (uma "parede de som") o volume quase não muda quando
    a nota muda; o que muda é qual semitom soa. top_db=40 ignora o que está muito
    abaixo do pico (senão, a cauda das notas vira "ataque").
    """
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=HOP, fmin=librosa.note_to_hz("E1"),
                             n_bins=84, bins_per_octave=12)) ** 2
    return librosa.onset.onset_strength(S=librosa.power_to_db(cqt, ref=np.max, top_db=40), sr=sr,
                                        hop_length=HOP, lag=2, max_size=3)


def onset_envelope(y: np.ndarray, sr: int, method: str = "energy", pitch_weight: float = 0.5) -> np.ndarray:
    """Envelope de ataques. method: "energy", "pitch" ou "both" (soma dos dois)."""
    if method == "energy":
        return _energy_envelope(y, sr)
    if method == "pitch":
        return _pitch_envelope(y, sr)
    energy, pitch = _energy_envelope(y, sr), _pitch_envelope(y, sr)
    n = min(len(energy), len(pitch))
    normalize = lambda e: e[:n] / (np.percentile(e[:n], 99) + 1e-9)  # noqa: E731
    return normalize(energy) + pitch_weight * normalize(pitch)


def _lowest_by_class(spectrum: np.ndarray, lowest_midi: int, threshold: float = 0.25) -> np.ndarray:
    """Para cada classe de nota (Dó, Dó#...), a nota MIDI mais grave que soa forte (NaN se nenhuma)."""
    result = np.full(12, np.nan)
    for i in np.nonzero(spectrum >= threshold * spectrum.max())[0]:
        midi = lowest_midi + int(i)
        if np.isnan(result[midi % 12]):
            result[midi % 12] = midi
    return result


def detect_pitched_notes(y: np.ndarray, sr: int, cfg: PitchedDetection,
                         tuning: float = 0.0) -> list[DetectedNote]:
    """Detecta as notas de um stem mono `y` com taxa de amostragem `sr`.

    `tuning`: afinação da gravação em frações de semitom (ver pipeline.estimate_tuning).
    Os bins do espectro são centrados nas notas DESSA afinação; sem isso, uma
    gravação meio semitom fora do padrão faz a mesma nota "oscilar" entre duas.
    """
    if not np.any(y):
        return []

    # --- 1. Ataques ---------------------------------------------------------
    onset_env = onset_envelope(y, sr, cfg.onset_method)
    # wait: ataques a menos de 60 ms um do outro contam como um só (ex.: acorde
    # palhetado, em que as cordas soam uma logo após a outra).
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=HOP, delta=cfg.onset_delta,
        wait=int(0.06 * sr / HOP), units="frames",
    )
    if len(onset_frames) == 0:
        return []
    env_scale = np.percentile(onset_env[onset_frames], 95) + 1e-9

    # --- 2. Espectro CQT (um bin por semitom) ------------------------------
    fmin_midi = librosa.note_to_midi(cfg.fmin)
    cqt = np.abs(librosa.cqt(
        y, sr=sr, hop_length=HOP, fmin=librosa.midi_to_hz(fmin_midi),
        n_bins=12 * cfg.octaves, bins_per_octave=12, tuning=tuning,
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

        scores = lowest = None
        fifth = third = 0.0
        if cfg.recognize_chords:
            root = min(bins)
            base = spectrum[root] + 1e-12
            fifth = float(spectrum[root + 7] / base) if root + 7 < len(spectrum) else 0.0
            thirds = [spectrum[root + k] for k in (3, 4, 15, 16) if root + k < len(spectrum)]
            third = float(max(thirds) / base) if thirds else 0.0
            # Acordes: janela mais longa (até ~230 ms), para "ouvir" todas as cordas.
            chord_spectrum = cqt[:, a:max(b, min(frame + 20, next_frame))].mean(axis=1)
            scores = chord_scores(chord_spectrum, int(fmin_midi))
            lowest = _lowest_by_class(chord_spectrum, int(fmin_midi))

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
            chord_scores=scores,
            lowest_by_class=lowest,
            fifth_ratio=fifth,
            third_ratio=third,
        ))
    _fix_octave_spikes(notes)
    return notes


def _fix_octave_spikes(notes: list[DetectedNote]) -> None:
    """Corrige picos de oitava: uma nota solta exatamente 12 semitons longe das vizinhas.

    Em notas graves e distorcidas, o harmônico uma oitava acima às vezes soa mais
    forte que a nota de verdade, e o detector "ouve" a oitava errada. Na chart isso
    vira um pulo para o outro lado do braço e de volta. Se pelo menos 3 das 4
    vizinhas (2 antes, 2 depois) estão na mesma altura e a nota está a uma oitava
    exata delas, ela volta para a altura das vizinhas. Riffs de oitava de verdade
    (que alternam várias vezes) não são afetados: as vizinhas não concordam.
    """
    original = [n.pitch for n in notes]   # decide sempre pelos valores originais
    for i in range(2, len(notes) - 2):
        context = np.array(original[i - 2:i] + original[i + 1:i + 3])
        center = float(np.median(context))
        distance = original[i] - center
        agree = int(np.sum(np.abs(context - center) <= 1))
        if abs(abs(distance) - 12) <= 0.5 and agree >= 3:
            notes[i].pitch = original[i] - 12 * np.sign(distance)
