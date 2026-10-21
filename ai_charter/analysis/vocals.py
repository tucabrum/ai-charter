"""Transcrição da voz: encontra as notas cantadas (início, fim e afinação).

Usa o pYIN, um detector de afinação feito para voz e instrumentos de uma nota
só. Ele diz, a cada ~12 ms, a frequência da voz e se há voz soando ("voiced").
As notas são os trechos com voz, cortados onde:
  - a voz para (silêncio ou consoante surda, como "s", "p", "t");
  - há um novo ataque (consoante sonora, como "m", "l", "n");
  - a afinação muda de semitom (duas notas seguidas sem pausa).
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from ai_charter.config import VocalRules

HOP = 256


@dataclass
class SungNote:
    time: float   # início (s)
    end: float    # fim (s)
    pitch: float  # nota MIDI (pode ter fração; 60 = Dó central)


def _split_on_pitch(midi: np.ndarray, start: int, end: int, threshold: float, min_frames: int) -> list[int]:
    """Pontos dentro de [start, end) onde a afinação muda de nota."""
    cuts = []
    w = 3  # compara a mediana das ~35 ms antes com as ~35 ms depois
    i = start + max(w, min_frames)
    while i < end - max(w, min_frames):
        before = np.median(midi[i - w:i])
        after = np.median(midi[i:i + w])
        if abs(after - before) >= threshold:
            # Corta no frame de maior mudança dentro da vizinhança.
            local = np.abs(np.diff(midi[i - w:i + w]))
            cut = i - w + 1 + int(np.argmax(local))
            cuts.append(cut)
            i = cut + min_frames
        else:
            i += 1
    return cuts


def _fix_octave_errors(notes: list[SungNote], max_gap: float = 1.0) -> None:
    """Corrige erros de oitava: o detector às vezes "ouve" a nota 12 semitons acima/abaixo.

    Se uma nota está longe das vizinhas (mais de uma quinta, em média) e,
    deslocada uma oitava, fica bem mais perto delas, ela é corrigida. Saltos de
    oitava reais costumam ser confirmados pela nota seguinte (que fica perto da
    nota nova), então não são alterados.
    """
    original = [n.pitch for n in notes]  # compara sempre com os valores originais
    for i in range(1, len(notes) - 1):
        # Precisa das duas vizinhas (anterior e seguinte), a no máximo `max_gap` segundos.
        if notes[i].time - notes[i - 1].end > max_gap or notes[i + 1].time - notes[i].end > max_gap:
            continue
        neighbors = (original[i - 1], original[i + 1])
        distance = np.mean([abs(original[i] - p) for p in neighbors])
        if distance <= 7:
            continue
        for shift in (-12, 12):
            if np.mean([abs(original[i] + shift - p) for p in neighbors]) + 3 < distance:
                notes[i].pitch = original[i] + shift
                break


def detect_sung_notes(y: np.ndarray, sr: int, cfg: VocalRules) -> list[SungNote]:
    if not np.any(y):
        return []
    f0, voiced, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz(cfg.fmin), fmax=librosa.note_to_hz(cfg.fmax),
        sr=sr, frame_length=2048, hop_length=HOP,
    )
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    n = min(len(f0), len(rms))
    loud = librosa.amplitude_to_db(rms[:n], ref=np.max) > cfg.min_rms_db
    active = voiced[:n] & loud
    midi = librosa.hz_to_midi(np.where(active, f0[:n], np.nan))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    onsets = set(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP,
                                            delta=cfg.onset_delta, units="frames"))
    min_frames = max(1, int(cfg.min_note_seconds * sr / HOP))

    # 1. Trechos contínuos de voz.
    runs, i = [], 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        start = i
        while i < n and active[i]:
            i += 1
        runs.append((start, i))

    # 2. Corta cada trecho nos ataques e nas mudanças de afinação.
    notes: list[SungNote] = []
    for start, end in runs:
        cuts = {c for c in onsets if start + min_frames <= c <= end - min_frames}
        cuts |= set(_split_on_pitch(midi, start, end, cfg.pitch_split_semitones, min_frames))
        bounds = [start] + sorted(cuts) + [end]
        for a, b in zip(bounds, bounds[1:]):
            if b - a < min_frames:
                continue
            # Afinação = mediana do miolo da nota (ignora as pontas, onde a voz desliza).
            trim = (b - a) // 5
            pitch = float(np.nanmedian(midi[a + trim: b - trim]))
            notes.append(SungNote(
                time=float(librosa.frames_to_time(a, sr=sr, hop_length=HOP)),
                end=float(librosa.frames_to_time(b, sr=sr, hop_length=HOP)),
                pitch=pitch,
            ))
    _fix_octave_errors(notes)
    return notes
