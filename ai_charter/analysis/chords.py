"""Reconhecimento de acordes pelo "chroma" (quanto de cada uma das 12 notas soa).

Cada acorde tem uma "impressão digital": um acorde de Dó maior tem Dó, Mi e Sol.
Mas instrumentos reais também soam HARMÔNICOS (múltiplos da frequência de cada
nota): o 3º harmônico de Mi é Si, por exemplo. Por isso os modelos abaixo já
incluem os harmônicos esperados de cada nota do acorde.

Os modelos também incluem NOTAS SIMPLES (uma nota + seus harmônicos): assim a
mesma comparação decide se o trecho é um acorde ou uma nota só.
"""

from __future__ import annotations

import numpy as np

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
# Tipos de acorde: intervalos (em semitons) a partir da fundamental.
KINDS = {"": (0, 4, 7), "m": (0, 3, 7), "5": (0, 7), "note": (0,)}
# Harmônicos de cada nota, como (distância em semitons dentro da oitava, peso).
HARMONIC_CLASSES = ((0, 1.0), (0, 0.5), (7, 0.33), (0, 0.25), (4, 0.2))
HARMONIC_DECAY = 1.5   # quanto os harmônicos mais altos perdem força

CHORDS: list[tuple[int, str]] = [(root, kind) for root in range(12) for kind in KINDS]


def _templates() -> np.ndarray:
    rows = []
    for root, kind in CHORDS:
        v = np.zeros(12)
        for interval in KINDS[kind]:
            for offset, weight in HARMONIC_CLASSES:
                v[(root + interval + offset) % 12] += weight ** HARMONIC_DECAY
        rows.append(v / np.linalg.norm(v))
    return np.array(rows)


TEMPLATES = _templates()


def chord_name(index: int) -> str:
    root, kind = CHORDS[index]
    return NOTE_NAMES[root] + ("" if kind == "note" else kind)


def is_single_note(index: int) -> bool:
    return CHORDS[index][1] == "note"


def chord_voices(index: int) -> int:
    """Quantas notas o acorde tem na chart (tríade = 3, power chord = 2, nota = 1)."""
    return len(KINDS[CHORDS[index][1]])


def chord_scores(spectrum: np.ndarray, lowest_midi: int) -> np.ndarray:
    """Semelhança (0 a 1) do trecho com cada acorde/nota de CHORDS.

    `spectrum`: energia por semitom (um bin por semitom, o primeiro é `lowest_midi`).
    """
    compressed = np.sqrt(np.maximum(spectrum, 0))  # compressão: notas fracas também contam
    chroma = np.zeros(12)
    for i, value in enumerate(compressed):
        chroma[(lowest_midi + i) % 12] += value
    norm = np.linalg.norm(chroma)
    return TEMPLATES @ (chroma / norm) if norm > 0 else np.zeros(len(CHORDS))
