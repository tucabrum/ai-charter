"""Carrega as regras de charting a partir de arquivos TOML.

As regras padrão ficam em `default_rules.toml`. O usuário pode passar um
arquivo próprio que sobrescreve apenas os valores que ele mudar.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from importlib import resources
from pathlib import Path
from typing import Any

# Ordem usada em todo o projeto: da mais difícil para a mais fácil.
DIFFICULTIES = ("expert", "hard", "medium", "easy")
PITCHED_INSTRUMENTS = ("guitar", "bass", "keys")


@dataclass
class GeneralRules:
    resolution: int
    allow_triplets: bool
    starpower_every_measures: int
    starpower_phrase_beats: int
    drum_fill_every_measures: int
    min_stem_energy: float
    sections: bool
    section_seconds: float


@dataclass
class PitchedDetection:
    fmin: str
    octaves: int
    onset_delta: float
    min_rms_db: float
    chord_ratio: float
    release_ratio: float
    lane_window: int


@dataclass
class DrumDetection:
    kick_band: list[float]
    snare_band: list[float]
    snare_snap_band: list[float]
    hihat_band: list[float]
    threshold: float
    dominance: float
    crash_sustain_ratio: float
    crash_min_strength: float


@dataclass
class GuitarSelectionRules:
    active_db: float
    weight_loudness: float
    weight_single_notes: float
    weight_pitch: float
    switch_cost: float
    inactive_cost: float
    min_hold_measures: int


@dataclass
class VocalRules:
    fmin: str
    fmax: str
    min_rms_db: float
    onset_delta: float
    pitch_split_semitones: float
    min_note_seconds: float
    phrase_gap_beats: float
    language: str
    whisper_model: str


@dataclass
class FretDifficultyRules:
    lanes: int
    max_chord: int
    min_gap_beats: float
    allow_offgrid: bool
    grid: int
    sustain_min_beats: float
    allow_hopos: bool
    min_reduction: float


@dataclass
class DrumDifficultyRules:
    kick_min_gap_beats: float
    snare_min_gap_beats: float
    hihat_min_gap_beats: float
    crash: bool
    max_hands: int
    allow_offgrid: bool
    grid: int
    min_reduction: float


@dataclass
class Rules:
    general: GeneralRules
    detection: dict[str, PitchedDetection]
    drum_detection: DrumDetection
    guitar_selection: GuitarSelectionRules
    vocals: VocalRules
    frets: dict[str, FretDifficultyRules]
    drums: dict[str, DrumDifficultyRules]


def _merge(base: dict, override: dict) -> dict:
    """Junta dois dicionários: valores de `override` substituem os de `base`."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _build(cls, data: dict[str, Any], where: str):
    """Cria um dataclass a partir de um dicionário, avisando sobre chaves erradas."""
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"Chave(s) desconhecida(s) em [{where}]: {', '.join(sorted(unknown))}")
    missing = names - set(data)
    if missing:
        raise ValueError(f"Chave(s) faltando em [{where}]: {', '.join(sorted(missing))}")
    return cls(**data)


def load_rules(path: str | Path | None = None) -> Rules:
    """Lê as regras padrão e, se `path` for informado, aplica as do usuário por cima."""
    default_text = resources.files("ai_charter").joinpath("default_rules.toml").read_text("utf-8")
    data = tomllib.loads(default_text)
    if path is not None:
        with open(path, "rb") as f:
            data = _merge(data, tomllib.load(f))

    return Rules(
        general=_build(GeneralRules, data["general"], "general"),
        detection={
            name: _build(PitchedDetection, data["detection"][name], f"detection.{name}")
            for name in PITCHED_INSTRUMENTS
        },
        drum_detection=_build(DrumDetection, data["drum_detection"], "drum_detection"),
        guitar_selection=_build(GuitarSelectionRules, data["guitar_selection"], "guitar_selection"),
        vocals=_build(VocalRules, data["vocals"], "vocals"),
        frets={d: _build(FretDifficultyRules, data["frets"][d], f"frets.{d}") for d in DIFFICULTIES},
        drums={d: _build(DrumDifficultyRules, data["drums"][d], f"drums.{d}") for d in DIFFICULTIES},
    )
