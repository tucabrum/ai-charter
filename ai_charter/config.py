"""Carrega as regras de charting a partir de arquivos TOML.

As regras padrão ficam em `default_rules.toml`. Por cima delas podem vir, nesta
ordem, uma predefinição de gênero musical (`genres/<gênero>.toml`, só com o que
muda) e um arquivo do usuário, que sobrescreve apenas os valores que ele mudar.
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
    onset_method: str
    onset_delta: float
    min_rms_db: float
    chord_ratio: float
    release_ratio: float
    lane_window: int
    recognize_chords: bool
    chord_fifth_ratio: float
    chord_third_ratio: float
    chord_smoothing: int
    chord_change_margin: float
    power_chord_third_margin: float
    power_chord_notes: int
    chord_notes: int
    accent_chord_notes: int
    accent_min_beats: float
    allowed_chords: list[str]   # acordes possíveis (ex.: ["G", "D", "Em", "C"]); vazio = qualquer um


@dataclass
class DrumDetection:
    use_model: bool
    onset_delta: float
    min_level: float
    max_sounds: int
    sound_similarity: float
    explained_error: float
    min_hits: int
    min_share: float
    threshold: float
    kick_max_hz: float
    floor_tom_max_hz: float
    tom_min_ring: float
    snare_min_rattle: float
    tom_share_in_snare: float
    drum_cymbal_air: float
    weak_share: float
    hat_over_snare_air: float
    cymbal_over_snare: float
    ride_sustain_ratio: float
    crash_sustain_ratio: float
    adtof_kick_trust: float = 0.0


@dataclass
class SanitizeRules:
    enabled: bool
    noise_ratio: float
    min_found: float
    max_extra: float
    min_groove_measures: int


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
    sung_min_ratio: float
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
    sanitize: SanitizeRules
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


@dataclass
class Genre:
    """Predefinição de regras para um gênero musical (arquivo genres/<id>.toml)."""
    id: str
    name: str
    description: str
    aliases: list[str]        # tags de gênero (em minúsculas) que correspondem a ele
    tempo: str | None         # mapa de tempo sugerido quando o usuário deixa "auto"
    rules: dict[str, Any]     # só o que muda em relação ao default_rules.toml


def list_genres() -> list[Genre]:
    """As predefinições de gênero que vêm com o ai-charter, na ordem do nome do arquivo."""
    genres = []
    folder = resources.files("ai_charter").joinpath("genres")
    for item in sorted(folder.iterdir(), key=lambda p: p.name):
        if not item.name.endswith(".toml"):
            continue
        data = tomllib.loads(item.read_text("utf-8"))
        preset = data.pop("preset", {})
        genres.append(Genre(
            id=item.name[:-len(".toml")], name=preset.get("name", item.name), description=preset.get("description", ""),
            aliases=[a.lower() for a in preset.get("aliases", [])], tempo=preset.get("tempo"), rules=data,
        ))
    return genres


def get_genre(genre: str) -> Genre:
    for g in list_genres():
        if g.id == genre:
            return g
    names = ", ".join(g.id for g in list_genres())
    raise ValueError(f"Gênero desconhecido: {genre} (disponíveis: {names})")


def genre_for_tag(tag: str | None) -> str | None:
    """O gênero cuja lista de apelidos tem a tag (ex.: "Thrash Metal" -> "metal"); None se nenhum."""
    tag = (tag or "").strip().lower()
    if not tag:
        return None
    return next((g.id for g in list_genres() if tag in g.aliases), None)


def load_rules(path: str | Path | None = None, genre: str | None = None) -> Rules:
    """Lê as regras padrão, aplica as do gênero (se houver) e, por fim, as do arquivo `path`."""
    default_text = resources.files("ai_charter").joinpath("default_rules.toml").read_text("utf-8")
    data = tomllib.loads(default_text)
    if genre:
        data = _merge(data, get_genre(genre).rules)
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
        sanitize=_build(SanitizeRules, data["sanitize"], "sanitize"),
        vocals=_build(VocalRules, data["vocals"], "vocals"),
        frets={d: _build(FretDifficultyRules, data["frets"][d], f"frets.{d}") for d in DIFFICULTIES},
        drums={d: _build(DrumDifficultyRules, data["drums"][d], f"drums.{d}") for d in DIFFICULTIES},
    )
