"""Bateria: descobrir os sons do kit e separar bumbo, caixa, tons e pratos (sem precisar do ffmpeg)."""

import mido
import numpy as np
import pytest
from synth import SR, add, drum

from ai_charter.analysis.drums import CRASH, HIHAT, KICK, SNARE, TOM_HIGH, TOM_LOW, detect_drum_hits
from ai_charter.analysis.tempo import TempoMap
from ai_charter.charting.common import Candidate
from ai_charter.charting.drums import build_drum_track
from ai_charter.config import load_rules
from ai_charter.revision import _tempo_map

RULES = load_rules()
BEAT = 0.5


def tom(f0: float, rng: np.random.Generator) -> np.ndarray:
    """Tom: altura definida que ressoa, mais o estalo curto da baqueta."""
    t = np.arange(int(0.6 * SR)) / SR
    body = (np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * f0 * 1.6 * t)) * np.exp(-t * 5)
    return body + rng.standard_normal(len(t)) * np.exp(-t * 80) * 0.3


def found(hits, at: float, piece: str) -> bool:
    return any(abs(h.time - at) < 0.03 and h.piece == piece for h in hits)


@pytest.fixture(scope="module")
def kit_song():
    """Groove de bumbo/caixa com chimbal em colcheias, prato no começo de cada bloco e uma
    virada de tom agudo + surdo no fim de cada bloco de 8 batidas."""
    rng = np.random.default_rng(1)
    y, truth = np.zeros(int(20 * SR)), []
    for b in range(32):
        at = 0.5 + b * BEAT
        if b % 8 in (6, 7):
            for j, (f0, piece) in enumerate(((210, TOM_HIGH), (100, TOM_LOW))):
                add(y, tom(f0, rng), at + j * BEAT / 2, 0.7)
                truth.append((at + j * BEAT / 2, piece))
            continue
        foot = KICK if b % 2 == 0 else SNARE
        add(y, drum(foot, rng), at, 0.8)
        add(y, drum("crash" if b % 8 == 0 else "hihat", rng), at)
        add(y, drum("hihat", rng), at + BEAT / 2)
        truth += [(at, foot), (at, CRASH if b % 8 == 0 else HIHAT), (at + BEAT / 2, HIHAT)]
    return detect_drum_hits(y, SR, RULES.drum_detection), truth


def test_every_drum_piece_is_found(kit_song):
    hits, truth = kit_song
    for piece in (KICK, SNARE, TOM_HIGH, TOM_LOW, CRASH):
        expected = [t for t, p in truth if p == piece]
        assert sum(found(hits, t, piece) for t in expected) >= 0.9 * len(expected), piece
    hats = [t for t, p in truth if p == HIHAT]
    assert sum(found(hits, t, HIHAT) for t in hats) >= 0.7 * len(hats)   # os logo após o prato são mascarados


def test_toms_are_never_called_kick(kit_song):
    hits, truth = kit_song
    tom_times = [t for t, p in truth if p in (TOM_HIGH, TOM_LOW)]
    assert not any(found(hits, t, KICK) for t in tom_times)
    wrong = [h for h in hits if not any(abs(h.time - t) < 0.03 and h.piece == p for t, p in truth)]
    assert len(wrong) <= 0.1 * len(hits)


def test_floor_tom_groove_is_not_kick():
    """O caso que motivou o transcritor: um surdo (~105 Hz, ressoa) tocando o ritmo, alternado
    com o bumbo (~50 Hz, curto). Detector por "pico nos graves" chamava os dois de bumbo."""
    rng = np.random.default_rng(2)
    y, kicks, floor = np.zeros(int(14 * SR)), [], []
    for b in range(24):
        at = 0.5 + b * BEAT
        if b % 3 == 2:
            add(y, tom(105, rng), at, 0.8)
            floor.append(at)
        else:
            add(y, drum("kick", rng), at, 0.8)
            kicks.append(at)
    hits = detect_drum_hits(y, SR, RULES.drum_detection)
    assert sum(found(hits, t, TOM_LOW) for t in floor) >= 0.9 * len(floor)
    assert sum(found(hits, t, KICK) for t in kicks) >= 0.9 * len(kicks)
    assert not any(found(hits, t, KICK) for t in floor)


def test_synthetic_song_drums_end_to_end(tmp_path):
    """O mesmo que o teste de ponta a ponta (que precisa do ffmpeg), direto pelo pipeline."""
    from synth import START, render

    from ai_charter.pipeline import chart_drums

    paths = render(tmp_path)
    res = RULES.general.resolution
    beat_times = START + np.arange(0, 40) * BEAT
    tempo = TempoMap(np.concatenate([[0.0], beat_times]), res)   # batida 1 = START
    expert = {n.tick: n for n in chart_drums(paths["drums"], tempo, RULES, []).notes["expert"]}
    beats = range(res, res + 24 * res, res)
    kick = [t for t in beats if t in expert and 0 in expert[t].lanes]
    snare = [t for t in beats if t in expert and 1 in expert[t].lanes]
    assert sum((t - res) // res % 2 == 0 for t in kick) >= 0.9 * 12
    assert sum((t - res) // res % 2 == 1 for t in snare) >= 0.9 * 12
    flags = {f for n in expert.values() for f in n.flags}
    assert {66, 68} <= flags   # chimbal (amarelo) e ataque (verde) marcados como prato


def test_charting_puts_toms_and_cymbals_on_their_lanes():
    res = RULES.general.resolution
    cands = [Candidate(0, 1.0, piece=KICK), Candidate(0, 1.0, piece=CRASH),
             Candidate(res, 1.0, piece=TOM_HIGH), Candidate(res, 0.5, piece=HIHAT),   # mesma lane (amarelo)
             Candidate(2 * res, 1.0, piece=TOM_LOW), Candidate(3 * res, 1.0, piece="ride")]
    notes = {n.tick: n for n in build_drum_track(cands, RULES.drums, res, [], 0).notes["expert"]}
    assert notes[0].lanes == (0, 4) and notes[0].flags == (68,)        # bumbo + ataque (verde)
    assert notes[res].lanes == (2,) and notes[res].flags == ()          # tom agudo ganha do chimbal
    assert notes[2 * res].lanes == (4,) and notes[2 * res].flags == ()  # surdo: verde sem prato
    assert notes[3 * res].lanes == (3,) and notes[3 * res].flags == (67,)  # condução: azul prato


def test_tempo_map_from_midi():
    midi = mido.MidiFile(type=1, ticks_per_beat=192)
    midi.tracks.append(mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500000, time=0),
                                       mido.MetaMessage("set_tempo", tempo=250000, time=384),
                                       mido.MetaMessage("end_of_track", time=768)]))
    tempo = _tempo_map(midi)
    assert tempo.beat_times[:6] == pytest.approx([0.0, 0.5, 1.0, 1.25, 1.5, 1.75])
    assert tempo.time_to_tick(1.25) == pytest.approx(576)


def test_shipped_drum_model_loads_and_matches_the_features():
    from ai_charter.analysis.drums import FEATURE_NAMES
    from ai_charter.drum_model import INPUT_NAMES, OUTPUTS, DrumModel, model_inputs

    model = DrumModel.load()
    assert model is not None, "ai_charter/drum_model.json não carregou (treine com ai-charter-bench train-drums)"
    assert model.w1.shape[0] == len(INPUT_NAMES) and model.w2.shape[1] == len(OUTPUTS)
    features = np.random.default_rng(0).normal(size=(20, len(FEATURE_NAMES))).astype(np.float32)
    p = model.predict(model_inputs(features, np.arange(20) * 48.0, 192))
    assert p.shape == (20, len(OUTPUTS)) and np.all((p >= 0) & (p <= 1))


def test_model_from_another_feature_version_is_ignored(tmp_path):
    import json

    from ai_charter.drum_model import DrumModel, MODEL_PATH

    params = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    params["features"] = params["features"][:-1]
    path = tmp_path / "velho.json"
    path.write_text(json.dumps(params), encoding="utf-8")
    assert DrumModel.load(path) is None
