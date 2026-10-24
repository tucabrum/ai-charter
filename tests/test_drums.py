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


def test_model_sees_ticks_in_the_tempo_map_grid(kit_song):
    """Charts do Rock Band têm 480 ticks por batida, não 192: o modelo tem de receber a grade do
    mapa de tempo, senão a posição de cada golpe no compasso sai embaralhada."""
    from ai_charter.drum_model import OUTPUTS
    from ai_charter.pipeline import model_drum_candidates

    seen = {}

    class Spy:
        thresholds = np.ones(len(OUTPUTS))

        def predict_song(self, features, ticks, resolution, adtof_feats=None):
            seen.update(ticks=ticks, resolution=resolution)
            return np.zeros((len(ticks), len(OUTPUTS)))

    rng = np.random.default_rng(3)
    y = np.zeros(int(6 * SR))
    for b in range(8):
        add(y, drum("kick", rng), 0.5 + b * BEAT, 0.8)
    tempo = TempoMap(np.arange(0, 14) * BEAT, 480)
    model_drum_candidates(y, SR, tempo, RULES, Spy())
    assert seen["resolution"] == 480
    beats = seen["ticks"] / 480
    assert np.median(np.abs(beats - np.round(beats))) < 0.05   # ataques caem nas batidas


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


@pytest.mark.parametrize("which", ["basico", "adtof"])
def test_shipped_drum_models_load_and_match_the_features(which):
    from ai_charter.analysis.drums import FEATURE_NAMES
    from ai_charter.drum_model import (ADTOF_MODEL_PATH, INPUT_NAMES, INPUT_NAMES_ADTOF, MODEL_PATH, OUTPUTS,
                                       DrumModel)

    path, names = (MODEL_PATH, INPUT_NAMES) if which == "basico" else (ADTOF_MODEL_PATH, INPUT_NAMES_ADTOF)
    model = DrumModel.load(path)
    assert model is not None, f"{path.name} não carregou (treine com ai-charter-bench train-drums)"
    assert model.w1.shape[0] == len(names) and model.w2.shape[1] == len(OUTPUTS)
    assert model.uses_adtof == (which == "adtof")
    rng = np.random.default_rng(0)
    features = rng.normal(size=(20, len(FEATURE_NAMES))).astype(np.float32)
    adtof_feats = rng.uniform(size=(20, 5)).astype(np.float32) if model.uses_adtof else None
    p = model.predict_song(features, np.arange(20) * 48.0, 192, adtof_feats)
    assert p.shape == (20, len(OUTPUTS)) and np.all((p >= 0) & (p <= 1))


def test_groove_inputs_see_the_bars_around():
    """A etapa da levada vê, para cada ataque, o que a etapa do som achou em cada semicolcheia em volta."""
    from ai_charter.drum_model import GROOVE_NAMES, GROOVE_OFFSETS, OUTPUTS, groove_inputs

    res = 480
    ticks = np.array([0, res, 4 * res, 4 * res + res // 4], dtype=float)   # 1, 2, compasso seguinte, +1/16
    probs = np.zeros((4, len(OUTPUTS)), dtype=np.float32)
    probs[:, OUTPUTS.index(HIHAT)] = [0.9, 0.8, 0.7, 0.6]
    x = groove_inputs(probs, ticks, res)
    assert x.shape == (4, len(GROOVE_NAMES))
    width = len(OUTPUTS) + 1
    at = lambda row, off, k: x[row, GROOVE_OFFSETS.index(off) * width + k]
    hat = OUTPUTS.index(HIHAT)
    assert at(0, 16, hat) == pytest.approx(0.7)       # um compasso depois
    assert at(0, 4, hat) == pytest.approx(0.8)        # uma batida depois
    assert at(2, 1, hat) == pytest.approx(0.6) and at(2, 1, width - 1) == 1.0
    assert at(1, 1, width - 1) == 0.0                 # nada logo depois da batida 2


def test_model_from_another_feature_version_is_ignored(tmp_path):
    import json

    from ai_charter.drum_model import DrumModel, MODEL_PATH

    params = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    params["features"] = params["features"][:-1]
    path = tmp_path / "velho.json"
    path.write_text(json.dumps(params), encoding="utf-8")
    assert DrumModel.load(path) is None


def test_adtof_chances_are_read_around_each_hit():
    from ai_charter.analysis.adtof import at_times

    act = np.zeros((300, 5), dtype=np.float32)
    act[100, 1] = 0.9      # caixa em 1,00 s
    act[203, 0] = 0.8      # bumbo 30 ms depois de 2,00 s
    feats = at_times(act, np.array([1.0, 2.0, 2.5]))
    assert feats[0, 1] == pytest.approx(0.9) and feats[1, 0] == pytest.approx(0.8)
    assert not feats[2].any()


def test_model_with_adtof_inputs_requires_them():
    from ai_charter.drum_model import INPUT_NAMES_ADTOF, OUTPUTS, DrumModel, GROOVE_NAMES

    rng = np.random.default_rng(0)
    layer = lambda n, m: {"mean": [0.0] * n, "std": [1.0] * n, "w1": rng.normal(size=(n, 4)).tolist(),
                          "b1": [0.0] * 4, "w2": rng.normal(size=(4, m)).tolist(), "b2": [0.0] * m}
    model = DrumModel({**layer(len(INPUT_NAMES_ADTOF), len(OUTPUTS)), "thresholds": [0.5] * len(OUTPUTS),
                       "features": INPUT_NAMES_ADTOF, "groove": layer(len(GROOVE_NAMES), len(OUTPUTS))})
    feats = rng.normal(size=(10, 46)).astype(np.float32)   # o retrato acústico (FEATURE_NAMES)
    ticks = np.arange(10) * 96.0
    assert model.uses_adtof
    with pytest.raises(ValueError):
        model.predict_song(feats, ticks, 192)
    p = model.predict_song(feats, ticks, 192, np.zeros((10, 5), dtype=np.float32))
    assert p.shape == (10, len(OUTPUTS))


def test_extra_onsets_join_the_detected_ones():
    from ai_charter.analysis.drums import HOP, MERGE_FRAMES, ONSET_LATENCY, _with_extra_onsets

    onsets = np.array([100, 200, 300])
    near, far = (200 + 2) * HOP / SR - ONSET_LATENCY, 250 * HOP / SR - ONSET_LATENCY
    merged, extra = _with_extra_onsets(onsets, np.array([near, far]), SR, 1000)
    assert merged.tolist() == [100, 200, 250, 300] and extra.tolist() == [False, False, True, False]
    assert _with_extra_onsets(onsets, None, SR, 1000)[0] is onsets
    assert MERGE_FRAMES >= 2


def test_reduction_features_see_the_notes_around():
    from ai_charter.charting.drum_reduce import FEATURE_NAMES, features

    res = 192
    notes = [(0, KICK), (0, HIHAT), (48, HIHAT), (96, SNARE)]
    x = features(notes, res, lambda _t: 120.0)
    assert x.shape == (4, len(FEATURE_NAMES))
    col = FEATURE_NAMES.index
    assert x[0, col("peca_kick")] == 1 and x[0, col("junto_hihat")] == 1 and x[0, col("posicao_0")] == 1
    assert x[1, col("depois_1/16_hihat")] == 1 and x[1, col("depois_1/8_snare")] == 1
    assert x[3, col("antes_1/8_hihat")] == 1 and x[3, col("bpm")] == pytest.approx(0.6)
