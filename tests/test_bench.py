"""Benchmark da bateria: comparação por tempo, grade e métricas de jogo."""

import numpy as np

from ai_charter.analysis.tempo import TempoMap
from ai_charter.bench import game_metrics, grid_score, match_times


def test_match_times_uses_each_official_hit_tolerance():
    theirs = [(1.00, "caixa", 0.03), (2.00, "caixa", 0.01), (3.00, "bumbo", 0.03)]
    ours = [(1.02, "caixa"), (2.02, "caixa"), (3.00, "caixa")]
    c = match_times(ours, theirs)
    assert c["caixa"] == [1, 3, 2]     # 2,02 fica fora da tolerância de 2,00; 3,00 é outra peça
    assert c["bumbo"] == [0, 0, 1]


def test_grid_score_spots_double_tempo_and_shifted_bars():
    official = TempoMap(np.arange(0, 40) * 0.5, 192)
    assert grid_score(official, official, 1, 15) == {"batidas": 1.0, "compassos": 1.0, "andamento": 1.0}
    double = TempoMap(np.arange(0, 80) * 0.25, 192)
    assert grid_score(double, official, 1, 15)["andamento"] == 2.0
    shifted = TempoMap(np.concatenate([[0.0], np.arange(1, 40) * 0.5 + 0.5]), 192)   # compasso começa 1 batida depois
    score = grid_score(shifted, official, 1, 15)
    assert score["batidas"] > 0.9 and score["compassos"] < 0.1


def test_game_metrics_on_a_plain_rock_beat():
    res = 192
    hits = set()
    for bar in range(4):
        for eighth in range(8):
            hits.add((bar * 4 * res + eighth * res // 2, "chimbal"))
        hits |= {(bar * 4 * res + res, "caixa"), (bar * 4 * res + 3 * res, "caixa"), (bar * 4 * res, "bumbo")}
    m = game_metrics(hits, res)
    assert m["caixa_fora_2_4"] == 0.0 and m["chimbal_buracos"] == 0.0 and m["tons"] == 0.0
    hits.discard((res // 2, "chimbal"))   # um buraco de 2 colcheias... continua par
    hits.discard((res, "chimbal"))        # ...e agora 3 colcheias = 6 semicolcheias: par
    hits.discard((3 * res // 2, "chimbal"))
    hits.add((5 * res // 4, "chimbal"))   # golpe fora do lugar: 5 + 3 semicolcheias
    assert game_metrics(hits, res)["chimbal_buracos"] > 0


def test_official_bars_follow_time_signatures():
    import mido

    from ai_charter.bench import official_bar_times

    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([mido.MetaMessage("time_signature", numerator=5, denominator=4, time=0),
                                       mido.MetaMessage("time_signature", numerator=4, denominator=4, time=5 * 480),
                                       mido.MetaMessage("end_of_track", time=8 * 480)]))
    tempo = TempoMap(np.arange(0, 20) * 0.5, 480)
    assert official_bar_times(midi, tempo)[:3].tolist() == [0.0, 2.5, 4.5]   # 5/4, depois 4/4
