"""Tempo de entrada: compassos de silêncio antes da primeira nota."""

import mido
import numpy as np
import pytest

from ai_charter.analysis.tempo import TempoMap
from ai_charter.charting.common import ChartNote, TrackChart
from ai_charter.export import lead_in
from ai_charter.export.chart_writer import SongInfo, write_chart
from ai_charter.export.midi_writer import write_midi

RES = 192


def tempo_120(beats=40):
    return TempoMap(np.arange(beats) * 0.5, RES)   # 120 BPM: compasso de 2 s


def track(first_tick):
    return TrackChart(notes={d: [ChartNote(first_tick, (0,)), ChartNote(first_tick + RES, (1,), RES)]
                             for d in ("expert", "hard", "medium", "easy")})


@pytest.mark.parametrize("first_tick, bars", [
    (0, 2),            # nota no 0 s: faltam 3 s -> 2 compassos (4 s)
    (2 * RES, 1),      # nota em 1 s: faltam 2 s -> 1 compasso
    (6 * RES, 0),      # nota em 3 s: já basta
])
def test_plan_uses_whole_bars(first_tick, bars):
    plan = lead_in.plan(tempo_120(), first_tick)
    assert plan.bars == bars and plan.ticks == bars * 4 * RES and plan.seconds == pytest.approx(bars * 2.0)


def test_plan_off_without_notes_or_with_zero():
    assert lead_in.plan(tempo_120(), None).bars == 0
    assert lead_in.plan(tempo_120(), 0, min_seconds=0).bars == 0
    assert lead_in.first_tick({"guitar": track(300), "bass": track(100)}) == 100


def notes_with_time(path):
    midi = mido.MidiFile(path)
    out = {}
    for t in midi.tracks:
        now = 0
        for msg in t:
            now += msg.time
            out.setdefault(t.name, []).append((now, msg))
    return midi, out


def test_shift_midi_keeps_setup_and_moves_everything_else(tmp_path):
    path = tmp_path / "notes.mid"
    tempo = TempoMap(np.concatenate([np.arange(4) * 0.5, 2.0 + np.arange(1, 30) * 0.6]), RES)   # 120 -> 100 BPM
    write_midi(path, "Música", tempo, {"guitar": track(0)}, [4 * RES], None, 30 * RES)
    _, before = notes_with_time(path)
    lead_in.shift_midi(path, 8 * RES)
    midi, after = notes_with_time(path)
    sync = [(t, m.type) for t, m in after[midi.tracks[0].name] if m.type in ("set_tempo", "time_signature")]
    assert sync[:2] == [(0, "time_signature"), (0, "set_tempo")]        # configuração no 0
    assert sync[2] == (8 * RES + 3 * RES, "set_tempo")                 # a mudança (na batida 3) anda
    for name, events in before.items():
        moved = [(t, m) for t, m in events if not (t == 0 and m.is_meta and m.type in lead_in.KEEP_AT_START)]
        kept = {(t + 8 * RES, str(m.copy(time=0))) for t, m in moved if m.type != "end_of_track"}
        assert kept <= {(t, str(m.copy(time=0))) for t, m in after[name]}
    first_note = min(t for t, m in after["PART GUITAR"] if m.type == "note_on" and m.velocity)
    assert first_note == 8 * RES
    # 8 batidas a 120 BPM = 4 s de entrada, no mesmo andamento
    assert mido.tick2second(first_note, RES, 500000) == pytest.approx(4.0)


def test_shift_chart(tmp_path):
    path = tmp_path / "notes.chart"
    info = SongInfo(name="Música", artist="Banda")
    write_chart(path, info, tempo_120(), {"guitar": track(0)}, [4 * RES])
    lead_in.shift_chart(path, 4 * RES)
    text = path.read_text(encoding="utf-8-sig")
    assert path.read_bytes().startswith(b"\xef\xbb\xbf") and "\r\n" in path.read_bytes().decode("utf-8-sig")
    assert "  0 = TS 4" in text and "  0 = B 120000" in text                   # o tick 0 fica
    assert f"  {4 * RES} = N 0 0" in text and "  0 = N 0 0" not in text       # a nota andou
    assert f"  {8 * RES} = E \"section" in text                               # a seção também
    assert "Resolution = 192" in text                                          # [Song] intacto


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="precisa do ffmpeg")
def test_pipeline_pads_audio_and_chart_together(tmp_path):
    """Música sintética que já começa tocando: a entrada vale no notes.mid, nos .ogg e no song.ini."""
    import shutil
    import subprocess

    import synth
    from ai_charter.pipeline import Options, run

    paths = synth.render(tmp_path)
    stems = tmp_path / "stems"
    stems.mkdir()
    for name in ("guitar", "bass", "drums"):
        shutil.move(paths[name], stems / f"{name}.wav")

    def length(path):
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                              str(path)], capture_output=True, text=True).stdout
        return float(out)

    results = {}
    for lead in (0, 3.0):
        r = run(Options(input=paths["mix"], output_dir=tmp_path / f"out{lead}", separator="folder", stems_dir=stems,
                        title="Synth", artist="Test", instruments=("guitar",), lead_in=lead), log=lambda *_: None)
        midi = mido.MidiFile(r.song_dir / "notes.mid")
        first = min(sum(m.time for m in t[:i + 1]) for t in midi.tracks if t.name == "PART GUITAR"
                    for i, m in enumerate(t) if m.type == "note_on" and m.velocity)
        ini = (r.song_dir / "song.ini").read_text(encoding="utf-8")
        preview = int(next(l.split("=")[1] for l in ini.splitlines() if l.startswith("preview_start_time")))
        results[lead] = (midi, first, length(r.song_dir / "guitar.ogg"), preview)
    (midi0, first0, len0, prev0), (midi3, first3, len3, prev3) = results[0], results[3.0]
    tempo = next(m.tempo for m in midi0.tracks[0] if m.type == "set_tempo")
    lead_s = (first3 - first0) / midi3.ticks_per_beat * tempo / 1e6       # no andamento do 1º compasso
    assert (first3 - first0) % (4 * midi3.ticks_per_beat) == 0               # compassos inteiros
    assert mido.tick2second(first3, midi3.ticks_per_beat, tempo) >= 3.0
    assert len3 - len0 == pytest.approx(lead_s, abs=0.03)                    # o áudio andou o mesmo
    assert (prev3 - prev0) / 1000 == pytest.approx(lead_s, abs=0.002)
