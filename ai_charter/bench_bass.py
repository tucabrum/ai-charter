"""Avaliação do baixo contra charts oficiais (Rock Band/Harmonix, que vêm com o stem do baixo).

Uso:
  ai-charter-bench-bass demucs-cache "pasta com músicas" D:\\ai-charter-cache [--test --limit 61]
  ai-charter-bench-bass run "pasta com músicas" D:\\ai-charter-cache --test [--audio demucs] [--report r.json]

O run avalia as músicas de teste que já estão no cache da bateria (mix.ogg), sempre as mesmas.

Duas fontes de áudio:
  - stem:   o stem oficial do baixo (rhythm.ogg); mede só o transcritor, sem o erro da separação;
  - demucs: o baixo que o Demucs separou do mix (cache/<pasta>/bass.ogg), como no uso real.
E dois mapas de tempo: o oficial (só o baixo) e o detectado por nós no mix (uso real).

Compara o Expert nota a nota, em segundos (tolerância de 1/32 de nota no tempo da oficial):
  - notas: precisão, revocação e F1 dos ataques, sem olhar a cor;
  - contorno: entre duas notas oficiais seguidas que achamos, a cor sobe, desce ou repete como na
    oficial (a mesma nota na mesma cor; nota mais aguda mais à direita). As lanes são relativas, então
    é isto, e não a cor exata, o que diz se o desenho da linha de baixo está certo;
  - cor exata: a mesma lane da oficial (só informativo);
  - sustain: notas longas na oficial (>= 1/3 de batida) que também são longas na nossa, e o contrário;
  - densidade: nossas notas / oficiais; acordes: % de notas com mais de uma lane;
  - altura (só nas músicas com Pro Bass, que traz corda e casa): a nota que o transcritor ouviu,
    antes das lanes, é a do Pro Bass? Exata, oitava errada, vizinha (até 2 semitons) ou outra.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mido
import numpy as np

from ai_charter.bench import MIX_SR, cache_dir_for, is_test, song_folders, song_mix
from ai_charter.revision import _split, _tempo_map

EXPERT = 96
OPEN = 7                 # nota aberta no .chart: fica abaixo do verde no contorno
SUSTAIN_BEATS = 1 / 3    # na oficial, nota com pelo menos isto é sustain (as curtas vêm com 1/16)
OPEN_STRINGS = (28, 33, 38, 43, 47, 52)   # Pro Bass: Mi1 Lá1 Ré2 Sol2 (e as cordas de cima do formato de 6)


def bass_stem(folder: Path) -> Path | None:
    for p in sorted(folder.iterdir()):
        if p.name.lower().startswith(("rhythm", "bass")) and p.suffix.lower() in (".ogg", ".opus", ".wav", ".mp3"):
            return p
    return None


def official_notes(midi: mido.MidiFile) -> list[tuple[int, tuple[int, ...], int]] | None:
    """(tick, lanes, duração em ticks) do Expert oficial do baixo; None se não tem baixo."""
    track = next((t for t in midi.tracks if t.name == "PART BASS"), None)
    if track is None:
        return None
    notes, _ = _split(track)
    by_tick: dict[int, list] = {}
    for n in notes:
        if EXPERT <= n.pitch <= EXPERT + 4:
            by_tick.setdefault(n.start, []).append((n.pitch - EXPERT, n.end - n.start))
    return [(t, tuple(sorted(l for l, _ in v)), max(d for _, d in v)) for t, v in sorted(by_tick.items())] or None


def _low(lanes: tuple[int, ...]) -> int:
    return min(-1 if l == OPEN else l for l in lanes)


def pair_notes(ours: list[float], theirs: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """(índice nosso, índice oficial): cada nota oficial com a nossa mais próxima dentro da tolerância
    dela, sem repetir. `theirs`: (tempo, tolerância)."""
    widest = max((tol for _, tol in theirs), default=0.0)
    used, pairs, j = set(), [], 0
    for i, t in enumerate(ours):
        while j < len(theirs) and theirs[j][0] < t - widest:
            j += 1
        best, k = None, j
        while k < len(theirs) and theirs[k][0] <= t + widest:
            if k not in used and abs(theirs[k][0] - t) <= theirs[k][1]:
                if best is None or abs(theirs[k][0] - t) < abs(theirs[best][0] - t):
                    best = k
            k += 1
        if best is not None:
            used.add(best)
            pairs.append((i, best))
    return pairs


def compare(ours, theirs) -> dict:
    """`ours`/`theirs`: [(tempo s, lanes, sustain?)], em ordem; `theirs` com a tolerância no fim."""
    pairs = pair_notes([o[0] for o in ours], [(t[0], t[3]) for t in theirs])
    of = {k: i for i, k in pairs}
    contour, matrix = [0, 0], [[0] * 3 for _ in range(3)]   # matriz[oficial][nossa]: desce, repete, sobe
    for k in range(len(theirs) - 1):
        if k in of and k + 1 in of and of[k + 1] > of[k]:
            want = np.sign(_low(theirs[k + 1][1]) - _low(theirs[k][1]))
            got = np.sign(_low(ours[of[k + 1]][1]) - _low(ours[of[k]][1]))
            contour[0] += int(want == got)
            contour[1] += 1
            matrix[int(want) + 1][int(got) + 1] += 1
    exact = sum(_low(ours[i][1]) == _low(theirs[k][1]) for i, k in pairs)
    sus_t = [k for _, k in pairs if theirs[k][2]]
    sus_o = [i for i, _ in pairs if ours[i][2]]
    return {"acertos": len(pairs), "nossas": len(ours), "oficiais": len(theirs),
            "contorno": contour, "contorno_matriz": matrix, "cor_exata": [exact, len(pairs)],
            "sustain_oficial_achado": [sum(ours[of[k]][2] for k in sus_t), len(sus_t)],
            "sustain_nosso_certo": [sum(theirs[k][2] for i, k in pairs if ours[i][2]), len(sus_o)],
            "acordes": [sum(len(o[1]) > 1 for o in ours), sum(len(t[1]) > 1 for t in theirs)],
            "sustains": [sum(o[2] for o in ours), sum(t[2] for t in theirs)]}


def pro_bass(midi: mido.MidiFile) -> list[tuple[int, int, int]] | None:
    """(tick, corda, casa) do Expert do Pro Bass (cordas 0 = Mi grave em diante; casa = velocity - 100)."""
    track = next((t for t in midi.tracks if t.name == "PART REAL_BASS"), None)
    if track is None:
        return None
    notes, _ = _split(track)
    return sorted((n.start, n.pitch - EXPERT, n.velocity - 100) for n in notes
                  if EXPERT <= n.pitch < EXPERT + len(OPEN_STRINGS) and n.velocity >= 100) or None


def _fold(semitones: float) -> int:
    """Desvio de afinação plausível (-7 a +4 semitons): uma oitava de diferença é erro, não afinação."""
    return int((round(semitones) + 7) % 12 - 7)


def pitch_score(detected: list[tuple[float, float]], truth: list[tuple[float, int, int, float]]) -> dict:
    """A altura das notas detectadas contra o Pro Bass. `detected`: (tempo, nota MIDI), em ordem;
    `truth`: (tempo, corda, casa, tolerância). A afinação de cada corda (Drop D, meio tom abaixo...) não
    vem na chart: é a mediana da diferença nas notas casadas, dobrada para -7..+4 (ver `_fold`)."""
    pairs = pair_notes([t for t, _ in detected], [(t, tol) for t, _, _, tol in truth])
    standard = [(detected[i][1], OPEN_STRINGS[truth[k][1]] + truth[k][2], truth[k][1]) for i, k in pairs]
    if not standard:
        return {"notas": 0}
    song = _fold(np.median([d - s for d, s, _ in standard]))
    shift = {}
    for string in {s for _, _, s in standard}:
        diffs = [d - s for d, s, st in standard if st == string]
        shift[string] = _fold(np.median(diffs)) if len(diffs) >= 10 else song
    err = np.array([round(d) - s - shift[st] for d, s, st in standard])
    return {"notas": len(err), "casadas_de": len(truth), "afinação": {str(k): v for k, v in sorted(shift.items())},
            "exata": int(np.sum(err == 0)), "oitava": int(np.sum((err != 0) & (err % 12 == 0))), "oitava_acima": int(np.sum((err > 0) & (err % 12 == 0))),
            "vizinha": int(np.sum(np.isin(err, (-2, -1, 1, 2)))),
            "outra": int(np.sum((err % 12 != 0) & ~np.isin(err, (-2, -1, 1, 2))))}


def evaluate(folder: Path, cache: Path, rules, audio: str, tempo_source: str) -> dict | None:
    import librosa

    from ai_charter.analysis.tempo import estimate_tempo_map
    from ai_charter.pipeline import ANALYSIS_SR, chart_pitched, estimate_tuning

    out = cache_dir_for(cache, folder)
    stem = bass_stem(folder) if audio == "stem" else out / "bass.ogg"
    if stem is None or not stem.is_file() or not (out / "mix.ogg").is_file():
        return None
    midi = mido.MidiFile(folder / "notes.mid")
    official = official_notes(midi)
    if not official:
        return None
    mix = librosa.load(out / "mix.ogg", sr=ANALYSIS_SR, mono=True)[0]
    off_tempo = _tempo_map(midi)
    if tempo_source == "oficial":
        # Os ticks do baixo seguem a resolução das regras; com a grade da chart (480), ela tem de ser a mesma.
        rules = dataclasses.replace(rules, general=dataclasses.replace(rules.general, resolution=off_tempo.resolution))
        tempo = off_tempo
    else:
        tempo = estimate_tempo_map(mix, ANALYSIS_SR, rules.general.resolution, "auto")
    track, _ = chart_pitched([stem], "bass", tempo, rules, [], log=lambda *_: None,
                             tuning=estimate_tuning(mix, ANALYSIS_SR))
    res = midi.ticks_per_beat
    theirs = [(float(off_tempo.tick_to_time(t)), lanes, d >= SUSTAIN_BEATS * res,
               float(off_tempo.tick_to_time(t + res / 8) - off_tempo.tick_to_time(t)))
              for t, lanes, d in official]
    ours = [(float(tempo.tick_to_time(n.tick)), n.lanes, n.length > 0) for n in track.notes.get("expert", [])]
    row = compare(ours, theirs)
    if pro_bass(midi):
        row["altura"] = evaluate_pitch(folder, rules, stem, estimate_tuning(mix, ANALYSIS_SR))
    return row


def evaluate_pitch(folder: Path, rules, stem: Path | None = None, tuning: float | None = None) -> dict | None:
    """Só a altura (o transcritor, antes das lanes; é o que a Fase 2 melhora), contra o Pro Bass.
    Sem `stem`, o stem oficial; sem `tuning`, a afinação medida no próprio stem."""
    import librosa

    from ai_charter.analysis.pitched import detect_pitched_notes
    from ai_charter.pipeline import ANALYSIS_SR, estimate_tuning

    midi = mido.MidiFile(folder / "notes.mid")
    pro = pro_bass(midi)
    stem = stem or bass_stem(folder)
    if not pro or stem is None:
        return None
    tempo, res = _tempo_map(midi), midi.ticks_per_beat
    y = librosa.load(stem, sr=ANALYSIS_SR, mono=True)[0]
    notes = detect_pitched_notes(y, ANALYSIS_SR, rules.detection["bass"],
                                 estimate_tuning(y, ANALYSIS_SR) if tuning is None else tuning)
    lowest: dict[int, tuple[int, int]] = {}
    for tick, string, fret in pro:
        if tick not in lowest or OPEN_STRINGS[string] + fret < OPEN_STRINGS[lowest[tick][0]] + lowest[tick][1]:
            lowest[tick] = (string, fret)
    truth = [(float(tempo.tick_to_time(t)), s, f, float(tempo.tick_to_time(t + res / 8) - tempo.tick_to_time(t)))
             for t, (s, f) in sorted(lowest.items())]
    return pitch_score([(n.time, n.pitch) for n in notes], truth)


def _one(args):
    folder, cache, rules_path, audio, tempo_source = args
    from ai_charter.config import load_rules

    start = time.time()
    try:
        row = evaluate(folder, cache, load_rules(rules_path), audio, tempo_source)
    except Exception as exc:   # uma música com problema não derruba a medição inteira
        return folder.name, {"erro": repr(exc)}, time.time() - start
    return folder.name, row, time.time() - start


def summary(rows: dict[str, dict]) -> dict[str, float]:
    ok = [r for r in rows.values() if r and "erro" not in r]
    tot = lambda key, i=None: sum(r[key] if i is None else r[key][i] for r in ok)
    p = tot("acertos") / max(1, tot("nossas"))
    r = tot("acertos") / max(1, tot("oficiais"))
    ratio = lambda key: tot(key, 0) / max(1, tot(key, 1))
    f1s = [2 * x["acertos"] / max(1, x["nossas"] + x["oficiais"]) for x in ok]
    return {"músicas": len(ok), "precisão": p, "revocação": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "f1_mediana": float(np.median(f1s)) if f1s else 0.0,
            "contorno": ratio("contorno"),
            "contorno_matriz": np.sum([r["contorno_matriz"] for r in ok], axis=0).tolist() if ok else [],
            "cor_exata": ratio("cor_exata"),
            "sustain_oficial_achado": ratio("sustain_oficial_achado"),
            "sustain_nosso_certo": ratio("sustain_nosso_certo"),
            "densidade": tot("nossas") / max(1, tot("oficiais")),
            "acordes_nossos": tot("acordes", 0) / max(1, tot("nossas")),
            "acordes_oficiais": tot("acordes", 1) / max(1, tot("oficiais")),
            "sustains_nossos": tot("sustains", 0) / max(1, tot("nossas")),
            "sustains_oficiais": tot("sustains", 1) / max(1, tot("oficiais")),
            **_pitch_summary([r["altura"] for r in ok if r.get("altura", {}).get("notas")])}


def _pitch_summary(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    total = sum(r["notas"] for r in rows)
    return {"altura_músicas": len(rows), **{f"altura_{k}": sum(r[k] for r in rows) / total
                                           for k in ("exata", "oitava", "oitava_acima", "vizinha", "outra")}}


def run(folders: list[Path], cache: Path, rules_path: str | None, audio: str, tempo_source: str,
        workers: int = 1) -> dict:
    jobs = [(f, cache, rules_path, audio, tempo_source) for f in folders]
    rows: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (name, row, secs) in enumerate(pool.map(_one, jobs), 1):
            rows[name] = row
            if row and "erro" not in row:
                f1 = 2 * row["acertos"] / max(1, row["nossas"] + row["oficiais"])
                print(f"[{i}/{len(jobs)}] {name}: F1 {f1:.3f}  ({secs:.0f}s)", flush=True)
            else:
                print(f"[{i}/{len(jobs)}] {name}: {row.get('erro') if row else 'sem dados'}", flush=True)
    return {"resumo": summary(rows), "músicas": rows}


def demucs_cache(folders: list[Path], cache: Path) -> None:
    """Para cada música: mix -> Demucs -> baixo (e guitarra, para a avaliação dela depois) no cache.
    Músicas já no cache são puladas (dá para parar e continuar)."""
    import soundfile as sf

    from ai_charter.export.audio import require_ffmpeg, to_ogg
    from ai_charter.separation import separate_demucs

    require_ffmpeg()
    for i, folder in enumerate(folders, 1):
        out = cache_dir_for(cache, folder)
        if (out / "bass.ogg").is_file():
            continue
        start = time.time()
        mix = song_mix(folder)
        if mix is None:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            mix_wav = Path(tmp) / "mix.wav"
            sf.write(mix_wav, mix.T, MIX_SR)
            stems = separate_demucs(mix_wav, Path(tmp))
            out.mkdir(parents=True, exist_ok=True)
            if not (out / "mix.ogg").is_file():
                to_ogg(mix_wav, out / "mix.ogg")
            if "guitar" in stems and not (out / "guitar.ogg").is_file():
                to_ogg(stems["guitar"][0], out / "guitar.part.ogg")
                shutil.move(out / "guitar.part.ogg", out / "guitar.ogg")
            to_ogg(stems["bass"][0], out / "bass.part.ogg")
            shutil.move(out / "bass.part.ogg", out / "bass.ogg")   # só conta como pronta no fim
        print(f"[{i}/{len(folders)}] {folder.name}: {time.time() - start:.0f}s", flush=True)


def _pitch_one(args):
    folder, rules_path = args
    from ai_charter.config import load_rules

    try:
        return folder.name, evaluate_pitch(folder, load_rules(rules_path))
    except Exception as exc:
        return folder.name, {"erro": repr(exc)}


def run_pitch(folders: list[Path], rules_path: str | None, workers: int = 1) -> dict:
    """A altura nas músicas com Pro Bass (o stem oficial e o mapa de tempo da chart: não precisa de cache)."""
    rows: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (name, row) in enumerate(pool.map(_pitch_one, [(f, rules_path) for f in folders]), 1):
            rows[name] = row
            if row and row.get("notas"):
                print(f"[{i}/{len(folders)}] {name}: exata {row['exata'] / row['notas']:.2f}  "
                      f"oitava {row['oitava'] / row['notas']:.2f}  afinação {row['afinação']}", flush=True)
    return {"resumo": _pitch_summary([r for r in rows.values() if r and r.get("notas")]), "músicas": rows}


def has_pro_bass(folder: Path) -> bool:
    """Sem abrir o MIDI inteiro: o nome da pista aparece em texto no arquivo."""
    return b"PART REAL_BASS" in (folder / "notes.mid").read_bytes()


def _songs(args) -> list[Path]:
    folders = song_folders(Path(args.songs), args.charter)
    if args.test:
        folders = [f for f in folders if is_test(f.name)]
    if args.command == "altura":
        folders = [f for f in folders if bass_stem(f) is not None and has_pro_bass(f)]
    else:
        folders = [f for f in folders if (cache_dir_for(Path(args.cache), f) / "mix.ogg").is_file()]
    return folders[: args.limit] if args.limit else folders


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ai-charter-bench-bass", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("demucs-cache", "run", "altura"):
        p = sub.add_parser(name)
        p.add_argument("songs")
        if name != "altura":
            p.add_argument("cache")
        p.add_argument("--charter", default="Harmonix")
        p.add_argument("--test", action="store_true", help="só as músicas de teste")
        p.add_argument("--limit", type=int, default=0)
        if name == "run":
            p.add_argument("--audio", choices=("stem", "demucs"), default="stem")
            p.add_argument("--tempo", choices=("oficial", "nosso"), default="oficial")
        if name != "demucs-cache":
            p.add_argument("--rules", default=None)
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    folders = _songs(args)
    if args.command == "demucs-cache":
        demucs_cache(folders, Path(args.cache))
        return
    if args.command == "altura":
        result = run_pitch(folders, args.rules, args.workers)
    else:
        result = run(folders, Path(args.cache), args.rules, args.audio, args.tempo, args.workers)
    print(json.dumps(result["resumo"], ensure_ascii=False, indent=1))
    if args.report:
        Path(args.report).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
