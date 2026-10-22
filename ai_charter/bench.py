"""Avaliação da bateria contra charts oficiais (ex.: Rock Band, que vêm com os stems da bateria).

Uso:
  ai-charter-bench drums "pasta com músicas" [--limit 40] [--test] [--report relatorio.json]
  ai-charter-bench collect "pasta com músicas" exemplos.npz      # ataques + gabarito, para treinar
  ai-charter-bench train-drums exemplos.npz                      # treina ai_charter/drum_model.json

As músicas se dividem sempre do mesmo jeito em treino (~80%) e teste (~20%, pelo nome): o modelo
nunca vê as de teste, e `drums --test` mede só nelas.

Para cada música (pasta com notes.mid e drums*.ogg): soma os stems da bateria (como o stem único
que o ai-charter recebe), transcreve e chartea com o mapa de tempo da chart oficial e compara o
Expert golpe a golpe, peça por peça, com tolerância de 1/32. Mostra precisão (dos nossos golpes,
quantos estão certos), revocação (dos golpes oficiais, quantos achamos) e F1, por peça e no total.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import zlib
from collections import defaultdict
from pathlib import Path

import mido
import numpy as np

from ai_charter.revision import _split, _tempo_map

# Peças como a chart Pro Drums mostra: lane + prato/tom
PIECES = ("bumbo", "caixa", "chimbal", "tom amarelo", "condução", "tom azul", "ataque", "surdo")
LANE_PIECES = {0: ("bumbo", "bumbo"), 1: ("caixa", "caixa"), 2: ("chimbal", "tom amarelo"),
               3: ("condução", "tom azul"), 4: ("ataque", "surdo")}
EXPERT, TOM_MARKER = 96, {2: 110, 3: 111, 4: 112}


def official_hits(midi: mido.MidiFile) -> set[tuple[int, str]] | None:
    """(tick, peça) da bateria Expert oficial; None se a chart não tem bateria."""
    track = next((t for t in midi.tracks if t.name == "PART DRUMS"), None)
    if track is None:
        return None
    notes, _ = _split(track)
    if any(n.pitch == EXPERT + 5 for n in notes):   # bateria de 5 peças (Guitar Hero): outra distribuição
        return None
    toms = {lane: [(n.start, n.end) for n in notes if n.pitch == marker] for lane, marker in TOM_MARKER.items()}
    hits = set()
    for n in notes:
        lane = 0 if n.pitch == 95 else n.pitch - EXPERT   # 95 = bumbo duplo (Expert+)
        if not 0 <= lane <= 4:
            continue
        is_tom = any(a <= n.start < b for a, b in toms.get(lane, []))
        hits.add((n.start, LANE_PIECES[lane][1 if is_tom else 0]))
    return hits


def our_hits(track) -> set[tuple[int, str]]:
    hits = set()
    for note in track.notes.get("expert", []):
        for lane in note.lanes:
            cymbal = lane in (0, 1) or 64 + lane in note.flags
            hits.add((note.tick, LANE_PIECES[lane][0 if cymbal else 1]))
    return hits


def match(ours: set, theirs: set, tolerance: int) -> dict[str, list[int]]:
    """Por peça: [acertos, nossos, oficiais], casando cada golpe com o mais próximo dentro da tolerância."""
    out = {p: [0, 0, 0] for p in PIECES}
    for piece in PIECES:
        a = sorted(t for t, p in ours if p == piece)
        b = sorted(t for t, p in theirs if p == piece)
        out[piece][1], out[piece][2] = len(a), len(b)
        used, j = set(), 0
        for t in a:
            while j < len(b) and b[j] < t - tolerance:
                j += 1
            k = j
            while k < len(b) and b[k] <= t + tolerance:
                if k not in used:
                    used.add(k)
                    out[piece][0] += 1
                    break
                k += 1
    return out


def score(counts: dict[str, list[int]]) -> dict[str, dict[str, float]]:
    rows = {}
    for piece, (hit, ours, theirs) in counts.items():
        p = hit / ours if ours else 0.0
        r = hit / theirs if theirs else 0.0
        rows[piece] = {"precisão": p, "revocação": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                       "nossos": ours, "oficiais": theirs}
    return rows


def drum_audio(folder: Path, sr: int) -> np.ndarray | None:
    import librosa

    stems = sorted(p for p in folder.iterdir() if p.name.lower().startswith("drums") and p.suffix.lower() in (".ogg", ".opus", ".wav", ".mp3"))
    if not stems:
        return None
    parts = [librosa.load(p, sr=sr, mono=True)[0] for p in stems]
    y = np.zeros(max(len(p) for p in parts), dtype=np.float32)
    for p in parts:
        y[: len(p)] += p
    return y


def evaluate(folder: Path, rules, bleed: tuple[float, float] | None = None) -> dict | None:
    """Placar de uma música (ou None se não dá para avaliar)."""
    import soundfile as sf

    from ai_charter.pipeline import DRUMS_SR, chart_drums

    midi = mido.MidiFile(folder / "notes.mid")
    theirs = official_hits(midi)
    y = drum_audio(folder, DRUMS_SR) if theirs else None
    if not theirs or y is None:
        return None
    if bleed:   # o mesmo vazamento sorteado da coleta: a situação real (bateria separada do mix)
        others = other_audio(folder, DRUMS_SR, len(y))
        if others is not None:
            y = y + np.random.default_rng(zlib.crc32(folder.name.encode("utf-8"))).uniform(*bleed) * others
    with tempfile.TemporaryDirectory() as tmp:   # chart_drums lê de um arquivo; nada vai para a pasta da música
        path = Path(tmp) / "drums.wav"
        sf.write(path, y, DRUMS_SR)
        ours = our_hits(chart_drums(path, _tempo_map(midi), rules, []))
    return match(ours, theirs, midi.ticks_per_beat // 8)


# Charts feitas por programas (como o próprio ai-charter) nunca servem de gabarito
NOT_OFFICIAL = ("ai-charter", "autochart")
# Músicas sempre no teste (casos que acompanhamos de perto)
HOLDOUT = {"phoenix - lasso"}


def charter_of(folder: Path) -> str:
    ini = folder / "song.ini"
    if not ini.is_file():
        return ""
    for line in ini.read_text(encoding="utf-8", errors="ignore").splitlines():
        key, _, value = line.partition("=")
        if key.strip().lower() == "charter":
            return value.strip()
    return ""


def song_folders(root: Path, charter: str | None = None) -> list[Path]:
    """Pastas com notes.mid; sem charts geradas por programa e, se pedido, só de um charter."""
    out = []
    for p in sorted(root.rglob("notes.mid")):
        who = charter_of(p.parent).lower()
        if any(bad in who for bad in NOT_OFFICIAL):
            continue
        if charter and charter.lower() not in who:
            continue
        out.append(p.parent)
    return out


def song_key(name: str) -> str:
    """A música por trás do nome da pasta: sem jogo na frente, sem "(2x Bass Pedal)", em minúsculas."""
    import re

    name = name.replace("\\", "/").split("/")[-1]
    return re.sub(r"\s*[\(\[].*?[\)\]]", "", name).strip().lower()


def is_test(name: str) -> bool:
    """A música é do conjunto de teste? Fixo pelo nome normalizado: a mesma música, em jogos
    diferentes, cai sempre do mesmo lado."""
    key = song_key(name)
    return key in HOLDOUT or zlib.crc32(key.encode("utf-8")) % 5 == 0


def _collect_one(folder: Path, rules, bleed: tuple[float, float]):
    """Ataques de uma música: (retrato, ticks em 192/batida, gabarito, golpes perdidos, golpes)."""
    from ai_charter.analysis.drums import ONSET_LATENCY, HOP, analyze_kit
    from ai_charter.pipeline import DRUMS_SR

    midi = mido.MidiFile(folder / "notes.mid")
    theirs = official_hits(midi)
    y = drum_audio(folder, DRUMS_SR) if theirs else None
    if y is None:
        return None
    gain = np.random.default_rng(zlib.crc32(folder.name.encode("utf-8"))).uniform(*bleed)
    others = other_audio(folder, DRUMS_SR, len(y))
    if others is not None:
        y = y + gain * others
    kit = analyze_kit(y, DRUMS_SR, rules.drum_detection)
    tempo, res = _tempo_map(midi), midi.ticks_per_beat
    ticks = np.asarray(tempo.time_to_tick(kit.onsets * HOP / DRUMS_SR - ONSET_LATENCY), dtype=float)
    labels = np.zeros((len(ticks), len(PIECES)), dtype=np.float32)
    missed = 0
    for t, piece in theirs:
        near = np.flatnonzero(np.abs(ticks - t) <= res // 8)
        if len(near):
            labels[near[np.argmin(np.abs(ticks[near] - t))], PIECES.index(piece)] = 1.0
        else:
            missed += 1
    return kit.features, ticks * 192 / res, labels, missed, len(theirs), gain


def _collect_safe(args):
    folder, rules, bleed = args
    try:
        return folder, _collect_one(folder, rules, bleed), None
    except Exception as err:   # uma música com arquivo estranho não para a coleta
        return folder, None, str(err)


def collect(folders: list[Path], out: Path, rules, bleed: tuple[float, float] = (0.05, 0.4), workers: int = 1) -> None:
    """Para cada ataque: retrato acústico, tick e as peças da chart oficial.

    A bateria do Rock Band é limpa (gravada à parte); a que o ai-charter recebe é separada do mix e
    vaza guitarra, baixo e voz. Por isso cada música entra com os outros stems dela misturados à
    bateria, num volume sorteado (sempre o mesmo para a mesma música) dentro de `bleed`.
    Com workers > 1, várias músicas são analisadas ao mesmo tempo (um processo cada)."""
    from concurrent.futures import ProcessPoolExecutor

    feats, ticks_all, ys, songs, missed, total = [], [], [], [], 0, 0
    jobs = [(f, rules, bleed) for f in folders]
    results = (map(_collect_safe, jobs) if workers <= 1
               else ProcessPoolExecutor(workers).map(_collect_safe, jobs, chunksize=1))
    for i, (folder, res, err) in enumerate(results, 1):
        if err or res is None:
            if err:
                print(f"[{i}/{len(folders)}] {folder.name}: erro ({err})", flush=True)
            continue
        f, t, y, m, n, gain = res
        feats.append(f)
        ticks_all.append(t)
        ys.append(y)
        missed += m
        total += n
        songs += [f"{folder.parent.name}/{folder.name}"] * len(t)   # duas cópias da mesma música não se misturam
        if i % 25 == 0 or i == len(folders):
            print(f"[{i}/{len(folders)}] {folder.name}: {len(t)} ataques (vazamento {gain:.2f})", flush=True)
    np.savez_compressed(out, features=np.vstack(feats), ticks=np.concatenate(ticks_all), y=np.vstack(ys),
                        songs=np.array(songs))
    print(f"{len(set(songs))} músicas, {sum(len(x) for x in feats)} ataques; golpes oficiais sem ataque detectado: "
          f"{missed}/{total} ({missed / max(1, total):.1%})")


def other_audio(folder: Path, sr: int, length: int) -> np.ndarray | None:
    """Os outros stems da música (sem bateria e sem público), somados e com o volume da bateria."""
    import librosa

    stems = [p for p in folder.iterdir() if p.suffix.lower() in (".ogg", ".opus", ".wav", ".mp3")
             and not p.name.lower().startswith(("drums", "crowd", "preview"))]
    if not stems:
        return None
    y = np.zeros(length, dtype=np.float32)
    for p in stems:
        part = librosa.load(p, sr=sr, mono=True)[0][:length]
        y[: len(part)] += part
    return y


# Peças da chart (8) -> saídas do modelo (6): os três tons viram "tom"
def _outputs(y8: np.ndarray) -> np.ndarray:
    idx = {p: i for i, p in enumerate(PIECES)}
    cols = [idx["bumbo"], idx["caixa"], idx["chimbal"], idx["condução"], idx["ataque"]]
    tom = y8[:, [idx["tom amarelo"], idx["tom azul"], idx["surdo"]]].max(axis=1, keepdims=True)
    return np.hstack([y8[:, cols], tom])


def _epochs(rows: int) -> int:
    """Épocas para dar ~70 mil passos de treino: muitas com poucas músicas, poucas com milhares."""
    return int(np.clip(round(70_000 * 512 / max(1, rows)), 8, 60))


def train_drums(data: Path, folds: int = 3) -> None:
    """Treina as duas etapas do modelo de bateria (drum_model.py).

    A etapa da levada aprende com o que a etapa do som acha de músicas que ela NÃO viu no treino
    (validação cruzada em `folds` partes): se aprendesse com as respostas da etapa do som para as
    próprias músicas de treino (quase sempre certas), confiaria demais nela e não aprenderia a
    corrigir os erros que aparecem numa música nova."""
    from ai_charter import drum_model as dm

    d = np.load(data)
    songs = d["songs"]
    cut = np.flatnonzero(songs[1:] != songs[:-1]) + 1    # os ataques de cada música vêm juntos
    starts, ends = np.concatenate([[0], cut]), np.concatenate([cut, [len(songs)]])
    names = [str(songs[s]) for s in starts]
    del songs
    ticks, y = d["ticks"], _outputs(d["y"])
    feats = d["features"]
    x1 = np.empty((len(ticks), len(dm.INPUT_NAMES)), dtype=np.float32)
    for s, e in zip(starts, ends):
        x1[s:e] = dm.model_inputs(feats[s:e], ticks[s:e], 192)
    del feats
    song_of = np.repeat(np.arange(len(names)), ends - starts)
    crc = np.array([zlib.crc32(n.encode("utf-8")) for n in names])
    test = np.array([is_test(n) for n in names])[song_of]
    val = (crc % 10 == 3)[song_of] & ~test
    fit = ~test & ~val
    fold = (crc % folds)[song_of]
    print(f"treino {len(set(song_of[fit]))} músicas, validação {len(set(song_of[val]))}, "
          f"teste {len(set(song_of[test]))}; {fit.sum()} ataques de treino", flush=True)

    epochs = _epochs(int(fit.sum()))
    p1 = np.zeros(y.shape, dtype=np.float32)
    for k in range(folds):
        print(f"etapa do som, parte {k + 1}/{folds} (para a etapa da levada aprender com erros reais)", flush=True)
        params = dm.train(x1, y, np.flatnonzero(fit & (fold != k)), epochs=epochs)
        held = np.flatnonzero(fit & (fold == k))
        p1[held] = dm.mlp(params, x1, held)
    print("etapa do som, modelo final", flush=True)
    sound = dm.train(x1, y, np.flatnonzero(fit), epochs=epochs)
    rest = np.flatnonzero(~fit)
    p1[rest] = dm.mlp(sound, x1, rest)
    del x1

    x2 = np.empty((len(ticks), len(dm.GROOVE_NAMES)), dtype=np.float16)
    for s, e in zip(starts, ends):
        x2[s:e] = dm.groove_inputs(p1[s:e], ticks[s:e], 192)
    print("etapa da levada", flush=True)
    groove = dm.train(x2, y, np.flatnonzero(fit), hidden=64, epochs=epochs)

    # Limiar de cada saída: o que dá o melhor F1 na validação
    val_rows, test_rows = np.flatnonzero(val), np.flatnonzero(test)
    pv, pt = dm.mlp(groove, x2, val_rows), dm.mlp(groove, x2, test_rows)
    pv1, pt1 = p1[val_rows], p1[test_rows]
    grid = np.arange(0.1, 0.9, 0.05)

    def best(p: np.ndarray, k: int) -> float:
        return float(max(grid, key=lambda th: _f1(p[:, k] >= th, y[val_rows, k] > 0)))

    thresholds = [best(pv, k) for k in range(y.shape[1])]
    thresholds1 = [best(pv1, k) for k in range(y.shape[1])]
    print(f"{'saída':10s} {'só som':>7s} {'+levada':>8s}   (F1 por ataque no teste)")
    report = {}
    for k, piece in enumerate(dm.OUTPUTS):
        f_sound = _f1(pt1[:, k] >= thresholds1[k], y[test_rows, k] > 0)
        report[piece] = round(_f1(pt[:, k] >= thresholds[k], y[test_rows, k] > 0), 3)
        print(f"{piece:10s} {f_sound:7.3f} {report[piece]:8.3f}")
    dm.save(sound, np.array(thresholds), {"teste_f1_por_ataque": report,
                                          "musicas_treino": int(len(set(song_of[fit])))}, groove=groove)
    print(f"modelo salvo em {dm.MODEL_PATH}")


def _f1(pred: np.ndarray, true: np.ndarray) -> float:
    tp = float((pred & true).sum())
    return 2 * tp / max(1.0, float(pred.sum() + true.sum()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ai-charter-bench", description="Avalia a bateria contra charts oficiais.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("drums", help="compara a bateria gerada com o Expert oficial de cada música")
    p.add_argument("folder", type=Path)
    p.add_argument("--limit", type=int, help="usa só N músicas (espalhadas pela lista)")
    p.add_argument("--test", action="store_true", help="só as músicas do conjunto de teste")
    p.add_argument("--bleed", action="store_true", help="mistura os outros stems à bateria (vazamento, como no uso real)")
    p.add_argument("--rules", type=Path, help="arquivo .toml com regras personalizadas")
    p.add_argument("--report", type=Path, help="grava o placar por música e o total em JSON")
    c = sub.add_parser("collect", help="junta ataques + gabarito das charts oficiais para treinar")
    c.add_argument("folder", type=Path)
    c.add_argument("out", type=Path)
    c.add_argument("--rules", type=Path)
    c.add_argument("--workers", type=int, default=1, help="músicas analisadas ao mesmo tempo")
    c.add_argument("--charter", help="só charts desse charter (ex.: Harmonix)")
    p.add_argument("--charter", help="só charts desse charter (ex.: Harmonix)")
    t = sub.add_parser("train-drums", help="treina o modelo de bateria (ai_charter/drum_model.json)")
    t.add_argument("data", type=Path)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from ai_charter.config import load_rules

    if args.command == "train-drums":
        train_drums(args.data)
        return
    rules = load_rules(args.rules)
    folders = song_folders(args.folder, args.charter)
    if args.command == "collect":
        collect(folders, args.out, rules, workers=args.workers)
        return
    if args.test:
        folders = [f for f in folders if is_test(f.name)]
    if args.limit and args.limit < len(folders):
        folders = [folders[i] for i in np.linspace(0, len(folders) - 1, args.limit).astype(int)]
    total: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    per_song = {}
    for i, folder in enumerate(folders, 1):
        start = time.time()
        try:
            counts = evaluate(folder, rules, (0.05, 0.4) if args.bleed else None)
        except Exception as err:   # uma música com arquivo estranho não para a avaliação
            print(f"[{i}/{len(folders)}] {folder.name}: erro ({err})")
            continue
        if counts is None:
            continue
        for piece, c in counts.items():
            for k in range(3):
                total[piece][k] += c[k]
        hit = sum(c[0] for c in counts.values())
        ours, theirs = sum(c[1] for c in counts.values()), sum(c[2] for c in counts.values())
        f1 = 2 * hit / (ours + theirs) if ours + theirs else 0.0
        per_song[folder.name] = {"f1": f1, "pecas": score(counts)}
        print(f"[{i}/{len(folders)}] {folder.name}: F1 {f1:.2f} ({time.time() - start:.0f}s)", flush=True)
    rows = score(total)
    hit, ours, theirs = (sum(c[k] for c in total.values()) for k in range(3))
    print(f"\n{len(per_song)} músicas · F1 geral {2 * hit / max(1, ours + theirs):.3f}")
    print(f"{'peça':12s} {'precisão':>9s} {'revocação':>10s} {'F1':>6s} {'nossos':>7s} {'oficiais':>9s}")
    for piece, r in rows.items():
        print(f"{piece:12s} {r['precisão']:9.2f} {r['revocação']:10.2f} {r['f1']:6.2f} {r['nossos']:7d} {r['oficiais']:9d}")
    if args.report:
        args.report.write_text(json.dumps({"total": rows, "musicas": per_song}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
