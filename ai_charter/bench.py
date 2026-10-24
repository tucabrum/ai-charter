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


def _collect_one(folder: Path, rules, bleed: tuple[float, float], adtof_cache: Path | None = None,
                 demucs_cache: Path | None = None):
    """Ataques de uma música: (retrato, ticks em 192/batida, gabarito, golpes perdidos, golpes).
    Com `adtof_cache`, os golpes que o ADTOF ouviu no mix entram também como ataques. Com
    `demucs_cache`, a bateria é a que o Demucs separou do mix (demucs-cache), como no uso real,
    em vez da bateria limpa do jogo com vazamento simulado."""
    from ai_charter.analysis.drums import ONSET_LATENCY, HOP, analyze_kit
    from ai_charter.pipeline import DRUMS_SR

    midi = mido.MidiFile(folder / "notes.mid")
    theirs = official_hits(midi)
    if demucs_cache is not None:
        import librosa

        separated = cache_dir_for(demucs_cache, folder) / "drums.ogg"
        if not theirs or not separated.is_file():
            return None
        y, gain = librosa.load(separated, sr=DRUMS_SR, mono=True)[0], 0.0
    else:
        y = drum_audio(folder, DRUMS_SR) if theirs else None
        if y is None:
            return None
        gain = np.random.default_rng(zlib.crc32(folder.name.encode("utf-8"))).uniform(*bleed)
        others = other_audio(folder, DRUMS_SR, len(y))
        if others is not None:
            y = y + gain * others
    extra = None
    if adtof_cache is not None and (cache_dir_for(adtof_cache, folder) / "adtof_mix.npy").is_file():
        from ai_charter.analysis import adtof

        extra = adtof.hit_times(np.load(cache_dir_for(adtof_cache, folder) / "adtof_mix.npy"))
    kit = analyze_kit(y, DRUMS_SR, rules.drum_detection, extra)
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
    folder, rules, bleed, adtof_cache, demucs_cache = args
    try:
        return folder, _collect_one(folder, rules, bleed, adtof_cache, demucs_cache), None
    except Exception as err:   # uma música com arquivo estranho não para a coleta
        return folder, None, str(err)


def collect(folders: list[Path], out: Path, rules, bleed: tuple[float, float] = (0.05, 0.4), workers: int = 1,
            adtof_cache: Path | None = None, demucs_cache: Path | None = None) -> None:
    """Para cada ataque: retrato acústico, tick e as peças da chart oficial.

    A bateria do Rock Band é limpa (gravada à parte); a que o ai-charter recebe é separada do mix e
    vaza guitarra, baixo e voz. Por isso cada música entra com os outros stems dela misturados à
    bateria, num volume sorteado (sempre o mesmo para a mesma música) dentro de `bleed`.
    Com workers > 1, várias músicas são analisadas ao mesmo tempo (um processo cada)."""
    from concurrent.futures import ProcessPoolExecutor

    feats, ticks_all, ys, songs, missed, total = [], [], [], [], 0, 0
    jobs = [(f, rules, bleed, adtof_cache, demucs_cache) for f in folders]
    tag = " [demucs]" if demucs_cache else ""   # outra versão da mesma música: grupo próprio, mesmo lado da divisão
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
        songs += [f"{folder.parent.name}/{folder.name}{tag}"] * len(t)   # duas cópias da mesma música não se misturam
        if i % 25 == 0 or i == len(folders):
            print(f"[{i}/{len(folders)}] {folder.name}: {len(t)} ataques (vazamento {gain:.2f})", flush=True)
    np.savez_compressed(out, features=np.vstack(feats), ticks=np.concatenate(ticks_all), y=np.vstack(ys),
                        songs=np.array(songs))
    print(f"{len(set(songs))} músicas, {sum(len(x) for x in feats)} ataques; golpes oficiais sem ataque detectado: "
          f"{missed}/{total} ({missed / max(1, total):.1%})")


MIX_SR = 44100


def cache_dir_for(cache: Path, folder: Path) -> Path:
    return cache / f"{folder.parent.name} - {folder.name}"


def song_mix(folder: Path) -> np.ndarray | None:
    """Todos os stems da música somados (sem público e prévia): o "MP3" que o usuário teria."""
    import librosa

    stems = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in (".ogg", ".opus", ".wav", ".mp3")
             and not p.name.lower().startswith(("crowd", "preview"))]
    parts = []
    for p in stems:
        y = librosa.load(p, sr=MIX_SR, mono=False)[0]
        parts.append(np.vstack([y, y]) if y.ndim == 1 else y[:2])
    if not parts:
        return None
    mix = np.zeros((2, max(p.shape[1] for p in parts)), dtype=np.float32)
    for p in parts:
        mix[:, : p.shape[1]] += p
    return mix / max(1.0, float(np.abs(mix).max()) / 0.99)   # sem estourar


def demucs_cache(folders: list[Path], cache: Path) -> None:
    """Para cada música: mix de todos os stems -> Demucs (o mesmo do ai-charter) -> bateria separada.
    Guarda mix.ogg e drums.ogg em cache/<pasta>; músicas já no cache são puladas (dá para parar e
    continuar)."""
    import shutil

    import soundfile as sf

    from ai_charter.export.audio import require_ffmpeg, to_ogg
    from ai_charter.separation import separate_demucs

    require_ffmpeg()
    cache.mkdir(parents=True, exist_ok=True)
    for i, folder in enumerate(folders, 1):
        out = cache_dir_for(cache, folder)
        if (out / "drums.ogg").is_file():
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
            # OGG pelo ffmpeg, como no export (o libsndfile trava gravando OGG longo)
            to_ogg(mix_wav, out / "mix.ogg")
            to_ogg(stems["drums"][0], out / "drums.part.ogg")
            shutil.move(out / "drums.part.ogg", out / "drums.ogg")   # só conta como pronta no fim
        print(f"[{i}/{len(folders)}] {folder.name}: {time.time() - start:.0f}s", flush=True)


def _adtof_one(args):
    folder, cache = args
    from ai_charter.analysis import adtof

    out = cache_dir_for(cache, folder) / "adtof_mix.npy"
    if out.is_file():
        return folder, None
    try:
        import torch

        torch.set_num_threads(2)
        mix = song_mix(folder)
        if mix is None:
            return folder, "sem áudio"
        act = adtof.activations(mix, MIX_SR)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, act)
        return folder, None
    except Exception as err:   # uma música com arquivo estranho não para a coleta
        return folder, str(err)


def adtof_cache(folders: list[Path], cache: Path, workers: int = 1) -> None:
    """Saídas do ADTOF no mix de cada música (cache/<pasta>/adtof_mix.npy), para treinar e avaliar."""
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(f, cache) for f in folders]
    results = map(_adtof_one, jobs) if workers <= 1 else ProcessPoolExecutor(workers).map(_adtof_one, jobs, chunksize=1)
    for i, (folder, err) in enumerate(results, 1):
        if err:
            print(f"[{i}/{len(folders)}] {folder.name}: erro ({err})", flush=True)
        elif i % 50 == 0 or i == len(folders):
            print(f"[{i}/{len(folders)}] {folder.name}", flush=True)


def match_times(ours: list[tuple[float, str]], theirs: list[tuple[float, str, float]]) -> dict[str, list[int]]:
    """Como `match`, mas em segundos: cada golpe oficial tem a sua tolerância (1/32 de nota no tempo
    dele). É o jeito de comparar quando a nossa grade (BPM detectado) não é a da chart oficial."""
    out = {p: [0, 0, 0] for p in PIECES}
    for piece in PIECES:
        a = sorted(t for t, p in ours if p == piece)
        b = sorted((t, tol) for t, p, tol in theirs if p == piece)
        out[piece][1], out[piece][2] = len(a), len(b)
        widest = max((tol for _, tol in b), default=0.0)
        used, j = set(), 0
        for t in a:
            while j < len(b) and b[j][0] < t - widest:
                j += 1
            k = j
            while k < len(b) and b[k][0] <= t + widest:
                if k not in used and abs(b[k][0] - t) <= b[k][1]:
                    used.add(k)
                    out[piece][0] += 1
                    break
                k += 1
    return out


def official_bar_times(midi: mido.MidiFile, tempo) -> np.ndarray:
    """Inícios de compasso da chart oficial, seguindo as fórmulas de compasso (5/4, 3/4, 2/4...)."""
    res, tick, changes = midi.ticks_per_beat, 0, []
    for msg in midi.tracks[0]:
        tick += msg.time
        if msg.type == "time_signature":
            changes.append((tick, msg.numerator * 4 / msg.denominator * res))
    changes = sorted(changes) or [(0, 4 * res)]
    end = max(sum(m.time for m in t) for t in midi.tracks)
    bars, t, k = [], 0.0, 0
    while t <= end:
        while k + 1 < len(changes) and changes[k + 1][0] <= t:
            k += 1
        bars.append(t)
        t += changes[k][1]
    return np.asarray(tempo.tick_to_time(np.array(bars)), dtype=float)


def grid_score(ours, official, start: float, end: float, window: float = 0.07,
               official_bars: np.ndarray | None = None) -> dict[str, float]:
    """A nossa grade contra a oficial entre `start` e `end`: batidas e inícios de compasso que caem a
    até `window` segundos dos oficiais (F1), e a razão entre os andamentos (2 = dobro, 0,5 = metade).
    Sem `official_bars`, os compassos oficiais são contados de 4 em 4 batidas."""
    def f1(a, b):
        a, b = a[(a >= start) & (a <= end)], b[(b >= start) & (b <= end)]
        if not len(a) or not len(b):
            return 0.0
        hit = sum(np.min(np.abs(b - t)) <= window for t in a)
        return float(2 * hit / (len(a) + len(b)))

    a, b = np.asarray(ours.beat_times), np.asarray(official.beat_times)
    inside = lambda x: x[(x >= start) & (x <= end)]
    ratio = float(np.median(np.diff(inside(b))) / np.median(np.diff(inside(a)))) if len(inside(a)) > 2 else 0.0
    bars = b[::4] if official_bars is None else np.asarray(official_bars)
    return {"batidas": f1(a, b), "compassos": f1(a[::4], bars), "andamento": ratio}


def game_metrics(hits: set[tuple[int, str]], res: int) -> dict[str, float]:
    """O que se sente jogando, na grade da própria chart: caixa fora do 2 e do 4, buracos estranhos na
    linha de chimbal (intervalos de 3, 5 ou 7 semicolcheias), quanto é tom."""
    step = res / 4
    snare = [t for t, p in hits if p == "caixa" and abs(t / step - round(t / step)) < 0.25]
    off = sum(round(t / step) % 16 not in (4, 12) for t in snare) / max(1, len(snare))
    hats = sorted(t for t, p in hits if p == "chimbal")
    gaps = [round((b - a) / step) for a, b in zip(hats, hats[1:]) if 0 < b - a <= 4 * res]
    odd = sum(g in (3, 5, 7) for g in gaps) / max(1, len(gaps))
    toms = sum(p in ("tom amarelo", "tom azul", "surdo") for _, p in hits) / max(1, len(hits))
    return {"golpes": float(len(hits)), "caixa_fora_2_4": off, "chimbal_buracos": odd, "tons": toms}


def evaluate_real(folder: Path, cache: Path, rules, tempo_source: str = "nosso") -> dict | None:
    """Uma música como no uso real: a bateria do Demucs (cache), e o mapa de tempo detectado por nós
    no mix (tempo_source="nosso") ou o oficial ("oficial", para separar o erro da grade do resto)."""
    import librosa

    from ai_charter.pipeline import ANALYSIS_SR, chart_drums
    from ai_charter.analysis.tempo import estimate_tempo_map

    out = cache_dir_for(cache, folder)
    if not (out / "drums.ogg").is_file():
        return None
    midi = mido.MidiFile(folder / "notes.mid")
    theirs = official_hits(midi)
    if not theirs:
        return None
    official = _tempo_map(midi)
    if tempo_source == "oficial":
        tempo = official
    else:
        mix = librosa.load(out / "mix.ogg", sr=ANALYSIS_SR, mono=True)[0]
        tempo = estimate_tempo_map(mix, ANALYSIS_SR, rules.general.resolution, "auto")
    act_path = out / "adtof_mix.npy"
    ours = our_hits(chart_drums(out / "drums.ogg", tempo, rules, [], mix=out / "mix.ogg",
                                adtof_act=np.load(act_path) if act_path.is_file() else None))
    eighth = midi.ticks_per_beat / 8
    theirs_t = [(float(official.tick_to_time(t)), p,
                 float(official.tick_to_time(t + eighth) - official.tick_to_time(t))) for t, p in theirs]
    ours_t = [(float(tempo.tick_to_time(t)), p) for t, p in ours]
    times = [t for t, _, _ in theirs_t]
    return {"counts": match_times(ours_t, theirs_t),
            "grade": grid_score(tempo, official, min(times), max(times),
                                official_bars=official_bar_times(midi, official)),
            "jogo_nosso": game_metrics(ours, tempo.resolution),
            "jogo_oficial": game_metrics(theirs, midi.ticks_per_beat)}


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


def _adtof_for_songs(names, starts, ends, ticks, cache: Path, root: Path):
    """Chances do ADTOF (no mix) em cada ataque das músicas da coleta; None nas sem cache."""
    from ai_charter.analysis import adtof

    index = {f"{f.parent.name}/{f.name}": f for f in song_folders(root)}
    out = np.zeros((len(ticks), len(adtof.CLASSES)), dtype=np.float32)
    have = np.zeros(len(names), dtype=bool)
    for i, (name, s, e) in enumerate(zip(names, starts, ends)):
        folder = index.get(name.removesuffix(" [demucs]"))
        act_path = cache_dir_for(cache, folder) / "adtof_mix.npy" if folder else None
        if act_path is None or not act_path.is_file():
            continue
        midi = mido.MidiFile(folder / "notes.mid")
        tempo = _tempo_map(midi)
        times = tempo.tick_to_time(ticks[s:e] * midi.ticks_per_beat / 192)
        out[s:e] = adtof.at_times(np.load(act_path), np.asarray(times))
        have[i] = True
    return out, have


def _load_collections(paths) -> dict[str, np.ndarray]:
    """Uma coleta (.npz) ou várias juntadas, na mesma forma."""
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    parts = [np.load(p) for p in paths]
    return {k: np.concatenate([d[k] for d in parts]) for k in ("features", "ticks", "y", "songs")}


def train_drums(data, folds: int = 3, adtof_cache: Path | None = None, songs_root: Path | None = None) -> None:
    """Treina as duas etapas do modelo de bateria (drum_model.py).

    A etapa da levada aprende com o que a etapa do som acha de músicas que ela NÃO viu no treino
    (validação cruzada em `folds` partes): se aprendesse com as respostas da etapa do som para as
    próprias músicas de treino (quase sempre certas), confiaria demais nela e não aprenderia a
    corrigir os erros que aparecem numa música nova.

    Com `adtof_cache` (e `songs_root`, a pasta das músicas), o modelo também recebe as chances do
    ADTOF no mix (bench adtof-cache) e vai para drum_model_adtof.json; músicas sem cache ficam de fora."""
    from ai_charter import drum_model as dm

    d = _load_collections(data)
    songs = d["songs"]
    cut = np.flatnonzero(songs[1:] != songs[:-1]) + 1    # os ataques de cada música vêm juntos
    starts, ends = np.concatenate([[0], cut]), np.concatenate([cut, [len(songs)]])
    names = [str(songs[s]) for s in starts]
    del songs
    ticks, y = d["ticks"], _outputs(d["y"])
    feats = d["features"]
    adtof_feats, have = None, np.ones(len(names), dtype=bool)
    if adtof_cache is not None:
        adtof_feats, have = _adtof_for_songs(names, starts, ends, ticks, adtof_cache, songs_root)
        print(f"ADTOF: {have.sum()} de {len(names)} músicas com cache", flush=True)
    input_names = dm.INPUT_NAMES_ADTOF if adtof_feats is not None else dm.INPUT_NAMES
    x1 = np.empty((len(ticks), len(input_names)), dtype=np.float32)
    for s, e in zip(starts, ends):
        x1[s:e] = dm.model_inputs(feats[s:e], ticks[s:e], 192, None if adtof_feats is None else adtof_feats[s:e])
    del feats, adtof_feats
    song_of = np.repeat(np.arange(len(names)), ends - starts)
    crc = np.array([zlib.crc32(n.encode("utf-8")) for n in names])
    test = np.array([is_test(n) for n in names])[song_of] & have[song_of]
    val = (crc % 10 == 3)[song_of] & ~test & have[song_of]
    fit = ~test & ~val & have[song_of]
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
    path = dm.ADTOF_MODEL_PATH if input_names == dm.INPUT_NAMES_ADTOF else dm.MODEL_PATH
    dm.save(sound, np.array(thresholds), {"teste_f1_por_ataque": report,
                                          "musicas_treino": int(len(set(song_of[fit])))},
            path=path, groove=groove, features=input_names)
    print(f"modelo salvo em {path}")


def train_tom_lanes(data, adtof_cache: Path, songs_root: Path) -> None:
    """Treina a escolha da cor de cada tom (amarelo, azul, verde) com as charts oficiais:
    drum_tom_lanes.json. Só os golpes que a chart tem como tom entram; o teste mede a cor certa."""
    from ai_charter import drum_model as dm

    d = _load_collections(data)
    songs = d["songs"]
    cut = np.flatnonzero(songs[1:] != songs[:-1]) + 1
    starts, ends = np.concatenate([[0], cut]), np.concatenate([cut, [len(songs)]])
    names = [str(songs[s]) for s in starts]
    del songs
    ticks, feats = d["ticks"], d["features"]
    lanes = d["y"][:, [PIECES.index("tom amarelo"), PIECES.index("tom azul"), PIECES.index("surdo")]]
    adtof_feats, have = _adtof_for_songs(names, starts, ends, ticks, adtof_cache, songs_root)
    xs, ys, tests = [], [], []
    for i, (s, e) in enumerate(zip(starts, ends)):
        rows = np.flatnonzero(lanes[s:e].sum(axis=1) == 1)   # um tom só (sem dois tons juntos)
        if not have[i] or not len(rows):
            continue
        x = dm.tom_lane_inputs(feats[s:e], ticks[s:e], 192, adtof_feats[s:e])
        xs.append(x[rows])
        ys.append(lanes[s:e][rows])
        tests.append(np.full(len(rows), is_test(names[i])))
    x, y, test = np.vstack(xs), np.vstack(ys).astype(np.float32), np.concatenate(tests)
    fit = np.flatnonzero(~test)
    print(f"{len(fit)} golpes de tom para treinar, {int(test.sum())} de teste", flush=True)
    params = dm.train(x, y, fit, hidden=32, epochs=_epochs(len(fit)))
    pred = dm.mlp(params, x, np.flatnonzero(test)).argmax(axis=1)
    acc = float((pred == y[test].argmax(axis=1)).mean())
    print(f"cor certa no teste: {acc:.2f} (ao acaso: 0,33)")
    out = dm._jsonable(params)
    out.update(features=dm.INPUT_NAMES_ADTOF + dm.TOM_CONTEXT_NAMES, lanes=list(dm.TOM_LANES),
               info={"cor_certa_teste": round(acc, 3)})
    dm.TOM_LANE_PATH.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"modelo salvo em {dm.TOM_LANE_PATH}")


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
    c.add_argument("--adtof-cache", type=Path, help="pasta do adtof-cache: golpes do ADTOF entram como ataques")
    c.add_argument("--demucs-cache", type=Path, help="pasta do demucs-cache: usa a bateria separada pelo Demucs")
    p.add_argument("--charter", help="só charts desse charter (ex.: Harmonix)")
    p.add_argument("--cache", type=Path, help="como no uso real: a bateria que o Demucs separou do mix (demucs-cache)")
    p.add_argument("--tempo", choices=("nosso", "oficial"), default="nosso",
                   help="com --cache: mapa de tempo detectado no mix (padrão) ou o da chart oficial")
    t = sub.add_parser("train-drums", help="treina o modelo de bateria (ai_charter/drum_model.json)")
    t.add_argument("data", type=Path, nargs="+", help="uma ou mais coletas (juntadas)")
    t.add_argument("--adtof-cache", type=Path, help="pasta do adtof-cache: treina o modelo com ADTOF")
    t.add_argument("--songs", type=Path, help="com --adtof-cache: a pasta das músicas usada na coleta")
    tl = sub.add_parser("train-tom-lanes", help="treina a cor de cada tom (ai_charter/drum_tom_lanes.json)")
    tl.add_argument("data", type=Path, nargs="+")
    tl.add_argument("--adtof-cache", type=Path, required=True)
    tl.add_argument("--songs", type=Path, required=True)
    m = sub.add_parser("demucs-cache", help="mix de cada música -> Demucs -> bateria separada, guardada em cache")
    m.add_argument("folder", type=Path)
    m.add_argument("cache", type=Path)
    m.add_argument("--charter", help="só charts desse charter (ex.: Harmonix)")
    m.add_argument("--test", action="store_true", help="só as músicas do conjunto de teste")
    m.add_argument("--train", action="store_true", help="só as músicas de treino (fora do teste)")
    m.add_argument("--limit", type=int, help="usa só N músicas (espalhadas pela lista)")
    m.add_argument("--rules", type=Path)
    a = sub.add_parser("adtof-cache", help="saídas do ADTOF no mix de cada música, guardadas em cache")
    a.add_argument("folder", type=Path)
    a.add_argument("cache", type=Path)
    a.add_argument("--charter", help="só charts desse charter (ex.: Harmonix)")
    a.add_argument("--workers", type=int, default=1, help="músicas analisadas ao mesmo tempo")
    a.add_argument("--rules", type=Path)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from ai_charter.config import load_rules

    if args.command == "train-tom-lanes":
        train_tom_lanes(args.data, args.adtof_cache, args.songs)
        return
    if args.command == "train-drums":
        train_drums(args.data, adtof_cache=args.adtof_cache, songs_root=args.songs)
        return
    rules = load_rules(args.rules)
    folders = song_folders(args.folder, args.charter)
    if args.command == "collect":
        if args.demucs_cache:
            folders = [f for f in folders if (cache_dir_for(args.demucs_cache, f) / "drums.ogg").is_file()]
        collect(folders, args.out, rules, workers=args.workers, adtof_cache=args.adtof_cache,
                demucs_cache=args.demucs_cache)
        return
    if args.command == "adtof-cache":
        adtof_cache(folders, args.cache, workers=args.workers)
        return
    if args.test:
        folders = [f for f in folders if is_test(f.name)]
    if getattr(args, "train", False):
        folders = [f for f in folders if not is_test(f.name)]
    if getattr(args, "cache", None) and args.command == "drums":   # sempre as mesmas: as que estão no cache
        folders = [f for f in folders if (cache_dir_for(args.cache, f) / "drums.ogg").is_file()]
    if args.limit and args.limit < len(folders):
        folders = [folders[i] for i in np.linspace(0, len(folders) - 1, args.limit).astype(int)]
    if args.command == "demucs-cache":
        demucs_cache(folders, args.cache)
        return
    total: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    per_song, extras = {}, defaultdict(list)
    for i, folder in enumerate(folders, 1):
        start = time.time()
        try:
            if args.cache:
                real = evaluate_real(folder, args.cache, rules, args.tempo)
                counts = real and real.pop("counts")
            else:
                real, counts = None, evaluate(folder, rules, (0.05, 0.4) if args.bleed else None)
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
        per_song[folder.name] = {"f1": f1, "pecas": score(counts), **(real or {})}
        for group, values in (real or {}).items():
            extras[group].append(values)
        print(f"[{i}/{len(folders)}] {folder.name}: F1 {f1:.2f} ({time.time() - start:.0f}s)", flush=True)
    rows = score(total)
    hit, ours, theirs = (sum(c[k] for c in total.values()) for k in range(3))
    print(f"\n{len(per_song)} músicas · F1 geral {2 * hit / max(1, ours + theirs):.3f}")
    print(f"{'peça':12s} {'precisão':>9s} {'revocação':>10s} {'F1':>6s} {'nossos':>7s} {'oficiais':>9s}")
    for piece, r in rows.items():
        print(f"{piece:12s} {r['precisão']:9.2f} {r['revocação']:10.2f} {r['f1']:6.2f} {r['nossos']:7d} {r['oficiais']:9d}")
    for group, rows_ in extras.items():   # grade e métricas de jogo: média das músicas
        print(f"{group:13s}", "  ".join(f"{k} {np.mean([r[k] for r in rows_]):.2f}" for k in rows_[0]))
    if args.report:
        args.report.write_text(json.dumps({"total": rows, "musicas": per_song}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
