"""Avaliação dos vocais contra charts oficiais (Rock Band/Harmonix, que vêm com o stem da voz).

Uso:
  ai-charter-bench-vocals demucs-cache "pasta com músicas" D:\\ai-charter-cache [--test --limit 60]
  ai-charter-bench-vocals run "pasta com músicas" D:\\ai-charter-cache --test [--report r.json]

O run avalia as músicas que o demucs-cache já separou (o conjunto não muda quando a biblioteca cresce).

Como no uso real: a voz que o Demucs separou do mix (cache/<pasta>/vocals.ogg) e a letra oficial,
tirada da própria chart (já dividida em sílabas, como o usuário escreveria com hífens). Os vocais
são gerados duas vezes, com o mapa de tempo oficial e com o detectado por nós no mix, e comparados
com a chart oficial, sílaba por sílaba, em segundos (os dois mapas de tempo valem igual):
  - colocadas: das sílabas oficiais, quantas aparecem na nossa chart;
  - erro de tempo: distância entre o início da nossa sílaba e o da oficial (mediana e % até 100 ms);
  - nota: a afinação certa, ignorando a oitava (só sílabas cantadas nas duas), no começo da sílaba e
    ao longo das notas oficiais (a cada 50 ms: pega notas que oscilam no meio de um sustain);
  - frases: inícios de frase na sílaba certa (F1) e frases certas no tempo (sobreposição >= 50%);
  - gafes: sílabas no mesmo tick (ou duas numa nota), palavras partidas entre frases e notas sobrepostas.
"""

from __future__ import annotations

import argparse
import bisect
import difflib
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import mido
import numpy as np

from ai_charter.bench import MIX_SR, cache_dir_for, is_test, song_folders, song_mix
from ai_charter.revision import _split, _tempo_map

PHRASE, PHRASE_P2 = 105, 106
MIN_PITCH, MAX_PITCH = 36, 84
ON_TIME = 0.1      # sílaba "no tempo": até 100 ms da oficial
LASSO = Path("Rock Band 3") / "Phoenix - Lasso"


class Syllable:
    """Uma sílaba (sem os "+", que só continuam a anterior): onde e como ela é cantada."""
    __slots__ = ("tick", "time", "text", "pitch", "phrase", "end")

    def __init__(self, tick: int, text: str, pitch: int | None, phrase: int, end: int):
        self.tick, self.text, self.pitch, self.phrase, self.end = tick, text, pitch, phrase, end
        self.time = 0.0


def clean_lyric(text: str) -> str:
    """Tira as marcas do Rock Band: "#"/"^" (falado), "$" (escondida), "=" (hífen de verdade)."""
    text = text.strip().replace("=", "-").replace("§", " ")
    return re.sub(r"[#^*$%]", "", text).strip()


def official_vocals(midi: mido.MidiFile) -> tuple[list[Syllable], list[tuple[int, int]]] | None:
    """Sílabas e frases (tick, fim) da PART VOCALS oficial; None se não houver."""
    track = next((t for t in midi.tracks if t.name == "PART VOCALS"), None)
    if track is None:
        return None
    notes, other = _split(track)
    # As frases ficam na nota 105 ou na 106 (algumas charts usam as duas, com frases diferentes).
    phrases = []
    for start, end in sorted((n.start, n.end) for n in notes if n.pitch in (PHRASE, PHRASE_P2)):
        if phrases and start < phrases[-1][1] and start - phrases[-1][0] < (phrases[-1][1] - phrases[-1][0]) / 2:
            phrases[-1] = (phrases[-1][0], max(end, phrases[-1][1]))   # a mesma frase nas duas notas
        else:
            phrases.append((start, end))
    pitched = {n.start: n for n in notes if MIN_PITCH <= n.pitch <= MAX_PITCH}
    syllables = []
    for tick, msg in other:
        if not msg.is_meta or msg.type not in ("lyrics", "text"):
            continue
        raw = msg.text.strip()
        if not raw or raw.startswith("[") or raw == "+" or tick not in pitched:
            continue
        text = clean_lyric(raw)
        if not text or text == "-":
            continue
        note = pitched[tick]
        spoken = "#" in raw or "^" in raw
        syllables.append(Syllable(tick, text, None if spoken else note.pitch,
                                  _phrase_of(phrases, tick), note.end))
    return (syllables, phrases) if syllables else None


def _phrase_of(phrases: list[tuple[int, int]], tick: int) -> int:
    """Índice da frase que contém o tick (a última que começou antes dele)."""
    return max(0, bisect.bisect_right([p[0] for p in phrases], tick) - 1)


def lyric_lines(syllables: list[Syllable]) -> list[list[str]]:
    """A letra oficial no formato do ai-charter: uma lista de sílabas por frase."""
    lines: list[list[str]] = []
    last = None
    for s in syllables:
        if s.phrase != last:
            lines.append([])
            last = s.phrase
        lines[-1].append(s.text)
    return lines


def our_vocals(track) -> list[Syllable]:
    """As sílabas da nossa VocalTrack (notas com letra e trechos falados)."""
    phrases = sorted((t, t + n) for t, n in track.phrases)
    out = [Syllable(n.tick, n.lyric, n.pitch, _phrase_of(phrases, n.tick), n.tick + n.length)
           for n in track.notes if n.lyric and n.lyric != "+"]
    out += [Syllable(t, text, None, _phrase_of(phrases, t), t) for t, text in track.spoken]
    return sorted(out, key=lambda s: s.tick)


def official_notes(midi: mido.MidiFile) -> list[tuple[int, int, int]]:
    """(início, fim, altura) das notas cantadas oficiais, "+" incluídos."""
    track = next((t for t in midi.tracks if t.name == "PART VOCALS"), None)
    notes, _ = _split(track) if track is not None else ([], [])
    return sorted((n.start, n.end, n.pitch) for n in notes if MIN_PITCH <= n.pitch <= MAX_PITCH)


def pitch_over_time(ours: list[tuple[float, float, int]], theirs: list[tuple[float, float, int]],
                    step: float = 0.05) -> tuple[int, int]:
    """Em instantes a cada `step` s dentro das notas oficiais (s), quantos têm a nossa nota soando com a
    mesma altura (ignorando a oitava). Devolve (certos, total)."""
    starts = [a for a, _, _ in ours]
    hit = total = 0
    for a, b, pitch in theirs:
        for t in np.arange(a + step / 2, b, step):
            total += 1
            k = bisect.bisect_right(starts, t) - 1
            if k >= 0 and ours[k][1] > t and (ours[k][2] - pitch) % 12 == 0:
                hit += 1
    return hit, total


def _token(text: str) -> str:
    return re.sub(r"\W", "", text.lower())


def gaffes(syllables: list[Syllable], notes: list[tuple[int, int]]) -> dict[str, int]:
    """Gafes visíveis no jogo. `notes`: (início, fim) de cada nota cantada, em ticks."""
    stacked = sum(1 for a, b in zip(syllables, syllables[1:]) if a.tick == b.tick)
    stacked += sum(1 for s in syllables if " " in s.text.strip())     # duas palavras numa nota
    split = sum(1 for a, b in zip(syllables, syllables[1:])
                if a.text.endswith("-") and a.phrase != b.phrase)
    notes = sorted(notes)
    overlap = sum(1 for a, b in zip(notes, notes[1:]) if a[1] > b[0])
    return {"mesmo_tick": stacked, "palavra_partida": split, "sobrepostas": overlap}


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    return max(0.0, inter) / max(1e-9, max(a[1], b[1]) - min(a[0], b[0]))


def compare(ours: list[Syllable], theirs: list[Syllable],
            our_phrases: list[tuple[float, float]] = (), their_phrases: list[tuple[float, float]] = ()) -> dict:
    """Compara as sílabas (com `time` em segundos) e devolve contagens que se somam entre músicas.
    `*_phrases`: (início, fim) de cada frase em segundos, para ver se elas estão no lugar certo."""
    sm = difflib.SequenceMatcher(None, [_token(s.text) for s in theirs], [_token(s.text) for s in ours],
                                 autojunk=False)
    pairs = [(a + k, b + k) for a, b, size in sm.get_matching_blocks() for k in range(size)]
    errors = [abs(ours[j].time - theirs[i].time) for i, j in pairs]
    pitched = [(theirs[i].pitch, ours[j].pitch) for i, j in pairs
               if theirs[i].pitch is not None and ours[j].pitch is not None]
    # Inícios de frase, contados nas sílabas oficiais.
    start_t = {i for i in range(len(theirs)) if i == 0 or theirs[i].phrase != theirs[i - 1].phrase}
    start_o = {i for i, j in pairs if j == 0 or ours[j].phrase != ours[j - 1].phrase}
    hit = start_t & start_o
    # Frase certa no tempo: alguma nossa cobre pelo menos metade dela (e não muito mais).
    whole = sum(1 for p in their_phrases if any(_iou(p, q) >= 0.5 for q in our_phrases))
    return {"oficiais": len(theirs), "nossas": len(ours), "colocadas": len(pairs),
            "erros": errors, "no_tempo": sum(e <= ON_TIME for e in errors),
            "com_nota": len(pitched), "nota_certa": sum((a - b) % 12 == 0 for a, b in pitched),
            "frases_oficiais": len(start_t), "frases_nossas": len(start_o), "inicio_certo": len(hit),
            "frases_tempo": len(their_phrases), "frase_certa": whole}


def evaluate(folder: Path, cache: Path, rules, aligner: str = "ctc") -> dict | None:
    """Uma música: a voz do Demucs (cache) com a letra oficial, nos dois mapas de tempo.
    `aligner`: "ctc" (alinhamento forçado, o do ai-charter) ou "atual" (o antigo, notas x sílabas)."""
    import librosa

    from ai_charter.analysis.tempo import estimate_tempo_map
    from ai_charter.analysis.vocals import detect_sung_notes, detect_voice_segments
    from ai_charter.analysis.align import align_syllables
    from ai_charter.charting.vocals import build_aligned_track, build_vocal_track
    from ai_charter.pipeline import ANALYSIS_SR, estimate_tuning

    out = cache_dir_for(cache, folder)
    if not (out / "vocals.ogg").is_file():
        return None
    midi = mido.MidiFile(folder / "notes.mid")
    found = official_vocals(midi)
    if found is None:
        return None
    theirs, phrases = found
    official = _tempo_map(midi)
    for s in theirs:
        s.time = float(official.tick_to_time(s.tick))
    sung_phrases = sorted({s.phrase for s in theirs})
    their_phrases = [tuple(float(t) for t in official.tick_to_time(np.array(phrases[k]))) for k in sung_phrases]
    their_notes = [(*(float(x) for x in official.tick_to_time(np.array([a, b]))), p) for a, b, p in official_notes(midi)]
    lines = lyric_lines(theirs)
    mix = librosa.load(out / "mix.ogg", sr=ANALYSIS_SR, mono=True)[0]
    tuning = estimate_tuning(mix, ANALYSIS_SR)
    detected = estimate_tempo_map(mix, ANALYSIS_SR, rules.general.resolution, "auto")
    y, sr = librosa.load(out / "vocals.ogg", sr=ANALYSIS_SR, mono=True)
    sung = detect_sung_notes(y, sr, rules.vocals, tuning)
    def listen():   # como no ai-charter: o Whisper só entra numa música mal ouvida
        from ai_charter.analysis.transcribe import anchor_words

        try:
            return anchor_words(out / "vocals.ogg", rules.vocals.whisper_model)
        except RuntimeError:   # sem o faster-whisper
            return None

    spans = align_syllables(y, sr, lines, listen) if aligner == "ctc" else None
    result = {"oficial_gafes": gaffes(theirs, [(s.tick, s.end) for s in theirs if s.pitch is not None])}
    for name, tempo in (("oficial", official), ("nosso", detected)):
        if spans is not None:
            track = build_aligned_track(spans, sung, lines, tempo, tempo.resolution, rules.vocals)
        else:
            segments = detect_voice_segments(y, sr, rules.vocals, rules.vocals.phrase_gap_beats * 60 / tempo.average_bpm)
            track = build_vocal_track(sung, segments, lines, tempo, tempo.resolution,
                                      rules.general.allow_triplets, rules.vocals)
        ours = our_vocals(track)
        for s in ours:
            s.time = float(tempo.tick_to_time(s.tick))
        our_phrases = [tuple(float(x) for x in tempo.tick_to_time(np.array([t, t + n]))) for t, n in track.phrases]
        result[name] = compare(ours, theirs, our_phrases, their_phrases)
        result[name]["gafes"] = gaffes(ours, [(n.tick, n.tick + n.length) for n in track.notes])
        result[name]["nota_tempo"], result[name]["amostras"] = pitch_over_time(
            [(*(float(x) for x in tempo.tick_to_time(np.array([n.tick, n.tick + n.length]))), n.pitch)
             for n in track.notes], their_notes)
    return result


def summary(rows: list[dict]) -> dict[str, float]:
    """Totais de várias músicas (cada sílaba pesa igual)."""
    tot = {k: sum(r[k] for r in rows) for k in rows[0] if k not in ("erros", "gafes")}
    errors = [e for r in rows for e in r["erros"]]
    gaff = {k: sum(r["gafes"][k] for r in rows) for k in rows[0]["gafes"]}
    p = tot["inicio_certo"] / max(1, tot["frases_nossas"])
    r = tot["inicio_certo"] / max(1, tot["frases_oficiais"])
    return {
        "colocadas": tot["colocadas"] / max(1, tot["oficiais"]),
        "erro_mediano_ms": 1000 * float(np.median(errors)) if errors else float("nan"),
        "no_tempo": tot["no_tempo"] / max(1, tot["oficiais"]),
        "nota_certa": tot["nota_certa"] / max(1, tot["com_nota"]),
        "nota_no_tempo": tot.get("nota_tempo", 0) / max(1, tot.get("amostras", 0)),
        "frases_f1": 2 * p * r / max(1e-9, p + r),
        "frases_certas": tot["frase_certa"] / max(1, tot["frases_tempo"]),
        **{f"gafe_{k}": v for k, v in gaff.items()},
    }


def demucs_cache(folders: list[Path], cache: Path) -> None:
    """Separa a voz do mix (o mesmo Demucs do ai-charter) e guarda cache/<pasta>/vocals.ogg.
    Reaproveita o mix.ogg do cache da bateria; músicas já prontas são puladas."""
    import librosa
    import soundfile as sf

    from ai_charter.export.audio import require_ffmpeg, to_ogg
    from ai_charter.separation import separate_demucs

    require_ffmpeg()
    for i, folder in enumerate(folders, 1):
        out = cache_dir_for(cache, folder)
        if (out / "vocals.ogg").is_file():
            continue
        start = time.time()
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            mix_wav = Path(tmp) / "mix.wav"
            if (out / "mix.ogg").is_file():
                sf.write(mix_wav, librosa.load(out / "mix.ogg", sr=MIX_SR, mono=False)[0].T, MIX_SR)
            else:
                mix = song_mix(folder)
                if mix is None:
                    continue
                sf.write(mix_wav, mix.T, MIX_SR)
            stems = separate_demucs(mix_wav, Path(tmp))
            out.mkdir(parents=True, exist_ok=True)
            if not (out / "mix.ogg").is_file():
                to_ogg(mix_wav, out / "mix.ogg")
            # OGG pelo ffmpeg (o libsndfile trava gravando OGG longo); só conta como pronta no fim
            to_ogg(stems["vocals"][0], out / "vocals.part.ogg")
            shutil.move(out / "vocals.part.ogg", out / "vocals.ogg")
        print(f"[{i}/{len(folders)}] {folder.name}: {time.time() - start:.0f}s", flush=True)


def select_folders(root: Path, test: bool, limit: int | None) -> list[Path]:
    """As músicas da Harmonix com voz (PART VOCALS + vocals.ogg), escolhidas como na bateria
    (`--test --limit`), mais a Lasso."""
    folders = song_folders(root, "Harmonix")
    if test:
        folders = [f for f in folders if is_test(f.name)]
    if limit and limit < len(folders):
        folders = [folders[i] for i in np.linspace(0, len(folders) - 1, limit).astype(int)]
    lasso = root / LASSO
    if (lasso / "notes.mid").is_file() and lasso not in folders:
        folders.append(lasso)
    keep = []
    for f in folders:
        try:
            names = {t.name for t in mido.MidiFile(f / "notes.mid").tracks}
        except Exception:
            continue
        if "PART VOCALS" in names and (f / "vocals.ogg").is_file():
            keep.append(f)
    return keep


def _evaluate_safe(args):
    folder, cache, rules_path, aligner = args
    from ai_charter.config import load_rules

    start = time.time()
    try:
        return folder, evaluate(folder, cache, load_rules(rules_path), aligner), time.time() - start, None
    except Exception as err:   # uma música com arquivo estranho não para a avaliação
        return folder, None, time.time() - start, err


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ai-charter-bench-vocals", description="Avalia os vocais contra charts oficiais.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in (("run", "gera os vocais com a letra oficial e compara com a chart oficial"),
                        ("demucs-cache", "mix de cada música -> Demucs -> voz separada, guardada em cache")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("folder", type=Path)
        p.add_argument("cache", type=Path)
        p.add_argument("--test", action="store_true", help="só as músicas do conjunto de teste")
        p.add_argument("--limit", type=int, help="demucs-cache: usa só N músicas (espalhadas pela lista); "
                       "run: no máximo N das que já estão no cache")
        if name == "run":
            p.add_argument("--rules", type=Path, help="arquivo .toml com regras personalizadas")
            p.add_argument("--aligner", choices=("ctc", "atual"), default="ctc",
                           help="ctc = alinhamento forçado da letra (padrão); atual = o antigo, notas x sílabas")
            p.add_argument("--workers", type=int, default=1, help="músicas analisadas ao mesmo tempo")
            p.add_argument("--report", type=Path, help="grava o placar por música e o total em JSON")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.command == "demucs-cache":
        demucs_cache(select_folders(args.folder, args.test, args.limit), args.cache)
        return
    # As músicas avaliadas são as que já têm a voz no cache: músicas novas na biblioteca mudariam a
    # escolha espalhada do --limit, e os números deixariam de ser comparáveis.
    folders = [f for f in select_folders(args.folder, args.test, None)
               if (cache_dir_for(args.cache, f) / "vocals.ogg").is_file()][:args.limit]

    from concurrent.futures import ProcessPoolExecutor

    jobs = [(f, args.cache, args.rules, args.aligner) for f in folders]
    per_song: dict[str, dict] = {}
    pool = ProcessPoolExecutor(args.workers) if args.workers > 1 else None
    results = pool.map(_evaluate_safe, jobs) if pool else map(_evaluate_safe, jobs)
    for i, (folder, res, secs, err) in enumerate(results, 1):
        if err is not None:
            print(f"[{i}/{len(folders)}] {folder.name}: erro ({err})", flush=True)
        elif res is not None:
            per_song[folder.name] = res
            s = summary([res["oficial"]])
            print(f"[{i}/{len(folders)}] {folder.name}: no tempo {s['no_tempo']:.2f}, "
                  f"nota {s['nota_certa']:.2f}, frases {s['frases_f1']:.2f} ({secs:.0f}s)", flush=True)
    if pool:
        pool.shutdown()
    if not per_song:
        print("nenhuma música avaliada (rode antes o demucs-cache)")
        return
    totals = {name: summary([r[name] for r in per_song.values()]) for name in ("oficial", "nosso")}
    official_gaffes = {k: sum(r["oficial_gafes"][k] for r in per_song.values())
                       for k in next(iter(per_song.values()))["oficial_gafes"]}
    print(f"\n{len(per_song)} músicas")
    print(f"{'mapa de tempo':14s}" + "".join(f"{k:>16s}" for k in totals["oficial"]))
    for name, row in totals.items():
        print(f"{name:14s}" + "".join(f"{v:16.3f}" if isinstance(v, float) else f"{v:16d}" for v in row.values()))
    print("gafes na chart oficial:", official_gaffes)
    if args.report:
        for r in per_song.values():
            for name in ("oficial", "nosso"):
                r[name]["erros"] = [round(e, 3) for e in r[name]["erros"]]
        args.report.write_text(json.dumps({"total": totals, "gafes_oficial": official_gaffes, "musicas": per_song},
                                          ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
