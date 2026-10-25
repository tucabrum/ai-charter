"""Alinhamento forçado da letra com a voz: onde começa e termina cada sílaba.

Usa o MMS_FA do torchaudio (wav2vec2 multilíngue da Meta, treinado em mais de 1000 línguas,
português e inglês incluídos). A rede ouve a voz e diz, a cada 20 ms, a chance de cada letra
estar soando; o alinhamento CTC acha o caminho mais provável que passa por TODAS as letras da
letra, em ordem. Assim cada sílaba cai onde ela é cantada, mesmo sem pausas que separem as frases.

Depois, os trechos mal ouvidos (confiança baixa) são alinhados de novo só entre as linhas bem
ouvidas vizinhas (realign_gaps).

Entre as linhas pode ir um "*" ("qualquer coisa"), que absorve o que não está na letra (gritos,
vocal de apoio). O "*" do torchaudio tem probabilidade 1 em todo quadro e engole trechos inteiros
de letra mal ouvida; aqui ele tem uma probabilidade fixa e baixa (STAR_LOGP).

O modelo (1,26 GB) é baixado na primeira vez e fica no cache do torch.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

SAMPLE_RATE = 16000
FRAME = 0.02              # cada saída da rede cobre 20 ms
CHUNK_SECONDS = 10.0      # a rede ouve a música em pedaços (com pedaços longos ela "ensurdece" no meio)
CONTEXT_SECONDS = 1.5     # com um pouco de contexto dos lados
STAR_LOGP: float | None = -4.0   # log-probabilidade do "*" entre as linhas (None = sem "*")
SILENCE_DB = -35.0        # voz abaixo disso (em relação ao pico) é silêncio: nenhuma letra ali
ISLAND_SECONDS = 1.5      # som mais curto que isso no meio de um silêncio longo...
ISLAND_GAP = 4.0          # ...(pelo menos isso de cada lado) é vazamento, não canto: conta como silêncio
ANCHOR_SCORE = 0.3       # linha com confiança abaixo disso é alinhada de novo (realign_gaps)
WORD_SIMILARITY = 0.7     # palavra da letra e palavra ouvida pelo Whisper "batem" a partir disso (0-1)
WINDOW_MARGIN = 1.0       # s de folga na janela de uma linha, antes e depois das palavras que bateram
EDGE_WORD_SECONDS = 0.4   # e mais isso para cada palavra da ponta da linha que o Whisper não ouviu
MIN_ANCHOR_LETTERS = 6    # a linha só ganha janela se as palavras que bateram somam isso de letras
SECONDS_PER_WORD = 0.8    # e se elas cabem em (palavras da linha x isso + 2 s): senão o casamento é furado
NEAR_SECONDS = 20.0       # desempate nas letras repetidas: perto de onde o alinhamento já pôs a palavra
NEAR_BONUS = 0.3
WHISPER_BELOW = 0.2       # música com confiança média abaixo disso foi mal ouvida: chama o Whisper
TRUST_WHISPER_BELOW = 0.3   # só linhas com confiança abaixo disso podem ser corrigidas pelo Whisper
OUTSIDE_SECONDS = 1.0     # linha alinhada a mais que isso fora da janela do Whisper é corrigida
MIN_SYLLABLE_SECONDS = 0.15   # espaço mínimo por sílaba ao alinhar dentro de uma janela
VOWELS = set("aeiouy")

_model = None


@dataclass
class SyllableSpan:
    start: float   # s: começo da sílaba (a primeira consoante)
    end: float     # s
    score: float   # confiança média das letras (0-1)
    vowel: float | None = None   # s: começo da primeira vogal


def model_downloaded() -> bool:
    """O modelo já está no cache do torch? (Senão, o primeiro uso baixa 1,26 GB.)"""
    import torch
    from torchaudio.pipelines import MMS_FA

    return (Path(torch.hub.get_dir()) / "checkpoints" / Path(MMS_FA._path).name).is_file()


def _load_model():
    global _model
    if _model is None:
        from torchaudio.pipelines import MMS_FA

        if not model_downloaded():
            print("    vocals: baixando o modelo de alinhamento MMS_FA (1,26 GB, só na primeira vez)...")
        _model = MMS_FA.get_model(with_star=False).eval()
    return _model


def romanize(text: str) -> str:
    """Só as letras que o modelo conhece (a-z e apóstrofo), sem acentos: "Lá," -> "la"."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if "a" <= ch <= "z" or ch == "'")


def emissions(y: np.ndarray, sr: int, peak: float | None = None) -> np.ndarray:
    """Log-probabilidade de cada letra a cada 20 ms (quadros x símbolos), música inteira.
    `peak`: volume de referência para o silêncio (ver mute_silence); padrão, o pico de `y`."""
    import librosa
    import torch

    model = _load_model()
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    hop = int(CHUNK_SECONDS * SAMPLE_RATE)
    ctx = int(CONTEXT_SECONDS * SAMPLE_RATE)
    per_frame = int(FRAME * SAMPLE_RATE)
    n_frames = len(y) // per_frame
    parts = []
    with torch.inference_mode():
        for start in range(0, len(y), hop):
            a, b = max(0, start - ctx), min(len(y), start + hop + ctx)
            if b - a < 400:      # menos que a janela da rede
                break
            out, _ = model(torch.from_numpy(np.ascontiguousarray(y[a:b], dtype=np.float32))[None])
            first = (start - a) // per_frame
            parts.append(out[0].numpy()[first:first + hop // per_frame])
    em = np.concatenate(parts) if parts else np.zeros((0, 28), dtype=np.float32)
    if len(em) < n_frames:   # completa as pontas com "silêncio" (o símbolo vazio)
        pad = np.full((n_frames - len(em), em.shape[1]), -30.0, dtype=np.float32)
        pad[:, 0] = 0.0
        em = np.concatenate([em, pad])
    return mute_silence(em[:n_frames], y, peak=peak)


def frame_rms(y: np.ndarray, sr: int) -> np.ndarray:
    """Volume (RMS) de cada quadro de 20 ms."""
    per_frame = int(FRAME * sr)
    n = len(y) // per_frame
    return np.sqrt(np.mean(y[:n * per_frame].reshape(n, per_frame) ** 2, axis=1))


def mute_silence(em: np.ndarray, y: np.ndarray, floor_db: float | None = None,
                 peak: float | None = None) -> np.ndarray:
    """Onde a voz está em silêncio (`floor_db` abaixo do pico), nenhuma letra pode soar: sem isso,
    uma letra que a rede não reconheceu pode cair em qualquer ponto de uma introdução muda.
    `y` a 16 kHz. `peak`: o volume máximo da MÚSICA INTEIRA, quando `y` é só um trecho dela (num
    solo instrumental, o pico do próprio trecho é o ruído, e nada seria silêncio)."""
    floor_db = SILENCE_DB if floor_db is None else floor_db
    rms = frame_rms(y, SAMPLE_RATE)[:len(em)]
    n = len(rms)
    if n == 0:
        return em
    ref = peak if peak is not None else float(rms.max())
    db = 20 * np.log10(np.maximum(rms, 1e-10) / max(ref, 1e-10))
    silent = np.zeros(len(em), dtype=bool)
    silent[:n] = _drop_islands(db < floor_db)
    em = em.copy()
    em[silent, 1:] = -30.0
    em[silent, 0] = 0.0
    return em


def _drop_islands(silent: np.ndarray) -> np.ndarray:
    """Marca como silêncio os trechos de som curtos (ISLAND_SECONDS) cercados de silêncio longo
    (ISLAND_GAP) dos dois lados, como um resto de guitarra no stem da voz durante um solo."""
    silent = silent.copy()
    runs, i = [], 0             # (início, fim, é silêncio)
    while i < len(silent):
        j = i
        while j < len(silent) and silent[j] == silent[i]:
            j += 1
        runs.append((i, j, bool(silent[i])))
        i = j
    island, gap = ISLAND_SECONDS / FRAME, ISLAND_GAP / FRAME
    for k in range(1, len(runs) - 1):
        a, b, quiet = runs[k]
        before, after = runs[k - 1][1] - runs[k - 1][0], runs[k + 1][1] - runs[k + 1][0]
        if not quiet and b - a < island and before >= gap and after >= gap:
            silent[a:b] = True
    return silent


def align_syllables(y: np.ndarray, sr: int, lines: list[list[str]],
                    listen: Callable[[], list[tuple[str, float, float]] | None] | None = None,
                    ) -> list[list[SyllableSpan | None]]:
    """Início e fim de cada sílaba (None se ela não tem letra que o modelo conheça, ex.: "...").

    `listen`: devolve as palavras que o Whisper ouviu (texto, início, fim). Só é chamado quando a
    música inteira foi mal ouvida (confiança média < WHISPER_BELOW): aí o alinhamento letra a
    letra se perde fácil (voz com efeito, vazamento de instrumento num solo), e as linhas inseguras
    que caíram longe de onde o Whisper as ouviu são refeitas (fix_with_windows). Nas músicas bem
    ouvidas o Whisper não ajuda: medido nas charts oficiais, ele erra tanto quanto corrige."""
    em = emissions(y, sr)
    spans = align_emissions(em, lines)
    windows: list[tuple[float, float] | None] = [None] * len(lines)
    if listen is not None and lines and np.mean([line_score(sp) for sp in spans]) < WHISPER_BELOW:
        heard = listen()
        if heard:
            windows = line_windows(lines, heard, len(em) * FRAME, word_times(lines, spans))
            spans = fix_with_windows(em, lines, spans, windows)
    peak = float(frame_rms(y, sr).max()) if len(y) else None
    fixed = {k for k, w in enumerate(windows) if w is not None and not outside(spans[k], w)}
    return realign_gaps(spans, lines, lambda a, b: emissions(y[int(a * sr):int(b * sr)], sr, peak), len(y) / sr,
                        fixed=fixed)


def lyric_words(lines: list[list[str]]) -> list[tuple[int, str]]:
    """As palavras da letra (sílabas juntadas), com a linha de cada uma."""
    words: list[tuple[int, str]] = []
    for li, line in enumerate(lines):
        current = ""
        for syl in line:
            current += syl[:-1] if syl.endswith("-") else syl
            if not syl.endswith("-"):
                words.append((li, current))
                current = ""
        if current:
            words.append((li, current))
    return words


def word_times(lines: list[list[str]], spans: list[list[SyllableSpan | None]]) -> list[float | None]:
    """Início de cada palavra da letra (lyric_words) num alinhamento já feito."""
    times: list[float | None] = []
    for line, sp in zip(lines, spans):
        start = None
        for syl, s in zip(line, sp):
            if start is None and s is not None:
                start = s.start
            if not syl.endswith("-"):
                times.append(start)
                start = None
        if line and line[-1].endswith("-"):
            times.append(start)
    return times


def match_words(ours: list[str], heard: list[str], ours_at: list[float | None] | None = None,
                heard_at: list[float] | None = None) -> list[tuple[int, int]]:
    """Pares (palavra da letra, palavra ouvida) que batem, em ordem, com a maior semelhança total.
    Com os tempos (`ours_at`: onde o alinhamento pôs cada palavra da letra; `heard_at`: onde o
    Whisper a ouviu), numa letra com trechos repetidos ganha a repetição mais perto no tempo."""
    import difflib

    a, b = [romanize(w) for w in ours], [romanize(w) for w in heard]
    n, m = len(a), len(b)
    best = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = difflib.SequenceMatcher(None, a[i - 1], b[j - 1]).ratio() if a[i - 1] and b[j - 1] else 0.0
            if ours_at is not None and heard_at is not None and ours_at[i - 1] is not None:
                sim += NEAR_BONUS * max(0.0, 1 - abs(ours_at[i - 1] - heard_at[j - 1]) / NEAR_SECONDS)
            take = best[i - 1, j - 1] + sim if sim >= WORD_SIMILARITY else -1.0
            best[i, j] = max(best[i - 1, j], best[i, j - 1], take)
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        if best[i, j] == best[i - 1, j]:
            i -= 1
        elif best[i, j] == best[i, j - 1]:
            j -= 1
        else:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
    return pairs[::-1]


def line_windows(lines: list[list[str]], heard: list[tuple[str, float, float]], duration: float,
                 expected: list[float | None] | None = None) -> list[tuple[float, float] | None]:
    """Para cada linha da letra, a janela de tempo (s) em que ela pode cair, pelas palavras que o
    Whisper ouviu e que batem com as dela; None para as linhas sem casamento confiável.
    `expected`: onde o alinhamento sem âncoras pôs cada palavra (desempate nas repetições)."""
    words = lyric_words(lines)
    pairs = match_words([w for _, w in words], [w for w, _, _ in heard], expected, [h[1] for h in heard])
    hits: dict[int, list[tuple[int, float, float]]] = {}
    for i, j in pairs:
        hits.setdefault(words[i][0], []).append((i, heard[j][1], heard[j][2]))
    first_word = {}
    for i, (li, _) in enumerate(words):
        first_word.setdefault(li, i)
    windows: list[tuple[float, float] | None] = [None] * len(lines)
    for li, found in hits.items():
        count = sum(1 for l2, _ in words if l2 == li)
        letters = sum(len(romanize(words[i][1])) for i, _, _ in found)
        if letters < MIN_ANCHOR_LETTERS or found[-1][2] - found[0][1] > count * SECONDS_PER_WORD + 2:
            continue
        before = found[0][0] - first_word[li]                      # palavras antes da 1ª que bateu
        after = first_word[li] + count - 1 - found[-1][0]          # e depois da última
        windows[li] = (max(0.0, found[0][1] - WINDOW_MARGIN - EDGE_WORD_SECONDS * before),
                       min(duration, found[-1][2] + WINDOW_MARGIN + EDGE_WORD_SECONDS * after))
    # Janelas em ordem: uma linha não começa antes das palavras ouvidas da anterior acabarem.
    last_heard = 0.0
    for li in range(len(lines)):
        if windows[li] is not None:
            a, b = windows[li]
            windows[li] = (max(a, last_heard), max(b, last_heard + 0.5))
            last_heard = max(last_heard, hits[li][-1][2])
    return windows


def outside(spans: list[SyllableSpan | None], window: tuple[float, float] | None) -> bool:
    """A linha alinhada caiu fora da janela do Whisper (com OUTSIDE_SECONDS de tolerância), e com
    pouca confiança? Bem ouvida, ela fica onde está: quando os dois discordam numa linha bem ouvida,
    o Whisper (que casa palavras soltas e marca o tempo com folga) erra tanto quanto acerta."""
    found = [s for s in spans if s is not None]
    if window is None or not found or line_score(spans) >= TRUST_WHISPER_BELOW:
        return False
    return found[0].start < window[0] - OUTSIDE_SECONDS or found[-1].start > window[1] + OUTSIDE_SECONDS


def fix_with_windows(em: np.ndarray, lines: list[list[str]], spans: list[list[SyllableSpan | None]],
                     windows: list[tuple[float, float] | None]) -> list[list[SyllableSpan | None]]:
    """Corrige o alinhamento da música inteira só onde ele discorda do Whisper: as linhas que caíram
    fora da janela delas. As linhas em que os dois concordam ficam como estão e servem de borda;
    cada trecho entre duas bordas que tem uma linha fora do lugar é alinhado de novo, com as janelas.
    (Alinhar toda linha só dentro da janela piora: a folga da janela pega pedaços das vizinhas.)"""
    out = [list(sp) for sp in spans]
    trusted = [w is not None and not outside(sp, w) for sp, w in zip(spans, windows)]
    total = len(em) * FRAME
    li = 0
    while li < len(lines):
        if trusted[li]:
            li += 1
            continue
        j = li
        while j < len(lines) and not trusted[j]:
            j += 1
        if any(outside(spans[k], windows[k]) for k in range(li, j)):
            before = [s for s in out[li - 1] if s is not None] if li > 0 else []
            after = [s for s in out[j] if s is not None] if j < len(lines) else []
            a = before[-1].end if before else 0.0
            b = after[0].start if after else total
            fa, fb = int(a / FRAME), max(int(a / FRAME) + 1, int(b / FRAME))
            local = [None if w is None else (max(0.0, w[0] - a), min(b, w[1]) - a) for w in windows[li:j]]
            fixed = align_in_windows(em[fa:fb], lines[li:j], local)
            t0 = fa * FRAME
            for k, sp in zip(range(li, j), fixed):
                out[k] = [None if s is None else
                          SyllableSpan(s.start + t0, s.end + t0, s.score, None if s.vowel is None else s.vowel + t0)
                          for s in sp]
        li = j
    return out


def align_in_windows(em: np.ndarray, lines: list[list[str]],
                     windows: list[tuple[float, float] | None]) -> list[list[SyllableSpan | None]]:
    """Alinha cada linha com janela dentro dela, e as linhas sem janela no espaço entre as janelas
    vizinhas. Sem nenhuma janela, é o alinhamento da música inteira."""
    out: list[list[SyllableSpan | None]] = [[None] * len(line) for line in lines]
    total = len(em) * FRAME

    def place(first: int, last: int, a: float, b: float) -> None:
        # Espaço mínimo para as sílabas: com janelas apertadas, as linhas ficariam sem tempo.
        b = min(total, max(b, a + MIN_SYLLABLE_SECONDS * sum(len(line) for line in lines[first:last])))
        fa, fb = int(a / FRAME), max(int(a / FRAME) + 1, int(b / FRAME))
        found = align_emissions(em[fa:fb], lines[first:last])
        t0 = fa * FRAME
        for k, sp in zip(range(first, last), found):
            out[k] = [None if s is None else
                      SyllableSpan(s.start + t0, s.end + t0, s.score, None if s.vowel is None else s.vowel + t0)
                      for s in sp]

    li, prev_end = 0, 0.0
    while li < len(lines):
        if windows[li] is not None:
            a, b = windows[li]
            place(li, li + 1, max(a, prev_end), b)
            found = [s for s in out[li] if s is not None]
            prev_end = found[-1].end if found else max(a, prev_end)
            li += 1
            continue
        j = li
        while j < len(lines) and windows[j] is None:
            j += 1
        nxt = windows[j][0] if j < len(lines) else total
        place(li, j, prev_end, max(nxt, prev_end + FRAME))
        found = [s for k in range(li, j) for s in out[k] if s is not None]
        prev_end = found[-1].end if found else prev_end
        li = j
    return out


def line_score(spans: list[SyllableSpan | None]) -> float:
    found = [s for s in spans if s is not None]
    return float(np.mean([s.score for s in found])) if found else 1.0


def realign_gaps(spans: list[list[SyllableSpan | None]], lines: list[list[str]], emit, duration: float,
                 anchor: float = ANCHOR_SCORE, fixed: set[int] = frozenset()) -> list[list[SyllableSpan | None]]:
    """Segunda passada: as linhas bem ouvidas (confiança >= `anchor`) viram âncoras, e cada trecho de
    linhas mal ouvidas entre duas âncoras é alinhado de novo, só com o áudio entre elas.

    A rede ouve de novo aquele trecho, com os pedaços começando nele; às vezes ela entende o que não
    tinha entendido no meio da música inteira. Fica o resultado com a maior confiança.
    `emit(início, fim)`: emissões do áudio entre os dois instantes (s).
    """
    out = [list(sp) for sp in spans]
    good = [line_score(sp) >= anchor or k in fixed for k, sp in enumerate(spans)]   # `fixed`: já ancoradas
    li = 0
    while li < len(lines):
        if good[li]:
            li += 1
            continue
        j = li
        while j < len(lines) and not good[j]:
            j += 1
        before = [s for s in out[li - 1] if s is not None] if li > 0 else []
        after = [s for s in out[j] if s is not None] if j < len(lines) else []
        t0 = before[-1].end if before else 0.0
        t1 = after[0].start if after else duration
        if t1 - t0 >= 0.5:
            new = align_emissions(emit(t0, t1), lines[li:j])
            if np.mean([line_score(sp) for sp in new]) > np.mean([line_score(out[k]) for k in range(li, j)]):
                for k, sp in zip(range(li, j), new):
                    out[k] = [None if s is None else
                              SyllableSpan(s.start + t0, s.end + t0, s.score, None if s.vowel is None else s.vowel + t0)
                              for s in sp]
        li = j
    return out


def align_emissions(em: np.ndarray, lines: list[list[str]],
                    star_logp: float | None = STAR_LOGP) -> list[list[SyllableSpan | None]]:
    import torch
    import torchaudio.functional as F
    from torchaudio.pipelines import MMS_FA

    vocab = MMS_FA.get_dict(star=None)
    star = len(vocab)
    vowel_ids = {vocab[v] for v in VOWELS}
    tokens: list[int] = []
    owner: list[tuple[int, int] | None] = []      # (linha, sílaba) de cada símbolo; None = "*"
    for li, line in enumerate(lines):
        if star_logp is not None:
            tokens.append(star)
            owner.append(None)
        for si, syl in enumerate(line):
            for ch in romanize(syl):
                tokens.append(vocab[ch])
                owner.append((li, si))
    if star_logp is not None:
        tokens.append(star)
        owner.append(None)
        em = np.concatenate([em, np.full((len(em), 1), star_logp, dtype=em.dtype)], axis=1)
    result: list[list[SyllableSpan | None]] = [[None] * len(line) for line in lines]
    # O CTC precisa de pelo menos um quadro por símbolo (mais um entre símbolos repetidos).
    needed = len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))
    if all(o is None for o in owner) or needed > len(em):
        return result
    path, scores = F.forced_align(torch.from_numpy(np.ascontiguousarray(em))[None],
                                  torch.tensor([tokens], dtype=torch.int32), blank=0)
    spans = F.merge_tokens(path[0], scores[0].exp())
    for k, span in enumerate(spans):      # um span por símbolo, na ordem
        who = owner[k]
        if who is None:
            continue
        li, si = who
        start, end = span.start * FRAME, span.end * FRAME
        is_vowel = tokens[k] in vowel_ids
        old = result[li][si]
        if old is None:
            result[li][si] = SyllableSpan(start, end, span.score, start if is_vowel else None)
        else:                             # soma as confianças; a média vem abaixo
            old.end, old.score = end, old.score + span.score
            if old.vowel is None and is_vowel:
                old.vowel = start
    for li, line in enumerate(lines):
        for si, syl in enumerate(line):
            s = result[li][si]
            if s is not None:
                s.score /= len(romanize(syl))
    return result
