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

import numpy as np

SAMPLE_RATE = 16000
FRAME = 0.02              # cada saída da rede cobre 20 ms
CHUNK_SECONDS = 10.0      # a rede ouve a música em pedaços (com pedaços longos ela "ensurdece" no meio)
CONTEXT_SECONDS = 1.5     # com um pouco de contexto dos lados
STAR_LOGP: float | None = -4.0   # log-probabilidade do "*" entre as linhas (None = sem "*")
SILENCE_DB = -35.0        # voz abaixo disso (em relação ao pico) é silêncio: nenhuma letra ali
ANCHOR_SCORE = 0.3       # linha com confiança abaixo disso é alinhada de novo (realign_gaps)
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


def emissions(y: np.ndarray, sr: int) -> np.ndarray:
    """Log-probabilidade de cada letra a cada 20 ms (quadros x símbolos), música inteira."""
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
    return mute_silence(em[:n_frames], y)


def mute_silence(em: np.ndarray, y: np.ndarray, floor_db: float | None = None) -> np.ndarray:
    """Onde a voz está em silêncio (`floor_db` abaixo do pico), nenhuma letra pode soar: sem isso,
    uma letra que a rede não reconheceu pode cair em qualquer ponto de uma introdução muda.
    `y` a 16 kHz."""
    floor_db = SILENCE_DB if floor_db is None else floor_db
    per_frame = int(FRAME * SAMPLE_RATE)
    n = min(len(em), len(y) // per_frame)
    if n == 0:
        return em
    rms = np.sqrt(np.mean(y[:n * per_frame].reshape(n, per_frame) ** 2, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-10) / max(float(rms.max()), 1e-10))
    silent = np.zeros(len(em), dtype=bool)
    silent[:n] = db < floor_db
    em = em.copy()
    em[silent, 1:] = -30.0
    em[silent, 0] = 0.0
    return em


def align_syllables(y: np.ndarray, sr: int, lines: list[list[str]]) -> list[list[SyllableSpan | None]]:
    """Início e fim de cada sílaba (None se ela não tem letra que o modelo conheça, ex.: "...")."""
    spans = align_emissions(emissions(y, sr), lines)
    return realign_gaps(spans, lines, lambda a, b: emissions(y[int(a * sr):int(b * sr)], sr), len(y) / sr)


def line_score(spans: list[SyllableSpan | None]) -> float:
    found = [s for s in spans if s is not None]
    return float(np.mean([s.score for s in found])) if found else 1.0


def realign_gaps(spans: list[list[SyllableSpan | None]], lines: list[list[str]], emit, duration: float,
                 anchor: float = ANCHOR_SCORE) -> list[list[SyllableSpan | None]]:
    """Segunda passada: as linhas bem ouvidas (confiança >= `anchor`) viram âncoras, e cada trecho de
    linhas mal ouvidas entre duas âncoras é alinhado de novo, só com o áudio entre elas.

    A rede ouve de novo aquele trecho, com os pedaços começando nele; às vezes ela entende o que não
    tinha entendido no meio da música inteira. Fica o resultado com a maior confiança.
    `emit(início, fim)`: emissões do áudio entre os dois instantes (s).
    """
    out = [list(sp) for sp in spans]
    good = [line_score(sp) >= anchor for sp in spans]
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
