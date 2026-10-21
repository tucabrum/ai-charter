"""Letras: leitura do arquivo de texto e divisão das palavras em sílabas.

No Rock Band / YARG / Clone Hero, cada nota de vocal recebe um pedaço da letra:
  - sílaba no meio da palavra termina com "-"   ("ca-" "mi-" "nho")
  - "+" numa nota = continuação da sílaba anterior (melisma: uma sílaba, várias notas)

O arquivo de letra é texto simples: uma linha por frase cantada. Se alguma
palavra for dividida errado, escreva-a com hífens (ex.: "sa-ú-de") e eles
serão respeitados.
"""

from __future__ import annotations

import re
import unicodedata

VOWELS = set("aeiouáàâãéêíóôõúüy")
# Encontros consonantais que ficam juntos na mesma sílaba (pr, bl, tr...).
CLUSTERS = {a + b for a in "bcdfgkptv" for b in "lr"} - {"dl", "tl", "vl"}
DIGRAPHS = ("ch", "lh", "nh", "qu", "gu", "qü", "gü")  # "qu"/"gu" só antes de vogal
# Ditongos decrescentes: vogal + i/u (semivogal) ficam na mesma sílaba.
GLIDES = set("iu")
STRONG = set("aeoáéóâêôãõ")


def _is_vowel(ch: str) -> bool:
    return ch.lower() in VOWELS


def _split_consonants(cluster: str) -> int:
    """Quantas consoantes do grupo entre duas vogais vão para a sílaba SEGUINTE."""
    low = cluster.lower()
    if len(low) <= 1:
        return len(low)
    if low[-2:] in DIGRAPHS or low[-2:] in CLUSTERS:
        return 2
    return 1


def syllabify_pt(word: str) -> list[str]:
    """Divide uma palavra em português em sílabas (regras simplificadas da gramática)."""
    if "-" in word:
        return [p for p in word.split("-") if p]
    low = word.lower()
    # "qu"/"gu" antes de vogal: o u se junta à consoante (que, gui, á-gua, a-guen-tar).
    is_vowel = [_is_vowel(c) for c in low]
    for i in range(1, len(low) - 1):
        if low[i] in "uü" and low[i - 1] in "qg" and _is_vowel(low[i + 1]):
            is_vowel[i] = False

    # Núcleos: grupos de vogais seguidas, separando hiatos.
    nuclei: list[tuple[int, int]] = []  # (início, fim exclusivo)
    i = 0
    while i < len(low):
        if not is_vowel[i]:
            i += 1
            continue
        start = i
        i += 1
        while i < len(low) and is_vowel[i]:
            prev, cur = low[i - 1], low[i]
            # Ditongo: vogal forte + i/u (sem acento), ou "ão", "õe", "ãe", "ui", "iu".
            diphthong = (prev in STRONG and cur in GLIDES) or (prev + cur in ("ão", "õe", "ãe", "ui", "iu"))
            # Depois de um ditongo, uma nova vogal começa outra sílaba (ex.: "sai-a").
            if not diphthong or i - start >= 2:
                break
            i += 1
        nuclei.append((start, i))

    if len(nuclei) <= 1:
        return [word]

    # Corta entre os núcleos, decidindo quantas consoantes vão para a sílaba seguinte.
    cuts = []
    for (_, end), (next_start, _) in zip(nuclei, nuclei[1:]):
        cluster = word[end:next_start]
        cuts.append(next_start - _split_consonants(cluster))
    pieces, prev = [], 0
    for cut in cuts:
        pieces.append(word[prev:cut])
        prev = cut
    pieces.append(word[prev:])
    return [p for p in pieces if p]


EN_DIGRAPHS = ("th", "sh", "ch", "ph", "wh", "ck", "gh")
EN_CONTRACTIONS = ("'s", "'re", "'ll", "'ve", "'d", "'m", "n't", "'t")


def syllabify_en(word: str) -> list[str]:
    """Heurística simples para inglês: um núcleo por grupo de vogais ("e" final é mudo)."""
    if "-" in word:
        return [p for p in word.split("-") if p]
    # Contrações (there's, I'll, don't): divide só a palavra base e cola o final de volta.
    for ending in EN_CONTRACTIONS:
        if word.lower().endswith(ending) and len(word) > len(ending):
            parts = syllabify_en(word[: -len(ending)])
            parts[-1] += word[-len(ending):]
            return parts
    low = word.lower()
    groups = [m.span() for m in re.finditer(r"[aeiouy]+", low)]
    if len(groups) > 1 and low.endswith("e") and groups[-1] == (len(low) - 1, len(low)) \
            and not low.endswith(("le", "ee")):
        groups.pop()
    if len(groups) <= 1:
        return [word]
    cuts = []
    for (_, end), (next_start, _) in zip(groups, groups[1:]):
        cluster = low[end:next_start]
        if cluster[-2:] in EN_DIGRAPHS:       # "th", "sh"... ficam juntos: no-thing, wa-shing
            take = 2
        elif cluster[:2] in EN_DIGRAPHS and len(cluster) == 2:
            take = 2
        else:
            take = min(len(cluster), _split_consonants(cluster) if len(cluster) > 1 else 1)
        cuts.append(next_start - take)
    pieces, prev = [], 0
    for cut in cuts:
        pieces.append(word[prev:cut])
        prev = cut
    pieces.append(word[prev:])
    # "-le" final leva a consoante anterior junto: lit-tle, ta-ble.
    if len(pieces) > 1 and pieces[-1].lower() == "le" and not _is_vowel(pieces[-2][-1]):
        pieces[-2], pieces[-1] = pieces[-2][:-1], pieces[-2][-1] + pieces[-1]
    return [p for p in pieces if p]


def _strip_punctuation(token: str) -> tuple[str, str, str]:
    """Separa pontuação do começo e do fim da palavra: ("(Oh,", ) -> ("(", "Oh", ",")."""
    m = re.match(r"^(\W*)(.*?)(\W*)$", token, flags=re.UNICODE)
    return m.group(1), m.group(2), m.group(3)


def line_syllables(line: str, language: str = "pt") -> list[str]:
    """Sílabas de uma linha, já no formato do jogo ("ca-" "mi-" "nho")."""
    syllabify = syllabify_pt if language == "pt" else syllabify_en
    result: list[str] = []
    for token in line.split():
        token = unicodedata.normalize("NFC", token)
        before, word, after = _strip_punctuation(token)
        parts = syllabify(word) if word else [token]
        parts[0] = before + parts[0]
        parts[-1] = parts[-1] + after
        result += [p + "-" for p in parts[:-1]] + [parts[-1]]
    return result


def read_lyrics(path, language: str = "pt") -> list[list[str]]:
    """Lê o arquivo de letra: uma lista de sílabas por linha (linhas vazias são ignoradas)."""
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return [line_syllables(line, language) for line in lines if line]
