"""Cifra (como as do Cifra Club): a letra e os acordes da música, para guiar a IA.

Uso:
  ai-charter musica.flac --cifra cifra.txt      # texto copiado do Cifra Club

Uma cifra é a letra com os acordes escritos na linha de cima:

    [Intro] G  D  Em  C

    [Primeira Parte]
    G              D
      Quando eu te vi
    Em           C
      Fiquei sem chão

Daqui saem:
  - a LETRA (só as linhas cantadas, sem acordes, rótulos e tablaturas), que vira a faixa
    de vocal quando não há outra letra;
  - o VOCABULÁRIO de acordes: o reconhecimento de acordes da guitarra e do teclado só
    escolhe entre os acordes da cifra (G7, G7M, G/B e Gadd9 contam como G; Am7 como Am).
    O tom da cifra pode não ser o da gravação (capotraste, afinação meio tom abaixo,
    cifra num tom mais fácil): a cifra é transposta para o tom em que soa a gravação.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NOTE_INDEX = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
# Um acorde: fundamental, sufixo (m, 7, 7M, 9, sus4, dim, °, +, (9), (b5)...) e baixo opcional.
CHORD = re.compile(r"^([A-G])([#b]?)((?:maj|min|m|M|dim|aug|sus|add|º|°|\+|-|[0-9]|\((?:[#b+\-]?\d+M?/?)+\)|[#b](?=\d))*)"
                   r"(?:/([A-G][#b]?))?$")
SECTION = re.compile(r"^\s*\[([^\]]+)\]\s*(.*)$")                 # [Refrão], [Intro] G D Em
TAB_LINE = re.compile(r"^\s*[A-Ga-g][#b]?\s*\|.*[-|]")            # E|---3---|
CAPO = re.compile(r"capotraste\D*(\d+)|capo\D*(\d+)", re.IGNORECASE)
KEY = re.compile(r"^\s*tom\s*:\s*([A-G][#b]?m?)", re.IGNORECASE)
NOISE = re.compile(r"^\s*(parte \d+ de \d+|composição de .*|\(?\d+x\)?|x\d+|tablatura.*|solo.*:)\s*$",
                   re.IGNORECASE)
FILLER = re.compile(r"[()|\-–]|^\d+x$|^x\d+$|^\.+$", re.IGNORECASE)


@dataclass
class Cifra:
    lyrics: list[str] = field(default_factory=list)   # linhas cantadas ("" separa estrofes)
    chords: list[str] = field(default_factory=list)   # acordes na ordem em que aparecem (como escritos)
    capo: int = 0                                     # casa do capotraste (0 = sem)
    key: str | None = None                            # "Tom:" da cifra, se houver

    @property
    def vocabulary(self) -> list[str]:
        """Acordes da música (fundamental + maior/menor), em ordem de aparição, sem repetir."""
        seen = []
        for chord in self.chords:
            simple = simplify(chord)
            if simple and simple not in seen:
                seen.append(simple)
        return seen


def parse_chord(token: str) -> tuple[int, str] | None:
    """ "F#m7(b5)" -> (6, "m"); "Bb/D" -> (10, ""); "G5" -> (7, "5"); não-acorde -> None."""
    m = CHORD.match(token.strip())
    if not m:
        return None
    root = (NOTE_INDEX[m.group(1)] + {"#": 1, "b": -1, "": 0}[m.group(2)]) % 12
    suffix = m.group(3) or ""
    if suffix.startswith("5") and not re.search(r"[mM]", suffix):
        kind = "5"
    elif re.match(r"^(m(?!aj)|min|-|dim|º|°)", suffix):
        kind = "m"      # menor, meio-diminuto e diminuto (na chart, o mais perto é o menor)
    else:
        kind = ""       # maior, com sétima, nona, sus, add, aumentado...
    return root, kind


def simplify(token: str) -> str | None:
    parsed = parse_chord(token)
    return None if parsed is None else NOTE_NAMES[parsed[0]] + parsed[1]


def transpose_name(name: str, semitones: int) -> str:
    root, kind = parse_chord(name)
    return NOTE_NAMES[(root + semitones) % 12] + kind


def _chord_tokens(text: str) -> list[str] | None:
    """Os acordes da linha, se ela for só de acordes (parênteses, "x2" e barras são ignorados)."""
    tokens = [t for t in re.split(r"[\s|]+", text.replace("(", " ").replace(")", " ")) if t]
    chords, other = [], 0
    for t in tokens:
        if parse_chord(t):
            chords.append(t)
        elif not FILLER.search(t) and not re.match(r"^\d+x$|^x\d+$", t, re.IGNORECASE):
            other += 1
    return chords if chords and other == 0 else None


def parse_cifra(text: str) -> Cifra:
    cifra = Cifra()
    blank = True
    started = False   # antes do 1º acorde ou rótulo de seção é o cabeçalho (título, artista...)
    header: list[str] = []
    for raw in text.replace("\r", "").split("\n"):
        line = raw.rstrip()
        if not line.strip():
            if not blank:
                (cifra.lyrics if started else header).append("")
            blank = True
            continue
        capo = CAPO.search(line)
        if capo and len(line) < 60:
            cifra.capo = int(capo.group(1) or capo.group(2))
            continue
        key = KEY.match(line)
        if key:
            cifra.key = key.group(1)
            continue
        section = SECTION.match(line)
        if section:
            rest = section.group(2)
            chords = _chord_tokens(rest) if rest.strip() else []
            cifra.chords += chords or []
            started = True
            if not blank:          # rótulo de seção começa outra estrofe
                cifra.lyrics.append("")
                blank = True
            continue
        if TAB_LINE.match(line) or NOISE.match(line):
            continue
        label, sep, after = line.partition(":")
        chords = _chord_tokens(line) or (_chord_tokens(after) if sep and len(label) < 20 else None)
        if chords:
            cifra.chords += chords
            started = True
            continue
        if not started:
            header.append(line.strip())
            blank = False
            continue
        cifra.lyrics.append(line.strip())
        blank = False
    if not started:   # texto sem nenhum acorde: é só a letra
        cifra.lyrics = header
    while cifra.lyrics and not cifra.lyrics[0]:
        cifra.lyrics.pop(0)
    while cifra.lyrics and not cifra.lyrics[-1]:
        cifra.lyrics.pop()
    return cifra


def read_cifra(path: Path) -> Cifra:
    return parse_cifra(Path(path).read_text(encoding="utf-8-sig", errors="replace"))


def sounding_vocabulary(cifra: Cifra) -> list[str]:
    """Os acordes como soam com o capotraste (formato G na casa 2 soa A)."""
    return [transpose_name(c, cifra.capo) for c in cifra.vocabulary]


def best_transposition(vocabulary: list[str], chroma: np.ndarray) -> tuple[int, float]:
    """Quantos semitons somar aos acordes para bater com a gravação, e a confiança (0-1).

    `chroma`: quanto soa de cada uma das 12 notas na música inteira (média). Compara com a
    soma das notas dos acordes da cifra nos 12 tons possíveis."""
    profile = np.zeros(12)
    for name in vocabulary:
        root, kind = parse_chord(name)
        for interval in {"": (0, 4, 7), "m": (0, 3, 7), "5": (0, 7)}[kind]:
            profile[(root + interval) % 12] += 1
    chroma = chroma / (np.linalg.norm(chroma) or 1)
    scores = []
    for shift in range(12):
        p = np.roll(profile, shift)
        scores.append(float(p @ chroma / (np.linalg.norm(p) or 1)))
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    return (best if best <= 6 else best - 12), scores[best] - scores[int(order[1])]


def prepare(path: Path, y: np.ndarray, sr: int, log=print) -> tuple[Cifra, list[str]]:
    """Lê a cifra e devolve o vocabulário no tom da gravação."""
    import librosa

    cifra = read_cifra(path)
    vocabulary = sounding_vocabulary(cifra)
    if not vocabulary:
        log(f"Cifra {Path(path).name}: nenhum acorde encontrado")
        return cifra, []
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=2048).mean(axis=1)
    shift, margin = best_transposition(vocabulary, chroma)
    if shift and margin > 0.01:
        vocabulary = [transpose_name(c, shift) for c in vocabulary]
    else:
        shift = 0
    log(f"Cifra {Path(path).name}: {len(cifra.lyrics)} linhas de letra, acordes {' '.join(vocabulary)}"
        + (f" (capotraste na casa {cifra.capo})" if cifra.capo else "")
        + (f" (transposta {shift:+d} semitons para o tom da gravação)" if shift else ""))
    return cifra, vocabulary


def lyrics_text(cifra: Cifra) -> str:
    return "\n".join(cifra.lyrics) + "\n"
