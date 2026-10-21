"""Segunda música de teste: violão, guitarra (riff e solo), baixo, bateria e voz.

Estrutura (100 BPM, compassos de 4 batidas):
  intro    4 compassos  violão
  verso 1  8 compassos  violão + voz
  riff     4 compassos  guitarra elétrica (power chords), sem violão
  solo     8 compassos  guitarra solo POR CIMA do violão (com 1 compasso de pausa no meio)
  verso 2  8 compassos  violão + voz
  final    2 compassos  violão

O violão usa síntese Karplus-Strong (corda pinçada), a guitarra usa distorção,
e a voz usa síntese de formantes (ressonâncias de cada vogal), cantando a LETRA abaixo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, sosfilt

from synth import SR, add, drum

BPM = 100.0
BEAT = 60.0 / BPM
START = 4 * BEAT  # a música começa no 2º compasso do arquivo (alinhada ao grid)
BAR = 4 * BEAT

# (nome, compassos, tem violão, tem guitarra, tem voz)
SECTIONS = [
    ("intro", 4, True, False, False),
    ("verso1", 8, True, False, True),
    ("riff", 4, False, True, False),
    ("solo", 8, True, True, False),
    ("verso2", 8, True, False, True),
    ("final", 2, True, False, False),
]
SOLO_REST_BAR = 4  # dentro do solo, o 5º compasso é só pausa da guitarra

# Letra original (gerada por IA). Cada verso ocupa 2 compassos.
LYRICS = [
    "No caminho da manhã",
    "o sol desperta devagar",
    "levo meu canto pelo mar",
    "sem pressa de chegar",
    "Na janela da cidade",
    "brilha tudo que sonhei",
    "toda noite tem cantiga",
    "pela luz do meu lugar",
]
# Sílabas esperadas (é o que o separador de sílabas do programa deve produzir).
SYLLABLES = [
    ["No", "ca", "mi", "nho", "da", "ma", "nhã"],
    ["o", "sol", "des", "per", "ta", "de", "va", "gar"],
    ["le", "vo", "meu", "can", "to", "pe", "lo", "mar"],
    ["sem", "pres", "sa", "de", "che", "gar"],
    ["Na", "ja", "ne", "la", "da", "ci", "da", "de"],
    ["bri", "lha", "tu", "do", "que", "so", "nhei"],
    ["to", "da", "noi", "te", "tem", "can", "ti", "ga"],
    ["pe", "la", "luz", "do", "meu", "lu", "gar"],
]
# Duração de cada sílaba em colcheias (cada verso soma 14 = 7 batidas + 1 de pausa).
RHYTHMS = {6: [2, 2, 2, 2, 2, 4], 7: [2, 1, 1, 2, 2, 2, 4], 8: [1, 1, 2, 1, 1, 2, 2, 4]}
# Melodia (notas MIDI) de cada verso.
MELODIES = [
    [60, 62, 64, 64, 62, 60, 62],
    [57, 60, 62, 64, 65, 64, 62, 60],
    [60, 62, 64, 67, 65, 64, 62, 64],
    [62, 60, 59, 60, 62, 60],
    [60, 62, 64, 64, 65, 64, 62, 60],
    [64, 65, 67, 65, 64, 62, 60],
    [60, 62, 64, 67, 69, 67, 65, 64],
    [62, 64, 65, 64, 62, 59, 60],
]

CHORDS = {  # violão (cordas soltas) e baixo
    "C": [48, 52, 55, 60, 64], "G": [43, 47, 50, 55, 59, 67],
    "Am": [45, 52, 57, 60, 64], "F": [41, 48, 53, 57, 60, 65],
}
PROGRESSION = ["C", "G", "Am", "F"]
ROOTS = {"C": 36, "G": 43, "Am": 45, "F": 41}

# Solo da guitarra: (batida dentro do compasso, duração em batidas, nota) por compasso.
SOLO = [
    [(0, 1, 72), (1, 0.5, 74), (1.5, 0.5, 76), (2, 2, 79)],
    [(0, 0.5, 77), (0.5, 0.5, 76), (1, 0.5, 74), (1.5, 0.5, 72), (2, 2, 74)],
    [(0, 1, 76), (1, 1, 77), (2, 2, 79)],                 # pausa curta: nada nas batidas 4
    [(0, 0.5, 81), (0.5, 0.5, 79), (1, 0.5, 77), (1.5, 0.5, 76), (2, 1, 74), (3, 1, 72)],
    [],                                                    # compasso inteiro de pausa
    [(0, 1, 72), (1, 1, 74), (2, 1, 76), (3, 1, 77)],
    [(0, 0.5, 79), (0.5, 0.5, 81), (1, 1, 79), (2, 2, 77)],
    [(0, 4, 76)],
]
RIFF = [(0, 40), (0.5, 40), (1, 43), (1.5, 43), (2, 45), (2.5, 45), (3, 43), (3.5, 40)]

# Formantes (Hz) das vogais: F1, F2, F3.
FORMANTS = {"a": (800, 1200, 2700), "e": (450, 1900, 2600), "i": (320, 2200, 3000),
            "o": (480, 850, 2600), "u": (340, 750, 2400)}
VOWEL_MAP = str.maketrans("áàâãéêíóôõúü", "aaaaeeiooouu")


def section_start(name: str) -> float:
    t = START
    for n, bars, *_ in SECTIONS:
        if n == name:
            return t
        t += bars * BAR
    raise KeyError(name)


def total_bars() -> int:
    return sum(s[1] for s in SECTIONS)


# ----------------------------------------------------------------- instrumentos
def pluck(midi: float, length: float, rng: np.random.Generator, brightness: float = 0.5) -> np.ndarray:
    """Corda pinçada (Karplus-Strong): ruído curto realimentado num filtro."""
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    delay = max(2, int(round(SR / f0)))
    n = int(length * SR)
    burst = np.zeros(n)
    burst[:delay] = lfilter([brightness], [1, brightness - 1], rng.uniform(-1, 1, delay))
    a = np.zeros(delay + 2)
    a[0], a[delay], a[delay + 1] = 1.0, -0.4985, -0.4985
    return lfilter([1.0], a, burst)


def strum(chord: str, down: bool, rng: np.random.Generator) -> np.ndarray:
    notes = CHORDS[chord] if down else CHORDS[chord][-4:][::-1]
    out = np.zeros(int(1.6 * SR))
    for i, midi in enumerate(notes):
        add(out, pluck(midi, 1.5, rng), 0.012 * i, 0.35 if down else 0.25)
    return out


def electric(midis: list[float], length: float, vibrato: bool = False) -> np.ndarray:
    """Guitarra distorcida: onda rica em harmônicos passando por saturação."""
    t = np.arange(int(length * SR)) / SR
    wave = np.zeros_like(t)
    for midi in midis:
        f0 = 440.0 * 2 ** ((midi - 69) / 12)
        if vibrato:
            f0 = f0 * 2 ** (0.25 * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - 0.25) / 0.2, 0, 1) / 12)
        phase = 2 * np.pi * np.cumsum(np.ones_like(t) * f0) / SR
        wave += sum(np.sin(h * phase) / h for h in range(1, 8))
    env = np.minimum(1.0, t / 0.004) * np.exp(-t * 0.8) * np.clip((length - t) / 0.03, 0, 1)
    return np.tanh(3.0 * wave * env) * 0.5


CONSONANT = 0.07  # duração das consoantes (s); elas vêm ANTES do tempo, a vogal cai no tempo


def sing(syllable: str, midi: float, length: float, rng: np.random.Generator) -> np.ndarray:
    """Uma sílaba cantada: consoante (ruído ou murmúrio) + vogal com formantes.

    O áudio começa CONSONANT segundos antes da vogal (mesmo sem consoante, fica silêncio).
    """
    text = syllable.lower().translate(VOWEL_MAP)
    vowel = next((ch for ch in text if ch in FORMANTS), "a")
    onset = text[: text.index(vowel)] if vowel in text else ""
    length += CONSONANT
    t = np.arange(int(length * SR)) / SR
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    f0 = f0 * 2 ** (0.3 * np.sin(2 * np.pi * 5.0 * t) * np.clip((t - 0.2) / 0.2, 0, 1) / 12)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    source = sum(np.sin(h * phase) / h ** 1.2 for h in range(1, 40) if f0[0] * h < 5000)
    voice = np.zeros_like(t)
    for freq, gain in zip(FORMANTS[vowel], (1.0, 0.6, 0.3)):
        b, a = butter(2, [freq * 0.9, freq * 1.1], btype="bandpass", fs=SR)
        voice += gain * lfilter(b, a, source)

    env = np.clip((t - CONSONANT) / 0.02, 0, 1) * np.clip((length - t) / 0.05, 0, 1)
    out = voice * env
    if onset:
        noise = rng.standard_normal(int(CONSONANT * SR))
        if onset[0] in "sçcxzfvj" or onset.startswith("ch"):
            burst = sosfilt(butter(4, 3000, btype="highpass", fs=SR, output="sos"), noise) * 0.25
        elif onset[0] in "ptkqbdg":
            burst = np.zeros_like(noise)
            burst[-int(0.015 * SR):] = noise[-int(0.015 * SR):] * 0.4
        else:  # m, n, l, r: murmúrio grave
            burst = sosfilt(butter(2, 400, btype="lowpass", fs=SR, output="sos"), noise) * 0.3
        out[: len(burst)] += burst
    return out / (np.abs(voice).max() + 1e-9) * 0.5


# ----------------------------------------------------------------- renderização
def render(folder: Path) -> dict[str, Path]:
    """Cria mix.flac, os stems (violao, guitarra, bass, drums, vocals) e letra.txt."""
    rng = np.random.default_rng(7)
    total = START + total_bars() * BAR + 3.0
    tracks = {name: np.zeros(int(total * SR)) for name in ("violao", "guitarra", "bass", "drums", "vocals")}

    bar = 0
    verse = 0
    for name, bars, has_violao, has_guitar, has_voice in SECTIONS:
        for b in range(bars):
            at = START + (bar + b) * BAR
            chord = PROGRESSION[b % 4]
            if has_violao:
                pattern = [(0, True), (1, True), (1.5, False), (2.5, False), (3, True), (3.5, False)]
                if name == "final" and b == bars - 1:
                    pattern = [(0, True)]
                for beat, down in pattern:
                    add(tracks["violao"], strum(chord, down, rng), at + beat * BEAT,
                        0.7 if name == "solo" else 1.0)
            if has_guitar and name == "riff":
                for beat, root in RIFF:
                    add(tracks["guitarra"], electric([root, root + 7, root + 12], BEAT * 0.45),
                        at + beat * BEAT, 0.7)
            if has_guitar and name == "solo":
                for beat, dur, midi in SOLO[b]:
                    add(tracks["guitarra"], electric([midi], dur * BEAT * 0.95, vibrato=dur >= 1),
                        at + beat * BEAT, 0.8)
            add(tracks["bass"], electric([ROOTS[chord]], BEAT * 1.9), at, 0.5)
            add(tracks["bass"], electric([ROOTS[chord]], BEAT * 1.9), at + 2 * BEAT, 0.5)
            for beat in range(4):
                add(tracks["drums"], drum("kick" if beat % 2 == 0 else "snare", rng), at + beat * BEAT, 0.7)
                add(tracks["drums"], drum("crash" if (b == 0 and beat == 0) else "hihat", rng), at + beat * BEAT, 0.6)
                add(tracks["drums"], drum("hihat", rng), at + (beat + 0.5) * BEAT, 0.6)
        if has_voice:
            for line in range(4):
                syllables, melody = SYLLABLES[verse], MELODIES[verse]
                t = START + (bar + line * 2) * BAR
                for syl, midi, eighths in zip(syllables, melody, RHYTHMS[len(syllables)]):
                    dur = eighths * BEAT / 2
                    add(tracks["vocals"], sing(syl, midi, dur, rng), t - CONSONANT, 0.75)
                    t += dur
                verse += 1
        bar += bars

    # A mesma escala para mix e stems: assim os stems somados dão a mix, sem cortes.
    mix = sum(tracks.values())
    scale = 1.0 / (np.abs(mix).max() * 1.1)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, signal in tracks.items():
        paths[name] = folder / f"{name}.wav"
        sf.write(paths[name], np.stack([signal, signal], axis=1).astype(np.float32) * scale, SR)
    mix = mix * scale
    paths["mix"] = folder / "mix.flac"
    sf.write(paths["mix"], np.stack([mix, mix], axis=1).astype(np.float32), SR)
    paths["lyrics"] = folder / "letra.txt"
    paths["lyrics"].write_text("\n".join(LYRICS) + "\n", encoding="utf-8")
    return paths
