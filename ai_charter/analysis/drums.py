"""Transcrição de bateria: stem da bateria -> batidas de cada peça.

Funciona como os conversores de bateria para MIDI:
  1. acha os ataques (onsets) no stem, em graves, médios e agudos separadamente (um bumbo
     baixinho não some debaixo de um chimbal forte);
  2. descobre QUAIS sons de bateria a música usa: o espectro do que "chega" em cada ataque é
     decomposto por NMF (fatoração em matrizes não negativas) em alguns moldes, um por som do
     kit da gravação (o bumbo, a caixa, cada tom, chimbal, condução, ataque). Os moldes começam
     em formas típicas e nos ataques que essas formas não explicam, e se adaptam ao kit;
  3. dá nome a cada molde pelo que ele é: onde está a energia, a altura (pico nos graves) e quanto
     ressoa depois do ataque. É isso que separa um surdo (altura ~100 Hz, ressoa) do bumbo (abaixo
     de ~95 Hz e curto), e o chimbal (curto) do prato de ataque (ressoa);
  4. cada ataque recebe as peças cujos moldes explicam uma parte relevante dele: bumbo e chimbal
     juntos viram duas batidas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.optimize import nnls

from ai_charter.config import DrumDetection

HOP = 256
LONG_FFT = 4096         # janela longa (93 ms): resolução de ~11 Hz nos graves, para o ESPECTRO
SHORT_FFT = 1024        # janela curta (23 ms): o INSTANTE do ataque e quanto o som ressoa
BANDS_PER_OCTAVE = 6
FMIN, FMAX = 30.0, 16000.0
# O detector marca o ataque ~13 ms ANTES do início real (a janela da STFT "enxerga"
# o ataque antes do seu centro); compensamos esse adiantamento.
ONSET_LATENCY = -0.013
# Quadros (de HOP amostras) em volta do ataque usados para descrever cada som
PRE = slice(-10, -7)          # janela longa, antes do ataque (ela ainda não o alcança)
ATTACK = slice(6, 10)         # janela longa, ~35-55 ms depois: o som que chegou
SHORT_PRE = slice(-4, -2)     # janela curta, ~15-20 ms antes
SHORT_ATTACK = slice(1, 4)    # janela curta, logo depois do ataque
SHORT_LATE = slice(13, 15)    # janela curta, ~80 ms depois (antes da próxima semicolcheia)
CYMBAL_TAIL = slice(25, 52)   # janela curta, ~150-300 ms depois: só prato de ataque ainda soa
# O que já soava antes do golpe também decai um pouco até o ataque ser medido;
# só conta como som novo o que passar disso.
CARRY = 0.85

KICK, SNARE, HIHAT, RIDE, CRASH = "kick", "snare", "hihat", "ride", "crash"
TOM_HIGH, TOM_MID, TOM_LOW = "tom_high", "tom_mid", "tom_low"
TOMS = (TOM_HIGH, TOM_MID, TOM_LOW)
TOM = "tom"             # rótulo provisório, antes de saber quantos tons o kit tem
CYMBAL = "cymbal"       # som de prato: chimbal, condução ou ataque, decidido em cada golpe


@dataclass
class DrumHit:
    time: float
    piece: str       # KICK, SNARE, HIHAT, RIDE, CRASH, TOM_HIGH, TOM_MID ou TOM_LOW
    strength: float  # ~0 a 1, relativo às batidas mais fortes da mesma peça


@dataclass
class DrumSound:
    """Um som do kit da gravação (um molde do NMF)."""
    piece: str
    pitch: float            # Hz do pico abaixo de 600 Hz
    ring: float             # fração do volume que ainda soa ~80 ms depois do ataque
    hits: int               # quantos golpes formaram o grupo desse som
    template: np.ndarray = field(repr=False)
    also: str = ""          # molde de duas peças que sempre soam juntas (ex.: bumbo + prato): a 2ª


# ------------------------------------------------------------------ espectros
def _band_matrix(sr: int, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    """Filtros triangulares de 1/6 de oitava sobre os bins da FFT; devolve (matriz, centros)."""
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n = int(np.floor(np.log2(min(FMAX, sr / 2) / FMIN) * BANDS_PER_OCTAVE))
    centers = FMIN * 2.0 ** (np.arange(n) / BANDS_PER_OCTAVE)
    edges = FMIN * 2.0 ** ((np.arange(-1, n + 1)) / BANDS_PER_OCTAVE)
    m = np.zeros((n, len(freqs)), dtype=np.float32)
    for k in range(n):
        lo, c, hi = edges[k], edges[k + 1], edges[k + 2]
        up = (freqs - lo) / (c - lo)
        down = (hi - freqs) / (hi - c)
        m[k] = np.maximum(0.0, np.minimum(up, down))
        if not m[k].any():
            m[k, np.argmin(np.abs(freqs - c))] = 1.0
    return m, centers


def _band_spectrogram(y: np.ndarray, sr: int, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    """Espectrograma em bandas de 1/6 de oitava (magnitude), em pedaços para caber na memória.

    O quadro i fica centrado em i·HOP amostras (como no librosa), com qualquer n_fft."""
    bands, centers = _band_matrix(sr, n_fft)
    padded = np.pad(y, n_fft // 2)
    n_frames = 1 + len(y) // HOP
    out = np.zeros((len(centers), n_frames), dtype=np.float32)
    window = np.hanning(n_fft).astype(np.float32)
    for start in range(0, n_frames, 2048):
        stop = min(n_frames, start + 2048)
        segment = padded[start * HOP: (stop - 1) * HOP + n_fft]
        if len(segment) < n_fft:
            segment = np.pad(segment, (0, n_fft - len(segment)))
        frames = librosa.util.frame(segment, frame_length=n_fft, hop_length=HOP)[:, : stop - start]
        spectrum = np.abs(np.fft.rfft(frames * window[:, None], axis=0))
        out[:, start: start + spectrum.shape[1]] = bands @ spectrum
    return out, centers


def _onsets(S: np.ndarray, freqs: np.ndarray, delta: float) -> np.ndarray:
    """Quadros dos ataques: picos do fluxo espectral em graves, médios e agudos separadamente.

    S: magnitude da STFT de janela curta."""
    sr_half = freqs[-1]
    log = np.log1p(100.0 * S / (S.max() + 1e-12))
    found: list[int] = []
    for lo, hi in ((20, 200), (200, 3000), (3000, sr_half + 1)):
        band = log[(freqs >= lo) & (freqs < hi)]
        # "superflux": compara com o máximo das frequências vizinhas 2 quadros antes,
        # para a ressonância de um som (que oscila um pouco) não parecer um ataque novo
        ref = maximum_filter1d(band, size=3, axis=0)
        flux = np.maximum(0.0, band[:, 2:] - ref[:, :-2]).mean(axis=0)
        flux = np.concatenate([[0.0, 0.0], flux])
        flux /= np.percentile(flux, 99.5) + 1e-9
        found += librosa.util.peak_pick(flux, pre_max=3, post_max=3, pre_avg=12, post_avg=4,
                                        delta=delta, wait=5).tolist()
    # Os picos de cada faixa juntos; dois a menos de ~20 ms são o mesmo ataque
    peaks: list[int] = []
    for p in sorted(found):
        if not peaks or p - peaks[-1] > 3:
            peaks.append(p)
    peaks = np.array(peaks, dtype=int)
    return peaks[(peaks + PRE.start >= 0) & (peaks + SHORT_LATE.stop < S.shape[1])]


# ------------------------------------------------------------------ sons do kit
def _update_h(V: np.ndarray, W: np.ndarray, H: np.ndarray, iterations: int) -> np.ndarray:
    """Quanto de cada molde há em cada ataque (NMF por divergência KL com os moldes fixos)."""
    ones = W.sum(axis=0)[:, None] + 1e-12
    for _ in range(iterations):
        H *= (W.T @ (V / (W @ H + 1e-12))) / ones
    return H


def _fit_error(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Erro relativo de explicar cada ataque como soma (não negativa) dos moldes."""
    err = np.empty(V.shape[1])
    for i in range(V.shape[1]):
        _, residual = nnls(W, V[:, i])
        err[i] = residual / (np.linalg.norm(V[:, i]) + 1e-12)
    return err


def _discover_sounds(V: np.ndarray, cfg: DrumDetection) -> list[tuple[np.ndarray, np.ndarray]]:
    """Os sons da gravação: grupos de ataques com espectro quase igual, do mais frequente ao
    mais raro. Cada som novo é procurado só entre os ataques que os já achados não explicam.

    Devolve [(molde, ataques do grupo)]. O molde é a média dos golpes do grupo, então o bumbo
    leva junto o próprio corpo (não é picotado em "subgrave" + "corpo" que parecem tons)."""
    shape = np.sqrt(V)
    shape /= np.linalg.norm(shape, axis=0, keepdims=True) + 1e-12
    pick = np.arange(V.shape[1])
    if len(pick) > 3000:          # músicas longas: uma amostra basta para descobrir os sons
        pick = np.linspace(0, len(pick) - 1, 3000).astype(int)
    X, Vp = shape[:, pick].astype(np.float32), V[:, pick]
    similar = (X.T @ X) >= cfg.sound_similarity
    sounds: list[tuple[np.ndarray, np.ndarray]] = []
    open_ = np.ones(len(pick), dtype=bool)
    while len(sounds) < cfg.max_sounds:
        cand = np.flatnonzero(open_)
        counts = similar[np.ix_(cand, cand)].sum(axis=1)
        if not len(cand) or counts.max() < cfg.min_hits:
            break
        seed = cand[int(np.argmax(counts))]
        center = X[:, cand[similar[seed, cand]]].mean(axis=1)
        group = cand[(X[:, cand].T @ (center / np.linalg.norm(center))) >= cfg.sound_similarity]
        if len(group) < cfg.min_hits:
            break
        template = np.median(Vp[:, group] / Vp[:, group].sum(axis=0), axis=1) + 1e-6
        sounds.append((template / template.sum(), pick[group]))
        W = np.stack([t for t, _ in sounds], axis=1)
        open_ = _fit_error(Vp, W) > cfg.explained_error
        open_[group] = False
    return _drop_combinations(sounds, cfg.explained_error)


def _drop_combinations(sounds: list[tuple[np.ndarray, np.ndarray]], tolerance: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tira moldes que são só a soma de outros (ex.: bumbo + chimbal, que tocam muito juntos).

    Como os moldes vêm de golpes reais, a soma de dois deles encaixa bem num molde combinado."""
    changed = True
    while changed and len(sounds) > 2:
        changed = False
        for k in range(len(sounds) - 1, -1, -1):
            others = np.stack([t for j, (t, _) in enumerate(sounds) if j != k], axis=1)
            coef, residual = nnls(others, sounds[k][0])
            if residual / np.linalg.norm(sounds[k][0]) < tolerance and (coef > 0.15 * coef.max()).sum() >= 2:
                del sounds[k]
                changed = True
                break
    return sounds


# ------------------------------------------------------------------ nomes dos sons
def _label(w: np.ndarray, centers: np.ndarray, ring: float, cfg: DrumDetection) -> tuple[str, str, float]:
    """(peça, peça extra, altura) de um molde; ring: quanto ele ressoa ~80 ms depois."""
    air = float(w[centers >= 6000].sum())
    tonal = (centers >= 35) & (centers < 600)
    pitch = float(centers[tonal][np.argmax(w[tonal])])
    # Pratos vivem acima de ~6 kHz; as esteiras da caixa ficam mais embaixo (2-6 kHz).
    # Qual prato (chimbal, condução ou ataque) é decidido golpe a golpe, pela ressonância.
    if air >= 0.6:
        return CYMBAL, "", pitch
    # Muito agudo junto com outra peça: as duas sempre tocam juntas (bumbo + chimbal,
    # caixa + chimbal, bumbo + prato no começo das frases) e viraram um som só.
    also = CYMBAL if air >= 0.3 else ""
    body = np.where(centers < 6000, w, 0.0)
    body /= body.sum() + 1e-12
    share = lambda lo, hi: float(body[(centers >= lo) & (centers < hi)].sum())   # noqa: E731
    if share(0, 300) >= 0.5:                            # graves: bumbo ou tom
        # Bumbo: altura bem grave. Um som um pouco mais agudo que ainda ressoa é um surdo.
        rings_like_tom = ring >= cfg.tom_min_ring and pitch > 0.75 * cfg.kick_max_hz
        return (KICK if pitch <= cfg.kick_max_hz and not rings_like_tom else TOM), also, pitch
    # Médios: caixa x tom agudo. As esteiras da caixa somam muito ruído em 2-8 kHz ao corpo;
    # a ressonância sozinha não separa (uma caixa com reverb ressoa tanto quanto um tom).
    wires = share(2000, 8000) / (share(80, 600) + 1e-12)
    if wires < cfg.snare_min_wires and ring >= cfg.tom_min_ring and share(80, 500) >= 0.35:
        return TOM, also, pitch
    return SNARE, also, pitch


def _name_toms(sounds: list[DrumSound], cfg: DrumDetection) -> None:
    """Surdo, tom médio e tom agudo pela altura de cada tom encontrado."""
    toms = sorted((s for s in sounds if s.piece == TOM), key=lambda s: -s.pitch)
    racks = [s for s in toms if s.pitch >= cfg.floor_tom_max_hz]
    for s in toms:
        if s.pitch < cfg.floor_tom_max_hz:
            s.piece = TOM_LOW
    distinct = sorted({round(np.log2(s.pitch) * 12) for s in racks}, reverse=True)   # em semitons
    for s in racks:
        semis = round(np.log2(s.pitch) * 12)
        if len(distinct) == 1:
            s.piece = TOM_HIGH if s.pitch >= 200 else TOM_MID
        else:   # o mais agudo é o tom agudo; os outros, o médio
            s.piece = TOM_HIGH if semis >= distinct[0] - 1 else TOM_MID


def _support(w: np.ndarray, mass: float = 0.8) -> np.ndarray:
    """As bandas onde o som vive: as mais fortes do molde, somando `mass` da energia."""
    order = np.argsort(w)[::-1]
    keep = order[: int(np.searchsorted(np.cumsum(w[order]), mass * w.sum())) + 1]
    mask = np.zeros(len(w), dtype=bool)
    mask[keep] = True
    return mask


def _explained(V: np.ndarray, w: np.ndarray, h: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Para cada ataque: que fração do que chegou NAS BANDAS `rows` o som (molde w, força h) explica.

    Medir só nas bandas do próprio som (e não no ataque inteiro) é o que deixa um bumbo aparecer
    debaixo de um prato forte, que domina a energia total do ataque."""
    if not rows.any():
        return np.zeros(V.shape[1])
    part = np.minimum(w[rows][:, None] * h[None, :], V[rows])
    return part.sum(axis=0) / (V[rows].sum(axis=0) + 1e-12)


@dataclass
class KitAnalysis:
    sounds: list[DrumSound]
    onsets: np.ndarray       # quadros (de HOP amostras) dos ataques
    explained: np.ndarray    # [som, ataque]: fração do ataque, nas bandas do som, que ele explica
    explained_also: np.ndarray  # o mesmo para a parte de prato dos sons combinados (só agudos)
    strength: np.ndarray     # [som, ataque]: força do golpe, relativa a um golpe típico desse som
    cymbal_tail: np.ndarray  # [ataque]: quanto dos agudos do golpe ainda soa em ~150-300 ms


def analyze_kit(y: np.ndarray, sr: int, cfg: DrumDetection) -> KitAnalysis:
    """Descobre os sons do kit da gravação e quanto cada um aparece em cada ataque."""
    S = np.abs(librosa.stft(y, n_fft=SHORT_FFT, hop_length=HOP))
    onsets = _onsets(S, librosa.fft_frequencies(sr=sr, n_fft=SHORT_FFT), cfg.onset_delta)
    if len(onsets) < 4:
        empty = np.zeros((0, len(onsets)))
        return KitAnalysis([], onsets, empty, empty, empty, np.zeros(len(onsets)))
    L, centers = _band_spectrogram(y, sr, LONG_FFT)
    Ls, centers_short = _band_spectrogram(y, sr, SHORT_FFT)
    at = lambda M, s: np.stack([M[:, i + s.start: i + s.stop].mean(axis=1) for i in onsets], axis=1)   # noqa: E731
    # O som novo de cada ataque (janela longa, para ter o espectro detalhado)
    V = np.maximum(at(L, ATTACK) - CARRY * at(L, PRE), 0.0).astype(np.float64) + 1e-9
    # Quanto dele sobra ~80 ms depois (janela curta, para não chegar na próxima batida)
    short_pre = at(Ls, SHORT_PRE)
    fresh = np.maximum(at(Ls, SHORT_ATTACK) - CARRY * short_pre, 0.0)
    left = np.maximum(at(Ls, SHORT_LATE) - CARRY * short_pre, 0.0)
    # Pratos: o menor nível dos agudos entre ~150 e 300 ms (as batidas seguintes não contam,
    # porque entre elas o nível cai). Chimbal e esteiras somem; prato de ataque continua soando.
    air_short = Ls[centers_short >= 6000].sum(axis=0)
    air_pre = np.array([air_short[i + SHORT_PRE.start: i + SHORT_PRE.stop].mean() for i in onsets])
    air_new = np.array([air_short[i + SHORT_ATTACK.start: i + SHORT_ATTACK.stop].mean() for i in onsets]) - CARRY * air_pre
    air_tail = np.array([air_short[i + CYMBAL_TAIL.start: i + CYMBAL_TAIL.stop].min() for i in onsets]) - CARRY * air_pre
    tail = np.clip(np.maximum(air_tail, 0.0) / np.maximum(air_new, 1e-9), 0.0, 1.5)
    # Ataques quase sem som novo (ex.: o fim de um prato) não ensinam nada.
    loud = V.sum(axis=0) >= cfg.min_level * np.percentile(V.sum(axis=0), 95)
    onsets, V, fresh, left, tail = onsets[loud], V[:, loud], fresh[:, loud], left[:, loud], tail[loud]
    found = _discover_sounds(V, cfg)
    if not found:
        empty = np.zeros((0, len(onsets)))
        return KitAnalysis([], onsets, empty, empty, empty, np.zeros(len(onsets)))
    W = np.stack([t for t, _ in found], axis=1)
    H = _update_h(V, W, np.ones((W.shape[1], V.shape[1])), 100)
    explained, explained_also = np.zeros_like(H), np.zeros_like(H)
    strength = np.zeros_like(H)
    air = centers >= 6000

    sounds = []
    for k in range(W.shape[1]):
        members = found[k][1]
        rows = _support(W[:, k])
        # Quanto do som NOVO ainda soa ~80 ms depois, nas bandas desse som. O que já soava
        # antes do golpe é descontado: um chimbal tocado enquanto um prato ressoa não pode
        # "herdar" a ressonância do prato.
        w = W[rows, k]
        rings = np.clip((w @ left[rows]) / (w @ fresh[rows] + 1e-12), 0.0, 1.5)
        ring = float(np.median(rings[members]))   # nos golpes do próprio grupo: sem mistura
        piece, also, pitch = _label(W[:, k], centers, ring, cfg)
        # Som combinado (ex.: bumbo + chimbal): cada parte é conferida nas suas bandas.
        explained[k] = _explained(V, W[:, k], H[k], rows & ~air if also else rows)
        explained_also[k] = _explained(V, W[:, k], H[k], rows & air) if also else 0.0
        strength[k] = np.clip(H[k] / (np.median(H[k][members]) + 1e-12), 0.0, 1.0)
        sounds.append(DrumSound(piece, pitch, ring, len(members), W[:, k], also))
    _name_toms(sounds, cfg)
    return KitAnalysis(sounds, onsets, explained, explained_also, strength, tail)


def detect_drum_hits(y: np.ndarray, sr: int, cfg: DrumDetection) -> list[DrumHit]:
    if not np.any(y):
        return []
    kit = analyze_kit(y, sr, cfg)
    hits: dict[tuple[int, str], float] = {}
    for k, sound in enumerate(kit.sounds):
        for i in np.flatnonzero(kit.strength[k] >= cfg.threshold):
            parts = [(sound.piece, kit.explained[k][i]), (sound.also, kit.explained_also[k][i])]
            for piece, share in parts:
                if not piece or share < cfg.min_share:
                    continue
                if piece == CYMBAL:   # chimbal some; condução ainda soa um pouco; ataque, muito
                    ring = kit.cymbal_tail[i]
                    piece = CRASH if ring >= cfg.crash_sustain_ratio else RIDE if ring >= cfg.ride_sustain_ratio else HIHAT
                key = (int(kit.onsets[i]), piece)
                hits[key] = max(hits.get(key, 0.0), float(kit.strength[k][i]))
    to_time = lambda f: float(librosa.frames_to_time(f, sr=sr, hop_length=HOP)) - ONSET_LATENCY  # noqa: E731
    return sorted((DrumHit(to_time(f), piece, s) for (f, piece), s in hits.items()), key=lambda h: h.time)
