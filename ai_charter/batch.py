"""Geração em lote: várias músicas de uma vez, cada uma com a sua partitura, letra e cifra.

Uso:
  ai-charter-batch "D:\\Músicas\\Álbum"                      # mostra os pares e gera tudo
  ai-charter-batch musicas\\ partituras\\ --dry-run          # só mostra os pares
  ai-charter-batch a.flac b.mp3 a.gp5 b.mid -o "D:\\Clone Hero\\Songs\\IA"

Os arquivos podem vir misturados ou em pastas separadas (as subpastas também são olhadas):
músicas, partituras (MIDI ou Guitar Pro) e .txt (letra ou cifra do Cifra Club, reconhecida
pelas linhas de acordes). Links do YouTube (vídeos ou playlists) também valem como músicas: o
áudio é baixado quando a geração de cada um começa (youtube.py). Cada partitura, letra e cifra é casada com a música de nome mais
parecido (nome do arquivo e as tags de título e artista). Cada arquivo vai para uma música só;
o que não se parece o bastante com nenhuma fica de fora e a música é gerada sem ele.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

AUDIO_EXTENSIONS = (".mp3", ".flac", ".wav", ".ogg", ".opus", ".m4a")
SCORE_EXTENSIONS = (".mid", ".midi", ".kar", ".gp3", ".gp4", ".gp5", ".gpx", ".gp", ".gp6", ".gp7", ".gp8")
TEXT_EXTENSIONS = (".txt",)
CHART_FILES = ("notes.mid", "notes.chart")   # pasta de música já chartada: os .ogg dela são stems
MIN_SCORE = 0.5      # abaixo disso o arquivo não é casado
HIGH, MEDIUM = 0.85, 0.65   # confiança: alta, média (e baixa até MIN_SCORE)
AMBIGUOUS = 0.05     # outra música quase tão parecida: a confiança cai para baixa
TITLE_FIRST = 0.7    # o artista só soma quando os títulos já são parecidos assim
MIN_CIFRA_CHORDS = 4   # um .txt com tantas linhas de acordes é cifra; senão, letra

# Palavras que não dizem qual é a música (versão, qualidade, programa da partitura...).
NOISE_WORDS = {
    "official", "oficial", "audio", "video", "clipe", "lyric", "lyrics", "letra", "hq", "hd", "4k",
    "remaster", "remastered", "remasterizado", "remasterizada", "version", "versao", "mono", "stereo",
    "explicit", "tab", "tabs", "tablatura", "guitar", "pro", "gp", "gp3", "gp4", "gp5", "gpx", "midi",
    "cifra", "cifraclub", "partitura", "score", "full", "band",
}
STOP_WORDS = {"the", "a", "an", "of", "and", "o", "os", "as", "e", "de", "da", "do", "das", "dos", "el", "la"}
# Trechos entre parênteses ou colchetes que são só versão: "(Ao Vivo)", "[Remastered 2011]"...
NOISE_BRACKET = re.compile(r"[(\[{][^)\]}]*\b(live|ao vivo|remaster\w*|official|oficial|video|audio|lyric\w*|"
                           r"version|vers[aã]o|mono|stereo|explicit|hq|hd|demo|edit|mix)\b[^)\]}]*[)\]}]",
                           re.IGNORECASE)
TRACK_NUMBER = re.compile(r"^\s*(\d{1,3}|[a-d]\d{1,2})\s*([-._)]\s*|\s+)")


# ---------------------------------------------------------------------------
# Comparação de nomes
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """ "01 - Mötley Crüe - Kickstart (Remastered 2003)_v2" -> "motley crue kickstart" """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = NOISE_BRACKET.sub(" ", text)
    text = re.sub(r"\bremaster\w*\s+\d{4}\b", " ", text)
    text = TRACK_NUMBER.sub("", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    words = [w for w in text.split() if w not in NOISE_WORDS and not re.fullmatch(r"v\d+", w)]
    return " ".join(words)


def _words(text: str) -> list[str]:
    words = text.split()
    return [w for w in words if w not in STOP_WORDS] or words


def _same_word(a: str, b: str) -> bool:
    return a == b or (min(len(a), len(b)) >= 4 and SequenceMatcher(None, a, b).ratio() >= 0.85)


def similarity(a: str, b: str) -> float:
    """Semelhança (0 a 1) entre dois nomes já normalizados.

    Junta a semelhança das letras (erros de digitação, palavras grudadas) com a das palavras,
    que aceita um nome dentro do outro: "Lasso" casa com "Phoenix Lasso"."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    letters = SequenceMatcher(None, a, b).ratio()
    wa, wb = _words(a), _words(b)
    small, large = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
    common = sum(any(_same_word(w, x) for x in large) for w in small)
    inside = common / len(small)                     # quanto do nome menor está no maior
    dice = 2 * common / (len(wa) + len(wb))          # quanto os dois têm em comum
    return max(letters, (inside + dice) / 2)


@dataclass(frozen=True)
class Name:
    artist: str   # normalizados; "" quando o nome não diz o artista
    title: str

    @property
    def full(self) -> str:
        return f"{self.artist} {self.title}".strip()


SEPARATOR = re.compile(r"\s+[-–—]+\s+|\s*_-_\s*|\s+-|-\s+")


def parse_name(text: str) -> Name:
    """ "01 - Phoenix - Lasso (Ao Vivo)" -> Name("phoenix", "lasso"); "Lasso" -> Name("", "lasso")."""
    text = TRACK_NUMBER.sub("", text.replace("_", " ").strip() if " - " not in text else text.strip())
    parts = [normalize(p) for p in SEPARATOR.split(text)]
    parts = [p for p in parts if p]
    if len(parts) >= 2:   # "Artista - Título" (ou "Artista - Álbum - Título")
        return Name(parts[0], parts[-1])
    return Name("", parts[0] if parts else "")


def display_name(stem: str) -> tuple[str, str]:
    """ "01 - Alpargatos - A Viagem" -> ("Alpargatos", "A Viagem"); sem separador -> ("", "")."""
    parts = [p.strip() for p in re.split(r"\s+[-–—]+\s+", TRACK_NUMBER.sub("", stem)) if p.strip()]
    return (parts[0], parts[-1]) if len(parts) >= 2 else ("", "")


def name_similarity(a: Name, b: Name) -> float:
    """Semelhança entre dois nomes, com o título pesando mais que o artista.

    Num lote de um álbum todas as músicas têm o mesmo artista: casar só por ele trocaria as
    partituras. O título vale 70%, e o artista só conta quando o título já se parece ("Lisbon" e
    "Lasso" têm letras em comum, mas não são a mesma música). A ordem "Título - Artista" também
    é tentada."""
    def weighted(title: float, artist: float) -> float:
        return 0.7 * title + (0.3 * artist if title >= TITLE_FIRST else 0.0)

    if a.artist and b.artist:
        return max(weighted(similarity(a.title, b.title), similarity(a.artist, b.artist)),
                   weighted(similarity(a.title, b.artist), similarity(a.artist, b.title)))
    return max(similarity(a.title, b.title), similarity(a.full, b.full))


def audio_names(path: Path, tags: dict | None = None) -> list[Name]:
    """Os nomes pelos quais a música pode aparecer: o do arquivo e o das tags."""
    tags = tags or {}
    names = [parse_name(path.stem)]
    title, artist = tags.get("title", "").strip(), tags.get("artist", "").strip()
    if title:
        names.append(Name(normalize(artist), normalize(title)))
    out = []
    for name in names:
        if name.title and name not in out:
            out.append(name)
    return out


def match_score(names: list[Name], other: Path, audio: Path | None = None) -> float:
    """Semelhança entre uma música (seus nomes) e um arquivo de partitura, letra ou cifra."""
    if audio is not None and other.stem.lower() == audio.stem.lower():
        return 1.0   # mesmo nome do arquivo de música
    target = parse_name(other.stem)
    return max((name_similarity(n, target) for n in names), default=0.0)


def confidence(score: float) -> str:
    return "alta" if score >= HIGH else "media" if score >= MEDIUM else "baixa"


# ---------------------------------------------------------------------------
# Arquivos do lote
# ---------------------------------------------------------------------------
@dataclass
class Files:
    audio: list[Path] = field(default_factory=list)
    scores: list[Path] = field(default_factory=list)
    lyrics: list[Path] = field(default_factory=list)
    cifras: list[Path] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)   # links do YouTube (vídeo ou playlist)


def is_cifra(path: Path) -> bool:
    """Um .txt é cifra quando tem linhas de acordes (como as do Cifra Club)."""
    from ai_charter.cifra import read_cifra

    try:
        return len(read_cifra(path).chords) >= MIN_CIFRA_CHORDS
    except OSError:
        return False


def collect(paths: list) -> Files:
    """Separa os arquivos (e o conteúdo das pastas, com subpastas) por tipo, e os links."""
    from ai_charter.youtube import is_url

    found: list[Path] = []
    urls: list[str] = []
    for raw in paths:
        if is_url(raw):
            if str(raw).strip() not in urls:
                urls.append(str(raw).strip())
            continue
        path = Path(raw)
        if path.is_dir():
            for folder, dirs, files in os.walk(path):
                if {f.lower() for f in files} & set(CHART_FILES):
                    dirs[:] = []   # pasta de uma chart pronta: song.ogg, guitar.ogg... são stems
                    continue
                dirs.sort()
                found += [Path(folder) / f for f in sorted(files)]
        elif path.is_file():
            found.append(path)
    out, seen = Files(urls=urls), set()
    for path in found:
        key = os.path.normcase(str(path.resolve()))
        if key in seen:
            continue
        seen.add(key)
        suffix = path.suffix.lower()
        if suffix in AUDIO_EXTENSIONS:
            out.audio.append(path)
        elif suffix in SCORE_EXTENSIONS:
            out.scores.append(path)
        elif suffix in TEXT_EXTENSIONS:
            (out.cifras if is_cifra(path) else out.lyrics).append(path)
    return out


@dataclass
class Pick:
    """O arquivo casado com uma música e quanto ele se parece com ela."""
    path: str
    score: float
    confidence: str   # alta, media ou baixa


@dataclass
class Item:
    input: str
    title: str = ""
    artist: str = ""
    genre_tag: str = ""
    genre: str | None = None
    reference: Pick | None = None
    lyrics: Pick | None = None
    cifra: Pick | None = None


@dataclass
class Plan:
    items: list[Item]
    scores: list[str]       # todas as partituras, letras e cifras do lote (para escolher outra na tela)
    lyrics: list[str]
    cifras: list[str]
    unmatched: dict[str, list[str]]   # as que não casaram com nenhuma música

    def to_dict(self) -> dict:
        return asdict(self)


def assign(names: list[list[str]], audio: list[Path], others: list[Path]) -> dict[int, Pick]:
    """Casa cada arquivo com no máximo uma música (e cada música com no máximo um arquivo).

    Os pares mais parecidos são fixados primeiro. Se outra música (ainda livre) também se parece
    quase tanto com o arquivo, o par fica com confiança baixa, para ser revisado."""
    table = [[match_score(names[i], o, audio[i]) for o in others] for i in range(len(audio))]
    pairs = sorted(((table[i][j], i, j) for i in range(len(audio)) for j in range(len(others))
                    if table[i][j] >= MIN_SCORE), reverse=True)
    picks: dict[int, Pick] = {}
    used: set[int] = set()
    for score, i, j in pairs:
        if i in picks or j in used:
            continue
        rivals = [table[k][j] for k in range(len(audio)) if k != i and k not in picks]
        rivals += [table[i][k] for k in range(len(others)) if k != j and k not in used]
        level = confidence(score)
        same_file = others[j].stem.lower() == audio[i].stem.lower()
        if not same_file and rivals and max(rivals) >= score - AMBIGUOUS:
            level = "baixa"
        picks[i] = Pick(str(others[j]), round(score, 2), level)
        used.add(j)
    return picks


def plan(files: Files, tags: dict[str, dict] | None = None, videos: list | None = None) -> Plan:
    """Os pares (música, partitura, letra, cifra) propostos para o lote.

    `tags`: as tags de cada música (pelo caminho), lidas com o ffprobe; sem elas vale só o nome.
    `videos`: os vídeos dos links (youtube.Video), que entram depois dos arquivos, pelo título."""
    from ai_charter.config import genre_for_tag

    tags = tags or {}
    videos = videos or []
    audio = sorted(files.audio, key=lambda p: (p.parent.as_posix().lower(), p.name.lower()))
    names = [audio_names(p, tags.get(str(p))) for p in audio]
    n_files = len(audio)
    for v in videos:   # o vídeo casa pelo nome "Artista - Título", como se fosse o nome de um arquivo
        audio.append(Path(re.sub(r'[<>:"/\\|?*]', "", v.name) or v.id))
        names.append([Name(normalize(v.artist), normalize(v.title))])
    refs = assign(names, audio, files.scores)
    lyrics = assign(names, audio, files.lyrics)
    cifras = assign(names, audio, files.cifras)
    items = []
    for i, path in enumerate(audio):
        if i >= n_files:
            v = videos[i - n_files]
            items.append(Item(input=v.url, title=v.title, artist=v.artist,
                              reference=refs.get(i), lyrics=lyrics.get(i), cifra=cifras.get(i)))
            continue
        t = tags.get(str(path)) or {}
        artist, title = t.get("artist", "").strip(), t.get("title", "").strip()
        if not title:   # sem tags: "Artista - Título" do nome do arquivo (senão vira "Unknown Artist")
            artist, title = display_name(path.stem)
        items.append(Item(input=str(path), title=title, artist=artist,
                          genre_tag=t.get("genre", ""), genre=genre_for_tag(t.get("genre")),
                          reference=refs.get(i), lyrics=lyrics.get(i), cifra=cifras.get(i)))

    def left(all_files: list[Path], picks: dict[int, Pick]) -> list[str]:
        taken = {p.path for p in picks.values()}
        return [str(f) for f in all_files if str(f) not in taken]

    return Plan(items=items, scores=[str(p) for p in files.scores], lyrics=[str(p) for p in files.lyrics],
                cifras=[str(p) for p in files.cifras],
                unmatched={"scores": left(files.scores, refs), "lyrics": left(files.lyrics, lyrics),
                           "cifras": left(files.cifras, cifras)})


def read_all_tags(paths: list[Path]) -> dict[str, dict]:
    """As tags de cada música (vazio sem ffprobe)."""
    import shutil

    from ai_charter.export import audio

    if not shutil.which("ffprobe"):
        return {}
    return {str(p): audio.read_tags(p) for p in paths}


def scan(paths: list) -> Plan:
    """Os pares do lote para os caminhos e links dados (os links são consultados no YouTube)."""
    files = collect(paths)
    videos, seen = [], set()
    if files.urls:
        from ai_charter.youtube import expand
        for url in files.urls:
            for v in expand(url):
                if v.id not in seen:
                    seen.add(v.id)
                    videos.append(v)
    return plan(files, read_all_tags(files.audio), videos)


# ---------------------------------------------------------------------------
# Do par ao pedido de geração
# ---------------------------------------------------------------------------
ITEM_OPTIONS = ("instruments", "genre", "reference_fidelity", "reference_tracks", "title", "artist")


def item_form(item: dict, common: dict) -> dict:
    """O formulário de uma geração (o mesmo da aba Nova chart) para uma música do lote.

    `common` tem as opções do lote todo; as da música (`item`) valem por cima delas."""
    data = {k: v for k, v in common.items() if k not in ("items",)}
    data.update({k: item[k] for k in ITEM_OPTIONS if item.get(k) not in (None, "")})
    data["input"] = item["input"]
    data["reference"] = item.get("reference") or ""
    data["lyrics"] = item.get("lyrics") or ""
    cifra = item.get("cifra") or ""
    if cifra:
        data["cifra_text"] = Path(cifra).read_text(encoding="utf-8-sig", errors="replace")
        data["lyrics_from_cifra"] = not data["lyrics"]   # sem letra, os vocais usam a da cifra
    data["open_editor"] = False   # num lote, abrir uma aba por música atrapalha
    return data


def _pick_path(pick) -> str:
    return pick.path if isinstance(pick, Pick) else (pick or "")


def plan_items(p: Plan) -> list[dict]:
    """Os pares do plano no formato que item_form recebe."""
    return [{"input": it.input, "reference": _pick_path(it.reference), "lyrics": _pick_path(it.lyrics),
             "cifra": _pick_path(it.cifra), "genre": it.genre or "", "title": it.title, "artist": it.artist}
            for it in p.items]


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------
LEVEL = {"alta": "", "media": " (média)", "baixa": " (BAIXA: confira)"}


def print_plan(p: Plan) -> None:
    for n, it in enumerate(p.items, 1):
        link = it.input.startswith(("http://", "https://"))
        name = " - ".join(x for x in (it.artist, it.title) if x)
        print(f"{n:>3}. {f'{name}  ({it.input})' if link else Path(it.input).name}")
        for label, pick in (("partitura", it.reference), ("letra", it.lyrics), ("cifra", it.cifra)):
            if pick is not None:
                print(f"       {label}: {Path(pick.path).name}  [{pick.score:.2f}{LEVEL[pick.confidence]}]")
    for kind, label in (("scores", "Partituras"), ("lyrics", "Letras"), ("cifras", "Cifras")):
        if p.unmatched[kind]:
            print(f"\n{label} sem música parecida (ficam de fora):")
            for f in p.unmatched[kind]:
                print(f"       {Path(f).name}")


def main(argv: list[str] | None = None) -> None:
    from ai_charter.config import list_genres

    parser = argparse.ArgumentParser(
        prog="ai-charter-batch",
        description="Gera charts de várias músicas de uma vez, casando cada partitura, letra e cifra "
                    "com a música de nome mais parecido.")
    parser.add_argument("paths", nargs="+",
                        help="músicas, partituras (.mid/.gp*), letras e cifras (.txt), pastas com eles, "
                             "ou links do YouTube (vídeos ou playlists)")
    parser.add_argument("-o", "--output", type=Path, default=Path("output"),
                        help="pasta onde as músicas serão criadas (padrão: ./output)")
    parser.add_argument("--instruments", default="guitar,bass,drums,keys",
                        help="instrumentos a gerar, separados por vírgula (padrão: guitar,bass,drums,keys)")
    parser.add_argument("--genre", choices=[g.id for g in list_genres()], metavar="GENERO",
                        help="gênero para todas as músicas (padrão: o da tag de cada uma, se houver)")
    parser.add_argument("--reference-fidelity", choices=("seguir", "completar", "corrigir"), default="seguir",
                        help="quanto a partitura manda nas notas (padrão: seguir)")
    parser.add_argument("--separator", choices=("demucs", "none"), default="demucs",
                        help="como separar os stems (padrão: demucs)")
    parser.add_argument("--tempo", choices=("auto", "constant", "variable"), default="auto")
    parser.add_argument("--format", choices=("mid", "chart"), default="mid")
    parser.add_argument("--overwrite", action="store_true",
                        help="gera de novo as músicas que já têm chart (padrão: elas são puladas)")
    parser.add_argument("--no-references", action="store_true", help="não usa as partituras")
    parser.add_argument("--video", action="store_true",
                        help="links do YouTube: o vídeo também vira o fundo de cada música (video.webm)")
    parser.add_argument("--dry-run", action="store_true", help="só mostra os pares, sem gerar")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from ai_charter.youtube import is_url
    missing = [p for p in args.paths if not is_url(p) and not Path(p).exists()]
    if missing:
        parser.error("não encontrado: " + ", ".join(missing))

    try:
        p = scan(args.paths)
    except RuntimeError as exc:   # link que o YouTube recusou, yt-dlp não instalado...
        parser.error(str(exc))
    if not p.items:
        parser.error("nenhuma música (.mp3, .flac, .wav, .ogg, .opus, .m4a) nos caminhos dados")
    print_plan(p)
    if args.dry_run:
        return

    from ai_charter.app import AlreadyExists, BadRequest, parse_request, run_request

    common = {"instruments": [i.strip() for i in args.instruments.split(",") if i.strip()],
              "genre": args.genre or "", "reference_fidelity": args.reference_fidelity,
              "separator": args.separator, "tempo": args.tempo, "format": args.format,
              "overwrite": args.overwrite, "output": str(args.output), "video": args.video}
    settings = {"output": str(args.output)}
    results: list[tuple[str, str, str]] = []
    items = plan_items(p)
    for n, item in enumerate(items, 1):
        if args.no_references:
            item["reference"] = ""
        if args.genre:
            item["genre"] = ""
        name = Path(item["input"]).name
        print(f"\n=== [{n}/{len(items)}] {name} ===")
        try:
            song_dir = run_request(parse_request(item_form(item, common), settings), print)
            results.append(("pronta", name, str(song_dir)))
        except AlreadyExists as exc:
            print(f"Pulada: {exc}")
            results.append(("pulada", name, "já existe uma chart (use --overwrite)"))
        except (BadRequest, RuntimeError, ValueError, OSError) as exc:
            print(f"Erro: {exc}", file=sys.stderr)
            results.append(("erro", name, str(exc)))
        except KeyboardInterrupt:
            results.append(("interrompida", name, ""))
            break
    print("\nResumo do lote:")
    for status, name, detail in results:
        print(f"  {status:<12} {name}  {detail}")
    counts = {s: sum(r[0] == s for r in results) for s in ("pronta", "pulada", "erro")}
    print(f"\n{counts['pronta']} pronta(s), {counts['pulada']} pulada(s), {counts['erro']} com erro"
          + (f", {len(items) - len(results)} não começada(s)" if len(results) < len(items) else ""))
    if counts["erro"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
