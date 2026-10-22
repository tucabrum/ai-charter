"""Interface do ai-charter no navegador: cria charts e abre o resultado no editor.

Uso:
  ai-charter-app                                   # abre a interface no navegador
  ai-charter-app --library "D:\\Clone Hero\\Songs"   # e já aponta a biblioteca de músicas

Sobe um servidor local (só nesta máquina) que entrega a página da interface
(viewer/app.html) e uma pequena API usada por ela:

  GET  /api/state        configurações, instrumentos e as gerações desta sessão
  GET  /api/jobs/<id>    andamento de uma geração (?since=N: só as linhas novas do log)
  GET  /api/library      músicas da biblioteca (pastas com notes.mid ou notes.chart)
  GET  /api/song?dir=    o que a pasta de uma música já tem (partes, stems, letra e gênero)
  GET  /api/tags?file=   tags do arquivo de música e o gênero que corresponde à tag
  GET  /api/reference?file=  pistas de uma partitura (MIDI ou Guitar Pro) e o palpite de cada uma
  POST /api/jobs         põe uma geração na fila (uma de cada vez: o Demucs é pesado)
  POST /api/add          põe na fila o acréscimo de instrumentos a uma chart (augment.py)
  POST /api/editor       abre uma pasta de música no editor (o mesmo do ai-charter-view)
  POST /api/pick         janela do Windows para escolher arquivo ou pasta
  POST /api/settings     grava a biblioteca e a pasta de saída (~/.ai-charter/app.json)

O editor de cada música roda num servidor próprio do ai-charter-view (view.py), numa
porta livre, e é aberto numa aba nova.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import tempfile
import threading
import traceback
import uuid
import webbrowser
from dataclasses import asdict, dataclass, field
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from ai_charter import view
from ai_charter.config import genre_for_tag, list_genres

PAGE = Path(__file__).parent / "viewer" / "app.html"
SETTINGS_PATH = Path.home() / ".ai-charter" / "app.json"
INSTRUMENTS = ("guitar", "bass", "drums", "keys")   # os de pipeline.ALL_INSTRUMENTS (sem importar o librosa)
SEPARATORS = ("demucs", "folder", "none")
FORMATS = ("mid", "chart")
TEMPO_MODES = ("auto", "constant", "variable")
LYRICS_NAME = "letra.txt"   # cópia da letra usada, guardada na pasta da música
MAX_BODY = 4 * 1024 * 1024
LIBRARY_DEPTH = 3           # subpastas olhadas na biblioteca (artista/álbum/música)
PICK_TYPES = {
    "audio": [("Música", "*.mp3 *.flac *.wav *.ogg *.opus *.m4a"), ("Todos", "*.*")],
    "lyrics": [("Letra", "*.txt"), ("Todos", "*.*")],
    "rules": [("Regras", "*.toml"), ("Todos", "*.*")],
    "score": [("Partitura", "*.mid *.midi *.kar *.gp3 *.gp4 *.gp5 *.gpx *.gp"), ("Todos", "*.*")],
}


class BadRequest(ValueError):
    """Pedido inválido: a mensagem vai para a página."""


# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
def load_settings(path: Path = SETTINGS_PATH) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return {"library": data.get("library") or "", "output": data.get("output") or ""}


def save_settings(settings: dict, path: Path = SETTINGS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Gerações (fila com um trabalhador só)
# ---------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    title: str
    status: str = "waiting"   # waiting, running, done, error
    lines: list[str] = field(default_factory=list)
    song_dir: str | None = None
    error: str | None = None
    editor_url: str | None = None

    def summary(self) -> dict:
        info = asdict(self)
        info["lines"] = len(self.lines)
        return info


@dataclass
class Request:
    """Uma geração pedida pela página, já validada."""
    input: Path
    output_dir: Path
    instruments: tuple[str, ...]
    separator: str = "demucs"
    stems_dir: Path | None = None
    rules_path: Path | None = None
    genre: str | None = None
    reference: Path | None = None
    reference_tracks: dict[int, str] | None = None
    tempo_mode: str = "auto"
    output_format: str = "mid"
    title: str | None = None
    artist: str | None = None
    lyrics: Path | None = None
    lyrics_text: str = ""
    cifra_text: str = ""             # cifra colada (Cifra Club)
    lyrics_from_cifra: bool = False  # os vocais usam a letra da cifra
    auto_lyrics: bool = False
    make_zip: bool = False
    overwrite: bool = False
    open_editor: bool = False


def _path(data: dict, key: str, kind: str, required: bool = False) -> Path | None:
    text = str(data.get(key) or "").strip().strip('"')
    if not text:
        if required:
            raise BadRequest(f"informe o campo {key}")
        return None
    path = Path(text).expanduser()
    if kind == "file" and not path.is_file():
        raise BadRequest(f"arquivo não encontrado: {path}")
    if kind == "dir" and not path.is_dir():
        raise BadRequest(f"pasta não encontrada: {path}")
    return path


def _choice(data: dict, key: str, choices: tuple[str, ...]) -> str:
    value = str(data.get(key) or choices[0])
    if value not in choices:
        raise BadRequest(f"{key} inválido: {value}")
    return value


def _genre(data: dict) -> str | None:
    genre = str(data.get("genre") or "").strip() or None
    if genre is not None and genre not in {g.id for g in list_genres()}:
        raise BadRequest(f"gênero inválido: {genre}")
    return genre


def _reference(data: dict) -> tuple[Path | None, dict[int, str] | None]:
    """A partitura de referência e as pistas escolhidas ({"2": "guitar"}; None = palpite)."""
    from ai_charter.reference import INSTRUMENTS as REF_INSTRUMENTS, read_score

    path = _path(data, "reference", "file")
    if path is None:
        return None, None
    try:
        tracks = {t.index for t in read_score(path).tracks}
    except (OSError, RuntimeError, ValueError) as exc:
        raise BadRequest(f"partitura: {exc}") from exc
    raw = data.get("reference_tracks")
    if raw is None:
        return path, None
    mapping = {}
    for index, instrument in dict(raw).items():
        if not str(index).isdigit() or int(index) not in tracks or instrument not in REF_INSTRUMENTS:
            raise BadRequest(f"pista da partitura inválida: {index}={instrument}")
        mapping[int(index)] = instrument
    return path, mapping


def parse_request(data: dict, settings: dict) -> Request:
    """Valida o formulário da página (os mesmos cuidados da CLI)."""
    instruments = tuple(data.get("instruments") or ())
    invalid = set(instruments) - set(INSTRUMENTS)
    if invalid:
        raise BadRequest(f"instrumento(s) inválido(s): {', '.join(sorted(invalid))}")
    lyrics_text = str(data.get("lyrics_text") or "").strip()
    lyrics = _path(data, "lyrics", "file")
    auto_lyrics = bool(data.get("auto_lyrics"))
    cifra_text = str(data.get("cifra_text") or "").strip()
    lyrics_from_cifra = bool(data.get("lyrics_from_cifra")) and bool(cifra_text)
    if not instruments and not (lyrics or lyrics_text or auto_lyrics or lyrics_from_cifra):
        raise BadRequest("escolha ao menos um instrumento ou informe a letra para os vocais")
    separator = _choice(data, "separator", SEPARATORS)
    reference, reference_tracks = _reference(data)
    stems_dir = _path(data, "stems", "dir", required=separator == "folder")
    output = _path(data, "output", "") or (Path(settings["output"]) if settings.get("output") else Path("output"))
    return Request(
        input=_path(data, "input", "file", required=True), output_dir=output, instruments=instruments,
        separator=separator, stems_dir=stems_dir, rules_path=_path(data, "rules", "file"), genre=_genre(data),
        reference=reference, reference_tracks=reference_tracks,
        tempo_mode=_choice(data, "tempo", TEMPO_MODES), output_format=_choice(data, "format", FORMATS),
        title=str(data.get("title") or "").strip() or None, artist=str(data.get("artist") or "").strip() or None,
        lyrics=lyrics, lyrics_text=lyrics_text, auto_lyrics=auto_lyrics, make_zip=bool(data.get("zip")),
        cifra_text=cifra_text, lyrics_from_cifra=lyrics_from_cifra,
        overwrite=bool(data.get("overwrite")), open_editor=bool(data.get("open_editor")),
    )


def run_request(req: Request, log) -> Path:
    """Roda o pipeline para um pedido e devolve a pasta da música criada."""
    from ai_charter.export import audio
    from ai_charter.pipeline import Options, _safe_name, run

    audio.require_ffmpeg()
    tags = audio.read_tags(req.input)
    title = req.title or tags.get("title") or req.input.stem
    artist = req.artist or tags.get("artist") or "Unknown Artist"
    song_dir = req.output_dir / _safe_name(f"{artist} - {title}")
    if not req.overwrite and any((song_dir / n).exists() for n in ("notes.mid", "notes.chart")):
        raise RuntimeError(f"já existe uma chart em {song_dir}. Marque \"substituir a chart existente\" "
                           "para gerar de novo por cima dela, ou use Acrescentar para gerar só "
                           "as partes que faltam.")
    old, backup = song_dir / view.CHART_NAME, song_dir / view.BACKUP_NAME
    if old.is_file() and not backup.exists():   # a chart substituída fica guardada, como no editor
        shutil.copy2(old, backup)
        log(f"Chart anterior guardada em {backup.name}")
    with tempfile.TemporaryDirectory(prefix="ai-charter-app-") as tmp:
        lyrics = req.lyrics
        if req.lyrics_text:
            lyrics = Path(tmp) / LYRICS_NAME
            lyrics.write_text(req.lyrics_text + "\n", encoding="utf-8")
        cifra = None
        if req.cifra_text:
            cifra = Path(tmp) / "cifra.txt"
            cifra.write_text(req.cifra_text + "\n", encoding="utf-8")
        opts = Options(
            input=req.input, output_dir=req.output_dir, separator=req.separator, stems_dir=req.stems_dir,
            rules_path=req.rules_path, genre=req.genre, reference=req.reference,
            reference_tracks=req.reference_tracks, instruments=req.instruments, tempo_mode=req.tempo_mode,
            title=title, artist=artist, lyrics=lyrics, auto_lyrics=req.auto_lyrics,
            output_format=req.output_format, make_zip=req.make_zip,
            cifra=cifra, cifra_lyrics=req.lyrics_from_cifra,
        )
        result = run(opts, log=log)
        # A letra fica junto da música, para refazer ou acrescentar os vocais depois.
        if lyrics is not None and lyrics.resolve() != (result.song_dir / LYRICS_NAME).resolve():
            shutil.copyfile(lyrics, result.song_dir / LYRICS_NAME)
    if result.zip_path:
        log(f"Arquivo .zip: {result.zip_path}")
    return result.song_dir


@dataclass
class AddRequest:
    """Instrumentos a acrescentar numa chart que já existe (ver augment.py)."""
    folder: Path
    instruments: tuple[str, ...]
    rules_path: Path | None = None
    genre: str | None = None
    reference: Path | None = None
    reference_tracks: dict[int, str] | None = None
    lyrics: Path | None = None
    lyrics_text: str = ""
    cifra_text: str = ""             # cifra colada (Cifra Club)
    lyrics_from_cifra: bool = False  # os vocais usam a letra da cifra
    auto_lyrics: bool = False
    replace: bool = False
    open_editor: bool = False


def parse_add_request(data: dict) -> AddRequest:
    from ai_charter.augment import INSTRUMENTS as ADD_INSTRUMENTS

    folder = _path(data, "folder", "dir", required=True)
    if not (folder / "notes.mid").is_file():
        raise BadRequest(f"a pasta não tem notes.mid: {folder}")
    instruments = tuple(data.get("instruments") or ())
    invalid = set(instruments) - set(ADD_INSTRUMENTS)
    if invalid:
        raise BadRequest(f"instrumento(s) inválido(s): {', '.join(sorted(invalid))}")
    if not instruments:
        raise BadRequest("escolha ao menos um instrumento para acrescentar")
    reference, reference_tracks = _reference(data)
    return AddRequest(
        folder=folder, instruments=instruments, rules_path=_path(data, "rules", "file"), genre=_genre(data),
        reference=reference, reference_tracks=reference_tracks,
        lyrics=_path(data, "lyrics", "file"), lyrics_text=str(data.get("lyrics_text") or "").strip(),
        auto_lyrics=bool(data.get("auto_lyrics")), replace=bool(data.get("replace")),
        open_editor=bool(data.get("open_editor")), cifra_text=str(data.get("cifra_text") or "").strip(),
        lyrics_from_cifra=bool(data.get("lyrics_from_cifra")),
    )


def run_add_request(req: AddRequest, log) -> Path:
    from ai_charter.augment import augment

    from ai_charter.augment import CIFRA_NAME
    from ai_charter.cifra import lyrics_text, read_cifra

    lyrics = req.lyrics
    if req.lyrics_text and "vocals" in req.instruments:
        lyrics = req.folder / LYRICS_NAME   # fica guardada na pasta, como numa chart nova
        lyrics.write_text(req.lyrics_text + "\n", encoding="utf-8")
    cifra = req.folder / CIFRA_NAME
    if req.cifra_text:   # a cifra também fica guardada na pasta
        cifra.write_text(req.cifra_text + "\n", encoding="utf-8")
    if req.lyrics_from_cifra and "vocals" in req.instruments:
        if not cifra.is_file() or not read_cifra(cifra).lyrics:
            raise RuntimeError("a cifra não tem letra para os vocais")
        lyrics = req.folder / LYRICS_NAME
        lyrics.write_text(lyrics_text(read_cifra(cifra)), encoding="utf-8")
    result = augment(req.folder, req.instruments, lyrics=lyrics, auto_lyrics=req.auto_lyrics,
                     rules_path=req.rules_path, genre=req.genre, reference=req.reference,
                     reference_tracks=req.reference_tracks, replace_existing=req.replace, log=log)
    if not result.added:
        raise RuntimeError("nada foi acrescentado: " + "; ".join(f"{k}: {v}" for k, v in result.skipped.items()))
    return req.folder


def song_info(folder: Path) -> dict:
    """O que a pasta de uma música já tem: partes da chart, stems e letra."""
    from ai_charter.augment import LYRICS_NAMES, existing_parts
    from ai_charter.separation import stems_from_folder

    if not (folder / "notes.mid").is_file():
        raise BadRequest(f"a pasta não tem notes.mid: {folder}"
                         + (" (charts .chart ainda não são suportadas)" if (folder / "notes.chart").is_file() else ""))
    try:
        stems = sorted(stems_from_folder(folder))
    except RuntimeError:
        stems = []
    ini = read_song_ini(folder / "song.ini")
    return {
        "folder": str(folder), "name": ini.get("name") or folder.name, "artist": ini.get("artist", ""),
        "parts": existing_parts(folder), "stems": stems,
        "lyrics": next((n for n in LYRICS_NAMES if (folder / n).is_file()), None),
        "cifra": (folder / "cifra.txt").is_file(),
        "genre_tag": ini.get("genre", ""), "genre": genre_for_tag(ini.get("genre")),
    }


class Jobs:
    def __init__(self, runner=run_request, editors: Editors | None = None):
        self.runner = runner   # roda um Request (nova chart)
        self.editors = editors
        self.items: dict[str, Job] = {}
        self._queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._work, daemon=True).start()

    def submit(self, req: Request) -> Job:
        return self.submit_task(req.title or req.input.stem, partial(self.runner, req), req.open_editor)

    def submit_task(self, title: str, task, open_editor: bool = False) -> Job:
        """Põe na fila uma tarefa task(log) que devolve a pasta da música."""
        job = Job(id=uuid.uuid4().hex[:8], title=title)
        self.items[job.id] = job
        self._queue.put((job, task, open_editor))
        return job

    def _work(self) -> None:
        while True:
            job, task, open_editor = self._queue.get()
            job.status = "running"

            def log(line, job=job):
                print(line)
                job.lines.append(str(line))

            try:
                song_dir = task(log)
                job.song_dir = str(song_dir)
                log(f"Pronto! Música em: {song_dir}")
                if open_editor and self.editors is not None:
                    job.editor_url = self.editors.open(song_dir, browser=True)
                job.status = "done"
            except (RuntimeError, ValueError, OSError) as exc:
                job.error = str(exc)
                job.status = "error"
            except Exception as exc:   # erro do programa: o traceback ajuda a relatar
                log(traceback.format_exc())
                job.error = f"{type(exc).__name__}: {exc}"
                job.status = "error"


# ---------------------------------------------------------------------------
# Editor (um servidor do ai-charter-view por pasta de música)
# ---------------------------------------------------------------------------
class Editors:
    def __init__(self):
        self.servers: dict[Path, ThreadingHTTPServer] = {}
        self._lock = threading.Lock()

    def open(self, song_dir: Path | str, browser: bool = False) -> str:
        song_dir = Path(song_dir).resolve()
        if not song_dir.is_dir():
            raise BadRequest(f"pasta não encontrada: {song_dir}")
        with self._lock:
            server = self.servers.get(song_dir)
            if server is None:
                server = ThreadingHTTPServer(("127.0.0.1", 0),
                                             partial(view._Handler, song_dir=song_dir, compare=None))
                threading.Thread(target=server.serve_forever, daemon=True).start()
                self.servers[song_dir] = server
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        if browser:
            webbrowser.open(url)
        return url

    def close(self) -> None:
        for server in self.servers.values():
            server.shutdown()
            server.server_close()


# ---------------------------------------------------------------------------
# Biblioteca
# ---------------------------------------------------------------------------
def read_song_ini(path: Path) -> dict[str, str]:
    values = {}
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return values
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key.strip().lower()] = value.strip()
    return values


def list_library(root: Path) -> list[dict]:
    """Pastas com chart dentro da biblioteca (até LIBRARY_DEPTH níveis), por artista e nome."""
    songs = []
    root = root.resolve()
    for folder, dirs, files in os.walk(root):
        depth = len(Path(folder).relative_to(root).parts)
        if depth >= LIBRARY_DEPTH:
            dirs[:] = []
        names = {f.lower() for f in files}
        chart = next((n for n in ("notes.mid", "notes.chart") if n in names), None)
        if chart is None:
            continue
        dirs[:] = []   # dentro de uma música não há outras
        ini = read_song_ini(Path(folder) / "song.ini")
        parts = [p for p in ("guitar", "bass", "drums", "keys", "vocals")
                 if ini.get(f"diff_{p}", "-1").lstrip("-").isdigit() and int(ini.get(f"diff_{p}", "-1")) >= 0]
        songs.append({
            "path": folder, "chart": chart, "name": ini.get("name") or Path(folder).name,
            "artist": ini.get("artist", ""), "parts": parts,
            "ai": "ai-charter" in ini.get("loading_phrase", ""),
            "modified": os.path.getmtime(Path(folder) / chart),
        })
    songs.sort(key=lambda s: (s["artist"].lower(), s["name"].lower()))
    return songs


# ---------------------------------------------------------------------------
# Janela do Windows para escolher arquivos (tkinter, numa thread só dela)
# ---------------------------------------------------------------------------
class Dialogs:
    def __init__(self):
        self._requests: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def ask(self, kind: str, start: str | None = None) -> str | None:
        if kind not in (*PICK_TYPES, "folder"):
            raise BadRequest(f"tipo inválido: {kind}")
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()
            reply: queue.Queue = queue.Queue(maxsize=1)
            self._requests.put((kind, start, reply))
            answer = reply.get()
        if isinstance(answer, Exception):
            raise RuntimeError(f"não deu para abrir a janela de arquivos ({answer}); cole o caminho no campo")
        return answer

    def _loop(self) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
        except Exception as exc:   # sem tkinter ou sem tela
            while True:
                self._requests.get()[2].put(exc)
        while True:
            kind, start, reply = self._requests.get()
            try:
                root.attributes("-topmost", True)   # a janela aparece na frente do navegador
                root.update()
                start = start if start and Path(start).exists() else None
                if kind == "folder":
                    path = filedialog.askdirectory(parent=root, initialdir=start)
                else:
                    path = filedialog.askopenfilename(parent=root, initialdir=start, filetypes=PICK_TYPES[kind])
                reply.put(str(Path(path)) if path else None)
            except Exception as exc:
                reply.put(exc)


# ---------------------------------------------------------------------------
# Servidor
# ---------------------------------------------------------------------------
@dataclass
class App:
    settings: dict
    settings_path: Path = SETTINGS_PATH
    editors: Editors = field(default_factory=Editors)
    dialogs: Dialogs = field(default_factory=Dialogs)
    jobs: Jobs | None = None
    add_runner: object = run_add_request   # roda um AddRequest (os testes trocam)

    def __post_init__(self):
        if self.jobs is None:
            self.jobs = Jobs(editors=self.editors)

    def state(self) -> dict:
        return {
            "settings": self.settings, "instruments": INSTRUMENTS, "separators": SEPARATORS,
            "formats": FORMATS, "tempo_modes": TEMPO_MODES,
            "genres": [{"id": g.id, "name": g.name, "description": g.description} for g in list_genres()],
            "ffmpeg": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
            "jobs": [j.summary() for j in self.jobs.items.values()],
        }


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, app: App, **kwargs):
        self.app = app
        super().__init__(*args, **kwargs)

    def log_message(self, *args) -> None:
        pass

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status: int = 200) -> None:
        self._send(json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        try:
            self._get()
        except Exception as exc:   # a página mostra o erro em vez de perder a conexão
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _get(self) -> None:
        url = urlparse(self.path)
        path, query = unquote(url.path), parse_qs(url.query)
        if path in ("/", "/index.html"):
            return self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/state":
            return self._json(self.app.state())
        if path.startswith("/api/jobs/"):
            job = self.app.jobs.items.get(path[len("/api/jobs/"):])
            if job is None:
                return self._json({"error": "geração não encontrada"}, 404)
            since = int((query.get("since") or ["0"])[0] or 0)
            return self._json({**job.summary(), "lines": job.lines[since:], "next": len(job.lines)})
        if path == "/api/tags":
            try:
                file = _path({"file": (query.get("file") or [""])[0]}, "file", "file", required=True)
            except BadRequest as exc:
                return self._json({"error": str(exc)}, 400)
            from ai_charter.export import audio
            tags = audio.read_tags(file) if shutil.which("ffprobe") else {}
            return self._json({"title": tags.get("title", ""), "artist": tags.get("artist", ""),
                               "genre_tag": tags.get("genre", ""), "genre": genre_for_tag(tags.get("genre"))})
        if path == "/api/reference":
            from ai_charter.reference import GUIDED, default_mapping, read_score
            try:
                file = _path({"file": (query.get("file") or [""])[0]}, "file", "file", required=True)
                score = read_score(file)
            except (BadRequest, OSError, RuntimeError, ValueError) as exc:
                return self._json({"error": str(exc)}, 400)
            return self._json({"tracks": [t.summary() for t in score.tracks], "guided": GUIDED,
                               "mapping": default_mapping(score), "beats": round(score.length_beats)})
        if path == "/api/song":
            try:
                folder = _path({"dir": (query.get("dir") or [""])[0]}, "dir", "dir", required=True)
                return self._json(song_info(folder))
            except BadRequest as exc:
                return self._json({"error": str(exc)}, 400)
        if path == "/api/library":
            root = (query.get("dir") or [self.app.settings.get("library", "")])[0]
            if not root or not Path(root).is_dir():
                return self._json({"error": "escolha a pasta da biblioteca de músicas", "songs": []})
            return self._json({"dir": root, "songs": list_library(Path(root))})
        self._send(b"not found", "text/plain", 404)

    def do_POST(self) -> None:
        # Mesma proteção do view.py: só a própria página (que manda o cabeçalho) pode pedir.
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        if self.headers.get("X-AI-Charter") != "app" or host not in ("127.0.0.1", "localhost"):
            return self._send(b"forbidden", "text/plain", 403)
        size = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(size)) if 0 < size <= MAX_BODY else {}
            if not isinstance(data, dict):
                raise BadRequest("pedido inválido")
            path = urlparse(self.path).path
            if path == "/api/jobs":
                job = self.app.jobs.submit(parse_request(data, self.app.settings))
                return self._json(job.summary())
            if path == "/api/add":
                req = parse_add_request(data)
                job = self.app.jobs.submit_task(f"{req.folder.name} (+ {', '.join(req.instruments)})",
                                                partial(self.app.add_runner, req), req.open_editor)
                return self._json(job.summary())
            if path == "/api/editor":
                return self._json({"url": self.app.editors.open(_path(data, "song_dir", "dir", required=True))})
            if path == "/api/pick":
                return self._json({"path": self.app.dialogs.ask(str(data.get("kind")), data.get("start"))})
            if path == "/api/settings":
                for key in ("library", "output"):
                    if key in data:
                        self.app.settings[key] = str(data[key] or "").strip().strip('"')
                save_settings(self.app.settings, self.app.settings_path)
                return self._json(self.app.settings)
        except (BadRequest, ValueError, RuntimeError) as exc:
            return self._json({"error": str(exc)}, 400)
        self._send(b"not found", "text/plain", 404)


class _Server(ThreadingHTTPServer):
    # No Windows, reaproveitar o endereço deixa dois servidores na mesma porta (e um rouba
    # os pedidos do outro, por exemplo de um ai-charter-view já aberto): aqui, a porta ocupada dá erro.
    allow_reuse_address = False


def make_server(app: App, port: int = 0) -> ThreadingHTTPServer:
    return _Server(("127.0.0.1", port), partial(_Handler, app=app))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ai-charter-app",
                                     description="Interface do ai-charter no navegador.")
    parser.add_argument("--library", type=Path, help="pasta de músicas do Clone Hero/YARG (fica salva)")
    parser.add_argument("--port", type=int, default=0, help="porta do servidor local (padrão: qualquer livre)")
    parser.add_argument("--no-browser", action="store_true", help="não abre o navegador sozinho")
    args = parser.parse_args(argv)
    app = App(settings=load_settings())
    if args.library:
        if not args.library.is_dir():
            parser.error(f"pasta não encontrada: {args.library}")
        app.settings["library"] = str(args.library.resolve())
        save_settings(app.settings)
    server = make_server(app, args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"ai-charter aberto em {url}  (Ctrl+C para fechar)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.editors.close()


if __name__ == "__main__":
    main()
