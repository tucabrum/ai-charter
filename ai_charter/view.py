"""Visualizador de charts no navegador: toca a música com a pista de notas (como no YARG).

Uso:
  ai-charter-view "output/Phoenix - Lasso"                       # abre a música
  ai-charter-view "output/Phoenix - Lasso" --compare oficial.mid  # e outra chart lado a lado

Sobe um servidor local (só nesta máquina) que entrega a página do visualizador
(viewer/index.html) e os arquivos da pasta da música, e abre o navegador. A
página também funciona sozinha: abra o index.html e use o botão "Abrir pasta".

O editor da página salva o notes.mid de volta na pasta da música; na primeira vez,
a original é guardada como notes.backup.mid. Junto vão o notes.locks.json (compassos
editados à mão) e, quando o usuário pede, o ai_request.json (pedido de revisão para a
IA; ver ai_charter/revision.py).
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

VIEWER = Path(__file__).parent / "viewer" / "index.html"
CHART_NAME = "notes.mid"
BACKUP_NAME = "notes.backup.mid"
LOCKS_NAME = "notes.locks.json"
REQUEST_NAME = "ai_request.json"
WRITABLE = (CHART_NAME, LOCKS_NAME, REQUEST_NAME)   # o que a página pode gravar na pasta da música
MAX_CHART_BYTES = 20 * 1024 * 1024


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, song_dir: Path | None, compare: Path | None, **kwargs):
        self.song_dir = song_dir
        self.compare = compare
        super().__init__(*args, **kwargs)

    def log_message(self, *args) -> None:   # sem uma linha no terminal por arquivo
        pass

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self._send(VIEWER.read_bytes(), "text/html; charset=utf-8")
        if path == "/song/__files__" and self.song_dir:
            names = sorted(p.name for p in self.song_dir.iterdir() if p.is_file())
            return self._send(json.dumps(names).encode(), "application/json")
        if path.startswith("/song/") and self.song_dir:
            return self._file(self.song_dir, path[len("/song/"):])
        if path == "/compare/__file__" and self.compare:
            return self._send(self.compare.name.encode(), "text/plain; charset=utf-8")
        if path.startswith("/compare/") and self.compare and path[len("/compare/"):] == self.compare.name:
            return self._file(self.compare.parent, self.compare.name)
        self._send(b"not found", "text/plain", 404)

    def do_POST(self) -> None:
        # Só a própria página grava: o cabeçalho X-AI-Charter obriga o navegador a pedir
        # permissão (preflight) a outros sites, e este servidor não a dá.
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        url = urlparse(self.path)
        name = parse_qs(url.query).get("file", [CHART_NAME])[0]
        if (url.path != "/song/save" or not self.song_dir or name not in WRITABLE
                or self.headers.get("X-AI-Charter") != "save" or host not in ("127.0.0.1", "localhost")):
            return self._send(b"forbidden", "text/plain", 403)
        size = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(size) if 0 < size <= MAX_CHART_BYTES else b""
        if name == CHART_NAME and not body.startswith(b"MThd"):
            return self._send("não é um arquivo MIDI".encode(), "text/plain; charset=utf-8", 400)
        if name != CHART_NAME:
            try:
                json.loads(body)
            except ValueError:
                return self._send("não é um JSON válido".encode(), "text/plain; charset=utf-8", 400)
        target, backup = self.song_dir / name, self.song_dir / BACKUP_NAME
        if name == CHART_NAME and target.is_file() and not backup.exists():
            shutil.copy2(target, backup)
        tmp = target.with_name(name + ".tmp")
        tmp.write_bytes(body)
        os.replace(tmp, target)
        info = {"saved": name, "backup": BACKUP_NAME if name == CHART_NAME and backup.exists() else None}
        self._send(json.dumps(info).encode(), "application/json")

    def _file(self, folder: Path, name: str) -> None:
        target = (folder / name).resolve()
        # Só arquivos de dentro da pasta (nada de "../" para fora dela).
        if target.parent != folder.resolve() or not target.is_file():
            return self._send(b"not found", "text/plain", 404)
        kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send(target.read_bytes(), kind)


def serve(song_dir: Path | None, compare: Path | None = None, port: int = 0, open_browser: bool = True) -> None:
    handler = partial(_Handler, song_dir=song_dir.resolve() if song_dir else None,
                      compare=compare.resolve() if compare else None)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Visualizador aberto em {url}  (Ctrl+C para fechar)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ai-charter-view",
                                     description="Toca uma música com a pista de notas, no navegador.")
    parser.add_argument("song", type=Path, nargs="?",
                        help="pasta da música (com notes.mid ou notes.chart e os .ogg)")
    parser.add_argument("--compare", type=Path, help="outra notes.mid/.chart para mostrar lado a lado")
    parser.add_argument("--port", type=int, default=0, help="porta do servidor local (padrão: qualquer livre)")
    parser.add_argument("--no-browser", action="store_true", help="não abre o navegador sozinho")
    args = parser.parse_args(argv)
    if args.song and not args.song.is_dir():
        parser.error(f"pasta não encontrada: {args.song}")
    if args.compare and not args.compare.is_file():
        parser.error(f"arquivo não encontrado: {args.compare}")
    serve(args.song, args.compare, args.port, not args.no_browser)


if __name__ == "__main__":
    main()
