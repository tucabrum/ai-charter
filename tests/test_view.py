"""Servidor do visualizador: entrega a página e só os arquivos da pasta da música."""

import json
import threading
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer

import pytest

from ai_charter.view import _Handler


@pytest.fixture
def server(tmp_path):
    song = tmp_path / "song"
    song.mkdir()
    (song / "notes.mid").write_bytes(b"MThd")
    (song / "song.ini").write_text("[song]\nname = Teste\n")
    (tmp_path / "secreto.txt").write_text("não pode sair")
    official = tmp_path / "oficial.mid"
    official.write_bytes(b"MThd-oficial")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Handler, song_dir=song, compare=official))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/"
    httpd.shutdown()


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.read()


def test_serves_viewer_song_files_and_comparison(server):
    assert b"Visualizador" in get(server)
    assert json.loads(get(server + "song/__files__")) == ["notes.mid", "song.ini"]
    assert get(server + "song/notes.mid") == b"MThd"
    assert get(server + "compare/__file__") == b"oficial.mid"
    assert get(server + "compare/oficial.mid") == b"MThd-oficial"


def test_files_outside_the_song_folder_are_not_served(server):
    for path in ("song/..%2Fsecreto.txt", "song/../secreto.txt", "compare/secreto.txt"):
        with pytest.raises(urllib.error.HTTPError):
            get(server + path)
