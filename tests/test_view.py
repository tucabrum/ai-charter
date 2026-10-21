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


def post(url, body, headers):
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_editor_saves_notes_mid_and_keeps_the_original(server, tmp_path):
    song = tmp_path / "song"
    headers = {"X-AI-Charter": "save", "Content-Type": "application/octet-stream"}
    assert post(server + "song/save", b"MThd-v1", headers) == {"saved": "notes.mid", "backup": "notes.backup.mid"}
    assert (song / "notes.mid").read_bytes() == b"MThd-v1"
    assert (song / "notes.backup.mid").read_bytes() == b"MThd"
    post(server + "song/save", b"MThd-v2", headers)
    assert (song / "notes.mid").read_bytes() == b"MThd-v2"
    assert (song / "notes.backup.mid").read_bytes() == b"MThd"   # a original não é sobrescrita


def test_save_needs_the_page_header_and_a_midi_file(server, tmp_path):
    song = tmp_path / "song"
    with pytest.raises(urllib.error.HTTPError) as err:
        post(server + "song/save", b"MThd-site-qualquer", {"Content-Type": "application/octet-stream"})
    assert err.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as err:
        post(server + "song/save", b"texto", {"X-AI-Charter": "save"})
    assert err.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as err:
        post(server + "song/song.ini", b"MThd", {"X-AI-Charter": "save"})
    assert err.value.code == 403
    assert (song / "notes.mid").read_bytes() == b"MThd"
    assert not (song / "notes.backup.mid").exists()
