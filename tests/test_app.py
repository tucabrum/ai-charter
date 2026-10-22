"""Interface no navegador (ai-charter-app): formulário, fila de gerações, biblioteca e editor."""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from ai_charter import app as app_mod
from ai_charter.app import App, BadRequest, Editors, Jobs, list_library, parse_request, run_request


def fake_runner(req, log):
    log("1/5 Convertendo o áudio...")
    if req.title == "falha":
        raise RuntimeError("deu ruim")
    song = req.output_dir / f"Artista - {req.title or req.input.stem}"
    song.mkdir(parents=True, exist_ok=True)
    (song / "notes.mid").write_bytes(b"MThd")
    return song


@pytest.fixture
def music(tmp_path):
    path = tmp_path / "musica.flac"
    path.write_bytes(b"fLaC")
    return path


@pytest.fixture
def server(tmp_path):
    editors = Editors()
    app = App(settings={"library": "", "output": str(tmp_path / "saida")},
              settings_path=tmp_path / "app.json", editors=editors,
              jobs=Jobs(runner=fake_runner, editors=editors))
    httpd = app_mod.make_server(app)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/", app
    httpd.shutdown()
    editors.close()


def get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read()) if "/api/" in url else r.read()


def post(url, data, header="app"):
    req = urllib.request.Request(url, json.dumps(data).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "X-AI-Charter": header})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def wait_job(base, job_id):
    for _ in range(100):
        info = get(f"{base}api/jobs/{job_id}")
        if info["status"] in ("done", "error"):
            return info
        time.sleep(0.05)
    raise AssertionError("a geração não terminou")


def test_page_and_state(server):
    base, _ = server
    assert b"Nova chart" in get(base)
    state = get(base + "api/state")
    assert state["instruments"] == ["guitar", "bass", "drums", "keys"]
    assert state["jobs"] == []


def test_job_runs_and_reports_log_and_folder(server, music, tmp_path):
    base, _ = server
    job = post(base + "api/jobs", {"input": str(music), "instruments": ["guitar", "drums"]})
    info = wait_job(base, job["id"])
    assert info["status"] == "done"
    assert info["song_dir"] == str(tmp_path / "saida" / "Artista - musica")
    assert info["lines"][0] == "1/5 Convertendo o áudio..."
    assert info["lines"][-1].startswith("Pronto!")
    assert get(f"{base}api/jobs/{job['id']}?since={info['next']}")["lines"] == []   # só as novas
    assert [j["id"] for j in get(base + "api/state")["jobs"]] == [job["id"]]


def test_job_error_is_reported(server, music):
    base, _ = server
    info = wait_job(base, post(base + "api/jobs", {"input": str(music), "title": "falha",
                                                    "instruments": ["bass"]})["id"])
    assert info["status"] == "error" and info["error"] == "deu ruim"


def test_invalid_form_is_rejected(server, tmp_path, music):
    base, _ = server
    for data, message in (
        ({"input": str(tmp_path / "nao_existe.mp3"), "instruments": ["guitar"]}, "arquivo não encontrado"),
        ({"input": str(music), "instruments": ["flauta"]}, "instrumento(s) inválido(s)"),
        ({"input": str(music), "instruments": []}, "ao menos um instrumento"),
        ({"input": str(music), "instruments": ["guitar"], "separator": "folder"}, "informe o campo stems"),
    ):
        with pytest.raises(urllib.error.HTTPError) as err:
            post(base + "api/jobs", data)
        assert err.value.code == 400
        assert message in json.loads(err.value.read())["error"]


def test_post_needs_the_page_header(server, music):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as err:
        post(base + "api/jobs", {"input": str(music), "instruments": ["guitar"]}, header="")
    assert err.value.code == 403


def test_lyrics_only_and_settings_are_saved(server, music, tmp_path):
    base, app = server
    req = parse_request({"input": str(music), "lyrics_text": "  primeira linha\nsegunda  "}, app.settings)
    assert req.instruments == () and req.lyrics_text == "primeira linha\nsegunda"
    assert req.output_dir == tmp_path / "saida"
    post(base + "api/settings", {"library": str(tmp_path), "output": '"C:\\musicas"'})
    assert json.loads((tmp_path / "app.json").read_text("utf-8")) == {"library": str(tmp_path),
                                                                     "output": "C:\\musicas"}


def test_editor_serves_the_song_folder(server, tmp_path):
    base, _ = server
    song = tmp_path / "Banda - Música"
    song.mkdir()
    (song / "notes.mid").write_bytes(b"MThd")
    url = post(base + "api/editor", {"song_dir": str(song)})["url"]
    assert url == post(base + "api/editor", {"song_dir": str(song)})["url"]   # reaproveita o servidor
    assert b"Visualizador" in get(url)
    with urllib.request.urlopen(url + "song/__files__") as r:
        assert json.loads(r.read()) == ["notes.mid"]


def test_library_lists_charts_with_parts(tmp_path):
    lib = tmp_path / "Songs"
    for folder, ini in (("Rock/Banda - Uma", "name = Uma\nartist = Banda\ndiff_guitar = 3\ndiff_vocals = -1\n"
                                             "loading_phrase = Chart gerada automaticamente pelo ai-charter.\n"),
                        ("Abba - Duas", "[Song]\nname = Duas\nartist = Abba\ndiff_drums = 2\ndiff_vocals = 1\n")):
        (lib / folder).mkdir(parents=True)
        (lib / folder / "notes.mid").write_bytes(b"MThd")
        (lib / folder / "song.ini").write_text(ini, encoding="utf-8")
    (lib / "sem chart").mkdir()
    songs = list_library(lib)
    assert [(s["artist"], s["name"], s["parts"], s["ai"]) for s in songs] == [
        ("Abba", "Duas", ["drums", "vocals"], False), ("Banda", "Uma", ["guitar"], True)]


def test_run_request_protects_existing_chart_and_keeps_lyrics(tmp_path, music, monkeypatch):
    from ai_charter import pipeline
    from ai_charter.export import audio

    monkeypatch.setattr(audio, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(audio, "read_tags", lambda path: {"title": "Música", "artist": "Banda"})
    used = {}

    def fake_run(opts, log):
        used["lyrics"] = opts.lyrics.read_text("utf-8")
        song = opts.output_dir / f"{opts.artist} - {opts.title}"
        (song / "notes.mid").write_bytes(b"MThd-novo")
        return pipeline.Result(song_dir=song)

    monkeypatch.setattr(pipeline, "run", fake_run)
    song = tmp_path / "Banda - Música"
    song.mkdir()
    (song / "notes.mid").write_bytes(b"MThd-antigo")
    req = parse_request({"input": str(music), "output": str(tmp_path), "instruments": ["guitar"],
                         "lyrics_text": "olá mundo"}, {"output": ""})
    with pytest.raises(RuntimeError, match="já existe uma chart"):
        run_request(req, print)
    req.overwrite = True
    assert run_request(req, print) == song
    assert used["lyrics"] == "olá mundo\n"
    assert (song / "letra.txt").read_text("utf-8") == "olá mundo\n"
    assert (song / "notes.backup.mid").read_bytes() == b"MThd-antigo"
    assert (song / "notes.mid").read_bytes() == b"MThd-novo"


def test_bad_pick_kind():
    with pytest.raises(BadRequest):
        app_mod.Dialogs().ask("executavel")
