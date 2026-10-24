"""Interface no navegador (ai-charter-app): formulário, fila de gerações, biblioteca e editor."""

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import mido
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
    assert {"id": "metal", "name": "Metal"}.items() <= next(g for g in state["genres"] if g["id"] == "metal").items()


def test_genre_is_passed_and_validated(server, music):
    base, app = server
    assert parse_request({"input": str(music), "instruments": ["guitar"], "genre": "punk"}, app.settings).genre == "punk"
    assert parse_request({"input": str(music), "instruments": ["guitar"], "genre": ""}, app.settings).genre is None
    with pytest.raises(urllib.error.HTTPError) as err:
        post(base + "api/jobs", {"input": str(music), "instruments": ["guitar"], "genre": "polca"})
    assert "gênero inválido" in json.loads(err.value.read())["error"]


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


def test_add_to_existing_chart(server, tmp_path):
    base, app = server
    song = tmp_path / "Banda - Música"
    song.mkdir()
    midi = mido.MidiFile()
    midi.tracks.append(mido.MidiTrack([mido.MetaMessage("track_name", name="PART GUITAR")]))
    midi.save(song / "notes.mid")
    for name in ("song.ogg", "guitar.ogg", "vocals.ogg"):
        (song / name).write_bytes(b"OggS")
    (song / "letra.txt").write_text("la la", encoding="utf-8")
    (song / "song.ini").write_text("[song]\nname = Música\ngenre = Bossa Nova\n", encoding="utf-8")
    info = get(base + "api/song?dir=" + urllib.parse.quote(str(song)))
    assert info["parts"] == ["guitar"] and info["stems"] == ["guitar", "other", "vocals"]
    assert info["lyrics"] == "letra.txt"
    assert (info["genre_tag"], info["genre"]) == ("Bossa Nova", "mpb")   # só sugestão: a página decide

    seen = []
    app.add_runner = lambda req, log: seen.append(req) or req.folder
    job = post(base + "api/add", {"folder": str(song), "instruments": ["vocals"], "lyrics_text": "oi"})
    assert wait_job(base, job["id"])["status"] == "done"
    assert seen[0].instruments == ("vocals",) and seen[0].lyrics_text == "oi"
    for data, message in (({"folder": str(tmp_path), "instruments": ["bass"]}, "não tem notes.mid"),
                          ({"folder": str(song), "instruments": []}, "ao menos um instrumento")):
        with pytest.raises(urllib.error.HTTPError) as err:
            post(base + "api/add", data)
        assert message in json.loads(err.value.read())["error"]


def test_reference_tracks_and_mapping(server, music, tmp_path):
    base, app = server
    midi = mido.MidiFile()
    for name, program in (("Guitarra", 29), ("Baixo", 33)):
        midi.tracks.append(mido.MidiTrack([
            mido.MetaMessage("track_name", name=name), mido.Message("program_change", program=program),
            mido.Message("note_on", note=50, velocity=90), mido.Message("note_off", note=50, time=480)]))
    ref = tmp_path / "ref.mid"
    midi.save(ref)
    info = get(base + "api/reference?file=" + urllib.parse.quote(str(ref)))
    assert [(t["index"], t["name"], t["guess"]) for t in info["tracks"]] == [(1, "Guitarra", "guitar"), (2, "Baixo", "bass")]
    assert info["mapping"] == {"1": "guitar", "2": "bass"}

    req = parse_request({"input": str(music), "instruments": ["guitar"], "reference": str(ref),
                         "reference_tracks": {"2": "guitar"}}, app.settings)
    assert req.reference == ref and req.reference_tracks == {2: "guitar"}
    assert parse_request({"input": str(music), "instruments": ["guitar"], "reference": str(ref)},
                         app.settings).reference_tracks is None   # sem tabela: palpite
    # fidelidade: uma para todos (padrão: seguir) ou uma por instrumento
    base_req = {"input": str(music), "instruments": ["guitar"], "reference": str(ref)}
    assert parse_request(base_req, app.settings).reference_fidelity == "seguir"
    assert parse_request({**base_req, "reference_fidelity": {"drums": "seguir", "guitar": "corrigir"}},
                         app.settings).reference_fidelity == {"drums": "seguir", "guitar": "corrigir"}
    with pytest.raises(BadRequest):
        parse_request({**base_req, "reference_fidelity": {"guitar": "inventar"}}, app.settings)
    for tracks, message in (({"7": "guitar"}, "pista da partitura inválida"), ({"1": "flauta"}, "inválida")):
        with pytest.raises(urllib.error.HTTPError) as err:
            post(base + "api/jobs", {"input": str(music), "instruments": ["guitar"], "reference": str(ref),
                                     "reference_tracks": tracks})
        assert message in json.loads(err.value.read())["error"]
    (tmp_path / "x.gpx").write_bytes(b"BCFS")
    with pytest.raises(urllib.error.HTTPError) as err:
        get(base + "api/reference?file=" + urllib.parse.quote(str(tmp_path / "x.gpx")))
    assert "score.gpif" in json.loads(err.value.read())["error"]


CIFRA = "Banda - Música\n\n[Intro] G D\n\nG        D\nLa la la\n"


def test_cifra_goes_to_the_pipeline(tmp_path, music, monkeypatch):
    from ai_charter import pipeline
    from ai_charter.export import audio

    monkeypatch.setattr(audio, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(audio, "read_tags", lambda path: {})
    used = {}

    def fake_run(opts, log):
        used["cifra"], used["cifra_lyrics"] = opts.cifra.read_text("utf-8"), opts.cifra_lyrics
        song = opts.output_dir / "x"
        song.mkdir(parents=True)
        return pipeline.Result(song_dir=song)

    monkeypatch.setattr(pipeline, "run", fake_run)
    # só a letra da cifra já basta para pedir uma chart (de vocais)
    req = parse_request({"input": str(music), "output": str(tmp_path), "cifra_text": CIFRA,
                         "lyrics_from_cifra": True}, {"output": ""})
    run_request(req, print)
    assert used == {"cifra": CIFRA.strip() + "\n", "cifra_lyrics": True}
    req = parse_request({"input": str(music), "output": str(tmp_path / "b"), "instruments": ["guitar"],
                         "cifra_text": CIFRA}, {"output": ""})
    run_request(req, print)
    assert used["cifra_lyrics"] is False   # "Sem vocais": a cifra só ajuda nos acordes


def test_add_uses_cifra_lyrics_and_keeps_the_cifra(tmp_path, monkeypatch):
    from ai_charter import app as app_module
    from ai_charter import augment as augment_module

    song = tmp_path / "song"
    song.mkdir()
    (song / "notes.mid").write_bytes(b"MThd")
    (song / "letra.txt").write_text("letra antiga\n", encoding="utf-8")
    seen = {}

    def fake_augment(folder, instruments, **kwargs):
        seen.update(kwargs)
        return augment_module.AugmentResult(added=list(instruments))

    monkeypatch.setattr(augment_module, "augment", fake_augment)
    req = app_module.AddRequest(folder=song, instruments=("vocals",), cifra_text=CIFRA.strip(),
                                lyrics_from_cifra=True)
    app_module.run_add_request(req, print)
    assert (song / "cifra.txt").read_text("utf-8") == CIFRA.strip() + "\n"
    assert seen["lyrics"] == song / "letra.txt"
    assert (song / "letra.txt").read_text("utf-8") == "La la la\n"   # a letra da cifra no lugar da antiga


def test_bad_pick_kind():
    with pytest.raises(BadRequest):
        app_mod.Dialogs().ask("executavel")
