"""Lote: casamento de partituras, letras e cifras com as músicas pelo nome."""

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from ai_charter import app as app_mod
from ai_charter import batch
from ai_charter.app import AlreadyExists, App, Editors, Jobs
from ai_charter.batch import Name, collect, item_form, name_similarity, normalize, parse_name, plan

CIFRA = "[Intro] G  D  Em  C\n\nG              D\n  Quando eu te vi\nEm             C\n  tudo mudou\n"
LETRA = "Quando eu te vi\ntudo mudou\n"


def touch(folder: Path, *names: str, text: str = "x") -> list[Path]:
    paths = []
    for name in names:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


@pytest.mark.parametrize("text, name", [
    ("01 - Mötley Crüe - Kickstart My Heart (Remastered 2003)", Name("motley crue", "kickstart my heart")),
    ("Motley_Crue_-_Kickstart_my_heart_v2", Name("motley crue", "kickstart my heart")),
    ("Escape The Fate - Situations [Official Video]", Name("escape the fate", "situations")),
    ("Alpargatos - A Viagem (Ao Vivo)", Name("alpargatos", "a viagem")),
    ("Rush - 2112", Name("rush", "2112")),             # número no título não é ano
    ("Lasso", Name("", "lasso")),
])
def test_parse_name(text, name):
    assert parse_name(text) == name


def test_normalize_drops_accents_and_noise():
    assert normalize("Ação & Reação (Lyric Video) GP5") == "acao and reacao"


@pytest.mark.parametrize("a, b, low, high", [
    ("Phoenix - Lasso", "Lasso", 0.99, 1.0),                       # só o título
    ("Phoenix - Lasso", "phoenix lasso tab", 0.99, 1.0),            # sem separador e com ruído
    ("Situations - Escape the Fate", "Escape The Fate - Situations", 0.99, 1.0),   # ordem trocada
    ("Metallica - Kickstart my heart", "Metallica - Kickstrat my heart", 0.9, 1.0),   # erro de digitação
    ("Phoenix - Lasso", "Phoenix - 1901", 0.0, 0.4),                # mesmo artista, outra música
    ("Metallica - One", "Metallica - Fade to Black", 0.0, 0.45),
])
def test_name_similarity(a, b, low, high):
    assert low <= name_similarity(parse_name(a), parse_name(b)) <= high


def test_collect_separates_kinds_and_skips_chart_folders(tmp_path):
    touch(tmp_path, "musicas/Phoenix - Lasso.flac", "musicas/Phoenix - 1901.mp3", "partituras/lasso.gp5",
          "partituras/1901.mid", "outro/capa.jpg")
    touch(tmp_path, "musicas/Phoenix - Lasso.txt", text=LETRA)
    touch(tmp_path, "cifras/1901.txt", text=CIFRA)
    touch(tmp_path, "Songs/Banda - Pronta/notes.mid", "Songs/Banda - Pronta/song.ogg")   # chart já feita
    files = collect([tmp_path, tmp_path / "musicas" / "Phoenix - 1901.mp3"])   # repetido não conta 2 vezes
    assert sorted(p.name for p in files.audio) == ["Phoenix - 1901.mp3", "Phoenix - Lasso.flac"]
    assert sorted(p.name for p in files.scores) == ["1901.mid", "lasso.gp5"]
    assert [p.name for p in files.lyrics] == ["Phoenix - Lasso.txt"]
    assert [p.name for p in files.cifras] == ["1901.txt"]


def test_plan_pairs_each_file_with_one_song(tmp_path):
    audio = touch(tmp_path, "Phoenix - Lasso.flac", "Phoenix - 1901.flac", "Phoenix - Rome.flac")
    scores = touch(tmp_path, "phoenix_1901.gp5", "Lasso (guitar pro).gpx", "Alpargatos - A Viagem.mid")
    p = plan(batch.Files(audio=audio, scores=scores))
    by_song = {Path(it.input).stem: it.reference for it in p.items}
    assert Path(by_song["Phoenix - Lasso"].path).name == "Lasso (guitar pro).gpx"
    assert Path(by_song["Phoenix - 1901"].path).name == "phoenix_1901.gp5"
    assert by_song["Phoenix - Lasso"].confidence == "alta"
    assert by_song["Phoenix - Rome"] is None                       # nenhuma parecida: sem partitura
    assert [Path(f).name for f in p.unmatched["scores"]] == ["Alpargatos - A Viagem.mid"]


def test_plan_uses_tags_and_flags_ambiguous_pairs(tmp_path):
    audio = touch(tmp_path, "faixa01.mp3", "faixa02.mp3")
    scores = touch(tmp_path, "Phoenix - Lasso.gp5", "Lasso.gp5")
    tags = {str(audio[0]): {"title": "Lasso", "artist": "Phoenix", "genre": "Indie Rock"},
            str(audio[1]): {"title": "Lisbon", "artist": "Phoenix"}}
    p = plan(batch.Files(audio=audio, scores=scores), tags)
    first = p.items[0]
    assert (first.title, first.artist) == ("Lasso", "Phoenix")
    assert first.reference is not None and first.reference.confidence == "baixa"   # duas partituras iguais
    assert p.items[1].reference is None


def test_same_file_name_wins(tmp_path):
    audio = touch(tmp_path, "a/musica.mp3", "b/outra musica.mp3")
    lyrics = touch(tmp_path, "a/musica.txt", text=LETRA)
    p = plan(batch.Files(audio=audio, lyrics=lyrics))
    assert p.items[0].lyrics.score == 1.0 and p.items[1].lyrics is None


def test_item_form_merges_common_options_and_reads_cifra(tmp_path):
    (cifra,) = touch(tmp_path, "cifra.txt", text=CIFRA)
    common = {"instruments": ["guitar", "bass"], "genre": "", "reference_fidelity": "seguir",
              "output": "saida", "overwrite": True, "open_editor": True}
    item = {"input": "a.mp3", "reference": "a.gp5", "lyrics": "", "cifra": str(cifra), "genre": "punk",
            "instruments": ["drums"]}
    data = item_form(item, common)
    assert data["genre"] == "punk" and data["instruments"] == ["drums"] and data["overwrite"] is True
    assert data["reference"] == "a.gp5" and data["cifra_text"].startswith("[Intro]")
    assert data["lyrics_from_cifra"] is True and data["open_editor"] is False
    assert item_form({"input": "a.mp3"}, common)["instruments"] == ["guitar", "bass"]


def test_cli_dry_run_prints_pairs(tmp_path, capsys):
    touch(tmp_path, "Phoenix - Lasso.flac", "lasso.gp5", "nada a ver.mid")
    batch.main([str(tmp_path), "--dry-run"])
    out = capsys.readouterr().out
    assert "Phoenix - Lasso.flac" in out and "partitura: lasso.gp5" in out
    assert "sem música parecida" in out and "nada a ver.mid" in out


# ---------------------------------------------------------------------------
# Fila e API do lote no app
# ---------------------------------------------------------------------------

GATE = threading.Event()   # segura a geração "lenta" até o teste liberar


def fake_runner(req, log):
    log("1/5 Convertendo o áudio...")
    if req.input.stem == "lenta":
        while not GATE.wait(0.01):
            log("    ...")   # o log é onde a geração para quando cancelada
    if req.input.stem == "falha":
        raise RuntimeError("deu ruim")
    if req.input.stem == "existe":
        raise AlreadyExists("já existe uma chart")
    song = req.output_dir / f"Artista - {req.input.stem}"
    song.mkdir(parents=True, exist_ok=True)
    return song


@pytest.fixture
def server(tmp_path):
    GATE.clear()
    editors = Editors()
    app = App(settings={"library": "", "output": str(tmp_path / "saida")},
              settings_path=tmp_path / "app.json", editors=editors,
              jobs=Jobs(runner=fake_runner, editors=editors))
    httpd = app_mod.make_server(app)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/", app
    GATE.set()
    httpd.shutdown()
    editors.close()


def post(url, data):
    req = urllib.request.Request(url, json.dumps(data).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "X-AI-Charter": "app"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def statuses(app, ids):
    return [app.jobs.items[i].status for i in ids]


def wait_until(check, what="a fila não chegou no estado esperado"):
    for _ in range(200):
        if check():
            return
        time.sleep(0.02)
    raise AssertionError(what)


def test_scan_and_batch_run_in_order_and_keep_going_after_errors(server, tmp_path):
    base, app = server
    touch(tmp_path / "lote", "boa.flac", "falha.flac", "existe.flac", "boa.gp5")
    plan = post(base + "api/batch/scan", {"paths": [str(tmp_path / "lote")]})
    assert [Path(i["input"]).name for i in plan["items"]] == ["boa.flac", "existe.flac", "falha.flac"]
    assert Path(plan["items"][0]["reference"]["path"]).name == "boa.gp5"
    items = [{"input": i["input"], "reference": ""} for i in plan["items"]]   # .gp5 falso: sem partitura
    res = post(base + "api/batch", {"items": items, "common": {"instruments": ["guitar"]}})
    ids = [j["id"] for j in res["jobs"]]
    assert all(j["batch"] == res["batch"] for j in res["jobs"])
    wait_until(lambda: all(s not in ("waiting", "running") for s in statuses(app, ids)))
    assert statuses(app, ids) == ["done", "skipped", "error"]


def test_batch_with_invalid_rows_queues_nothing(server, tmp_path):
    base, app = server
    (ok,) = touch(tmp_path, "boa.flac")
    with pytest.raises(urllib.error.HTTPError) as err:
        post(base + "api/batch", {"items": [{"input": str(ok)}, {"input": str(tmp_path / "sumiu.flac")}],
                                  "common": {"instruments": ["guitar"]}})
    body = json.loads(err.value.read())
    assert list(body["rows"]) == ["1"] and "arquivo não encontrado" in body["rows"]["1"]
    assert app.jobs.items == {}


def test_pause_cancel_and_stop(server, tmp_path):
    base, app = server
    slow, a, b = touch(tmp_path, "lenta.flac", "a.flac", "b.flac")
    res = post(base + "api/batch", {"items": [{"input": str(p)} for p in (slow, a, b)],
                                    "common": {"instruments": ["bass"]}})
    ids = [j["id"] for j in res["jobs"]]
    wait_until(lambda: statuses(app, ids)[0] == "running")
    assert post(base + "api/queue", {"action": "pause"}) == {"paused": True}
    post(base + f"api/jobs/{ids[2]}/cancel", {})            # tira a última da fila
    GATE.set()                                              # a lenta termina; a fila fica parada
    wait_until(lambda: statuses(app, ids)[0] == "done")
    time.sleep(0.1)
    assert statuses(app, ids) == ["done", "waiting", "canceled"]
    post(base + "api/queue", {"action": "resume"})
    wait_until(lambda: statuses(app, ids)[1] == "done")

    GATE.clear()
    slow2, c = touch(tmp_path, "x/lenta.flac", "c.flac")
    res = post(base + "api/batch", {"items": [{"input": str(slow2)}, {"input": str(c)}],
                                    "common": {"instruments": ["bass"]}})
    ids = [j["id"] for j in res["jobs"]]
    wait_until(lambda: statuses(app, ids)[0] == "running")
    post(base + f"api/batch/{res['batch']}/cancel", {})     # tira as que esperam...
    post(base + f"api/jobs/{ids[0]}/cancel", {})            # ...e para a atual no próximo log
    wait_until(lambda: statuses(app, ids) == ["canceled", "canceled"])
    assert app.jobs.items[ids[0]].lines[-1] == "Parada pelo usuário."


def test_title_and_artist_come_from_the_file_name_without_tags(tmp_path):
    audio = touch(tmp_path, "03 - Alpargatos - A Viagem.mp3", "faixa.mp3")
    items = plan(batch.Files(audio=audio)).items
    assert (items[0].artist, items[0].title) == ("Alpargatos", "A Viagem")
    assert (items[1].artist, items[1].title) == ("", "")   # o pipeline usa o nome do arquivo
