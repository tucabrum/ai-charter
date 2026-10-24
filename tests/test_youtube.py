"""Link do YouTube no lugar do arquivo de música (sem rede: o yt-dlp é trocado por um falso)."""

from pathlib import Path

import pytest

from ai_charter import app as app_mod
from ai_charter import batch, youtube
from ai_charter.app import parse_request

URL = "https://www.youtube.com/watch?v=abcdefghijk"
INFO = {"id": "abcdefghijk", "title": "Phoenix - Lasso (Official Video)", "channel": "PhoenixVEVO",
        "duration": 185, "webpage_url": URL}


class FakeYDL:
    """O mínimo do yt_dlp.YoutubeDL: extract_info e download (que cria o arquivo do outtmpl)."""
    calls: list = []

    def __init__(self, options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        FakeYDL.calls.append(("info", url))
        if "list=" in url:
            return {"entries": [{"id": "aaaaaaaaaaa", "title": "Lasso", "channel": "Phoenix - Topic",
                                 "url": "https://www.youtube.com/watch?v=aaaaaaaaaaa"},
                                {"id": "bbbbbbbbbbb", "title": "1901", "channel": "Phoenix - Topic"}]}
        if "erro" in url:
            raise Exception("ERROR: Video unavailable")
        return dict(INFO)

    def download(self, urls):
        FakeYDL.calls.append(("download", urls[0]))
        target = Path(self.options["outtmpl"].replace("%(id)s", "abcdefghijk").replace("%(ext)s", "opus"))
        target.write_bytes(b"OggS")
        target.with_suffix(".webp").write_bytes(b"RIFF")
        for hook in self.options["progress_hooks"]:
            hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100})


@pytest.fixture(autouse=True)
def fake_ydl(monkeypatch):
    FakeYDL.calls = []
    monkeypatch.setattr(youtube, "_ydl", FakeYDL)
    monkeypatch.setattr(youtube, "write_tags", lambda *a: None)   # ffmpeg não entra nos testes


@pytest.mark.parametrize("info, names", [
    ({"title": "Phoenix - Lasso (Official Video)", "channel": "PhoenixVEVO"}, ("Phoenix", "Lasso")),
    ({"title": "Lasso", "channel": "Phoenix - Topic"}, ("Phoenix", "Lasso")),
    ({"title": "Clipe novo [HD]", "artist": "Alpargatos", "track": "A Viagem"}, ("Alpargatos", "A Viagem")),
    ({"title": "Lasso [Legendado]", "uploader": "Fulano"}, ("Fulano", "Lasso")),
])
def test_names_from_info(info, names):
    assert youtube.names_from_info(info) == names


def test_urls_and_ids():
    assert youtube.is_url(URL) and not youtube.is_url(r"C:\Músicas\a.mp3")
    for url in (URL, "https://youtu.be/abcdefghijk?t=3", "https://youtube.com/shorts/abcdefghijk",
                "https://music.youtube.com/watch?v=abcdefghijk&list=RD"):
        assert youtube.video_id(url) == "abcdefghijk"


def test_download_keeps_original_audio_and_reuses_it(tmp_path):
    lines = []
    path, video = youtube.download(URL, tmp_path, lines.append)
    assert path == tmp_path / "Phoenix - Lasso [abcdefghijk].opus" and path.is_file()
    assert (video.artist, video.title) == ("Phoenix", "Lasso")
    assert "    baixando... 50%" in lines
    opts_pp = [c for c in FakeYDL.calls if c[0] == "download"]
    assert len(opts_pp) == 1
    again, _ = youtube.download("https://youtu.be/abcdefghijk", tmp_path, lines.append)
    assert again == path and lines[-1].startswith("    já baixado")
    assert len([c for c in FakeYDL.calls if c[0] == "download"]) == 1   # não baixou de novo
    assert youtube.thumbnail(path) == path.with_suffix(".webp")


def test_download_error_is_readable(tmp_path):
    with pytest.raises(RuntimeError, match="não deu para baixar o vídeo: Video unavailable"):
        youtube.download("https://www.youtube.com/watch?v=erroerroerr", tmp_path)


def test_expand_playlist():
    videos = youtube.expand("https://www.youtube.com/playlist?list=PL123")
    assert [(v.id, v.artist, v.title) for v in videos] == [("aaaaaaaaaaa", "Phoenix", "Lasso"),
                                                            ("bbbbbbbbbbb", "Phoenix", "1901")]
    assert videos[1].url == "https://www.youtube.com/watch?v=bbbbbbbbbbb"


def test_request_with_link_downloads_first(tmp_path, monkeypatch):
    req = parse_request({"input": URL, "instruments": ["guitar"], "downloads": str(tmp_path / "dl")},
                        {"output": str(tmp_path / "out")})
    assert req.url == URL and req.input is None and req.download_dir == tmp_path / "dl"
    seen = {}

    def fake_run(opts, log):   # o pipeline recebe o arquivo baixado
        seen["input"] = opts.input
        song = opts.output_dir / "Phoenix - Lasso"
        song.mkdir(parents=True)
        return type("R", (), {"song_dir": song, "zip_path": None})()

    monkeypatch.setattr("ai_charter.pipeline.run", fake_run)
    monkeypatch.setattr("ai_charter.export.audio.require_ffmpeg", lambda: None)
    monkeypatch.setattr("ai_charter.export.audio.read_tags", lambda p: {"title": "Lasso", "artist": "Phoenix"})
    monkeypatch.setattr(youtube, "save_cover", lambda audio, song: True)
    lines = []
    song = app_mod.run_request(req, lines.append)
    assert seen["input"] == tmp_path / "dl" / "Phoenix - Lasso [abcdefghijk].opus"
    assert song == tmp_path / "out" / "Phoenix - Lasso"
    assert lines[0] == "0/5 Baixando o áudio do vídeo..." and lines[-1].startswith("A miniatura")


def test_batch_scan_matches_scores_to_videos(tmp_path):
    (tmp_path / "lasso.gp5").write_text("x")
    (tmp_path / "Phoenix - 1901.mid").write_text("x")
    p = batch.scan([str(tmp_path), "https://www.youtube.com/playlist?list=PL123"])
    assert [(it.input, it.artist, it.title) for it in p.items] == [
        ("https://www.youtube.com/watch?v=aaaaaaaaaaa", "Phoenix", "Lasso"),
        ("https://www.youtube.com/watch?v=bbbbbbbbbbb", "Phoenix", "1901")]
    assert Path(p.items[0].reference.path).name == "lasso.gp5"
    assert Path(p.items[1].reference.path).name == "Phoenix - 1901.mid"
    item = {**batch.plan_items(p)[0], "reference": ""}   # o .gp5 do teste não é uma partitura de verdade
    form = batch.item_form(item, {"instruments": ["guitar"]})
    assert parse_request(form, {"output": str(tmp_path)}).url == p.items[0].input
