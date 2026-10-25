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


def test_set_ini_value_replaces_or_appends(tmp_path):
    ini = tmp_path / "song.ini"
    ini.write_text("[song]\nname = x\nvideo_start_time = 0\n", encoding="utf-8")
    youtube.set_ini_value(ini, "video_start_time", -4000)
    youtube.set_ini_value(ini, "delay", 0)
    assert ini.read_text(encoding="utf-8") == "[song]\nname = x\nvideo_start_time = -4000\ndelay = 0\n"


def test_background_video_converts_once_and_delays_by_the_lead_in(tmp_path, monkeypatch):
    converted = []

    def fake_vp8(src, dst, log=print):
        assert src.name == "Phoenix - Lasso [abcdefghijk].video-src.opus"   # o FakeYDL "baixa" .opus
        converted.append(dst)
        dst.write_bytes(b"webm")

    monkeypatch.setattr(youtube, "to_vp8", fake_vp8)
    song = tmp_path / "Phoenix - Lasso"
    song.mkdir()
    (song / "song.ini").write_text("[song]\nname = Lasso\n", encoding="utf-8")
    for _ in range(2):
        video = youtube.background_video(URL, tmp_path / "dl", song, 4.0, lambda *_: None)
    assert video == song / "video.webm" and video.read_bytes() == b"webm"
    assert len(converted) == 1                                   # a 2ª vez usa o já convertido
    assert not list((tmp_path / "dl").glob("*.video-src.*"))     # o original é apagado
    assert "video_start_time = -4000" in (song / "song.ini").read_text(encoding="utf-8")
    # o vídeo convertido não é confundido com o áudio já baixado
    assert youtube.cached(tmp_path / "dl", "abcdefghijk") is None


def test_request_with_video_runs_it_after_the_chart(tmp_path, monkeypatch):
    req = parse_request({"input": URL, "instruments": ["guitar"], "video": True}, {"output": str(tmp_path)})
    assert req.video and not parse_request({"input": URL, "instruments": ["guitar"]}, {}).video
    song = tmp_path / "Phoenix - Lasso"
    song.mkdir()
    result = type("R", (), {"song_dir": song, "lead_in": 2.5})()
    monkeypatch.setattr(app_mod, "_generate", lambda req, log: result)
    monkeypatch.setattr(youtube, "download", lambda url, folder, log: (tmp_path / "a.opus", None))
    monkeypatch.setattr(youtube, "save_cover", lambda audio, song: False)
    seen, lines = {}, []
    monkeypatch.setattr(youtube, "background_video",
                        lambda url, folder, song_dir, lead, log: seen.update(lead=lead, song=song_dir))
    assert app_mod.run_request(req, lines.append) == song
    assert seen == {"lead": 2.5, "song": song} and "começa 2.5 s depois" in lines[-1]

    def broken(*args):
        raise RuntimeError("sem rede")

    monkeypatch.setattr(youtube, "background_video", broken)
    lines.clear()
    assert app_mod.run_request(req, lines.append) == song        # a chart fica pronta mesmo assim
    assert "o vídeo não deu certo (sem rede)" in lines[-1]


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="precisa do ffmpeg")
def test_to_vp8_makes_a_silent_webm(tmp_path):
    import subprocess

    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1280x960:rate=10:duration=1",
                    "-f", "lavfi", "-i", "sine=duration=1", "-shortest", str(src)], check=True)
    lines = []
    youtube.to_vp8(src, tmp_path / "out.webm", lines.append)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,height",
                            "-of", "csv=p=0", str(tmp_path / "out.webm")], capture_output=True, text=True).stdout
    assert probe.strip() == "vp8,video,720"                    # só vídeo, VP8, até 720p
    assert lines[-1].endswith("100%")
