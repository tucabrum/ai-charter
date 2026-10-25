"""Áudio de um vídeo do YouTube (ou de outro site que o yt-dlp conheça) para gerar a chart.

O áudio é guardado como veio (Opus ou AAC, só trocando o recipiente para .opus ou .m4a): converter
para MP3 recomprimiria e perderia qualidade, e o charter lê os dois formatos. O arquivo fica na
pasta dos downloads com o id do vídeo no nome ("Artista - Música [dQw4w9WgXcQ].opus"), então o
mesmo link não é baixado duas vezes. Título e artista vão para as tags do arquivo.

Precisa de: pip install "ai-charter[youtube]"   (o yt-dlp)
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DIR = Path.home() / "Music" / "ai-charter"   # pasta padrão dos downloads
AUDIO_EXTENSIONS = (".opus", ".m4a", ".mp3", ".ogg", ".webm", ".flac", ".wav")
URL = re.compile(r"^https?://\S+$", re.IGNORECASE)
VIDEO_ID = re.compile(r"(?:youtu\.be/|[?&]v=|/shorts/|/live/|/embed/)([A-Za-z0-9_-]{11})")
# Pedaços do título que não são o nome da música: "(Official Video)", "[HD]", "| Ao Vivo"...
TITLE_NOISE = re.compile(r"\s*[(\[][^)\]]*\b(official|oficial|video|v[ií]deo|clipe|audio|[áa]udio|lyric\w*|letra|"
                         r"visuali[sz]er|hd|hq|4k|remaster\w*|legendado)\b[^)\]]*[)\]]", re.IGNORECASE)
CHANNEL_NOISE = re.compile(r"\s*(-\s*topic|vevo|official)$", re.IGNORECASE)
NOT_INSTALLED = 'para baixar do YouTube instale: pip install "ai-charter[youtube]"'


def is_url(text) -> bool:
    return bool(URL.match(str(text or "").strip()))


def video_id(url: str) -> str | None:
    m = VIDEO_ID.search(url)
    return m.group(1) if m else None


@dataclass
class Video:
    url: str
    id: str
    title: str      # nome da música (sem "Official Video"...)
    artist: str
    duration: float | None = None

    @property
    def name(self) -> str:
        return f"{self.artist} - {self.title}" if self.artist else self.title


def names_from_info(info: dict) -> tuple[str, str]:
    """(artista, título) a partir dos dados do vídeo.

    Na ordem: os campos de música do YouTube Music (artist/track), o "Artista - Música" do título
    e, por fim, o canal ("Fulano - Topic" e "FulanoVEVO" viram "Fulano")."""
    artist = (info.get("artist") or info.get("creator") or "").split(",")[0].strip()
    track = (info.get("track") or "").strip()
    if artist and track:
        return artist, track
    title = TITLE_NOISE.sub("", info.get("title") or "").strip()
    parts = re.split(r"\s+[-–—|]\s+", title, maxsplit=1)
    if len(parts) == 2 and all(p.strip() for p in parts):
        return parts[0].strip(), TITLE_NOISE.sub("", parts[1]).strip(" -|")
    channel = CHANNEL_NOISE.sub("", info.get("channel") or info.get("uploader") or "").strip()
    return channel, title


def video_from_info(info: dict, url: str) -> Video:
    artist, title = names_from_info(info)
    return Video(url=info.get("webpage_url") or url, id=str(info.get("id") or video_id(url) or ""),
                 title=title or str(info.get("id") or "video"), artist=artist, duration=info.get("duration"))


def _ydl(options: dict):
    """O YoutubeDL (os testes trocam esta função para não usar a rede)."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(NOT_INSTALLED) from exc
    return yt_dlp.YoutubeDL(options)


def _download_error(exc: Exception) -> RuntimeError:
    text = re.sub(r"\x1b\[[0-9;]*m", "", str(exc)).replace("ERROR: ", "").strip()
    return RuntimeError(f"não deu para baixar o vídeo: {text}")


BASE_OPTIONS = {"quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True}


def info(url: str) -> Video:
    """Título, artista e duração de um vídeo, sem baixar."""
    try:
        with _ydl(BASE_OPTIONS) as ydl:
            return video_from_info(ydl.extract_info(url, download=False), url)
    except RuntimeError:
        raise
    except Exception as exc:   # DownloadError e afins do yt-dlp
        raise _download_error(exc) from exc


def expand(url: str) -> list[Video]:
    """Os vídeos de um link: o próprio vídeo, ou todos os de uma playlist."""
    if video_id(url) and "list=" not in url:
        return [info(url)]
    try:
        with _ydl({**BASE_OPTIONS, "noplaylist": False, "extract_flat": "in_playlist"}) as ydl:
            data = ydl.extract_info(url, download=False)
    except RuntimeError:
        raise
    except Exception as exc:
        raise _download_error(exc) from exc
    entries = data.get("entries")
    if entries is None:
        return [video_from_info(data, url)]
    return [video_from_info(e, e.get("url") or f"https://www.youtube.com/watch?v={e.get('id')}")
            for e in entries if e and e.get("id")]


def cached(folder: Path, vid: str | None) -> Path | None:
    """O áudio desse vídeo já baixado na pasta (pelo id no nome do arquivo)."""
    if not vid or not Path(folder).is_dir():
        return None
    return next((p for p in sorted(Path(folder).glob(f"*[[]{vid}[]].*"))   # "[id].video.webm" é o vídeo
                 if p.suffix.lower() in AUDIO_EXTENSIONS and p.name.endswith(f"[{vid}]{p.suffix}")), None)


def _safe(text: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text).strip(" .")[:120] or "video"


def write_tags(path: Path, title: str, artist: str) -> None:
    """Grava título e artista nas tags do arquivo, sem recomprimir (ffmpeg -c copy)."""
    if not shutil.which("ffmpeg"):
        return
    tmp = path.with_name(path.stem + ".tags" + path.suffix)
    result = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-map", "0:a", "-c", "copy",
                             "-metadata", f"title={title}", "-metadata", f"artist={artist}", str(tmp)],
                            capture_output=True)
    if result.returncode == 0 and tmp.is_file():
        tmp.replace(path)
    else:
        tmp.unlink(missing_ok=True)


def thumbnail(audio: Path) -> Path | None:
    """A miniatura do vídeo, baixada ao lado do áudio."""
    return next((p for p in sorted(audio.parent.glob(f"{glob_escape(audio.stem)}.*"))
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")), None)


def glob_escape(text: str) -> str:
    return re.sub(r"([\[\]*?])", r"[\1]", text)


def download(url: str, folder: Path | None = None, log=print) -> tuple[Path, Video]:
    """Baixa o áudio (se ainda não estiver na pasta) e devolve o arquivo e os dados do vídeo."""
    folder = Path(folder or DEFAULT_DIR)
    vid = video_id(url)
    found = cached(folder, vid)
    if found is not None:
        log(f"    já baixado: {found.name}")
        return found, Video(url=url, id=vid or "", title="", artist="")
    video = info(url)
    found = cached(folder, video.id)
    if found is not None:
        log(f"    já baixado: {found.name}")
        return found, video
    folder.mkdir(parents=True, exist_ok=True)
    log(f"    {video.name} ({_duration(video.duration)})")
    state = {"step": -1}

    def progress(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            step = int(d.get("downloaded_bytes", 0) * 4 / total) if total else 0
            if step > state["step"]:   # uma linha a cada 25% (o log também é onde a geração pode parar)
                state["step"] = step
                log(f"    baixando... {min(step * 25, 100)}%")

    options = {
        **BASE_OPTIONS,
        "format": "bestaudio/best",
        "outtmpl": str(folder / f"{_safe(video.name)} [%(id)s].%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "best"}],   # só troca o recipiente
        "writethumbnail": True,
        "progress_hooks": [progress],
    }
    try:
        with _ydl(options) as ydl:
            ydl.download([video.url])
    except RuntimeError:
        raise
    except Exception as exc:
        raise _download_error(exc) from exc
    found = cached(folder, video.id)
    if found is None:
        raise RuntimeError("o download terminou, mas o arquivo de áudio não apareceu na pasta")
    write_tags(found, video.title, video.artist)
    log(f"    salvo em {found}")
    return found, video


def save_cover(audio: Path, song_dir: Path) -> bool:
    """A miniatura do vídeo vira a capa (album.png, quadrada) se a pasta da música ainda não tiver uma."""
    thumb, cover = thumbnail(audio), Path(song_dir) / "album.png"
    if thumb is None or cover.exists() or not shutil.which("ffmpeg"):
        return False
    result = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(thumb), "-vf",
                             "crop='min(iw,ih)':'min(iw,ih)',scale=512:512", "-frames:v", "1", str(cover)],
                            capture_output=True)
    return result.returncode == 0 and cover.exists()


# ---------------------------------------------------------------------------
# Vídeo de fundo (video.webm na pasta da música)
# ---------------------------------------------------------------------------
# O Clone Hero e o YARG recomendam VP8 .webm, o único que os dois tocam em qualquer sistema (o .mp4
# pesa ou trava o Clone Hero em alguns PCs). O YouTube não entrega VP8: o vídeo é recodificado.
VIDEO_HEIGHT = 720       # resolução máxima: fica atrás da pista, e a conversão é rápida
VIDEO_BITRATE = "2500k"
VIDEO_NAME = "video.webm"


def set_ini_value(path: Path, key: str, value) -> None:
    """Grava key = value no song.ini (troca a linha que já existir)."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines() if Path(path).is_file() else ["[song]"]
    out, done = [], False
    for line in lines:
        if line.split("=", 1)[0].strip().lower() == key:
            line, done = f"{key} = {value}", True
        out.append(line)
    if not done:
        out.append(f"{key} = {value}")
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")


def _probe_seconds(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def to_vp8(src: Path, dst: Path, log=print) -> None:
    """Recodifica o vídeo em VP8 .webm, sem som (o som da música está nos .ogg)."""
    total = _probe_seconds(src)
    tmp = dst.with_name(dst.stem + ".part" + dst.suffix)
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostats", "-progress", "pipe:1", "-i", str(src), "-an",
           "-vf", f"scale=-2:'min({VIDEO_HEIGHT},ih)'", "-c:v", "libvpx", "-b:v", VIDEO_BITRATE, "-crf", "10",
           "-quality", "good", "-cpu-used", "4", "-threads", "8", str(tmp)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    step = -1
    try:
        for line in proc.stdout:
            value = line.partition("=")[2].strip()
            if line.startswith("out_time_us=") and value.isdigit() and total > 0:
                done = int(value) / 1e6 / total
                if int(done * 4) > step:   # uma linha a cada 25% (o log também é onde a geração pode parar)
                    step = int(done * 4)
                    log(f"    convertendo o vídeo... {min(step * 25, 100)}%")
        proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        tmp.unlink(missing_ok=True)
        raise
    if proc.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"não deu para converter o vídeo: {proc.stderr.read().strip()[-300:]}")
    tmp.replace(dst)


def background_video(url: str, folder: Path | None, song_dir: Path, lead_seconds: float = 0.0, log=print) -> Path:
    """Baixa o vídeo, converte para VP8 e põe como video.webm na pasta da música.

    O vídeo convertido fica guardado na pasta dos downloads (o mesmo link não é convertido de novo).
    O video_start_time do song.ini atrasa o vídeo pelo tempo de entrada (lead_in.py), para ele
    andar junto com a música."""
    folder = Path(folder or DEFAULT_DIR)
    vid = video_id(url)
    ready = next(iter(sorted(folder.glob(f"*[[]{vid}[]].video.webm"))), None) if vid and folder.is_dir() else None
    if ready is not None:
        log(f"    vídeo já convertido: {ready.name}")
    else:
        video = info(url)
        folder.mkdir(parents=True, exist_ok=True)
        base = f"{_safe(video.name)} [{video.id}]"
        state = {"step": -1}

        def progress(d):
            if d.get("status") == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate")
                step = int(d.get("downloaded_bytes", 0) * 4 / total) if total else 0
                if step > state["step"]:
                    state["step"] = step
                    log(f"    baixando o vídeo... {min(step * 25, 100)}%")

        options = {**BASE_OPTIONS, "progress_hooks": [progress],
                   "format": f"bestvideo[height<={VIDEO_HEIGHT}]/best[height<={VIDEO_HEIGHT}]/bestvideo/best",
                   "outtmpl": str(folder / f"{base}.video-src.%(ext)s")}
        try:
            with _ydl(options) as ydl:
                ydl.download([video.url])
        except RuntimeError:
            raise
        except Exception as exc:
            raise _download_error(exc) from exc
        downloaded = sorted(folder.glob(f"{glob_escape(base)}.video-src.*"))
        src = next((p for p in downloaded if p.suffix.lower() not in (".part", ".jpg", ".png", ".webp")), None)
        if src is None:
            raise RuntimeError("o download do vídeo terminou, mas o arquivo não apareceu na pasta")
        ready = folder / f"{base}.video.webm"
        to_vp8(src, ready, log)
        for leftover in downloaded:   # o original não é mais preciso
            leftover.unlink(missing_ok=True)
    target = Path(song_dir) / VIDEO_NAME
    shutil.copyfile(ready, target)
    set_ini_value(Path(song_dir) / "song.ini", "video_start_time", -round(lead_seconds * 1000))
    return target


def _duration(seconds) -> str:
    if not seconds:
        return "duração desconhecida"
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"
