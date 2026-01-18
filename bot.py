import asyncio
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, List, Tuple
from urllib.parse import parse_qs, quote, urlparse

import requests

import yt_dlp
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

zbe = int(os.environ.get("TELEGRAM_MAX_BYTES", str(49 * 1024 * 1024)))
tbonmok = os.environ.get(
    "ALLOW_LARGE_FILE_LINKS", "1"
).lower() not in {"0", "false", "no", "off"}
LARGE_FILE_UPLOAD_ENDPOINT = os.environ.get(
    "LARGE_FILE_UPLOAD_ENDPOINT", "https://transfer.sh"
).rstrip("/")
LARGE_FILE_UPLOAD_TIMEOUT = int(os.environ.get("LARGE_FILE_UPLOAD_TIMEOUT", "900"))
lihwak = 3
PLAYLIST_PICK_LIMIT = 8
rbk = 600 * 1024 * 1024
CACHE_DIR = Path("cache")
DOWNLOAD_SEMAPHORE = asyncio.Semaphore(3)
START_TIME = time.monotonic()
ACTIVE_DOWNLOADS = 0

YOUTUBE_REGEX = re.compile(
    r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/", re.IGNORECASE
)
SPOTIFY_REGEX = re.compile(
    r"(https?://)?(open\.)?spotify\.com/|spotify:", re.IGNORECASE
)
SPOTIFY_TYPES = {"track", "album", "playlist"}
_SPOTIFY_CLIENT = None
_SPOTIFY_CLIENT_ERROR = None
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
COOKIES_FROM_BROWSER_RE = re.compile(
    r"""(?x)
    (?P<name>[^+:]+)
    (?:\s*\+\s*(?P<keyring>[^:]+))?
    (?:\s*:\s*(?!:)(?P<profile>.+?))?
    (?:\s*::\s*(?P<container>.+))?
    """
)
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/119.0.0.0 Safari/537.36"
)
DEFAULT_PLAYER_CLIENTS = ("android", "web")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("yt_mp3_bot")
HELP_TEXT = (
    "Send me a YouTube or Spotify link and I'll fetch the best-quality MP3 I can. "
    "Spotify links are matched to YouTube audio. "
    f"Playlists supported (pick a track or send first {lihwak}), "
    "and if a track is above Telegram's ~50MB bot limit I'll upload a direct link for you. "
    "Commands:\n"
    "/start - quick intro\n"
    "/help  - this help\n"
    "/about - bot info\n"
    "/status - bot stats"
)
ABOUT_TEXT = (
    "Built with yt-dlp + ffmpeg. Spotify links are matched via YouTube search. "
    "Large files get uploaded via shareable links automatically."
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT)


async def about_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(ABOUT_TEXT)


def _format_uptime(seconds: float) -> str:
    mins, sec = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    days, hrs = divmod(hrs, 24)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hrs:
        parts.append(f"{hrs}h")
    if mins:
        parts.append(f"{mins}m")
    parts.append(f"{sec}s")
    return " ".join(parts)


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cache_count, cache_bytes = _cache_stats()
    uptime = _format_uptime(time.monotonic() - START_TIME)
    await update.message.reply_text(
        f"Uptime: {uptime}\n"
        f"Active downloads: {ACTIVE_DOWNLOADS}\n"
        f"Cache: {cache_count} files, {_human_size(cache_bytes)}"
    )


async def _safe_edit_text(message, text: str) -> None:
    try:
        await message.edit_text(text)
    except Exception:
        pass


def _sanitize_title(title: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]", "_", title)
    return cleaned.strip() or "audio"


def _human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def _ensure_cache_dir() -> None:
    try:
        CACHE_DIR.mkdir(exist_ok=True)
    except Exception:
        pass


def _cache_path(video_id: str) -> Path:
    return CACHE_DIR / f"{video_id}.mp3"


def _find_cached_audio(video_id: str) -> Path | None:
    if not video_id:
        return None
    path = _cache_path(video_id)
    if path.exists() and path.is_file():
        return path
    return None


def _trim_cache() -> None:
    try:
        files = [p for p in CACHE_DIR.glob("*.mp3") if p.is_file()]
    except FileNotFoundError:
        return
    total = sum(p.stat().st_size for p in files)
    if total <= rbk:
        return
    for p in sorted(files, key=lambda x: x.stat().st_mtime):
        try:
            total -= p.stat().st_size
            p.unlink(missing_ok=True)
        except Exception:
            continue
        if total <= rbk:
            break


def _store_in_cache(src: Path, video_id: str) -> None:
    if not video_id:
        return
    _ensure_cache_dir()
    dest = _cache_path(video_id)
    try:
        if not dest.exists():
            shutil.copyfile(src, dest)
        dest.touch()
        _trim_cache()
    except Exception:
        LOGGER.warning("Failed to cache file for %s", video_id, exc_info=True)


def _cache_stats() -> tuple[int, int]:
    try:
        files = [p for p in CACHE_DIR.glob("*.mp3") if p.is_file()]
    except FileNotFoundError:
        return 0, 0
    total = sum(p.stat().st_size for p in files)
    return len(files), total


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text or "").strip()


def _parse_cookies_from_browser(value: str | None) -> tuple[str, str | None, str | None, str | None] | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    match = COOKIES_FROM_BROWSER_RE.fullmatch(value)
    if not match and "," in value:
        parts = [part.strip() for part in value.split(",", 3)]
        parts += [""] * (4 - len(parts))
        browser = parts[0].lower()
        profile = parts[1] or None
        keyring = parts[2].upper() if parts[2] else None
        container = parts[3] or None
        return (browser, profile, keyring, container)
    if not match:
        return None
    browser, keyring, profile, container = match.group(
        "name", "keyring", "profile", "container"
    )
    browser = browser.strip().lower()
    profile = profile.strip() if profile else None
    keyring = keyring.strip().upper() if keyring else None
    container = container.strip() if container else None
    return (browser, profile, keyring, container)


def _parse_player_clients(value: str | None) -> list[str]:
    raw = value or ",".join(DEFAULT_PLAYER_CLIENTS)
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def _yt_dlp_common_opts() -> dict:
    cookies_path = os.environ.get("YTDLP_COOKIES")
    raw_cookies_from_browser = os.environ.get("YTDLP_COOKIES_FROM_BROWSER")
    cookies_from_browser = _parse_cookies_from_browser(raw_cookies_from_browser)
    if raw_cookies_from_browser and not cookies_from_browser:
        LOGGER.warning(
            "Invalid YTDLP_COOKIES_FROM_BROWSER format: %s",
            raw_cookies_from_browser,
        )
    player_clients = _parse_player_clients(
        os.environ.get("YTDLP_PLAYER_CLIENTS")
    )
    headers = {
        "User-Agent": os.environ.get("YTDLP_USER_AGENT") or DEFAULT_USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
    }
    opts = {
        "quiet": True,
        "no_warnings": True,
        "http_headers": headers,
        "cookiefile": cookies_path or None,
        "cookiesfrombrowser": cookies_from_browser,
        "extractor_args": {"youtube": {"player_client": player_clients}},
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 20,
    }
    return {key: value for key, value in opts.items() if value is not None}


def _read_env_value(path: Path, key: str) -> str | None:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        env_key, env_val = stripped.split("=", 1)
        env_key = env_key.strip().lstrip("\ufeff")
        if env_key != key:
            continue
        value = env_val.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        return value or None
    return None


def _format_user_error(exc: Exception) -> str:
    text = _strip_ansi(str(exc))
    if "HTTP Error 403" in text or ("403" in text and "Forbidden" in text):
        hints = [
            "YouTube blocked the download (403).",
            "Update yt-dlp (python -m pip install -U yt-dlp).",
        ]
        if not os.environ.get("YTDLP_COOKIES") and not os.environ.get(
            "YTDLP_COOKIES_FROM_BROWSER"
        ):
            hints.append(
                "Set YTDLP_COOKIES_FROM_BROWSER=chrome or YTDLP_COOKIES=path\\cookies.txt."
            )
        return " ".join(hints)
    return text or "Unknown error."


def _upload_large_file(path: Path) -> str:
    """
    Upload oversized files to an external host so we can share links beyond
    Telegram's ~50MB bot cap. Uses transfer.sh by default.
    """
    target = f"{LARGE_FILE_UPLOAD_ENDPOINT}/{quote(path.name)}"
    try:
        with path.open("rb") as fp:
            resp = requests.put(target, data=fp, timeout=LARGE_FILE_UPLOAD_TIMEOUT)
    except Exception as exc:
        raise RuntimeError("Failed uploading large file for sharing.") from exc

    if resp.status_code >= 400:
        raise RuntimeError(
            f"Upload failed with HTTP {resp.status_code}: {resp.text.strip()}"
        )

    url = (resp.text or "").strip()
    return url or target


def _download_audio(
    url: str,
    workdir: Path,
    progress_hook: Callable | None = None,
    playlist_items: int = 1,
) -> Tuple[List[dict], dict]:
    """
    Blocking helper to run yt_dlp in a thread.
    Returns (list_of_audio_items, info). Each item has path/title/performer/source.
    """
    ydl_opts = _yt_dlp_common_opts()
    ydl_opts.update(
        {
            "noplaylist": playlist_items == 1,
            "playlist_items": f"1-{playlist_items}" if playlist_items > 1 else None,
            "outtmpl": str(workdir / "%(id)s.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "0",
                }
            ],
            "progress_hooks": [progress_hook] if progress_hook else [],
        }
    )

    ydl_opts = {k: v for k, v in ydl_opts.items() if v is not None}

    def _run(opts: dict) -> dict:
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=True)

    formats_to_try = [
        "bestaudio/best",
        "best",
        "bestvideo+bestaudio/best",
        "best*",
    ]
    info = None
    last_exc: Exception | None = None
    for fmt in formats_to_try:
        opts_try = dict(ydl_opts)
        opts_try["format"] = fmt
        try:
            info = _run(opts_try)
            break
        except Exception as exc:
            last_exc = exc
            text = _strip_ansi(str(exc))
            if "Requested format is not available" in text or "no such format" in text:
                LOGGER.warning(
                    "Format %s not available for %s; trying next", fmt, url
                )
                continue
            raise

    if info is None:
        if last_exc:
            raise last_exc
        raise RuntimeError("No suitable formats available for this video.")

    mp3_files = sorted(workdir.glob("*.mp3"), key=lambda p: p.stat().st_mtime)
    if not mp3_files:
        raise RuntimeError("Download finished but no mp3 was produced.")

    entries = info.get("entries") or []
    audio_items: List[dict] = []
    for idx, mp3_path in enumerate(mp3_files):
        entry = entries[idx] if idx < len(entries) else info
        track_id = entry.get("id")
        title = entry.get("title") or mp3_path.stem
        performer = entry.get("uploader") or "YouTube"
        source_url = entry.get("webpage_url") or url
        audio_items.append(
            {
                "path": mp3_path,
                "id": track_id,
                "title": title,
                "performer": performer,
                "source": source_url,
            }
        )

    return audio_items, info


async def _send_audio_or_link(
    mp3_path: Path,
    message,
    title: str,
    performer: str,
    caption: str,
    status_msg,
    cache_id: str | None = None,
) -> None:
    """
    Send the MP3 if within Telegram limits, otherwise upload externally and share a link.
    """
    file_size = mp3_path.stat().st_size
    if file_size > zbe:
        if not tbonmok:
            raise RuntimeError(
                f"File '{mp3_path.name}' is {_human_size(file_size)} which is above "
                "Telegram's limit and external uploads are disabled."
            )
        if status_msg:
            await _safe_edit_text(
                status_msg,
                f"File {_human_size(file_size)} exceeds Telegram; uploading link...",
            )
        try:
            download_url = await asyncio.wait_for(
                asyncio.to_thread(_upload_large_file, mp3_path),
                timeout=LARGE_FILE_UPLOAD_TIMEOUT + 60,
            )
        except Exception as exc:
            raise RuntimeError(
                f"File '{mp3_path.name}' is {_human_size(file_size)} (> Telegram limit) "
                "and the external upload failed."
            ) from exc

        await message.reply_text(
            f"{title} ({_human_size(file_size)}) is over Telegram's limit.\n"
            f"Download: {download_url}"
        )
    else:
        if status_msg:
            await _safe_edit_text(status_msg, f"Uploading: {title[:64]}...")
        with mp3_path.open("rb") as audio_fp:
            await message.reply_audio(
                audio=audio_fp,
                title=title[:128],
                performer=performer[:128],
                caption=caption[:1024],
            )

    if cache_id:
        _store_in_cache(mp3_path, cache_id)


def _looks_like_playlist(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    qs = parse_qs(parsed.query)
    return "list" in qs or "playlist" in parsed.path.lower()


def _extract_playlist_id(url: str) -> str | None:
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    qs = parse_qs(parsed.query)
    return qs.get("list", [None])[0]


def fetch_basic_info(url: str, flat: bool = False) -> dict:
    opts = _yt_dlp_common_opts()
    opts.update(
        {
            "skip_download": True,
            "extract_flat": flat,
            "noplaylist": not flat,
        }
    )
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)


def fetch_playlist_preview(url: str, limit: int) -> tuple[str | None, List[dict], int | None]:
    opts = _yt_dlp_common_opts()
    opts.update(
        {
            "skip_download": True,
            "extract_flat": True,
            "playlist_items": f"1-{limit}",
            "noplaylist": False,
        }
    )
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = info.get("entries") or []
    preview = []
    for idx, entry in enumerate(entries[:limit]):
        vid = entry.get("id") or entry.get("url")
        if not vid:
            continue
        title = entry.get("title") or f"Track {idx + 1}"
        preview.append({"id": vid, "title": title})
    playlist_id = info.get("id") or _extract_playlist_id(url)
    total = info.get("playlist_count")
    return playlist_id, preview, total


def _get_spotify_client() -> tuple[object | None, str | None]:
    global _SPOTIFY_CLIENT, _SPOTIFY_CLIENT_ERROR
    if _SPOTIFY_CLIENT:
        return _SPOTIFY_CLIENT, None
    if _SPOTIFY_CLIENT_ERROR:
        return None, _SPOTIFY_CLIENT_ERROR

    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        _SPOTIFY_CLIENT_ERROR = (
            "Spotify support needs SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET."
        )
        return None, _SPOTIFY_CLIENT_ERROR

    try:
        from spotipy import Spotify
        from spotipy.oauth2 import SpotifyClientCredentials
    except Exception:
        _SPOTIFY_CLIENT_ERROR = "Spotify support needs the spotipy package."
        return None, _SPOTIFY_CLIENT_ERROR

    try:
        auth = SpotifyClientCredentials(
            client_id=client_id, client_secret=client_secret
        )
        _SPOTIFY_CLIENT = Spotify(
            auth_manager=auth,
            requests_timeout=10,
            retries=2,
        )
    except Exception:
        LOGGER.exception("Failed to initialize Spotify client.")
        _SPOTIFY_CLIENT_ERROR = "Failed to initialize Spotify client."
        return None, _SPOTIFY_CLIENT_ERROR

    return _SPOTIFY_CLIENT, None


def _parse_spotify_ref(value: str) -> tuple[str, str] | None:
    value = (value or "").strip()
    if value.lower().startswith("spotify:"):
        parts = value.split(":")
        if len(parts) >= 3:
            item_type = parts[1].lower()
            item_id = parts[2]
            if item_type in SPOTIFY_TYPES and item_id:
                return item_type, item_id
        return None

    try:
        parsed = urlparse(value)
    except Exception:
        return None
    host = parsed.netloc.lower()
    if not host.endswith("spotify.com"):
        return None
    segments = [seg for seg in parsed.path.split("/") if seg]
    for idx, segment in enumerate(segments):
        segment_lower = segment.lower()
        if segment_lower in SPOTIFY_TYPES and idx + 1 < len(segments):
            return segment_lower, segments[idx + 1]
    return None


def _spotify_track_metadata(track: dict) -> tuple[str, str, str]:
    title = track.get("name") or "Spotify Track"
    artists = ", ".join(
        artist.get("name")
        for artist in track.get("artists") or []
        if artist.get("name")
    )
    performer = artists or "Spotify"
    query = f"{performer} - {title} audio"
    return title, performer, query


def fetch_spotify_collection_preview(
    spotify, item_type: str, item_id: str, limit: int
) -> tuple[List[dict], int | None]:
    if item_type == "album":
        result = spotify.album_tracks(item_id, limit=limit)
        total = result.get("total")
        tracks = result.get("items") or []
    else:
        result = spotify.playlist_items(
            item_id,
            limit=limit,
            fields="items(track(id,name,artists(name))),total",
            additional_types=("track",),
        )
        total = result.get("total")
        tracks = [item.get("track") for item in result.get("items") or []]

    preview = []
    for idx, track in enumerate(tracks):
        if not track:
            continue
        track_id = track.get("id")
        name = track.get("name") or f"Track {idx + 1}"
        artists = ", ".join(
            artist.get("name")
            for artist in track.get("artists") or []
            if artist.get("name")
        )
        label = f"{name} - {artists}" if artists else name
        if track_id:
            preview.append({"id": track_id, "title": label})

    return preview, total


def fetch_spotify_collection_tracks(
    spotify, item_type: str, item_id: str, limit: int
) -> List[dict]:
    if item_type == "album":
        result = spotify.album_tracks(item_id, limit=limit)
        tracks = result.get("items") or []
    elif item_type == "playlist":
        result = spotify.playlist_items(
            item_id,
            limit=limit,
            fields="items(track(id,name,artists(name)))",
            additional_types=("track",),
        )
        tracks = [item.get("track") for item in result.get("items") or []]
    else:
        return []

    return [track for track in tracks if track and track.get("id")]


def make_progress_hook(loop: asyncio.AbstractEventLoop, status_msg):
    """
    Build a yt_dlp progress hook that edits the Telegram status message.
    Throttled to avoid Telegram flood limits.
    """
    last_sent = {"ts": 0.0, "text": ""}

    def hook(d: dict) -> None:
        status = d.get("status")
        now = time.monotonic()
        if now - last_sent["ts"] < 1.0:
            return

        text = None
        if status == "downloading":
            percent = (d.get("_percent_str") or "").strip()
            speed = (d.get("_speed_str") or "").strip()
            eta = d.get("eta")

            parts = ["Downloading...", percent]
            if speed:
                parts.append(f"@ {speed}")
            if eta:
                parts.append(f"ETA {int(eta)}s")
            text = " ".join(parts)
        elif status == "finished":
            text = "Download complete. Converting to MP3..."

        if text and text != last_sent["text"]:
            last_sent["ts"] = now
            last_sent["text"] = text
            asyncio.run_coroutine_threadsafe(
                _safe_edit_text(status_msg, text), loop
            )

    return hook


async def download_and_send_search(
    search_query: str,
    message,
    title: str,
    performer: str,
    source_url: str,
    cache_id: str | None = None,
    status_prefix: str = "",
) -> None:
    """
    Search YouTube for a query, download the top match, then upload it.
    """
    global ACTIVE_DOWNLOADS

    await message.chat.send_action(action=ChatAction.TYPING)
    status_msg = await message.reply_text(
        status_prefix + "Searching YouTube..."
    )
    loop = asyncio.get_running_loop()
    progress_hook = make_progress_hook(loop, status_msg)

    try:
        async with DOWNLOAD_SEMAPHORE:
            ACTIVE_DOWNLOADS += 1
            if cache_id:
                cache_hit = _find_cached_audio(cache_id)
                if cache_hit:
                    await _send_audio_or_link(
                        cache_hit,
                        message,
                        title=_sanitize_title(title),
                        performer=performer,
                        caption=f"{title}\nFrom: {source_url}",
                        status_msg=status_msg,
                        cache_id=None,
                    )
                    return

            with tempfile.TemporaryDirectory() as tmpdir:
                workdir = Path(tmpdir)
                try:
                    audio_items, _info = await asyncio.wait_for(
                        asyncio.to_thread(
                            _download_audio,
                            f"ytsearch1:{search_query}",
                            workdir,
                            progress_hook,
                            1,
                        ),
                        timeout=600,
                    )
                except asyncio.TimeoutError as exc:
                    LOGGER.exception("Timed out downloading search %s", search_query)
                    raise RuntimeError("Download timed out.") from exc

                mp3_path = audio_items[0]["path"]
                safe_title = _sanitize_title(title)
                await _send_audio_or_link(
                    mp3_path,
                    message,
                    title=safe_title,
                    performer=performer,
                    caption=f"{title}\nFrom: {source_url}",
                    status_msg=status_msg,
                    cache_id=cache_id,
                )
    except Exception as exc:
        LOGGER.exception("Failed to fetch audio for search %s", search_query)
        await message.reply_text(
            f"Sorry, something went wrong: {_format_user_error(exc)}"
        )
    finally:
        ACTIVE_DOWNLOADS = max(ACTIVE_DOWNLOADS - 1, 0)
        try:
            await status_msg.delete()
        except Exception:
            pass


async def download_and_send(
    url: str,
    message,
    playlist_items: int = 1,
    status_prefix: str = "",
) -> None:
    """
    Core flow: optional cache check, download, size check, upload.
    """
    global ACTIVE_DOWNLOADS

    await message.chat.send_action(action=ChatAction.TYPING)
    status_msg = await message.reply_text(
        status_prefix + ("Checking cache..." if playlist_items == 1 else "Downloading...")
    )
    loop = asyncio.get_running_loop()
    progress_hook = make_progress_hook(loop, status_msg)

    try:
        async with DOWNLOAD_SEMAPHORE:
            ACTIVE_DOWNLOADS += 1
            try:
                info_basic = await asyncio.to_thread(
                    fetch_basic_info, url, False
                )
            except Exception:
                info_basic = {}
            video_id = info_basic.get("id")
            title_hint = info_basic.get("title") or "YouTube Audio"
            performer_hint = info_basic.get("uploader") or "YouTube"

            if playlist_items == 1 and video_id:
                cache_hit = _find_cached_audio(video_id)
                if cache_hit:
                    await _send_audio_or_link(
                        cache_hit,
                        message,
                        title=_sanitize_title(title_hint),
                        performer=performer_hint,
                        caption=f"{title_hint}\nFrom: {url}",
                        status_msg=status_msg,
                        cache_id=None,
                    )
                    return

            with tempfile.TemporaryDirectory() as tmpdir:
                workdir = Path(tmpdir)
                try:
                    audio_items, info = await asyncio.wait_for(
                        asyncio.to_thread(
                            _download_audio, url, workdir, progress_hook, playlist_items
                        ),
                        timeout=900 if playlist_items > 1 else 600,
                    )
                except asyncio.TimeoutError as exc:
                    LOGGER.exception("Timed out downloading %s", url)
                    raise RuntimeError("Download timed out.") from exc

                for item in audio_items:
                    mp3_path = item["path"]
                    title = _sanitize_title(item["title"] or title_hint)
                    performer = item["performer"] or performer_hint
                    caption = f"{title}\nFrom: {item['source']}"
                    cache_id = item.get("id") if playlist_items == 1 else None

                    await _send_audio_or_link(
                        mp3_path,
                        message,
                        title=title,
                        performer=performer,
                        caption=caption,
                        status_msg=status_msg,
                        cache_id=cache_id,
                    )

                if playlist_items > 1 and info.get("playlist_count"):
                    total = info.get("playlist_count")
                    if total and total > playlist_items:
                        await message.reply_text(
                            f"Sent first {playlist_items} tracks out of {total}. "
                            "Send a direct video link for a specific track."
                        )
    except Exception as exc:
        LOGGER.exception("Failed to fetch audio for %s", url)
        await message.reply_text(
            f"Sorry, something went wrong: {_format_user_error(exc)}"
        )
    finally:
        ACTIVE_DOWNLOADS = max(ACTIVE_DOWNLOADS - 1, 0)
        try:
            await status_msg.delete()
        except Exception:
            pass


def _spotify_cache_id(track_id: str) -> str:
    return f"sp_{track_id}"


async def download_spotify_track_data(
    track: dict,
    message,
    status_prefix: str = "",
) -> None:
    track_id = track.get("id")
    if not track_id:
        await message.reply_text("Spotify track data is missing an ID.")
        return
    title, performer, query = _spotify_track_metadata(track)
    source_url = f"https://open.spotify.com/track/{track_id}"
    await download_and_send_search(
        query,
        message,
        title=title,
        performer=performer,
        source_url=source_url,
        cache_id=_spotify_cache_id(track_id),
        status_prefix=status_prefix,
    )


async def download_spotify_track_id(
    spotify, track_id: str, message, status_prefix: str = ""
) -> None:
    try:
        track = await asyncio.to_thread(spotify.track, track_id)
    except Exception:
        LOGGER.exception("Failed to fetch Spotify track %s", track_id)
        await message.reply_text("Failed to fetch Spotify track details.")
        return

    await download_spotify_track_data(track, message, status_prefix=status_prefix)


async def download_spotify_collection(
    spotify, item_type: str, item_id: str, message
) -> None:
    try:
        tracks = await asyncio.to_thread(
            fetch_spotify_collection_tracks,
            spotify,
            item_type,
            item_id,
            lihwak,
        )
    except Exception:
        LOGGER.exception("Failed to fetch Spotify %s %s", item_type, item_id)
        await message.reply_text(
            f"Failed to fetch Spotify {item_type} tracks."
        )
        return

    if not tracks:
        await message.reply_text("No tracks found in that Spotify list.")
        return

    total = len(tracks)
    for idx, track in enumerate(tracks, start=1):
        await download_spotify_track_data(
            track,
            message,
            status_prefix=f"[{idx}/{total}] ",
        )


async def handle_spotify_link(url: str, message) -> None:
    spotify, error = _get_spotify_client()
    if not spotify:
        await message.reply_text(error or "Spotify support is not configured.")
        return

    parsed = _parse_spotify_ref(url)
    if not parsed:
        await message.reply_text(
            "Unsupported Spotify link. Use a track, album, or playlist link."
        )
        return
    item_type, item_id = parsed

    if item_type == "track":
        await download_spotify_track_id(spotify, item_id, message)
        return

    if item_type in {"album", "playlist"}:
        try:
            preview, total = await asyncio.to_thread(
                fetch_spotify_collection_preview,
                spotify,
                item_type,
                item_id,
                PLAYLIST_PICK_LIMIT,
            )
        except Exception:
            LOGGER.exception(
                "Failed to preview Spotify %s %s", item_type, item_id
            )
            await message.reply_text(
                f"Couldn't preview that Spotify {item_type}. "
                f"Sending first {lihwak} tracks."
            )
            await download_spotify_collection(
                spotify, item_type, item_id, message
            )
            return

        if not preview:
            await download_spotify_collection(
                spotify, item_type, item_id, message
            )
            return

        buttons = []
        for idx, item in enumerate(preview):
            label = f"{idx + 1}. {item['title'][:40]}"
            buttons.append(
                InlineKeyboardButton(
                    label, callback_data=f"sptrack:{item['id']}"
                )
            )
        rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
        rows.append(
            [
                InlineKeyboardButton(
                    f"Send first {lihwak}",
                    callback_data=f"spdl:{item_type}:{item_id}",
                )
            ]
        )

        total_text = f" ({total} total)" if total else ""
        await message.reply_text(
            f"Spotify {item_type} detected{total_text}. Pick a track to download:",
            reply_markup=InlineKeyboardMarkup(rows),
        )
        return

    await message.reply_text("Unsupported Spotify link type.")


async def handle_link(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message:
        return

    url = (message.text or "").strip()
    is_spotify = bool(SPOTIFY_REGEX.search(url))
    is_youtube = bool(YOUTUBE_REGEX.search(url))
    if not (is_spotify or is_youtube):
        await message.reply_text(
            "Please send a YouTube or Spotify link."
        )
        return

    if not shutil.which("ffmpeg"):
        await message.reply_text(
            "ffmpeg is not installed or not on PATH. Install it first, then try again."
        )
        return

    if is_spotify:
        await handle_spotify_link(url, message)
        return

    playlist_hint = _looks_like_playlist(url)
    if playlist_hint:
        try:
            playlist_id, preview, total = await asyncio.to_thread(
                fetch_playlist_preview, url, PLAYLIST_PICK_LIMIT
            )
        except Exception:
            await download_and_send(
                url,
                message,
                playlist_items=lihwak,
                status_prefix=f"Playlist detected; sending first {lihwak}... ",
            )
            return

        buttons = []
        for idx, item in enumerate(preview):
            label = f"{idx + 1}. {item['title'][:40]}"
            buttons.append(
                InlineKeyboardButton(label, callback_data=f"plvid:{item['id']}")
            )
        if not buttons:
            await download_and_send(
                url,
                message,
                playlist_items=lihwak,
                status_prefix=f"Playlist detected; sending first {lihwak}... ",
            )
            return
        rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]

        extra = []
        if playlist_id:
            extra.append(
                InlineKeyboardButton(
                    f"Send first {lihwak}",
                    callback_data=f"pldl:{playlist_id}",
                )
            )
        if extra:
            rows.append(extra)

        total_text = f" ({total} total)" if total else ""
        await message.reply_text(
            f"Playlist detected{total_text}. Pick a track to download:",
            reply_markup=InlineKeyboardMarkup(rows),
        )
        return

    await download_and_send(url, message)


async def handle_playlist_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    query = update.callback_query
    if not query or not query.data:
        return

    data = query.data
    await query.answer()
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception:
        pass

    if data.startswith("plvid:"):
        video_id = data.split(":", 1)[1]
        url = f"https://youtu.be/{video_id}"
        await download_and_send(url, query.message)
    elif data.startswith("pldl:"):
        playlist_id = data.split(":", 1)[1]
        playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"
        await download_and_send(
            playlist_url,
            query.message,
            playlist_items=lihwak,
            status_prefix=f"Playlist detected; sending first {lihwak}... ",
        )


    elif data.startswith("sptrack:"):
        track_id = data.split(":", 1)[1]
        spotify, error = _get_spotify_client()
        if not spotify:
            await query.message.reply_text(
                error or "Spotify support is not configured."
            )
            return
        await download_spotify_track_id(spotify, track_id, query.message)
    elif data.startswith("spdl:"):
        parts = data.split(":", 2)
        if len(parts) < 3:
            return
        item_type, item_id = parts[1], parts[2]
        spotify, error = _get_spotify_client()
        if not spotify:
            await query.message.reply_text(
                error or "Spotify support is not configured."
            )
            return
        if item_type not in {"album", "playlist"}:
            await query.message.reply_text("Unsupported Spotify link type.")
            return
        await download_spotify_collection(spotify, item_type, item_id, query.message)


def main() -> None:
    env_path = Path(__file__).with_name(".env")
    load_dotenv(dotenv_path=env_path, override=True)
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        token = _read_env_value(env_path, "TELEGRAM_TOKEN")
        if token:
            os.environ["TELEGRAM_TOKEN"] = token
    if not token:
        raise RuntimeError("Set TELEGRAM_TOKEN env var with your bot token.")

    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    application = (
        Application.builder()
        .token(token)
        .read_timeout(30)
        .write_timeout(30)
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("about", about_cmd))
    application.add_handler(CommandHandler("status", status_cmd))
    application.add_handler(
        CallbackQueryHandler(
            handle_playlist_callback, pattern="^(plvid|pldl|sptrack|spdl):"
        )
    )
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_link,
        )
    )

    LOGGER.info("Bot starting...")
    application.run_polling(stop_signals=None)


if __name__ == "__main__":
    main()
