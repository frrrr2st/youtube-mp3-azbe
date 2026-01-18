<<<<<<< HEAD
# youtube-mp3-azbe
ajmi skript khatir li ta7mil  youtube vids mp3 , max 50mb l telegram max shit
=======
﻿# Telegram YouTube + Spotify MP3 Bot

Simple Telegram bot that takes a YouTube or Spotify link and returns the highest-quality MP3 it can (via `yt-dlp` + `ffmpeg`).
Spotify links are matched to YouTube audio via search.

## Prerequisites
- Python 3.10+.
- `ffmpeg` available on `PATH` (needed for MP3 conversion).
- A Telegram bot token stored in `TELEGRAM_TOKEN` (already placed in `.env` for convenience).
- Spotify API credentials in `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` (required for Spotify links).

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
# optional: override token
set TELEGRAM_TOKEN=your-bot-token
set SPOTIFY_CLIENT_ID=your-spotify-client-id
set SPOTIFY_CLIENT_SECRET=your-spotify-client-secret
rem optional: help with YouTube 403 blocks
set YTDLP_COOKIES_FROM_BROWSER=chrome
rem or: set YTDLP_COOKIES=C:\path\cookies.txt
rem optional: override large-file handling (defaults to transfer.sh)
set LARGE_FILE_UPLOAD_ENDPOINT=https://transfer.sh
set ALLOW_LARGE_FILE_LINKS=1
python bot.py  # loads .env automatically
```

Send the bot any YouTube or Spotify URL. It downloads the best audio, converts to MP3, and uploads it. If Telegram's ~50MB bot limit is exceeded, the MP3 is uploaded externally (transfer.sh by default) and you get a direct download link.

## Notes
- `yt-dlp` requests `bestaudio/best` and `ffmpeg` converts it to MP3 using the best available quality.
- Downloads are done in a temp dir per request; nothing is persisted.
- Large files: set `ALLOW_LARGE_FILE_LINKS=0` to disable external uploads, or `LARGE_FILE_UPLOAD_ENDPOINT` to change the hosting service.
- Caching: previously downloaded tracks are reused automatically (stored in `cache/` up to ~600MB).
- Progress is shown in-chat; playlist links now show buttons so you can pick a track (or grab the first few).
- Spotify links are matched to the top YouTube search result for the track.
- If YouTube returns 403, update `yt-dlp` and set `YTDLP_COOKIES_FROM_BROWSER` or `YTDLP_COOKIES`.
- Commands: `/start`, `/help`, `/about`, `/status`.
- Keep your bot token private; set it via `TELEGRAM_TOKEN` instead of hard-coding.


>>>>>>> 61f3895 (chore: humanize youtube mp3 azbe)
