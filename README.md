music-to-video

Convert a music file and cover image into:
- A YouTube-ready player-style video
- A YouTube thumbnail image

Requirements
- Python 3.10+
- FFmpeg in PATH (`ffmpeg`, `ffprobe`)
- Pillow

Install
```bash
pip install pillow
```

Quick Start
```bash
python make_player_apple.py --song "Midnight whispers.wav" --cover "Midnight whispers.jpg" --title "Midnight whispers" --artist "Susan"
```

The command above generates:
- `output/Midnight whispers.mp4`
- `output/Midnight whispers_youtube_thumb.jpg`
- `output/Midnight whispers_filter.txt`

Usage
```bash
python make_player_apple.py \
  --song "song.wav" \
  --cover "cover.jpg" \
  --title "Song Title" \
  --artist "Artist Name" \
  --style youtube \
  --quality high \
  --genre "K-POP"
```

Title-Only Mode (Auto Input Lookup)
If you only pass `--title`, the script tries to find matching audio/image files automatically.

```bash
python make_player_apple.py --title "Midnight whispers" --artist "Susan" --genre "K-POP"
```

Audio extensions checked:
- `.wav`, `.mp3`, `.flac`, `.m4a`, `.aac`, `.ogg`

Image extensions checked:
- `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`

Options
- `--song`: audio file path (optional if `--title` is used)
- `--cover`: cover image path (optional if `--title` is used)
- `--title`: track title; also used for auto file lookup
- `--artist`: artist display text (default: `Artist`)
- `--style`: `youtube` (default) or `classic`
- `--quality`: `standard`, `high`, `ultra`
- `--genre`: thumbnail badge text (example: `K-POP`, `R&B`)
- `--tagline`: optional thumbnail subtitle text
- `--skip-thumbnail`: skip thumbnail generation
- `--outdir`: output directory (default: `output`)

Troubleshooting
- If you see `ffmpeg not found` or `ffprobe not found`, install FFmpeg and add it to PATH.
- If title-only mode fails, provide explicit `--song` and `--cover` paths.
