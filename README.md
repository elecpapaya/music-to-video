music-to-video

Turn a single track into YouTube-ready visual assets with one command.

What this tool produces:
- A full HD (`1920x1080`) player-style music video with dynamic UI motion
- A click-oriented YouTube thumbnail generated from the same cover art
- A saved FFmpeg filter graph for debugging and custom tuning

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
python make_player_apple.py --title "Midnight satellite" --artist "SUNO AI" --genre "K-POP"
```

The command above generates:
- `output/Midnight satellite.mp4`
- `output/Midnight satellite_youtube_thumb.jpg`
- `output/Midnight satellite_filter.txt`

Usage

Title-only mode (recommended)
```bash
python make_player_apple.py \
  --title "Midnight satellite" \
  --artist "SUNO AI" \
  --genre "K-POP"
```

Explicit file mode (`--song` / `--cover`)
```bash
python make_player_apple.py \
  --song "song.wav" \
  --cover "cover.jpg" \
  --title "Song Title" \
  --artist "Artist Name" \
  --style youtube \
  --quality high \
  --motion-smooth 0.18 \
  --knob-scale 1.0 \
  --genre "K-POP"
```

Title-Only Mode (Auto Input Lookup)
If you pass only `--title`, the script tries to find matching audio/image files automatically.

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
- `--title-max-lines`: max title line count in video/thumbnail (default: `3`)
- `--title-min-font`: minimum title font size bound (default: `44`)
- `--title-max-font`: maximum title font size bound (default: `82`)
- `--motion-smooth`: progress/knob easing amount `0.0..1.0` (default: `0.18`)
- `--knob-scale`: progress knob size multiplier (default: `1.0`)
- `--font-title`: optional custom title font path
- `--font-body`: optional custom body font path
- `--genre`: thumbnail badge text (example: `K-POP`, `R&B`)
- `--tagline`: optional thumbnail subtitle text
- `--skip-thumbnail`: skip thumbnail generation
- `--outdir`: output directory (default: `output`)

Thumbnail safety behavior
- Text is automatically constrained to safe margins for desktop/mobile readability.
- Text placement avoids YouTube's bottom-right timestamp region.

Standalone thumbnail script
```bash
python make_thumbnail_youtube.py \
  --cover "cover.jpg" \
  --title "Song Title" \
  --artist "Artist Name" \
  --genre "K-POP" \
  --title-max-lines 3 \
  --title-min-font 56 \
  --title-max-font 102
```

Troubleshooting
- If you see `ffmpeg not found` or `ffprobe not found`, install FFmpeg and add it to PATH.
- If title-only mode fails, provide explicit `--song` and `--cover` paths.

License
MIT (see LICENSE)
