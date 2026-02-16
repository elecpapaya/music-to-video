music-to-video

Turn one music track into YouTube-ready assets with one command.

What this tool produces:
- A full HD (`1920x1080`) player-style music video
- A click-oriented YouTube thumbnail
- A saved FFmpeg filter graph (`*_filter.txt`) for tuning

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

Generated files:
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

Explicit file mode
```bash
python make_player_apple.py \
  --song "song.wav" \
  --cover "cover.jpg" \
  --title "Song Title" \
  --artist "Artist Name"
```

Advanced Motion & Reactive Features

B-roll auto edit from clip folder:
```bash
python make_player_apple.py \
  --title "Midnight satellite" \
  --artist "SUNO AI" \
  --broll-dir "./broll" \
  --beat-cut auto \
  --camera-motion parallax \
  --reactive-level 0.40 \
  --color-preset cinema
```

Single B-roll loop video:
```bash
python make_player_apple.py \
  --title "Midnight satellite" \
  --broll-video "./bg_loop.mp4" \
  --broll-blend softlight \
  --broll-opacity 0.32
```

Color preset + LUT:
```bash
python make_player_apple.py \
  --title "Midnight satellite" \
  --color-preset neon \
  --lut "./luts/teal_orange.cube" \
  --lut-intensity 0.55
```

Title-only mode lookup
If `--song` / `--cover` are omitted, the script searches by title base name.

Audio extensions:
- `.wav`, `.mp3`, `.flac`, `.m4a`, `.aac`, `.ogg`

Image extensions:
- `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`

Main options
- `--style`: `youtube` or `classic`
- `--quality`: `standard`, `high`, `ultra`
- `--title-max-lines`, `--title-min-font`, `--title-max-font`
- `--motion-smooth`, `--knob-scale`
- `--font-title`, `--font-body`
- `--broll-dir`, `--broll-video`
- `--broll-opacity`, `--broll-blend`
- `--beat-cut`, `--beat-sensitivity`
- `--min-cut-interval`, `--max-cut-interval`
- `--camera-motion`, `--camera-strength`
- `--reactive-level`, `--reactive-glow`, `--reactive-blur`, `--reactive-shake`
- `--color-preset`, `--lut`, `--lut-intensity`
- `--skip-thumbnail`

Standalone thumbnail script
```bash
python make_thumbnail_youtube.py \
  --cover "cover.jpg" \
  --title "Song Title" \
  --artist "Artist Name" \
  --genre "K-POP"
```

Troubleshooting
- If `ffmpeg` / `ffprobe` are missing, install FFmpeg and add it to PATH.
- If title-only lookup fails, pass explicit `--song` and `--cover`.
- If LUT is invalid, video renders with preset-only color grading.

License
MIT (see LICENSE)
