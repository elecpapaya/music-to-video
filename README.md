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
Note: this command uses defaults. EQ default is `--eq-style legacy`, so prism-style color changes are not enabled unless you set `--eq-style prism`.

Generated files:
- `output/Midnight satellite_apple.mp4`
- `output/Midnight satellite_apple_thumb.jpg`
- `output/Midnight satellite_apple_filter.txt`

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

Professional EQ (Studio Analyzer):
```bash
python make_player_apple.py \
  --title "Midnight satellite" \
  --eq-style studio \
  --eq-intensity balanced \
  --eq-quality medium
```

Punchier EQ:
```bash
python make_player_apple.py \
  --title "Midnight satellite" \
  --eq-style studio \
  --eq-intensity punchy \
  --eq-glow 0.55
```

Low-load EQ:
```bash
python make_player_apple.py \
  --title "Midnight satellite" \
  --eq-style studio \
  --eq-intensity subtle \
  --eq-quality low
```

Prism EQ (Apple, opt-in):
```bash
python make_player_apple.py \
  --title "Midnight satellite" \
  --eq-style prism \
  --eq-intensity balanced \
  --eq-color-drive 0.55
```

Prism EQ (Glass, opt-in):
```bash
python make_player_glass.py \
  --title "Midnight satellite" \
  --eq-style prism \
  --eq-color-drive 0.55
```

Note: `make_player_glass.py --eq-style prism` runs one extra audio analysis pass (`astats`) to drive beat-synced hue motion.

Prism rendering (current):
- EQ panel fill tint is transparent (no full-area orange wash).
- Color motion is applied to EQ bars/glow only.
- 3-band split tint is removed to avoid vertical seam artifacts.

Higher-visibility Prism presets:
```bash
python make_player_apple.py \
  --title "Midnight satellite" \
  --eq-style prism \
  --eq-intensity punchy \
  --eq-glow 0.85 \
  --eq-opacity 1.00 \
  --eq-color-drive 0.70
```

```bash
python make_player_glass.py \
  --title "Midnight satellite" \
  --eq-style prism \
  --eq-color-drive 0.70 \
  --reactive-level 0.40 \
  --reactive-glow 0.55
```

Why EQ changes may look subtle
- If you run without `--eq-style prism`, the script stays on `legacy` EQ.
- Very low `--eq-color-drive` (for prism) can make color motion hard to notice.
- If you keep reusing the same output file, compare with a separate `--outdir` to avoid confusing old/new renders.
- Compare outputs with separate folders:
```bash
python make_player_apple.py --title "Midnight satellite" --eq-style legacy --outdir output/eq_legacy
python make_player_apple.py --title "Midnight satellite" --eq-style prism --eq-color-drive 0.75 --eq-intensity punchy --outdir output/eq_prism
```
- In console logs, confirm:
  - `EQ          : ... prism ... drive=...`
  - `Analysis    : need=True reasons=prism_pulse`

End CTA Frame:
```bash
python make_player_apple.py \
  --title "Midnight satellite" \
  --end-cta on \
  --end-cta-style fullscreen
```

End CTA off:
```bash
python make_player_apple.py \
  --title "Midnight satellite" \
  --end-cta off
```

Custom CTA text/duration:
```bash
python make_player_apple.py \
  --title "Midnight satellite" \
  --end-cta-text "Like & Subscribe" \
  --end-cta-duration 6
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
- `--eq-style`, `--eq-intensity`, `--eq-quality`
- `--eq-peak-hold`, `--eq-glow`, `--eq-opacity`, `--eq-color-drive`
- `--end-cta`, `--end-cta-text`, `--end-cta-duration`, `--end-cta-style`
- `--color-preset`, `--lut`, `--lut-intensity`
- `--skip-thumbnail`

Standalone thumbnail script
```bash
python make_thumbnail_apple.py \
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
