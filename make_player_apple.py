#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apple Music style minimal player video generator (v4).

Fixes vs v3:
- Avoids ffmpeg '@file' parsing edge-cases on Windows by passing filtergraph directly
  (subprocess list args, no shell escaping).
- Still writes output/<title>_filter.txt for inspection, but ffmpeg uses the in-memory graph.

Features:
- elapsed / total time (mm:ss / mm:ss)
- ultra-thin waveform
- auto accent color from cover (average color via Pillow)

Usage:
  python make_player_apple.py --song "song.wav" --cover "cover.jpg" --title "Title" --artist "Artist"
"""

from __future__ import annotations

import argparse
import colorsys
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageStat


@dataclass
class FFPaths:
    ffmpeg: str
    ffprobe: str


@dataclass
class ColorScheme:
    accent_hex: str
    mood_bg_hex: str
    mood_tint_hex: str
    point_hex: str


QUALITY_PRESETS = {
    "standard": {
        "crf": "21",
        "preset": "medium",
        "maxrate": "10M",
        "bufsize": "20M",
        "audio_bitrate": "224k",
        "fps": "30",
    },
    "high": {
        "crf": "18",
        "preset": "slow",
        "maxrate": "16M",
        "bufsize": "32M",
        "audio_bitrate": "256k",
        "fps": "60",
    },
    "ultra": {
        "crf": "16",
        "preset": "slow",
        "maxrate": "20M",
        "bufsize": "40M",
        "audio_bitrate": "320k",
        "fps": "60",
    },
}

THUMB_W = 1280
THUMB_H = 720
AUDIO_EXTS = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"]
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


def which_or_raise(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise FileNotFoundError(f"{name} not found in PATH. Install FFmpeg and ensure '{name}' is available.")
    return p


def run_capture(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n\nSTDERR:\n{proc.stderr}")
    return proc.stdout.strip()


def ff_paths() -> FFPaths:
    return FFPaths(ffmpeg=which_or_raise("ffmpeg"), ffprobe=which_or_raise("ffprobe"))


def get_duration_seconds(ffp: FFPaths, audio_path: Path) -> float:
    out = run_capture([
        ffp.ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ])
    return float(out)


def format_mmss(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total//60:02d}:{total%60:02d}"


def sanitize_filename(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    if not s:
        return "output"
    s = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", s)
    return s[:120]


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _rgb_float_to_hex(r: float, g: float, b: float) -> str:
    ri = int(round(_clamp01(r) * 255))
    gi = int(round(_clamp01(g) * 255))
    bi = int(round(_clamp01(b) * 255))
    return f"{ri:02X}{gi:02X}{bi:02X}"


def cover_color_scheme(image_path: Path) -> ColorScheme:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((80, 80))
    stat = ImageStat.Stat(img)
    mr, mg, mb = (float(x) / 255.0 for x in stat.mean[:3])
    h, l, s = colorsys.rgb_to_hls(mr, mg, mb)

    # Keep UI accent readable while deriving all tones from the cover image.
    accent_l = min(0.66, max(0.42, l * 0.95 + 0.08))
    accent_s = min(0.88, max(0.34, s * 1.28))
    accent_hex = _rgb_float_to_hex(*colorsys.hls_to_rgb(h, accent_l, accent_s))

    # Deep base for padded background area.
    mood_bg_l = min(0.18, max(0.07, l * 0.28))
    mood_bg_s = min(0.55, max(0.16, s * 0.58))
    mood_bg_hex = _rgb_float_to_hex(*colorsys.hls_to_rgb(h, mood_bg_l, mood_bg_s))

    # Slightly brighter tint to harmonize the blur layer.
    tint_h = (h + 0.02) % 1.0
    tint_l = min(0.32, max(0.16, l * 0.46 + 0.04))
    tint_s = min(0.84, max(0.30, s * 1.10 + 0.05))
    mood_tint_hex = _rgb_float_to_hex(*colorsys.hls_to_rgb(tint_h, tint_l, tint_s))

    # Brighter point color for visible background accents on dark scenes.
    point_l = min(0.78, max(0.62, accent_l + 0.16))
    point_s = min(0.92, max(0.48, accent_s * 1.05))
    point_hex = _rgb_float_to_hex(*colorsys.hls_to_rgb(h, point_l, point_s))

    return ColorScheme(accent_hex=accent_hex, mood_bg_hex=mood_bg_hex, mood_tint_hex=mood_tint_hex, point_hex=point_hex)


def ffmpeg_escape_drawtext_text(s: str) -> str:
    s = s.replace("\\", "\\\\")
    s = s.replace(":", r"\:")
    s = s.replace("'", r"\'")
    s = s.replace("%", r"\%")
    s = s.replace("\r", " ").replace("\n", " ")
    return s


def resolve_media_path(
    explicit_value: str | None,
    *,
    title_hint: str,
    exts: list[str],
    kind: str,
) -> Path:
    if explicit_value:
        raw = Path(explicit_value).expanduser()
        if raw.exists():
            return raw.resolve()
        if raw.suffix == "":
            for ext in exts:
                cand = raw.with_suffix(ext)
                if cand.exists():
                    return cand.resolve()
        tried = ", ".join(str(raw.with_suffix(ext)) for ext in exts) if raw.suffix == "" else str(raw)
        raise FileNotFoundError(f"{kind} not found: {raw} (tried: {tried})")

    hint = (title_hint or "").strip()
    if not hint:
        raise FileNotFoundError(f"{kind} is not specified and cannot be inferred without --title.")
    base = Path(hint).stem if Path(hint).suffix else hint
    for ext in exts:
        cand = Path(base + ext)
        if cand.exists():
            return cand.resolve()
    tried = ", ".join(base + ext for ext in exts)
    raise FileNotFoundError(f"{kind} not found for title '{base}'. tried: {tried}")


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    raw = (hex_color or "").strip().lstrip("#")
    if len(raw) != 6:
        return (255, 255, 255)
    try:
        return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))
    except ValueError:
        return (255, 255, 255)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if bold:
        candidates = [
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        ]

    for p in candidates:
        path = Path(p)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_cover_to_canvas(img: Image.Image, w: int, h: int) -> Image.Image:
    src_w, src_h = img.size
    src_ratio = src_w / src_h
    dst_ratio = w / h

    if src_ratio > dst_ratio:
        new_h = h
        new_w = int(h * src_ratio)
    else:
        new_w = w
        new_h = int(w / src_ratio)

    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x = (new_w - w) // 2
    y = (new_h - h) // 2
    return resized.crop((x, y, x + w, y + h))


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int, max_lines: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for w in words[1:]:
        trial = f"{current} {w}"
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = w
    lines.append(current)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if not lines[-1].endswith("..."):
            while draw.textlength(lines[-1] + "...", font=font) > max_width and len(lines[-1]) > 1:
                lines[-1] = lines[-1][:-1]
            lines[-1] += "..."
    return lines


def draw_multiline_with_stroke(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    stroke_fill: tuple[int, int, int, int],
    stroke_width: int,
    line_gap: int,
) -> int:
    cy = y
    for line in lines:
        draw.text((x, cy), line, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
        bbox = draw.textbbox((x, cy), line, font=font, stroke_width=stroke_width)
        cy = bbox[3] + line_gap
    return cy


def make_youtube_thumbnail(
    *,
    cover_path: Path,
    title: str,
    artist: str,
    genre: str,
    tagline: str,
    accent_hex: str,
    mood_bg_hex: str,
    out_path: Path,
) -> None:
    cover = Image.open(cover_path).convert("RGB")
    accent = _hex_to_rgb(accent_hex)
    deep = _hex_to_rgb(mood_bg_hex)
    cover_size = 548
    card_x = 700
    card_y = 86

    bg = fit_cover_to_canvas(cover, THUMB_W, THUMB_H)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=34))
    bg = ImageEnhance.Brightness(bg).enhance(0.46)
    bg = ImageEnhance.Contrast(bg).enhance(1.12)
    bg = ImageEnhance.Color(bg).enhance(0.86)
    canvas = bg.convert("RGBA")

    overlay = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    for x in range(THUMB_W):
        t = x / (THUMB_W - 1)
        alpha = int(225 * ((1.0 - t) ** 1.55))
        odraw.line([(x, 0), (x, THUMB_H)], fill=(0, 0, 0, alpha), width=1)
    odraw.rectangle((0, 0, THUMB_W, 145), fill=(*deep, 116))
    odraw.rectangle((0, THUMB_H - 170, THUMB_W, THUMB_H), fill=(0, 0, 0, 168))

    glow = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    gdraw.ellipse((730, 30, 1260, 560), fill=(*accent, 88))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=62))
    overlay = Image.alpha_composite(overlay, glow)
    odraw.rectangle((0, 0, THUMB_W, THUMB_H), outline=(*accent, 26), width=2)
    canvas = Image.alpha_composite(canvas, overlay)

    accents = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    adraw = ImageDraw.Draw(accents)
    adraw.polygon([(22, 128), (44, 128), (24, 690), (2, 690)], fill=(*accent, 150))
    adraw.polygon([(52, 178), (64, 178), (48, 690), (36, 690)], fill=(*accent, 95))
    adraw.polygon([(612, 520), (650, 532), (576, 720), (538, 708)], fill=(*accent, 112))
    adraw.polygon([(1240, 0), (1280, 0), (1280, 240), (1260, 220)], fill=(*accent, 85))
    accents = accents.filter(ImageFilter.GaussianBlur(radius=1.2))
    canvas = Image.alpha_composite(canvas, accents)

    cover_box = fit_cover_to_canvas(cover, cover_size, cover_size)
    cover_rgba = cover_box.convert("RGBA")
    shadow = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    sdraw.rectangle((card_x - 14, card_y - 14, card_x + cover_size + 14, card_y + cover_size + 14), fill=(0, 0, 0, 196))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=14))
    canvas = Image.alpha_composite(canvas, shadow)
    canvas.alpha_composite(cover_rgba, (card_x, card_y))

    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (card_x - 2, card_y - 2, card_x + cover_size + 2, card_y + cover_size + 2),
        radius=20,
        outline=(255, 255, 255, 122),
        width=2,
    )
    draw.rounded_rectangle(
        (card_x - 14, card_y - 14, card_x + cover_size + 14, card_y + cover_size + 14),
        radius=26,
        outline=(*accent, 124),
        width=2,
    )

    badge_font = load_font(28, bold=True)
    body_font = load_font(30, bold=False)
    badge_text = (genre or "").strip().upper() or "OUT NOW"
    badge_pad_x = 18
    badge_pad_y = 9
    bb = draw.textbbox((0, 0), badge_text, font=badge_font)
    bw = (bb[2] - bb[0]) + badge_pad_x * 2
    bh = (bb[3] - bb[1]) + badge_pad_y * 2
    bx, by = 62, 58
    draw.rounded_rectangle((bx, by, bx + bw, by + bh), radius=16, fill=(15, 18, 26, 198), outline=(*accent, 225), width=2)
    draw.polygon([(bx + bw, by + 10), (bx + bw + 22, by + bh // 2), (bx + bw, by + bh - 10)], fill=(*accent, 230))
    draw.text((bx + badge_pad_x, by + badge_pad_y - 1), badge_text, font=badge_font, fill=(255, 255, 255, 246))

    title_area_x = 64
    title_area_w = 620
    title_font_size = 102
    while title_font_size >= 58:
        tfont = load_font(title_font_size, bold=True)
        wrapped = wrap_text(draw, title, tfont, title_area_w, max_lines=3)
        total_h = 0
        for line in wrapped:
            tb = draw.textbbox((0, 0), line, font=tfont, stroke_width=4)
            total_h += (tb[3] - tb[1]) + 9
        if total_h <= 318:
            break
        title_font_size -= 4

    title_y = 168
    draw_multiline_with_stroke(
        draw=draw,
        lines=wrapped,
        x=title_area_x + 5,
        y=title_y + 6,
        font=tfont,
        fill=(*accent, 155),
        stroke_fill=(0, 0, 0, 0),
        stroke_width=0,
        line_gap=9,
    )
    next_y = draw_multiline_with_stroke(
        draw=draw,
        lines=wrapped,
        x=title_area_x,
        y=title_y,
        font=tfont,
        fill=(255, 255, 255, 255),
        stroke_fill=(0, 0, 0, 176),
        stroke_width=4,
        line_gap=9,
    )

    artist_font = load_font(44, bold=True)
    artist_box = draw.textbbox((0, 0), artist, font=artist_font)
    aw = artist_box[2] - artist_box[0]
    ax = title_area_x
    ay = next_y + 14
    draw.rounded_rectangle((ax - 2, ay - 8, ax + aw + 20, ay + 50), radius=13, fill=(0, 0, 0, 132))
    draw.text((ax + 8, ay), artist, font=artist_font, fill=(255, 255, 255, 236))
    if tagline.strip():
        draw.text((title_area_x, ay + 64), tagline, font=body_font, fill=(226, 226, 226, 215))

    play_cx, play_cy = 1166, 648
    play_r = 39
    draw.ellipse(
        (play_cx - play_r, play_cy - play_r, play_cx + play_r, play_cy + play_r),
        fill=(0, 0, 0, 124),
        outline=(255, 255, 255, 228),
        width=3,
    )
    draw.polygon([(play_cx - 9, play_cy - 14), (play_cx - 9, play_cy + 14), (play_cx + 16, play_cy)], fill=(*accent, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path, quality=95, subsampling=0, optimize=True)


def build_filter_graph(
    *,
    title: str,
    artist: str,
    accent_hex: str,
    mood_bg_hex: str,
    mood_tint_hex: str,
    point_hex: str,
    total_mmss: str,
    dur: float,
    style: str = "youtube",
) -> str:
    title_e = ffmpeg_escape_drawtext_text(title)
    artist_e = ffmpeg_escape_drawtext_text(artist)
    total_e = ffmpeg_escape_drawtext_text(total_mmss)
    elapsed_expr = r"%{eif\:t/60\:d\:2}\:%{eif\:mod(t\,60)\:d\:2}"

    if style == "youtube":
        cover_x, cover_y, cover_size = 150, 170, 620
        info_x, info_y = 840, 140
        info_w, info_h = 980, 800
        title_y = 255
        title_fs = 70
        artist_y = 395
        artist_fs = 38
        bar_x, bar_y, bar_w = 875, 785, 900
        time_y = 820
    else:
        cover_x, cover_y, cover_size = 220, 190, 560
        info_x, info_y = 860, 170
        info_w, info_h = 920, 740
        title_y = 275
        title_fs = 62
        artist_y = 412
        artist_fs = 34
        bar_x, bar_y, bar_w = 895, 765, 840
        time_y = 800

    eq_x = info_x + 32
    eq_y = artist_y + 125
    eq_w = info_w - 64
    eq_h = 96
    controls_y = bar_y + 58
    prev_x = bar_x + bar_w // 2 - 120
    play_x = bar_x + bar_w // 2 - 20
    next_x = bar_x + bar_w // 2 + 78

    chains = [
        # Keep cover intact: background uses padded full frame (no crop slicing)
        "[0:v]split=2[cvsrc][bgsrc]",
        f"[bgsrc]scale=1920:1080:force_original_aspect_ratio=decrease:flags=lanczos,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x{mood_bg_hex},format=rgba,split=2[bg_base][bg_tintsrc]",
        "[bg_base]gblur=sigma=34,eq=brightness=-0.50:contrast=1.10:saturation=0.92[bg_blur]",
        "[bg_tintsrc]gblur=sigma=58,eq=brightness=-0.30:contrast=1.02:saturation=1.25,colorchannelmixer=aa=0.22[bg_tint0]",
        f"[bg_tint0]drawbox=x=0:y=0:w=1920:h=1080:color=0x{mood_tint_hex}@0.16:t=fill[bg_tint]",
        "[bg_blur][bg_tint]overlay=0:0:format=auto[vbg]",
        "[vbg]vignette=PI/4:0.72[bg_v]",
        f"[bg_v]drawbox=x=0:y=0:w=1920:h=1080:color=0x{mood_tint_hex}@0.06:t=fill[bg_a]",
        f"[bg_a]drawbox=x=0:y=0:w=1920:h=260:color=0x{mood_bg_hex}@0.45:t=fill[bg_b]",
        f"[bg_b]drawbox=x=0:y=820:w=1920:h=260:color=0x{mood_bg_hex}@0.40:t=fill[bg_c]",
        f"color=c=black@0.0:s=1920x1080,format=rgba,drawbox=x=24:y=32:w=320:h=180:color=0x{point_hex}@0.42:t=fill,drawbox=x=30:y=900:w=520:h=140:color=0x{point_hex}@0.34:t=fill[bg_point0]",
        "[bg_point0]gblur=sigma=38[bg_point]",
        "[bg_c][bg_point]overlay=0:0:format=auto[bg_cp]",
        f"[bg_cp]drawbox=x={cover_x-24}:y={cover_y-24}:w={cover_size+48}:h={cover_size+48}:color=black@0.48:t=fill[plate]",
        "[plate]boxblur=12:2[plate_b]",
        f"[cvsrc]scale={cover_size}:{cover_size}:force_original_aspect_ratio=decrease:flags=lanczos,pad={cover_size}:{cover_size}:(ow-iw)/2:(oh-ih)/2,format=rgba[cv]",
        f"[plate_b][cv]overlay={cover_x}:{cover_y}:format=auto[base0]",
        # Info panel
        f"[base0]drawbox=x={info_x}:y={info_y}:w={info_w}:h={info_h}:color=black@0.32:t=fill[base3]",
        f"[base3]drawbox=x={info_x}:y={info_y}:w={info_w}:h=3:color=0x{accent_hex}@0.86:t=fill[base4]",
        f"[base4]drawtext=fontfile='C\\:/Windows/Fonts/segoeuib.ttf':text='NOW PLAYING':x={info_x+32}:y={info_y+28}:fontsize=25:fontcolor=white@0.62[base5]",
        f"[base5]drawtext=fontfile='C\\:/Windows/Fonts/segoeuib.ttf':text='{title_e}':x={info_x+32}:y={title_y}:fontsize={title_fs}:fontcolor=white:borderw=2:bordercolor=black@0.28[base6]",
        f"[base6]drawtext=fontfile='C\\:/Windows/Fonts/segoeui.ttf':text='{artist_e}':x={info_x+32}:y={artist_y}:fontsize={artist_fs}:fontcolor=white@0.82[base7]",
        f"[base7]drawbox=x={info_x+32}:y={artist_y+86}:w=220:h=4:color=0x{accent_hex}@0.95:t=fill[base8]",
        f"[base8]drawtext=fontfile='C\\:/Windows/Fonts/segoeui.ttf':text='EQUALIZER':x={eq_x}:y={eq_y-32}:fontsize=20:fontcolor=white@0.46[base8a]",
        f"[base8a]drawbox=x={eq_x}:y={eq_y-8}:w={eq_w}:h=1:color=white@0.18:t=fill[base8b]",
        f"[base8b]drawbox=x={eq_x}:y={eq_y}:w={eq_w}:h={eq_h}:color=white@0.05:t=fill[base8c]",
        f"[1:a]showfreqs=s={eq_w}x{eq_h}:mode=bar:fscale=log:ascale=sqrt:win_func=hann:colors=white,format=rgba,colorkey=0x000000:0.18:0.0,split=2[eq_raw][eq_top]",
        "[eq_raw]gblur=sigma=2.2:steps=1,colorchannelmixer=aa=0.30[eq_glow]",
        f"[base8c][eq_glow]overlay={eq_x}:{eq_y}:format=auto[base9]",
        f"[base9][eq_top]overlay={eq_x}:{eq_y}:format=auto[base10]",
        f"[base10]drawbox=x={bar_x}:y={bar_y}:w={bar_w}:h=5:color=white@0.20:t=fill[bar_bg]",
        f"[bar_bg]drawbox=x={bar_x}:y={bar_y}:w={bar_w}:h=5:color=white@0.93:t=fill[bar_fill_full]",
        f"color=c=black@0.82:s={bar_w}x5[bar_mask]",
        f"[bar_fill_full][bar_mask]overlay=x='{bar_x}+{bar_w}*min(t\\,{dur:.6f})/{dur:.6f}':y={bar_y}:eval=frame[bar_move]",
        f"color=c=0x{accent_hex}@0.32:s=24x32[bar_knob_glow_src]",
        "[bar_knob_glow_src]gblur=sigma=2.2:steps=1[bar_knob_glow]",
        f"[bar_move][bar_knob_glow]overlay=x='{bar_x}+{bar_w}*min(t\\,{dur:.6f})/{dur:.6f}-12':y={bar_y-13}:eval=frame[bar_move2]",
        "color=c=white@0.90:s=14x22[bar_knob_body]",
        f"[bar_move2][bar_knob_body]overlay=x='{bar_x}+{bar_w}*min(t\\,{dur:.6f})/{dur:.6f}-7':y={bar_y-9}:eval=frame[bar_move3]",
        f"color=c=0x{accent_hex}@0.82:s=4x14[bar_knob_core]",
        f"[bar_move3][bar_knob_core]overlay=x='{bar_x}+{bar_w}*min(t\\,{dur:.6f})/{dur:.6f}-2':y={bar_y-5}:eval=frame[base13]",
        f"[base13]drawtext=fontfile='C\\:/Windows/Fonts/segoeui.ttf':text='{elapsed_expr}':x={bar_x}:y={time_y}:fontsize=24:fontcolor=white@0.70[base14]",
        f"[base14]drawtext=fontfile='C\\:/Windows/Fonts/segoeui.ttf':text='{total_e}':x='{bar_x}+{bar_w}-tw':y={time_y}:fontsize=24:fontcolor=white@0.70[base15]",
        f"[base15]drawtext=fontfile='C\\:/Windows/Fonts/segoeuib.ttf':text='<<':x={prev_x}:y={controls_y}:fontsize=34:fontcolor=white@0.58[base16]",
        f"[base16]drawtext=fontfile='C\\:/Windows/Fonts/segoeuib.ttf':text='PLAY':x={play_x}:y={controls_y+2}:fontsize=28:fontcolor=white@0.92[base17]",
        f"[base17]drawtext=fontfile='C\\:/Windows/Fonts/segoeuib.ttf':text='>>':x={next_x}:y={controls_y}:fontsize=34:fontcolor=white@0.58[vout]",
    ]

    return ";".join(chains)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", default="", help="Audio path. If omitted, inferred from --title + audio extension.")
    ap.add_argument("--cover", default="", help="Cover path. If omitted, inferred from --title + image extension.")
    ap.add_argument("--title", default="", help="Track title. Also used for auto file lookup when --song/--cover omitted.")
    ap.add_argument("--artist", default="Artist")
    ap.add_argument("--genre", default="", help="Thumbnail badge text (e.g. K-POP)")
    ap.add_argument("--tagline", default="", help="Optional thumbnail supporting text")
    ap.add_argument("--skip-thumbnail", action="store_true", help="Skip thumbnail generation")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--style", choices=("youtube", "classic"), default="youtube",
                    help="Player visual style: youtube (default) or classic")
    ap.add_argument("--quality", choices=tuple(QUALITY_PRESETS), default="high")
    args = ap.parse_args()

    title_input = (args.title or "").strip()
    title_hint = title_input or (Path(args.song).stem if (args.song or "").strip() else "")
    song = resolve_media_path(args.song or None, title_hint=title_hint, exts=AUDIO_EXTS, kind="Audio")
    cover = resolve_media_path(args.cover or None, title_hint=title_hint or song.stem, exts=IMAGE_EXTS, kind="Cover")

    title = title_input or song.stem
    artist = args.artist or "Artist"

    ffp = ff_paths()
    dur = get_duration_seconds(ffp, song)
    total = format_mmss(dur)
    scheme = cover_color_scheme(cover)
    accent = scheme.accent_hex

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    safe_title = sanitize_filename(title)
    out_path = outdir / f"{safe_title}.mp4"
    thumb_path = outdir / f"{safe_title}_youtube_thumb.jpg"
    graph_path = outdir / f"{safe_title}_filter.txt"
    quality = QUALITY_PRESETS[args.quality]

    graph = build_filter_graph(
        title=title,
        artist=artist,
        accent_hex=accent,
        mood_bg_hex=scheme.mood_bg_hex,
        mood_tint_hex=scheme.mood_tint_hex,
        point_hex=scheme.point_hex,
        total_mmss=total,
        dur=dur,
        style=args.style,
    )
    graph_path.write_text(graph, encoding="utf-8")

    thumb_generated = False
    if not args.skip_thumbnail:
        try:
            make_youtube_thumbnail(
                cover_path=cover,
                title=title,
                artist=artist,
                genre=args.genre,
                tagline=args.tagline,
                accent_hex=scheme.accent_hex,
                mood_bg_hex=scheme.mood_bg_hex,
                out_path=thumb_path,
            )
            thumb_generated = True
        except Exception as e:
            print(f"[WARN] thumbnail generation failed: {e}")

    print("=====================================================")
    print("Apple Minimal (Python v12)")
    print(f"Song   : {song}")
    print(f"Cover  : {cover}")
    print(f"Title  : {title}")
    print(f"Artist : {artist}")
    print(f"Style  : {args.style}")
    print(f"Quality: {args.quality}")
    print(f"Dur(s) : {dur:.3f}   Total: {total}")
    print(f"Accent : #{accent}")
    print(f"MoodBG : #{scheme.mood_bg_hex}")
    print(f"MoodTi : #{scheme.mood_tint_hex}")
    print(f"Point  : #{scheme.point_hex}")
    print(f"Graph  : {graph_path}")
    print(f"Thumb  : {thumb_path} ({'generated' if thumb_generated else 'skipped'})")
    print(f"Out    : {out_path}")
    print("=====================================================")

    cmd = [
        ffp.ffmpeg, "-hide_banner", "-y",
        "-loop", "1", "-i", str(cover),
        "-i", str(song),
        "-filter_complex", graph,
        "-map", "[vout]", "-map", "1:a",
        "-shortest",
        "-r", quality["fps"],
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-profile:v", "high",
        "-level:v", "4.1",
        "-crf", quality["crf"],
        "-preset", quality["preset"],
        "-maxrate", quality["maxrate"],
        "-bufsize", quality["bufsize"],
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", quality["audio_bitrate"],
        "-ar", "48000",
        str(out_path),
    ]

    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print("\n[ERROR] ffmpeg failed.")
        return proc.returncode

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
