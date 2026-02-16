#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a click-oriented YouTube thumbnail (1280x720) from a cover image.

Usage:
  python make_thumbnail_youtube.py --cover "cover.jpg" --title "Midnight whispers" --artist "Susan" --genre "K-POP"
"""

from __future__ import annotations

import argparse
import colorsys
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageStat


THUMB_W = 1280
THUMB_H = 720


def sanitize_filename(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    if not s:
        return "thumbnail"
    s = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", s)
    return s[:120]


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


def cover_palette(cover: Image.Image) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    small = cover.convert("RGB").resize((120, 120), Image.Resampling.LANCZOS)
    stat = ImageStat.Stat(small)
    mr, mg, mb = (v / 255.0 for v in stat.mean[:3])
    h, l, s = colorsys.rgb_to_hls(mr, mg, mb)

    accent = colorsys.hls_to_rgb(h, min(0.64, max(0.46, l + 0.08)), min(0.9, max(0.45, s * 1.25)))
    deep = colorsys.hls_to_rgb(h, min(0.18, max(0.07, l * 0.3)), min(0.6, max(0.18, s * 0.6)))
    accent_rgb = tuple(int(round(c * 255)) for c in accent)
    deep_rgb = tuple(int(round(c * 255)) for c in deep)
    return accent_rgb, deep_rgb


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


def make_thumbnail(cover_path: Path, title: str, artist: str, tagline: str, genre: str, out_path: Path) -> None:
    cover = Image.open(cover_path).convert("RGB")
    accent, deep = cover_palette(cover)
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

    # Graphic accents for stronger thumbnail points.
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

    # Premium play cue: outlined ring + clean triangle
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cover", required=True, help="Input cover image path")
    ap.add_argument("--title", default="", help="Main text title")
    ap.add_argument("--artist", default="Artist", help="Artist text")
    ap.add_argument("--tagline", default="", help="Optional supporting text (leave blank to hide)")
    ap.add_argument("--genre", default="", help="Optional badge text, e.g. K-POP, R&B, LO-FI")
    ap.add_argument("--outdir", default="output", help="Output directory")
    args = ap.parse_args()

    cover = Path(args.cover).expanduser().resolve()
    if not cover.exists():
        raise FileNotFoundError(f"Cover not found: {cover}")

    title = args.title or cover.stem
    safe_title = sanitize_filename(title)
    outdir = Path(args.outdir).expanduser().resolve()
    out_path = outdir / f"{safe_title}_youtube_thumb.jpg"

    make_thumbnail(
        cover_path=cover,
        title=title,
        artist=args.artist,
        tagline=args.tagline,
        genre=args.genre,
        out_path=out_path,
    )

    print("============================================")
    print("YouTube Thumbnail Generator")
    print(f"Cover  : {cover}")
    print(f"Title  : {title}")
    print(f"Artist : {args.artist}")
    print(f"Genre  : {args.genre or 'OUT NOW'}")
    print(f"Tagline: {args.tagline}")
    print(f"Out    : {out_path}")
    print("============================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
