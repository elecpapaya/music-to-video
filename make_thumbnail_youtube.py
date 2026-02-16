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

from font_resolver import FontResolutionError, resolve_pillow_font_paths
from text_layout import fit_title_block, truncate_with_ellipsis


THUMB_W = 1280
THUMB_H = 720
COVER_RADIUS = 22
OUTER_RADIUS_DELTA = 6
GLASS_FILL_ALPHA = 56
GLASS_OUTLINE_ALPHA = 22
GLASS_BLUR_RADIUS = 10
PLAY_GLOW_ALPHA = 70
PLAY_GLOW_BLUR_RADIUS = 18
PLAY_GLOW_RADIUS_ADD = 20
BADGE_TAIL_SAT_SCALE = 0.82
BADGE_TAIL_LIGHT_SCALE = 0.95

# Safe text area for desktop/mobile overlays.
SAFE_LEFT = 64
SAFE_TOP = 48
SAFE_RIGHT = 640
SAFE_BOTTOM = 640

# YouTube timestamp area (bottom-right): keep text away.
TS_BLOCK_X = THUMB_W - 240
TS_BLOCK_Y = THUMB_H - 120


class UserFacingError(RuntimeError):
    pass


def raise_user_error(code: str, message: str, details: str = "", hint: str = "") -> None:
    parts = [f"{code}: {message}"]
    if details:
        parts.append(f"Details: {details}")
    if hint:
        parts.append(f"Hint: {hint}")
    raise UserFacingError("\n".join(parts))


def sanitize_filename(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    if not s:
        return "thumbnail"
    s = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", s)
    return s[:120]


def _load_font(font_path: str | None, size: int) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            return ImageFont.load_default()
    return ImageFont.load_default()


def _rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
    return mask


def _apply_rounded_alpha(img_rgba: Image.Image, radius: int) -> Image.Image:
    out = img_rgba.convert("RGBA")
    out.putalpha(_rounded_mask(out.size, radius))
    return out


def _desaturate_rgb(rgb: tuple[int, int, int], sat_scale: float = 0.88, light_scale: float = 0.97) -> tuple[int, int, int]:
    r, g, b = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = max(0.0, min(1.0, s * sat_scale))
    l = max(0.0, min(1.0, l * light_scale))
    rr, gg, bb = colorsys.hls_to_rgb(h, l, s)
    return (int(round(rr * 255)), int(round(gg * 255)), int(round(bb * 255)))


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
        bbox = draw.textbbox((x, cy), line if line else "Ag", font=font, stroke_width=stroke_width)
        cy = bbox[3] + line_gap
    return cy


def _fit_badge_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, title_font_path: str | None) -> tuple[ImageFont.ImageFont, int, int]:
    badge_pad_x = 18
    badge_pad_y = 9
    for size in range(28, 15, -1):
        font = _load_font(title_font_path, size)
        bb = draw.textbbox((0, 0), text, font=font)
        bw = (bb[2] - bb[0]) + badge_pad_x * 2
        bh = (bb[3] - bb[1]) + badge_pad_y * 2
        if bw <= max_width:
            return font, bw, bh
    font = _load_font(title_font_path, 16)
    bb = draw.textbbox((0, 0), text, font=font)
    bw = min(max_width, (bb[2] - bb[0]) + badge_pad_x * 2)
    bh = (bb[3] - bb[1]) + badge_pad_y * 2
    return font, bw, bh


def make_thumbnail(
    *,
    cover_path: Path,
    title: str,
    artist: str,
    tagline: str,
    genre: str,
    out_path: Path,
    title_max_lines: int = 3,
    title_min_font: int = 56,
    title_max_font: int = 102,
    font_title_path: str | None = None,
    font_body_path: str | None = None,
) -> None:
    cover = Image.open(cover_path).convert("RGB")
    accent, deep = cover_palette(cover)
    tail_accent = _desaturate_rgb(accent, sat_scale=BADGE_TAIL_SAT_SCALE, light_scale=BADGE_TAIL_LIGHT_SCALE)
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
    cover_rgba = _apply_rounded_alpha(cover_box.convert("RGBA"), COVER_RADIUS)
    shadow = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    sdraw.rounded_rectangle(
        (card_x - 14, card_y - 14, card_x + cover_size + 14, card_y + cover_size + 14),
        radius=COVER_RADIUS + 8,
        fill=(0, 0, 0, 196),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=14))
    canvas = Image.alpha_composite(canvas, shadow)
    canvas.alpha_composite(cover_rgba, (card_x, card_y))

    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (card_x - 2, card_y - 2, card_x + cover_size + 2, card_y + cover_size + 2),
        radius=COVER_RADIUS,
        outline=(255, 255, 255, 122),
        width=2,
    )
    draw.rounded_rectangle(
        (card_x - 14, card_y - 14, card_x + cover_size + 14, card_y + cover_size + 14),
        radius=COVER_RADIUS + OUTER_RADIUS_DELTA,
        outline=(*accent, 124),
        width=2,
    )

    badge_text = (genre or "").strip().upper() or "OUT NOW"
    badge_font, bw, bh = _fit_badge_font(draw, badge_text, SAFE_RIGHT - SAFE_LEFT, font_title_path)
    badge_pad_x = 18
    badge_pad_y = 9
    bx = SAFE_LEFT
    by = max(SAFE_TOP + 8, 58)
    draw.rounded_rectangle((bx, by, bx + bw, by + bh), radius=16, fill=(15, 18, 26, 198), outline=(*accent, 225), width=2)
    draw.polygon([(bx + bw, by + 10), (bx + bw + 22, by + bh // 2), (bx + bw, by + bh - 10)], fill=(*tail_accent, 230))
    draw.text((bx + badge_pad_x, by + badge_pad_y - 1), badge_text, font=badge_font, fill=(255, 255, 255, 246))

    title_area_x = SAFE_LEFT
    title_area_w = SAFE_RIGHT - SAFE_LEFT
    title_y = max(168, SAFE_TOP + 112)
    title_area_h = max(160, SAFE_BOTTOM - title_y - 150)
    line_gap = 9
    title_size, wrapped, _ = fit_title_block(
        draw=draw,
        text=title,
        font_loader=lambda sz: _load_font(font_title_path, sz),
        min_font=title_min_font,
        max_font=title_max_font,
        max_width=title_area_w,
        max_height=title_area_h,
        max_lines=title_max_lines,
        line_gap=line_gap,
        stroke_width=4,
    )
    title_font = _load_font(font_title_path, title_size)

    title_probe_y = title_y
    for line in wrapped:
        bb = draw.textbbox((title_area_x, title_probe_y), line if line else "Ag", font=title_font, stroke_width=4)
        title_probe_y = bb[3] + line_gap
    title_block_bottom = title_probe_y
    artist_font = _load_font(font_body_path, 44)
    artist_text = truncate_with_ellipsis(draw, artist, artist_font, title_area_w - 14)
    artist_y = min(title_block_bottom + 14, SAFE_BOTTOM - (90 if tagline.strip() else 50))

    text_bottom = artist_y + 50
    if tagline.strip():
        body_font_probe = _load_font(font_body_path, 30)
        tagline_probe = truncate_with_ellipsis(draw, tagline.strip(), body_font_probe, title_area_w)
        tagline_y_probe = artist_y + 64
        if tagline_y_probe <= SAFE_BOTTOM - 22 and title_area_x < TS_BLOCK_X:
            tbb = draw.textbbox((title_area_x, tagline_y_probe), tagline_probe if tagline_probe else "Ag", font=body_font_probe)
            text_bottom = max(text_bottom, tbb[3])

    gx0 = SAFE_LEFT - 12
    gx1 = SAFE_RIGHT + 10
    gy0 = max(SAFE_TOP + 6, title_y - 18)
    gy1 = min(SAFE_BOTTOM + 16, text_bottom + 18)
    if gy1 > gy0:
        glass = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
        gldraw = ImageDraw.Draw(glass)
        gldraw.rounded_rectangle(
            (gx0, gy0, gx1, gy1),
            radius=20,
            fill=(0, 0, 0, GLASS_FILL_ALPHA),
            outline=(255, 255, 255, GLASS_OUTLINE_ALPHA),
            width=1,
        )
        glass = glass.filter(ImageFilter.GaussianBlur(radius=GLASS_BLUR_RADIUS))
        canvas = Image.alpha_composite(canvas, glass)
        draw = ImageDraw.Draw(canvas)

    draw_multiline_with_stroke(
        draw=draw,
        lines=wrapped,
        x=title_area_x + 3,
        y=title_y + 4,
        font=title_font,
        fill=(*accent, 162),
        stroke_fill=(0, 0, 0, 0),
        stroke_width=0,
        line_gap=line_gap,
    )
    next_y = draw_multiline_with_stroke(
        draw=draw,
        lines=wrapped,
        x=title_area_x,
        y=title_y,
        font=title_font,
        fill=(255, 255, 255, 255),
        stroke_fill=(0, 0, 0, 176),
        stroke_width=4,
        line_gap=line_gap,
    )

    artist_y = min(next_y + 14, SAFE_BOTTOM - (90 if tagline.strip() else 50))
    draw.rounded_rectangle(
        (title_area_x - 2, artist_y - 8, title_area_x + draw.textlength(artist_text, font=artist_font) + 20, artist_y + 50),
        radius=13,
        fill=(0, 0, 0, 132),
    )
    draw.text((title_area_x + 8, artist_y), artist_text, font=artist_font, fill=(255, 255, 255, 236))

    if tagline.strip():
        body_font = _load_font(font_body_path, 30)
        tagline_text = truncate_with_ellipsis(draw, tagline.strip(), body_font, title_area_w)
        tagline_y = artist_y + 64
        if tagline_y <= SAFE_BOTTOM - 22 and title_area_x < TS_BLOCK_X:
            draw.text((title_area_x, tagline_y), tagline_text, font=body_font, fill=(226, 226, 226, 215))

    play_cx, play_cy = 1166, 648
    play_r = 39
    play_glow = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    pgdraw = ImageDraw.Draw(play_glow)
    glow_r = play_r + PLAY_GLOW_RADIUS_ADD
    pgdraw.ellipse((play_cx - glow_r, play_cy - glow_r, play_cx + glow_r, play_cy + glow_r), fill=(*accent, PLAY_GLOW_ALPHA))
    play_glow = play_glow.filter(ImageFilter.GaussianBlur(radius=PLAY_GLOW_BLUR_RADIUS))
    canvas = Image.alpha_composite(canvas, play_glow)
    draw = ImageDraw.Draw(canvas)
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
    ap.add_argument("--title-max-lines", type=int, default=3)
    ap.add_argument("--title-min-font", type=int, default=56)
    ap.add_argument("--title-max-font", type=int, default=102)
    ap.add_argument("--font-title", default="", help="Optional title font path override")
    ap.add_argument("--font-body", default="", help="Optional body font path override")
    ap.add_argument("--outdir", default="output", help="Output directory")
    args = ap.parse_args()

    try:
        if args.title_max_lines < 1:
            raise_user_error("INVALID_ARG", "--title-max-lines must be >= 1.")
        if args.title_min_font < 8 or args.title_max_font < 8:
            raise_user_error("INVALID_ARG", "--title-min-font and --title-max-font must be >= 8.")
        if args.title_min_font > args.title_max_font:
            raise_user_error(
                "INVALID_ARG",
                "--title-min-font cannot be greater than --title-max-font.",
                hint="Use values like --title-min-font 56 --title-max-font 102",
            )

        cover = Path(args.cover).expanduser().resolve()
        if not cover.exists():
            raise_user_error(
                "COVER_NOT_FOUND",
                f"Cover image was not found: {cover}",
                hint='Provide a valid file path, e.g. --cover "cover.jpg"',
            )

        try:
            pillow_fonts = resolve_pillow_font_paths(
                font_title=(args.font_title or None),
                font_body=(args.font_body or None),
            )
        except FontResolutionError as e:
            raise_user_error(
                "FONT_RESOLUTION_FAILED",
                "Could not resolve requested font path(s).",
                details=str(e),
                hint="Check --font-title/--font-body paths or remove them to use automatic fallback.",
            )

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
            title_max_lines=args.title_max_lines,
            title_min_font=args.title_min_font,
            title_max_font=args.title_max_font,
            font_title_path=pillow_fonts["title"],
            font_body_path=pillow_fonts["body"],
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

    except UserFacingError as e:
        print(f"\n[ERROR] {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
