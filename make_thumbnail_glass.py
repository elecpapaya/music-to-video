#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a glass-style YouTube thumbnail (1280x720) from a cover image.
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageStat

from font_resolver import FontResolutionError, resolve_pillow_font_paths
from player_core import UserFacingError, raise_user_error, sanitize_filename
from text_layout import fit_title_block


THUMB_W = 1280
THUMB_H = 720
SAFE_LEFT = 72
SAFE_TOP = 58
SAFE_RIGHT = 700
SAFE_BOTTOM = 648


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


def _fit_cover_to_canvas(img: Image.Image, w: int, h: int) -> Image.Image:
    src_w, src_h = img.size
    src_ratio = src_w / max(1, src_h)
    dst_ratio = w / max(1, h)
    if src_ratio > dst_ratio:
        new_h = h
        new_w = int(h * src_ratio)
    else:
        new_w = w
        new_h = int(w / max(0.0001, src_ratio))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x = (new_w - w) // 2
    y = (new_h - h) // 2
    return resized.crop((x, y, x + w, y + h))


def _fit_single_line_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font_path: str | None,
    start_size: int,
    min_size: int,
    max_width: int,
) -> tuple[str, ImageFont.ImageFont]:
    val = (text or "").strip()
    size = max(8, int(start_size))
    min_size = max(8, int(min_size))

    while size > min_size:
        font = _load_font(font_path, size)
        if draw.textlength(val, font=font) <= max_width:
            return val, font
        size -= 1

    font = _load_font(font_path, min_size)
    if draw.textlength(val, font=font) <= max_width:
        return val, font

    # Final fallback: hard-trim without ellipsis to avoid "..." in output.
    out = val
    while out and draw.textlength(out, font=font) > max_width:
        out = out[:-1].rstrip()
    return out, font


def _cover_palette(cover: Image.Image) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    small = cover.convert("RGB").resize((120, 120), Image.Resampling.LANCZOS)
    mr, mg, mb = (v / 255.0 for v in ImageStat.Stat(small).mean[:3])
    h, l, s = colorsys.rgb_to_hls(mr, mg, mb)

    accent = colorsys.hls_to_rgb(h, min(0.68, max(0.48, l + 0.10)), min(0.92, max(0.46, s * 1.18)))
    deep = colorsys.hls_to_rgb(h, min(0.20, max(0.08, l * 0.30)), min(0.58, max(0.18, s * 0.60)))
    return (
        tuple(int(round(c * 255)) for c in accent),
        tuple(int(round(c * 255)) for c in deep),
    )


def make_thumbnail(
    *,
    cover_path: Path,
    title: str,
    artist: str,
    tagline: str,
    genre: str,
    out_path: Path,
    title_max_lines: int = 5,
    title_min_font: int = 40,
    title_max_font: int = 102,
    font_title_path: str | None = None,
    font_body_path: str | None = None,
) -> None:
    cover = Image.open(cover_path).convert("RGB")
    accent, deep = _cover_palette(cover)

    bg = _fit_cover_to_canvas(cover, THUMB_W, THUMB_H)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=36))
    bg = ImageEnhance.Brightness(bg).enhance(0.40)
    bg = ImageEnhance.Contrast(bg).enhance(1.14)
    bg = ImageEnhance.Color(bg).enhance(0.76)
    canvas = bg.convert("RGBA")

    overlay = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    for x in range(THUMB_W):
        t = x / max(1, THUMB_W - 1)
        alpha = int(196 * ((1.0 - t) ** 1.46))
        od.line([(x, 0), (x, THUMB_H)], fill=(0, 0, 0, alpha), width=1)

    od.rectangle((0, 0, THUMB_W, 166), fill=(*deep, 112))
    od.rectangle((0, THUMB_H - 156, THUMB_W, THUMB_H), fill=(0, 0, 0, 152))
    od.rectangle((0, 0, THUMB_W, THUMB_H), outline=(255, 255, 255, 30), width=2)

    glow = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((736, 8, 1270, 560), fill=(*accent, 82))
    gd.ellipse((662, 466, 1250, 850), fill=(*accent, 40))
    gd.ellipse((20, 120, 360, 460), fill=(*accent, 26))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=62))
    overlay = Image.alpha_composite(overlay, glow)
    canvas = Image.alpha_composite(canvas, overlay)

    draw = ImageDraw.Draw(canvas)

    panel_x0, panel_y0 = 56, 52
    panel_x1, panel_y1 = 724, 664
    cover_size = 438
    cover_x = 788
    cover_y = 88

    # Back plate under cover for depth.
    plate = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    pd = ImageDraw.Draw(plate)
    pd.rounded_rectangle(
        (cover_x - 26, cover_y - 26, cover_x + cover_size + 26, cover_y + cover_size + 26),
        radius=40,
        fill=(0, 0, 0, 148),
        outline=(255, 255, 255, 42),
        width=1,
    )
    plate = plate.filter(ImageFilter.GaussianBlur(radius=12))
    canvas = Image.alpha_composite(canvas, plate)
    draw = ImageDraw.Draw(canvas)

    cover_card = _fit_cover_to_canvas(cover, cover_size, cover_size)
    cover_rgba = _apply_rounded_alpha(cover_card.convert("RGBA"), radius=30)
    canvas.alpha_composite(cover_rgba, (cover_x, cover_y))
    draw.rounded_rectangle(
        (cover_x - 2, cover_y - 2, cover_x + cover_size + 2, cover_y + cover_size + 2),
        radius=32,
        outline=(255, 255, 255, 148),
        width=2,
    )
    draw.rounded_rectangle(
        (cover_x - 16, cover_y - 16, cover_x + cover_size + 16, cover_y + cover_size + 16),
        radius=36,
        outline=(*accent, 124),
        width=2,
    )

    panel = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(panel)
    gdraw.rounded_rectangle(
        (panel_x0, panel_y0, panel_x1, panel_y1),
        radius=28,
        fill=(11, 16, 30, 92),
        outline=(255, 255, 255, 46),
        width=1,
    )
    gdraw.rounded_rectangle(
        (panel_x0 + 1, panel_y0 + 1, panel_x1 - 1, panel_y0 + 80),
        radius=28,
        fill=(255, 255, 255, 22),
    )
    gdraw.line([(panel_x0 + 1, panel_y0 + 82), (panel_x1 - 1, panel_y0 + 82)], fill=(255, 255, 255, 34), width=1)
    panel = panel.filter(ImageFilter.GaussianBlur(radius=6))
    canvas = Image.alpha_composite(canvas, panel)
    draw = ImageDraw.Draw(canvas)

    badge_text = (genre or "").strip().upper() or "OUT NOW"
    badge_font = _load_font(font_title_path, 26)
    bb = draw.textbbox((0, 0), badge_text, font=badge_font)
    badge_w = (bb[2] - bb[0]) + 30
    badge_h = (bb[3] - bb[1]) + 18
    bx = panel_x0 + 30
    by = panel_y0 + 26
    draw.rounded_rectangle((bx, by, bx + badge_w, by + badge_h), radius=13, fill=(8, 12, 20, 180), outline=(*accent, 220), width=2)
    draw.text((bx + 15, by + 8), badge_text, font=badge_font, fill=(255, 255, 255, 246))
    draw.text((bx + badge_w + 18, by + 9), "OFFICIAL RELEASE", font=_load_font(font_body_path, 24), fill=(230, 236, 248, 170))

    title_area_x = panel_x0 + 32
    title_area_w = (panel_x1 - panel_x0) - 64
    title_y = by + badge_h + 30
    meta_anchor_y = panel_y1 - (100 if tagline.strip() else 52)
    title_bottom_buffer = 84 if tagline.strip() else 56
    title_area_h = max(90, meta_anchor_y - title_y - title_bottom_buffer)

    line_gap = 9
    eff_line_gap = line_gap
    eff_min_font = max(16, int(title_min_font))
    eff_max_lines = max(1, int(title_max_lines))
    title_size, wrapped = title_min_font, [title]
    for _ in range(20):
        title_size, wrapped, _ = fit_title_block(
            draw=draw,
            text=title,
            font_loader=lambda sz: _load_font(font_title_path, sz),
            min_font=eff_min_font,
            max_font=title_max_font,
            max_width=title_area_w,
            max_height=max(90, title_area_h),
            max_lines=eff_max_lines,
            line_gap=eff_line_gap,
            stroke_width=4,
        )
        if not any(line.rstrip().endswith("...") for line in wrapped):
            break
        if eff_max_lines < 8:
            eff_max_lines += 1
            continue
        if eff_min_font > 16:
            eff_min_font -= 2
            continue
        if eff_line_gap > 6:
            eff_line_gap -= 1
            continue
        wrapped[-1] = wrapped[-1].replace("...", "").rstrip()
        break
    title_font = _load_font(font_title_path, title_size)

    y = title_y
    for line in wrapped:
        draw.text((title_area_x + 3, y + 4), line, font=title_font, fill=(*accent, 160))
        draw.text((title_area_x, y), line, font=title_font, fill=(255, 255, 255, 252), stroke_width=4, stroke_fill=(0, 0, 0, 170))
        line_bb = draw.textbbox((title_area_x, y), line if line else "Ag", font=title_font, stroke_width=4)
        y = line_bb[3] + line_gap

    artist_text, artist_font = _fit_single_line_text(
        draw,
        artist,
        font_path=font_body_path,
        start_size=40,
        min_size=24,
        max_width=title_area_w - 12,
    )
    artist_y = min(y + 14, meta_anchor_y)
    artist_box_w = int(draw.textlength(artist_text, font=artist_font)) + 56
    draw.rounded_rectangle(
        (title_area_x - 2, artist_y - 8, title_area_x + artist_box_w, artist_y + 52),
        radius=13,
        fill=(0, 0, 0, 132),
    )
    dot_x = title_area_x + 16
    dot_y = artist_y + 20
    draw.ellipse((dot_x - 6, dot_y - 6, dot_x + 6, dot_y + 6), fill=(*accent, 236))
    draw.text((title_area_x + 30, artist_y), artist_text, font=artist_font, fill=(255, 255, 255, 236))

    if tagline.strip():
        tagline_text, body_font = _fit_single_line_text(
            draw,
            tagline.strip(),
            font_path=font_body_path,
            start_size=29,
            min_size=18,
            max_width=title_area_w,
        )
        draw.text((title_area_x, artist_y + 64), tagline_text, font=body_font, fill=(224, 232, 240, 220))

    # Click-focused CTA badge (not a player UI).
    cta_text = "WATCH NOW"
    cta_font = _load_font(font_title_path, 26)
    cta_bb = draw.textbbox((0, 0), cta_text, font=cta_font)
    cta_w = (cta_bb[2] - cta_bb[0]) + 48
    cta_h = (cta_bb[3] - cta_bb[1]) + 22
    cta_x = panel_x0 + 30
    cta_y = panel_y1 - cta_h - 24
    draw.rounded_rectangle(
        (cta_x, cta_y, cta_x + cta_w, cta_y + cta_h),
        radius=14,
        fill=(10, 16, 26, 198),
        outline=(*accent, 232),
        width=2,
    )
    draw.text((cta_x + 18, cta_y + 10), cta_text, font=cta_font, fill=(255, 255, 255, 250))
    draw.polygon(
        [(cta_x + cta_w + 14, cta_y + cta_h // 2), (cta_x + cta_w + 2, cta_y + 9), (cta_x + cta_w + 2, cta_y + cta_h - 9)],
        fill=(*accent, 238),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path, quality=95, subsampling=0, optimize=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cover", required=True, help="Input cover image path")
    ap.add_argument("--title", default="", help="Main text title")
    ap.add_argument("--artist", default="Artist", help="Artist text")
    ap.add_argument("--tagline", default="", help="Optional supporting text")
    ap.add_argument("--genre", default="", help="Optional badge text")
    ap.add_argument("--title-max-lines", type=int, default=5)
    ap.add_argument("--title-min-font", type=int, default=40)
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
            raise_user_error("INVALID_ARG", "--title-min-font cannot be greater than --title-max-font.")

        cover = Path(args.cover).expanduser().resolve()
        if not cover.exists():
            raise_user_error("COVER_NOT_FOUND", f"Cover image was not found: {cover}")

        try:
            pillow_fonts = resolve_pillow_font_paths(
                font_title=(args.font_title or None),
                font_body=(args.font_body or None),
            )
        except FontResolutionError as e:
            raise_user_error("FONT_RESOLUTION_FAILED", "Could not resolve requested font path(s).", details=str(e))

        title = args.title or cover.stem
        safe_title = sanitize_filename(title)
        outdir = Path(args.outdir).expanduser().resolve()
        out_path = outdir / f"{safe_title}_glass_thumb.jpg"

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
        print("Glass Thumbnail Generator")
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
