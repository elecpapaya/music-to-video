#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable

from PIL import ImageDraw, ImageFont


def truncate_with_ellipsis(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if max_width <= 0:
        return ""
    if draw.textlength(text, font=font) <= max_width:
        return text

    ellipsis = "..."
    if draw.textlength(ellipsis, font=font) > max_width:
        return ""

    out = text.rstrip()
    while out and draw.textlength(out + ellipsis, font=font) > max_width:
        out = out[:-1].rstrip()
    return (out + ellipsis) if out else ellipsis


def wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> tuple[list[str], bool]:
    text = (text or "").replace("\r", " ").replace("\n", " ").strip()
    if not text:
        return [""], False
    words = text.split()
    if not words:
        return [""], False

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)

    truncated = False
    if len(lines) > max_lines:
        truncated = True
        lines = lines[:max_lines]
        lines[-1] = truncate_with_ellipsis(draw, lines[-1], font, max_width)
    else:
        lines[-1] = truncate_with_ellipsis(draw, lines[-1], font, max_width)

    return lines, truncated


def _block_height(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.ImageFont,
    line_gap: int,
    stroke_width: int,
) -> int:
    h = 0
    for idx, line in enumerate(lines):
        sample = line if line else "Ag"
        bb = draw.textbbox((0, 0), sample, font=font, stroke_width=stroke_width)
        line_h = max(1, bb[3] - bb[1])
        h += line_h
        if idx < len(lines) - 1:
            h += line_gap
    return h


def fit_title_block(
    *,
    draw: ImageDraw.ImageDraw,
    text: str,
    font_loader: Callable[[int], ImageFont.ImageFont],
    min_font: int,
    max_font: int,
    max_width: int,
    max_height: int,
    max_lines: int,
    line_gap: int,
    stroke_width: int = 0,
) -> tuple[int, list[str], bool]:
    min_font = max(8, int(min_font))
    max_font = max(min_font, int(max_font))
    max_lines = max(1, int(max_lines))
    max_width = max(1, int(max_width))
    max_height = max(1, int(max_height))

    best_size = min_font
    best_lines: list[str] = [text or ""]
    best_truncated = False

    lo, hi = min_font, max_font
    while lo <= hi:
        mid = (lo + hi) // 2
        font = font_loader(mid)
        lines, truncated = wrap_text_to_width(draw, text, font, max_width, max_lines)
        h = _block_height(draw, lines, font, line_gap, stroke_width)
        if h <= max_height:
            best_size = mid
            best_lines = lines
            best_truncated = truncated
            lo = mid + 1
        else:
            hi = mid - 1

    final_font = font_loader(best_size)
    final_lines, final_truncated = wrap_text_to_width(draw, text, final_font, max_width, max_lines)
    return best_size, final_lines, (best_truncated or final_truncated)
