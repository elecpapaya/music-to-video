#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Color preset resolver for FFmpeg filtergraph.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

from PIL import Image, ImageStat


COLOR_PRESETS = ("neutral", "cinema", "neon", "warm", "cool", "mono")


def choose_auto_preset(cover_path: Path) -> str:
    img = Image.open(cover_path).convert("RGB").resize((100, 100))
    stat = ImageStat.Stat(img)
    mr, mg, mb = (float(x) / 255.0 for x in stat.mean[:3])
    _, l, s = colorsys.rgb_to_hls(mr, mg, mb)

    if s < 0.18:
        return "cinema"
    if l < 0.33:
        return "warm"
    if l > 0.68:
        return "cool"
    return "neutral"


def resolve_color_preset(name: str, cover_path: Path) -> str:
    n = (name or "auto").strip().lower()
    if n == "auto":
        return choose_auto_preset(cover_path)
    if n not in COLOR_PRESETS:
        return "neutral"
    return n


def preset_filter_chain(name: str) -> str:
    n = (name or "neutral").strip().lower()
    if n == "neutral":
        return "eq=brightness=-0.02:contrast=1.05:saturation=1.04"
    if n == "cinema":
        return "eq=brightness=-0.04:contrast=1.12:saturation=1.08,colorbalance=rs=.04:gs=-.01:bs=-.05:rm=.03:gm=.00:bm=-.03"
    if n == "neon":
        return "eq=brightness=-0.03:contrast=1.20:saturation=1.30,colorbalance=rs=.02:gs=.01:bs=.08"
    if n == "warm":
        return "eq=brightness=-0.02:contrast=1.08:saturation=1.14,colorbalance=rs=.08:gs=.01:bs=-.05"
    if n == "cool":
        return "eq=brightness=-0.03:contrast=1.08:saturation=1.07,colorbalance=rs=-.03:gs=.01:bs=.08"
    if n == "mono":
        return "hue=s=0,eq=brightness=-0.03:contrast=1.16:saturation=0.0"
    return "eq=brightness=-0.02:contrast=1.05:saturation=1.04"

