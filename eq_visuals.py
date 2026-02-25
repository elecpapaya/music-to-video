#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared EQ color and motion helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EqPalette:
    name: str
    low_hex: str
    mid_hex: str
    high_hex: str
    accent_hex: str
    frame_hex: str
    progress_hex: str
    knob_hex: str
    label_hex: str
    hue_speed: float
    hue_depth: float
    pulse_boost: float


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _normalize_hex(value: str, fallback: str) -> str:
    raw = (value or "").strip().lstrip("#")
    if len(raw) == 6 and all(c in "0123456789abcdefABCDEF" for c in raw):
        return raw.upper()
    return fallback.upper()


def hex_to_rgbf(value: str) -> tuple[float, float, float]:
    hx = _normalize_hex(value, "FFFFFF")
    r = int(hx[0:2], 16) / 255.0
    g = int(hx[2:4], 16) / 255.0
    b = int(hx[4:6], 16) / 255.0
    return r, g, b


def get_eq_palette(style: str, *, accent_hex: str, point_hex: str) -> EqPalette:
    style_v = (style or "legacy").strip().lower()
    accent = _normalize_hex(accent_hex, "FFFFFF")
    point = _normalize_hex(point_hex, accent)

    if style_v == "prism":
        return EqPalette(
            name="prism",
            low_hex="FF6E4A",
            mid_hex="3ED8FF",
            high_hex="D9FF4B",
            accent_hex=point,
            frame_hex="78C4FF",
            progress_hex="42FFD5",
            knob_hex="FFE66B",
            label_hex="D9F7FF",
            hue_speed=0.82,
            hue_depth=1.16,
            pulse_boost=1.45,
        )

    return EqPalette(
        name="legacy",
        low_hex="FFF2E8",
        mid_hex="FFFFFF",
        high_hex="E8F5FF",
        accent_hex=accent,
        frame_hex="FFFFFF",
        progress_hex="FFFFFF",
        knob_hex=accent,
        label_hex="FFFFFF",
        hue_speed=0.0,
        hue_depth=0.0,
        pulse_boost=0.0,
    )


def build_pulse_enable_expr(pulse_times: list[float], width: float = 0.16, max_terms: int = 42) -> str:
    pts = sorted(set(round(float(x), 3) for x in pulse_times if float(x) >= 0.0))
    if not pts:
        return "0"
    terms: list[str] = []
    for p in pts[:max_terms]:
        a, b = max(0.0, p - width * 0.35), p + width
        terms.append(f"between(t\\,{a:.3f}\\,{b:.3f})")
    return "+".join(terms) if terms else "0"


def build_hue_expr(
    *,
    color_drive: float,
    pulse_enable_expr: str,
    base_speed: float,
    base_depth: float,
    pulse_boost: float,
) -> str:
    drive = clamp01(color_drive)
    if drive <= 0.0001:
        return "0"

    pulse_gate = f"min(1\\,{pulse_enable_expr})" if pulse_enable_expr != "0" else "0"
    speed = base_speed * (0.30 + drive * 1.10)
    depth = base_depth * (0.18 + drive * 0.92)
    beat = pulse_boost * drive * 0.55
    return f"{speed:.5f}*t+{depth:.5f}*sin(1.9*t)+{beat:.5f}*({pulse_gate})*sin(17*t)"
