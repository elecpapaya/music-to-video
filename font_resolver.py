#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import platform
from pathlib import Path


class FontResolutionError(RuntimeError):
    pass


def _platform_family() -> str:
    name = platform.system().lower()
    if "windows" in name:
        return "windows"
    if "darwin" in name or "mac" in name:
        return "mac"
    return "linux"


def _candidates_for_role(role: str) -> list[str]:
    role = "title" if role == "title" else "body"
    pf = _platform_family()

    if pf == "windows":
        if role == "title":
            return [
                "C:/Windows/Fonts/malgunbd.ttf",
                "C:/Windows/Fonts/segoeuib.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/calibrib.ttf",
            ]
        return [
            "C:/Windows/Fonts/malgun.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        ]

    if pf == "mac":
        if role == "title":
            return [
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            ]
        return [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        ]

    # Linux
    if role == "title":
        return [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Bold.otf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    return [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]


def _resolve_single_font(override_path: str | None, role: str, *, required: bool) -> str | None:
    if override_path:
        p = Path(override_path).expanduser()
        if not p.exists():
            raise FontResolutionError(f"Requested {role} font does not exist: {p}")
        return str(p.resolve())

    for candidate in _candidates_for_role(role):
        p = Path(candidate)
        if p.exists():
            return str(p.resolve())

    if required:
        raise FontResolutionError(
            f"Could not resolve a {role} font for this OS. "
            "Specify --font-title/--font-body explicitly."
        )
    return None


def resolve_pillow_font_paths(font_title: str | None = None, font_body: str | None = None) -> dict[str, str | None]:
    return {
        "title": _resolve_single_font(font_title, "title", required=False),
        "body": _resolve_single_font(font_body, "body", required=False),
    }


def resolve_ffmpeg_font_paths(font_title: str | None = None, font_body: str | None = None) -> dict[str, str]:
    return {
        "title": _resolve_single_font(font_title, "title", required=True) or "",
        "body": _resolve_single_font(font_body, "body", required=True) or "",
    }
