#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import colorsys
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageStat


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


class UserFacingError(RuntimeError):
    pass


QUALITY_PRESETS = {
    "standard": {"crf": "21", "preset": "medium", "maxrate": "10M", "bufsize": "20M", "audio_bitrate": "224k", "fps": "30"},
    "high": {"crf": "18", "preset": "slow", "maxrate": "16M", "bufsize": "32M", "audio_bitrate": "256k", "fps": "60"},
    "ultra": {"crf": "16", "preset": "slow", "maxrate": "20M", "bufsize": "40M", "audio_bitrate": "320k", "fps": "60"},
}

AUDIO_EXTS = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"]
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
VIDEO_EXTS = [".mp4", ".mov", ".mkv", ".webm"]


def raise_user_error(code: str, message: str, details: str = "", hint: str = "") -> None:
    lines = [f"{code}: {message}"]
    if details:
        lines.append(f"Details: {details}")
    if hint:
        lines.append(f"Hint: {hint}")
    raise UserFacingError("\n".join(lines))


def _supported_exts_text(exts: list[str]) -> str:
    return ", ".join(exts)


def which_or_raise(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise_user_error("FFMPEG_NOT_FOUND", f"`{name}` was not found in PATH.", hint="Install FFmpeg and ensure ffmpeg/ffprobe are available.")
    return p


def run_capture(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise_user_error("COMMAND_FAILED", "External command failed.", details=f"Command: {' '.join(cmd)}\nSTDERR:\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def ff_paths() -> FFPaths:
    return FFPaths(ffmpeg=which_or_raise("ffmpeg"), ffprobe=which_or_raise("ffprobe"))


def get_duration_seconds(ffp: FFPaths, media_path: Path) -> float:
    out = run_capture([
        ffp.ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ])
    try:
        return float(out)
    except ValueError:
        raise_user_error("DURATION_PARSE_FAILED", "Could not parse media duration.", details=f"ffprobe output: {out}")


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
    return f"{int(round(_clamp01(r)*255)):02X}{int(round(_clamp01(g)*255)):02X}{int(round(_clamp01(b)*255)):02X}"


def cover_color_scheme(image_path: Path) -> ColorScheme:
    img = Image.open(image_path).convert("RGB").resize((80, 80))
    mr, mg, mb = (float(x) / 255.0 for x in ImageStat.Stat(img).mean[:3])
    h, l, s = colorsys.rgb_to_hls(mr, mg, mb)

    accent_l = min(0.66, max(0.42, l * 0.95 + 0.08))
    accent_s = min(0.88, max(0.34, s * 1.28))
    accent_hex = _rgb_float_to_hex(*colorsys.hls_to_rgb(h, accent_l, accent_s))

    mood_bg_l = min(0.18, max(0.07, l * 0.28))
    mood_bg_s = min(0.55, max(0.16, s * 0.58))
    mood_bg_hex = _rgb_float_to_hex(*colorsys.hls_to_rgb(h, mood_bg_l, mood_bg_s))

    tint_h = (h + 0.02) % 1.0
    tint_l = min(0.32, max(0.16, l * 0.46 + 0.04))
    tint_s = min(0.84, max(0.30, s * 1.10 + 0.05))
    mood_tint_hex = _rgb_float_to_hex(*colorsys.hls_to_rgb(tint_h, tint_l, tint_s))

    point_l = min(0.78, max(0.62, accent_l + 0.16))
    point_s = min(0.92, max(0.48, accent_s * 1.05))
    point_hex = _rgb_float_to_hex(*colorsys.hls_to_rgb(h, point_l, point_s))

    return ColorScheme(accent_hex=accent_hex, mood_bg_hex=mood_bg_hex, mood_tint_hex=mood_tint_hex, point_hex=point_hex)


def resolve_media_path(explicit_value: str | None, *, title_hint: str, exts: list[str], kind: str) -> Path:
    if explicit_value:
        raw = Path(explicit_value).expanduser()
        if raw.suffix:
            if raw.suffix.lower() not in exts:
                raise_user_error("UNSUPPORTED_EXTENSION", f"{kind} extension is not supported: {raw.suffix}", details=f"Supported: {_supported_exts_text(exts)}")
            if not raw.exists():
                raise_user_error(f"{kind.upper()}_NOT_FOUND", f"{kind} file not found: {raw}")
            return raw.resolve()

        for ext in exts:
            cand = raw.with_suffix(ext)
            if cand.exists():
                return cand.resolve()
        raise_user_error(f"{kind.upper()}_NOT_FOUND", f"{kind} file not found: {raw}", details=", ".join(str(raw.with_suffix(ext)) for ext in exts))

    hint = (title_hint or "").strip()
    if not hint:
        raise_user_error(f"{kind.upper()}_MISSING", f"{kind} is not specified and cannot be inferred.")
    base = Path(hint).stem if Path(hint).suffix else hint
    for ext in exts:
        cand = Path(base + ext)
        if cand.exists():
            return cand.resolve()
    raise_user_error(f"{kind.upper()}_NOT_FOUND", f"{kind} file could not be inferred from title '{base}'.", details=", ".join(base + ext for ext in exts))


def ffmpeg_escape_drawtext_text(s: str, preserve_newlines: bool = False) -> str:
    s = (s or "").replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'").replace("%", r"\%")
    return s.replace("\r", "").replace("\n", r"\n") if preserve_newlines else s.replace("\r", " ").replace("\n", " ")


def ffmpeg_escape_path(path: str) -> str:
    return path.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
