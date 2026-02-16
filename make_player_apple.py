#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a YouTube-ready player video from an audio file + cover image.
Also generates a thumbnail by default.
"""

from __future__ import annotations

import argparse
import colorsys
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageStat

from font_resolver import FontResolutionError, resolve_ffmpeg_font_paths, resolve_pillow_font_paths
from make_thumbnail_youtube import make_thumbnail
from text_layout import fit_title_block, truncate_with_ellipsis


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

AUDIO_EXTS = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"]
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


def raise_user_error(code: str, message: str, details: str = "", hint: str = "") -> None:
    parts = [f"{code}: {message}"]
    if details:
        parts.append(f"Details: {details}")
    if hint:
        parts.append(f"Hint: {hint}")
    raise UserFacingError("\n".join(parts))


def _supported_exts_text(exts: list[str]) -> str:
    return ", ".join(exts)


def which_or_raise(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise_user_error(
            "FFMPEG_NOT_FOUND",
            f"`{name}` was not found in PATH.",
            details="This app requires FFmpeg binaries (`ffmpeg` and `ffprobe`).",
            hint="Install FFmpeg and ensure both commands are available in your shell PATH.",
        )
    return p


def run_capture(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise_user_error(
            "COMMAND_FAILED",
            "External command execution failed.",
            details=f"Command: {' '.join(cmd)}\nSTDERR:\n{proc.stderr.strip()}",
            hint="Check input media integrity and FFmpeg installation.",
        )
    return proc.stdout.strip()


def ff_paths() -> FFPaths:
    return FFPaths(ffmpeg=which_or_raise("ffmpeg"), ffprobe=which_or_raise("ffprobe"))


def get_duration_seconds(ffp: FFPaths, audio_path: Path) -> float:
    out = run_capture(
        [
            ffp.ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
    )
    try:
        return float(out)
    except ValueError:
        raise_user_error(
            "DURATION_PARSE_FAILED",
            "Could not read audio duration with ffprobe.",
            details=f"Raw ffprobe output: {out}",
            hint='Try with explicit input, e.g. --song "track.wav"',
        )


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
                raise_user_error(
                    "UNSUPPORTED_EXTENSION",
                    f"{kind} extension is not supported: {raw.suffix}",
                    details=f"Supported {kind.lower()} extensions: {_supported_exts_text(exts)}",
                    hint=f"Use a supported file type or omit extension to auto-try candidates.",
                )
            if not raw.exists():
                raise_user_error(
                    f"{kind.upper()}_NOT_FOUND",
                    f"{kind} file not found: {raw}",
                    hint=f'Provide a valid path, e.g. --{kind.lower()} "your_file{exts[0]}"',
                )
            return raw.resolve()

        for ext in exts:
            cand = raw.with_suffix(ext)
            if cand.exists():
                return cand.resolve()
        tried = ", ".join(str(raw.with_suffix(ext)) for ext in exts)
        raise_user_error(
            f"{kind.upper()}_NOT_FOUND",
            f"{kind} file not found: {raw}",
            details=f"Tried: {tried}",
            hint=f'Specify full path explicitly with --{kind.lower()}',
        )

    hint = (title_hint or "").strip()
    if not hint:
        raise_user_error(
            f"{kind.upper()}_MISSING",
            f"{kind} is not specified and cannot be inferred.",
            hint=f'Use --title "Track Name" or provide --{kind.lower()} explicitly.',
        )
    base = Path(hint).stem if Path(hint).suffix else hint
    for ext in exts:
        cand = Path(base + ext)
        if cand.exists():
            return cand.resolve()
    tried = ", ".join(base + ext for ext in exts)
    raise_user_error(
        f"{kind.upper()}_NOT_FOUND",
        f"{kind} file could not be inferred from title '{base}'.",
        details=f"Tried: {tried}",
        hint=f'Provide --{kind.lower()} explicitly or add a matching file in the working directory.',
    )


def ffmpeg_escape_drawtext_text(s: str, preserve_newlines: bool = False) -> str:
    s = (s or "").replace("\\", "\\\\")
    s = s.replace(":", r"\:")
    s = s.replace("'", r"\'")
    s = s.replace("%", r"\%")
    if preserve_newlines:
        s = s.replace("\r", "").replace("\n", r"\n")
    else:
        s = s.replace("\r", " ").replace("\n", " ")
    return s


def ffmpeg_escape_drawtext_path(path: str) -> str:
    p = path.replace("\\", "/")
    p = p.replace(":", r"\:")
    p = p.replace("'", r"\'")
    return p


def _load_pillow_font(font_path: str | None, size: int) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            return ImageFont.load_default()
    return ImageFont.load_default()


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
    style: str,
    font_title_path: str,
    font_body_path: str,
    title_max_lines: int,
    title_min_font: int,
    title_max_font: int,
    motion_smooth: float,
    knob_scale: float,
) -> str:
    elapsed_expr = r"%{eif\:t/60\:d\:2}\:%{eif\:mod(t\,60)\:d\:2}"
    total_e = ffmpeg_escape_drawtext_text(total_mmss)

    if style == "youtube":
        cover_x, cover_y, cover_size = 150, 170, 620
        info_x, info_y = 840, 140
        info_w, info_h = 980, 800
        title_y = 255
        artist_y = 395
        artist_fs = 38
        bar_x, bar_y, bar_w = 875, 785, 900
        time_y = 820
    else:
        cover_x, cover_y, cover_size = 220, 190, 560
        info_x, info_y = 860, 170
        info_w, info_h = 920, 740
        title_y = 275
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

    measure_img = Image.new("RGB", (4, 4), (0, 0, 0))
    draw = ImageDraw.Draw(measure_img)
    title_area_w = info_w - 64
    title_area_h = max(100, artist_y - title_y - 18)
    base_line_gap = 10

    title_size, title_lines, _ = fit_title_block(
        draw=draw,
        text=title,
        font_loader=lambda size: _load_pillow_font(font_title_path, size),
        min_font=title_min_font,
        max_font=title_max_font,
        max_width=title_area_w,
        max_height=title_area_h,
        max_lines=title_max_lines,
        line_gap=base_line_gap,
        stroke_width=2,
    )
    title_line_spacing = max(6, int(title_size * 0.16))
    title_text = "\n".join(title_lines)
    title_e = ffmpeg_escape_drawtext_text(title_text, preserve_newlines=True)

    artist_size = artist_fs
    while artist_size > 20:
        font = _load_pillow_font(font_body_path, artist_size)
        if draw.textlength(artist, font=font) <= title_area_w:
            break
        artist_size -= 1
    artist_font = _load_pillow_font(font_body_path, artist_size)
    artist_text = truncate_with_ellipsis(draw, artist, artist_font, title_area_w)
    artist_e = ffmpeg_escape_drawtext_text(artist_text)

    k = max(0.0, min(1.0, float(motion_smooth)))
    p_expr = f"(min(t\\,{dur:.6f})/{dur:.6f})"
    s_expr = f"(3*pow({p_expr}\\,2)-2*pow({p_expr}\\,3))"
    ps_expr = f"((1-{k:.6f})*{p_expr}+{k:.6f}*{s_expr})"
    progress_x_expr = f"({bar_x}+{bar_w}*{ps_expr})"

    ks = max(0.4, min(2.0, float(knob_scale)))
    knob_glow_w = max(10, int(round(24 * ks)))
    knob_glow_h = max(12, int(round(32 * ks)))
    knob_body_w = max(8, int(round(14 * ks)))
    knob_body_h = max(10, int(round(22 * ks)))
    knob_core_w = max(2, int(round(4 * ks)))
    knob_core_h = max(6, int(round(14 * ks)))
    knob_glow_dx = knob_glow_w // 2
    knob_glow_dy = knob_glow_h // 2 - 3
    knob_body_dx = knob_body_w // 2
    knob_body_dy = knob_body_h // 2 - 2
    knob_core_dx = knob_core_w // 2
    knob_core_dy = knob_core_h // 2 - 2

    font_title_ff = ffmpeg_escape_drawtext_path(font_title_path)
    font_body_ff = ffmpeg_escape_drawtext_path(font_body_path)

    chains = [
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
        f"[base0]drawbox=x={info_x}:y={info_y}:w={info_w}:h={info_h}:color=black@0.32:t=fill[base3]",
        f"[base3]drawbox=x={info_x}:y={info_y}:w={info_w}:h=3:color=0x{accent_hex}@0.86:t=fill[base4]",
        f"[base4]drawtext=fontfile='{font_title_ff}':text='NOW PLAYING':x={info_x+32}:y={info_y+28}:fontsize=25:fontcolor=white@0.62[base5]",
        f"[base5]drawtext=fontfile='{font_title_ff}':text='{title_e}':x={info_x+32}:y={title_y}:fontsize={title_size}:line_spacing={title_line_spacing}:fontcolor=white:borderw=2:bordercolor=black@0.28[base6]",
        f"[base6]drawtext=fontfile='{font_body_ff}':text='{artist_e}':x={info_x+32}:y={artist_y}:fontsize={artist_size}:fontcolor=white@0.82[base7]",
        f"[base7]drawbox=x={info_x+32}:y={artist_y+86}:w=220:h=4:color=0x{accent_hex}@0.95:t=fill[base8]",
        f"[base8]drawtext=fontfile='{font_body_ff}':text='EQUALIZER':x={eq_x}:y={eq_y-32}:fontsize=20:fontcolor=white@0.46[base8a]",
        f"[base8a]drawbox=x={eq_x}:y={eq_y-8}:w={eq_w}:h=1:color=white@0.18:t=fill[base8b]",
        f"[base8b]drawbox=x={eq_x}:y={eq_y}:w={eq_w}:h={eq_h}:color=white@0.05:t=fill[base8c]",
        f"[1:a]showfreqs=s={eq_w}x{eq_h}:mode=bar:fscale=log:ascale=sqrt:win_func=hann:colors=white,format=rgba,colorkey=0x000000:0.18:0.0,split=2[eq_raw][eq_top]",
        "[eq_raw]gblur=sigma=2.2:steps=1,colorchannelmixer=aa=0.30[eq_glow]",
        f"[base8c][eq_glow]overlay={eq_x}:{eq_y}:format=auto[base9]",
        f"[base9][eq_top]overlay={eq_x}:{eq_y}:format=auto[base10]",
        f"[base10]drawbox=x={bar_x}:y={bar_y}:w={bar_w}:h=5:color=white@0.20:t=fill[bar_bg]",
        f"[bar_bg]drawbox=x={bar_x}:y={bar_y}:w={bar_w}:h=5:color=white@0.93:t=fill[bar_fill_full]",
        f"color=c=black@0.82:s={bar_w}x5[bar_mask]",
        f"[bar_fill_full][bar_mask]overlay=x='{progress_x_expr}':y={bar_y}:eval=frame[bar_move]",
        f"color=c=0x{accent_hex}@0.32:s={knob_glow_w}x{knob_glow_h}[bar_knob_glow_src]",
        "[bar_knob_glow_src]gblur=sigma=2.2:steps=1[bar_knob_glow]",
        f"[bar_move][bar_knob_glow]overlay=x='{progress_x_expr}-{knob_glow_dx}':y={bar_y-knob_glow_dy}:eval=frame[bar_move2]",
        f"color=c=white@0.90:s={knob_body_w}x{knob_body_h}[bar_knob_body]",
        f"[bar_move2][bar_knob_body]overlay=x='{progress_x_expr}-{knob_body_dx}':y={bar_y-knob_body_dy}:eval=frame[bar_move3]",
        f"color=c=0x{accent_hex}@0.82:s={knob_core_w}x{knob_core_h}[bar_knob_core]",
        f"[bar_move3][bar_knob_core]overlay=x='{progress_x_expr}-{knob_core_dx}':y={bar_y-knob_core_dy}:eval=frame[base13]",
        f"[base13]drawtext=fontfile='{font_body_ff}':text='{elapsed_expr}':x={bar_x}:y={time_y}:fontsize=24:fontcolor=white@0.70[base14]",
        f"[base14]drawtext=fontfile='{font_body_ff}':text='{total_e}':x='{bar_x}+{bar_w}-tw':y={time_y}:fontsize=24:fontcolor=white@0.70[base15]",
        f"[base15]drawtext=fontfile='{font_title_ff}':text='<<':x={prev_x}:y={controls_y}:fontsize=34:fontcolor=white@0.58[base16]",
        f"[base16]drawtext=fontfile='{font_title_ff}':text='PLAY':x={play_x}:y={controls_y+2}:fontsize=28:fontcolor=white@0.92[base17]",
        f"[base17]drawtext=fontfile='{font_title_ff}':text='>>':x={next_x}:y={controls_y}:fontsize=34:fontcolor=white@0.58[vout]",
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
    ap.add_argument("--style", choices=("youtube", "classic"), default="youtube", help="Player visual style: youtube (default) or classic")
    ap.add_argument("--quality", choices=tuple(QUALITY_PRESETS), default="high")
    ap.add_argument("--title-max-lines", type=int, default=3)
    ap.add_argument("--title-min-font", type=int, default=44)
    ap.add_argument("--title-max-font", type=int, default=82)
    ap.add_argument("--motion-smooth", type=float, default=0.18, help="Progress/knob smoothing amount (0.0 ~ 1.0)")
    ap.add_argument("--knob-scale", type=float, default=1.0, help="Progress knob scale factor")
    ap.add_argument("--font-title", default="", help="Optional title font path override")
    ap.add_argument("--font-body", default="", help="Optional body font path override")
    ap.add_argument("--outdir", default="output")
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
                hint="Use values like --title-min-font 44 --title-max-font 82",
            )
        if args.motion_smooth < 0.0 or args.motion_smooth > 1.0:
            raise_user_error("INVALID_ARG", "--motion-smooth must be between 0.0 and 1.0.")
        if args.knob_scale <= 0.0:
            raise_user_error("INVALID_ARG", "--knob-scale must be greater than 0.")

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

        try:
            ff_fonts = resolve_ffmpeg_font_paths(
                font_title=(args.font_title or None),
                font_body=(args.font_body or None),
            )
            pillow_fonts = resolve_pillow_font_paths(
                font_title=(args.font_title or None),
                font_body=(args.font_body or None),
            )
        except FontResolutionError as e:
            raise_user_error(
                "FONT_RESOLUTION_FAILED",
                "Failed to resolve font paths.",
                details=str(e),
                hint="Check --font-title/--font-body paths or remove them for automatic fallback.",
            )

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
            font_title_path=ff_fonts["title"],
            font_body_path=ff_fonts["body"],
            title_max_lines=args.title_max_lines,
            title_min_font=args.title_min_font,
            title_max_font=args.title_max_font,
            motion_smooth=args.motion_smooth,
            knob_scale=args.knob_scale,
        )
        graph_path.write_text(graph, encoding="utf-8")

        thumb_generated = False
        if not args.skip_thumbnail:
            try:
                make_thumbnail(
                    cover_path=cover,
                    title=title,
                    artist=artist,
                    genre=args.genre,
                    tagline=args.tagline,
                    out_path=thumb_path,
                    title_max_lines=args.title_max_lines,
                    title_min_font=max(56, args.title_min_font),
                    title_max_font=max(56, args.title_max_font),
                    font_title_path=pillow_fonts["title"],
                    font_body_path=pillow_fonts["body"],
                )
                thumb_generated = True
            except Exception as e:
                print(f"[WARN] thumbnail generation failed: {e}")

        print("=====================================================")
        print("music-to-video (Python v13)")
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
            ffp.ffmpeg,
            "-hide_banner",
            "-y",
            "-loop",
            "1",
            "-i",
            str(cover),
            "-i",
            str(song),
            "-filter_complex",
            graph,
            "-map",
            "[vout]",
            "-map",
            "1:a",
            "-shortest",
            "-r",
            quality["fps"],
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-level:v",
            "4.1",
            "-crf",
            quality["crf"],
            "-preset",
            quality["preset"],
            "-maxrate",
            quality["maxrate"],
            "-bufsize",
            quality["bufsize"],
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            quality["audio_bitrate"],
            "-ar",
            "48000",
            str(out_path),
        ]

        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print("\n[ERROR] ffmpeg failed.")
            return proc.returncode

        print("\nDone.")
        return 0

    except UserFacingError as e:
        print(f"\n[ERROR] {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
