#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a YouTube-ready player video from audio + cover image."""

from __future__ import annotations

import argparse
import colorsys
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageStat

from audio_reactive import analyze_audio_reactivity
from color_presets import preset_filter_chain, resolve_color_preset
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


def _load_pillow_font(font_path: str | None, size: int) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            return ImageFont.load_default()
    return ImageFont.load_default()


def _ffconcat_quote_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "'\\''")


def _build_uniform_cut_times(duration: float, min_cut: float, max_cut: float) -> list[float]:
    step = max(min_cut, min(max_cut, 1.6))
    t, out = step, []
    while t < duration - 0.2:
        out.append(round(t, 3))
        t += step
    return out


def _collect_broll_clips(broll_dir: Path) -> list[Path]:
    if not broll_dir.exists() or not broll_dir.is_dir():
        raise_user_error("E_BROLL_NOT_FOUND", f"B-roll directory not found: {broll_dir}")
    clips = [p for p in sorted(broll_dir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    if not clips:
        raise_user_error("E_BROLL_EMPTY", f"No B-roll files found. Supported: {_supported_exts_text(VIDEO_EXTS)}")
    return clips


def _prepare_broll_from_dir(
    *,
    ffp: FFPaths,
    broll_dir: Path,
    outdir: Path,
    safe_title: str,
    duration: float,
    cut_times: list[float],
    min_cut_interval: float,
    max_cut_interval: float,
) -> tuple[Path, Path]:
    clips = _collect_broll_clips(broll_dir)
    durations: dict[Path, float] = {}
    valid = []
    for clip in clips:
        d = get_duration_seconds(ffp, clip)
        durations[clip] = d
        if d >= 0.25:
            valid.append(clip)
    if not valid:
        raise_user_error("E_BROLL_EMPTY", "All B-roll clips are too short (<0.25s).")

    cuts = sorted(t for t in cut_times if 0.2 < t < duration - 0.2)
    if not cuts:
        cuts = _build_uniform_cut_times(duration, min_cut_interval, max_cut_interval)

    bounds = [0.0] + cuts + [duration]
    segments = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] - bounds[i] >= 0.2]
    if not segments:
        segments = [(0.0, duration)]

    rng = random.Random(f"{safe_title}|{duration:.3f}")
    order = valid[:]
    rng.shuffle(order)

    concat_path = outdir / f"{safe_title}_broll.ffconcat"
    lines = ["ffconcat version 1.0"]
    idx = 0
    for s0, s1 in segments:
        seg_len = max(0.2, s1 - s0)
        clip = order[idx % len(order)]
        idx += 1
        clip_d = durations[clip]
        max_in = max(0.0, clip_d - seg_len - 0.02)
        inpoint = rng.uniform(0.0, max_in) if max_in > 0 else 0.0
        outpoint = min(clip_d - 0.01, inpoint + seg_len)
        if outpoint - inpoint < 0.18:
            inpoint, outpoint = 0.0, min(clip_d - 0.01, max(0.20, seg_len))
        lines.append(f"file '{_ffconcat_quote_path(clip)}'")
        lines.append(f"inpoint {inpoint:.3f}")
        lines.append(f"outpoint {outpoint:.3f}")

    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_path = outdir / f"{safe_title}_broll_temp.mp4"
    cmd = [
        ffp.ffmpeg, "-hide_banner", "-y", "-safe", "0", "-f", "concat", "-i", str(concat_path),
        "-an", "-vf", "scale=1920:1080:force_original_aspect_ratio=increase:flags=lanczos,crop=1920:1080,fps=30,format=yuv420p",
        "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", str(out_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise_user_error("E_BROLL_PREP_FAILED", "Failed to build B-roll timeline.", details=proc.stderr.strip())
    return out_path, concat_path


def _prepare_broll_from_video(*, ffp: FFPaths, broll_video: Path, outdir: Path, safe_title: str, duration: float) -> Path:
    if not broll_video.exists():
        raise_user_error("E_BROLL_NOT_FOUND", f"B-roll video not found: {broll_video}")
    if broll_video.suffix.lower() not in VIDEO_EXTS:
        raise_user_error("UNSUPPORTED_EXTENSION", f"Unsupported B-roll extension: {broll_video.suffix}")

    out_path = outdir / f"{safe_title}_broll_loop.mp4"
    cmd = [
        ffp.ffmpeg, "-hide_banner", "-y", "-stream_loop", "-1", "-i", str(broll_video),
        "-an", "-t", f"{duration:.3f}",
        "-vf", "scale=1920:1080:force_original_aspect_ratio=increase:flags=lanczos,crop=1920:1080,fps=30,format=yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", str(out_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise_user_error("E_BROLL_PREP_FAILED", "Failed to normalize B-roll video.", details=proc.stderr.strip())
    return out_path


def _build_pulse_enable_expr(pulse_times: list[float], width: float = 0.16, max_terms: int = 42) -> str:
    pts = sorted(set(round(float(x), 3) for x in pulse_times if x >= 0.0))
    if not pts:
        return "0"
    terms = []
    for p in pts[:max_terms]:
        a, b = max(0.0, p - width * 0.35), p + width
        terms.append(f"between(t\\,{a:.3f}\\,{b:.3f})")
    return "+".join(terms) if terms else "0"


def _append_motion_chain(chains: list[str], *, input_label: str, output_label: str, mode: str, strength: float, factor: float = 1.0) -> None:
    s = _clamp01(strength) * max(0.4, factor)
    if mode == "none" or s <= 0.001:
        chains.append(f"[{input_label}]null[{output_label}]")
        return
    if mode in ("drift", "parallax"):
        ax, ay = max(1, int(round(28.0 * s))), max(1, int(round(18.0 * s)))
        chains.append(
            f"[{input_label}]scale=2060:1168:flags=lanczos,crop=1920:1080:x='(iw-1920)/2+{ax}*sin(t*0.07)':y='(ih-1080)/2+{ay}*cos(t*0.05)'[{output_label}]"
        )
        return
    if mode == "zoom":
        amp = max(0.01, 0.085 * s)
        cw = f"1920-260*{amp:.5f}*(0.5+0.5*sin(t*0.06))"
        ch = f"1080-146*{amp:.5f}*(0.5+0.5*sin(t*0.06))"
        chains.append(f"[{input_label}]scale=2240:1260:flags=lanczos,crop=w='{cw}':h='{ch}':x='(iw-w)/2':y='(ih-h)/2',scale=1920:1080[{output_label}]")
        return
    chains.append(f"[{input_label}]null[{output_label}]")


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
    broll_enabled: bool,
    broll_blend: str,
    broll_opacity: float,
    camera_motion: str,
    camera_strength: float,
    reactive_level: float,
    reactive_glow: float,
    reactive_blur: float,
    reactive_shake: float,
    pulse_times: list[float],
    color_filter_chain: str,
    lut_path: Path | None,
    lut_intensity: float,
) -> str:
    elapsed_expr = r"%{eif\:t/60\:d\:2}\:%{eif\:mod(t\,60)\:d\:2}"
    total_e = ffmpeg_escape_drawtext_text(total_mmss)

    if style == "youtube":
        cover_x, cover_y, cover_size = 150, 170, 620
        info_x, info_y, info_w, info_h = 840, 140, 980, 800
        title_y, artist_y, artist_fs = 255, 395, 38
        bar_x, bar_y, bar_w, time_y = 875, 785, 900, 820
    else:
        cover_x, cover_y, cover_size = 220, 190, 560
        info_x, info_y, info_w, info_h = 860, 170, 920, 740
        title_y, artist_y, artist_fs = 275, 412, 34
        bar_x, bar_y, bar_w, time_y = 895, 765, 840, 800

    eq_x, eq_y, eq_w, eq_h = info_x + 32, artist_y + 125, info_w - 64, 96
    controls_y = bar_y + 58
    prev_x, play_x, next_x = bar_x + bar_w // 2 - 120, bar_x + bar_w // 2 - 20, bar_x + bar_w // 2 + 78

    draw = ImageDraw.Draw(Image.new("RGB", (4, 4), (0, 0, 0)))
    title_area_w = info_w - 64
    title_area_h = max(100, artist_y - title_y - 18)
    title_size, title_lines, _ = fit_title_block(
        draw=draw,
        text=title,
        font_loader=lambda size: _load_pillow_font(font_title_path, size),
        min_font=title_min_font,
        max_font=title_max_font,
        max_width=title_area_w,
        max_height=title_area_h,
        max_lines=title_max_lines,
        line_gap=10,
        stroke_width=2,
    )
    title_e = ffmpeg_escape_drawtext_text("\n".join(title_lines), preserve_newlines=True)
    title_line_spacing = max(6, int(title_size * 0.16))

    artist_size = artist_fs
    while artist_size > 20:
        if draw.textlength(artist, font=_load_pillow_font(font_body_path, artist_size)) <= title_area_w:
            break
        artist_size -= 1
    artist_text = truncate_with_ellipsis(draw, artist, _load_pillow_font(font_body_path, artist_size), title_area_w)
    artist_e = ffmpeg_escape_drawtext_text(artist_text)

    p_expr = f"(min(t\\,{dur:.6f})/{dur:.6f})"
    s_expr = f"(3*pow({p_expr}\\,2)-2*pow({p_expr}\\,3))"
    k = _clamp01(float(motion_smooth))
    ps_expr = f"((1-{k:.6f})*{p_expr}+{k:.6f}*{s_expr})"
    progress_x_expr = f"({bar_x}+{bar_w}*{ps_expr})"

    ks = max(0.4, min(2.0, float(knob_scale)))
    gw, gh = max(10, int(round(24 * ks))), max(12, int(round(32 * ks)))
    bw, bh = max(8, int(round(14 * ks))), max(10, int(round(22 * ks)))
    cw, ch = max(2, int(round(4 * ks))), max(6, int(round(14 * ks)))
    gdx, gdy = gw // 2, gh // 2 - 3
    bdx, bdy = bw // 2, bh // 2 - 2
    cdx, cdy = cw // 2, ch // 2 - 2

    font_title_ff = ffmpeg_escape_path(font_title_path)
    font_body_ff = ffmpeg_escape_path(font_body_path)

    chains = [
        "[0:v]split=2[cvsrc][bgsrc]",
        "[bgsrc]scale=1920:1080:force_original_aspect_ratio=increase:flags=lanczos,crop=1920:1080,format=rgba[bg_norm]",
        "[bg_norm]gblur=sigma=34,eq=brightness=-0.50:contrast=1.10:saturation=0.92[bg_blur]",
        "[bg_norm]gblur=sigma=58,eq=brightness=-0.30:contrast=1.02:saturation=1.25,colorchannelmixer=aa=0.22[bg_tint0]",
        f"[bg_tint0]drawbox=x=0:y=0:w=1920:h=1080:color=0x{mood_tint_hex}@0.16:t=fill[bg_tint]",
        "[bg_blur][bg_tint]overlay=0:0:format=auto[bg0]",
        "[bg0]vignette=PI/4:0.72[bg1]",
        f"[bg1]drawbox=x=0:y=0:w=1920:h=1080:color=0x{mood_tint_hex}@0.06:t=fill[bg2]",
        f"[bg2]drawbox=x=0:y=0:w=1920:h=260:color=0x{mood_bg_hex}@0.45:t=fill[bg3]",
        f"[bg3]drawbox=x=0:y=820:w=1920:h=260:color=0x{mood_bg_hex}@0.40:t=fill[bg4]",
        f"color=c=black@0.0:s=1920x1080,format=rgba,drawbox=x=24:y=32:w=320:h=180:color=0x{point_hex}@0.44:t=fill,drawbox=x=1460:y=110:w=360:h=220:color=0x{point_hex}@0.23:t=fill,drawbox=x=26:y=900:w=540:h=140:color=0x{point_hex}@0.34:t=fill[bg_point0]",
        "[bg_point0]gblur=sigma=38[bg_point]",
        "[bg4][bg_point]overlay=0:0:format=auto[bg_core]",
    ]

    _append_motion_chain(chains, input_label="bg_core", output_label="bg_motion", mode=camera_motion, strength=camera_strength)

    mode = (broll_blend or "softlight").strip().lower()
    if broll_enabled:
        chains.append("[2:v]scale=1920:1080:force_original_aspect_ratio=increase:flags=lanczos,crop=1920:1080,fps=30,format=rgba[broll_base]")
        if camera_motion == "parallax":
            _append_motion_chain(chains, input_label="broll_base", output_label="broll_motion", mode="drift", strength=min(1.0, camera_strength * 1.25), factor=1.25)
        else:
            _append_motion_chain(chains, input_label="broll_base", output_label="broll_motion", mode=camera_motion, strength=min(1.0, camera_strength * 1.08), factor=1.08)
        chains.append("[broll_motion]eq=brightness=-0.02:contrast=1.08:saturation=1.12[broll_fx]")
        op = _clamp01(broll_opacity)
        if mode == "normal":
            chains.append(f"[broll_fx]colorchannelmixer=aa={op:.3f}[broll_alpha]")
            chains.append("[bg_motion][broll_alpha]overlay=0:0:format=auto[bg_mix]")
        elif mode in ("overlay", "softlight", "screen"):
            chains.append(f"[bg_motion][broll_fx]blend=all_mode={mode}:all_opacity={op:.3f}[bg_mix]")
        else:
            chains.append(f"[bg_motion][broll_fx]blend=all_mode=softlight:all_opacity={op:.3f}[bg_mix]")
    else:
        chains.append("[bg_motion]null[bg_mix]")

    if color_filter_chain:
        chains.append(f"[bg_mix]{color_filter_chain}[bg_color]")
    else:
        chains.append("[bg_mix]null[bg_color]")
    if lut_path:
        lut_ff = ffmpeg_escape_path(str(lut_path))
        chains.extend([
            "[bg_color]split=2[bg_o][bg_l]",
            f"[bg_l]lut3d=file='{lut_ff}'[bg_lut]",
            f"[bg_o][bg_lut]blend=all_mode=normal:all_opacity={_clamp01(lut_intensity):.3f}[bg_grade]",
        ])
    else:
        chains.append("[bg_color]null[bg_grade]")

    rl, rg, rb, rs = _clamp01(reactive_level), _clamp01(reactive_glow), _clamp01(reactive_blur), _clamp01(reactive_shake)
    pulse_enable = _build_pulse_enable_expr(pulse_times, width=0.16, max_terms=42)
    pulse_gate = f"min(1\\,{pulse_enable})" if pulse_enable != "0" else "0"

    if rl > 0.0001:
        glow_a = 0.04 + rl * rg * 0.30
        chains.extend([
            f"[1:a]showspectrum=s=1920x1080:mode=combined:slide=scroll:color=rainbow:scale=lin,format=rgba,colorchannelmixer=aa={glow_a:.3f}[rx_spec0]",
            "[rx_spec0]gblur=sigma=22[rx_spec]",
            "[bg_grade][rx_spec]overlay=0:0:format=auto[rx_bg0]",
        ])
    else:
        chains.append("[bg_grade]null[rx_bg0]")

    if rl > 0.0001 and rb > 0.0001 and pulse_enable != "0":
        chains.extend([
            "[rx_bg0]split=2[rx_clean][rx_blur_src]",
            "[rx_blur_src]boxblur=luma_radius=8:luma_power=1:chroma_radius=6:chroma_power=1[rx_blur]",
            f"[rx_clean][rx_blur]overlay=0:0:enable='{pulse_enable}'[rx_bg1]",
        ])
    else:
        chains.append("[rx_bg0]null[rx_bg1]")

    if rl > 0.0001 and rs > 0.0001 and pulse_gate != "0":
        shake_px = min(4.4, 0.9 + rl * rs * 5.6)
        shake_alpha = min(0.30, 0.08 + rl * rs * 0.20)
        sx = f"{shake_px:.3f}*sin(43*t)*({pulse_gate})"
        sy = f"{(shake_px*0.72):.3f}*cos(37*t)*({pulse_gate})"
        chains.extend([
            "[rx_bg1]split=2[rx_main][rx_src]",
            f"[rx_src]colorchannelmixer=aa={shake_alpha:.3f}[rx_alpha]",
            f"[rx_main][rx_alpha]overlay=x='{sx}':y='{sy}':eval=frame[bg_react]",
        ])
    else:
        chains.append("[rx_bg1]null[bg_react]")

    chains.extend([
        f"[bg_react]drawbox=x={cover_x-24}:y={cover_y-24}:w={cover_size+48}:h={cover_size+48}:color=black@0.48:t=fill[plate]",
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
        f"color=c=0x{accent_hex}@0.32:s={gw}x{gh}[bar_knob_glow_src]",
        "[bar_knob_glow_src]gblur=sigma=2.2:steps=1[bar_knob_glow]",
        f"[bar_move][bar_knob_glow]overlay=x='{progress_x_expr}-{gdx}':y={bar_y-gdy}:eval=frame[bar_move2]",
        f"color=c=white@0.90:s={bw}x{bh}[bar_knob_body]",
        f"[bar_move2][bar_knob_body]overlay=x='{progress_x_expr}-{bdx}':y={bar_y-bdy}:eval=frame[bar_move3]",
        f"color=c=0x{accent_hex}@0.82:s={cw}x{ch}[bar_knob_core]",
        f"[bar_move3][bar_knob_core]overlay=x='{progress_x_expr}-{cdx}':y={bar_y-cdy}:eval=frame[base13]",
        f"[base13]drawtext=fontfile='{font_body_ff}':text='{elapsed_expr}':x={bar_x}:y={time_y}:fontsize=24:fontcolor=white@0.70[base14]",
        f"[base14]drawtext=fontfile='{font_body_ff}':text='{total_e}':x='{bar_x}+{bar_w}-tw':y={time_y}:fontsize=24:fontcolor=white@0.70[base15]",
        f"[base15]drawtext=fontfile='{font_title_ff}':text='<<':x={prev_x}:y={controls_y}:fontsize=34:fontcolor=white@0.58[base16]",
        f"[base16]drawtext=fontfile='{font_title_ff}':text='PLAY':x={play_x}:y={controls_y+2}:fontsize=28:fontcolor=white@0.92[base17]",
        f"[base17]drawtext=fontfile='{font_title_ff}':text='>>':x={next_x}:y={controls_y}:fontsize=34:fontcolor=white@0.58[preout]",
        "[preout]setsar=1[vout]",
    ])

    return ";".join(chains)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", default="")
    ap.add_argument("--cover", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--artist", default="Artist")
    ap.add_argument("--genre", default="")
    ap.add_argument("--tagline", default="")
    ap.add_argument("--skip-thumbnail", action="store_true")
    ap.add_argument("--style", choices=("youtube", "classic"), default="youtube")
    ap.add_argument("--quality", choices=tuple(QUALITY_PRESETS), default="high")

    ap.add_argument("--title-max-lines", type=int, default=3)
    ap.add_argument("--title-min-font", type=int, default=44)
    ap.add_argument("--title-max-font", type=int, default=82)
    ap.add_argument("--motion-smooth", type=float, default=0.18)
    ap.add_argument("--knob-scale", type=float, default=1.0)
    ap.add_argument("--font-title", default="")
    ap.add_argument("--font-body", default="")

    ap.add_argument("--broll-dir", default="")
    ap.add_argument("--broll-video", default="")
    ap.add_argument("--broll-opacity", type=float, default=0.32)
    ap.add_argument("--broll-blend", choices=("normal", "overlay", "softlight", "screen"), default="softlight")
    ap.add_argument("--beat-cut", choices=("auto", "off", "strict"), default="auto")
    ap.add_argument("--beat-sensitivity", type=float, default=0.62)
    ap.add_argument("--min-cut-interval", type=float, default=0.85)
    ap.add_argument("--max-cut-interval", type=float, default=2.40)
    ap.add_argument("--camera-motion", choices=("none", "drift", "zoom", "parallax"), default="drift")
    ap.add_argument("--camera-strength", type=float, default=0.18)
    ap.add_argument("--reactive-level", type=float, default=0.35)
    ap.add_argument("--reactive-glow", type=float, default=0.45)
    ap.add_argument("--reactive-blur", type=float, default=0.22)
    ap.add_argument("--reactive-shake", type=float, default=0.18)
    ap.add_argument("--color-preset", choices=("auto", "neutral", "cinema", "neon", "warm", "cool", "mono"), default="auto")
    ap.add_argument("--lut", default="")
    ap.add_argument("--lut-intensity", type=float, default=0.65)

    ap.add_argument("--outdir", default="output")
    args = ap.parse_args()

    try:
        if args.title_min_font > args.title_max_font:
            raise_user_error("INVALID_ARG", "--title-min-font cannot be greater than --title-max-font.")
        if args.knob_scale <= 0.0:
            raise_user_error("INVALID_ARG", "--knob-scale must be > 0.")
        if args.min_cut_interval > args.max_cut_interval:
            raise_user_error("INVALID_ARG", "--min-cut-interval cannot exceed --max-cut-interval.")
        for name in ("broll_opacity", "camera_strength", "reactive_level", "reactive_glow", "reactive_blur", "reactive_shake", "lut_intensity"):
            if not (0.0 <= float(getattr(args, name)) <= 1.0):
                raise_user_error("INVALID_ARG", f"--{name.replace('_', '-')} must be 0.0~1.0")

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

        try:
            ff_fonts = resolve_ffmpeg_font_paths(font_title=(args.font_title or None), font_body=(args.font_body or None))
            pillow_fonts = resolve_pillow_font_paths(font_title=(args.font_title or None), font_body=(args.font_body or None))
        except FontResolutionError as e:
            raise_user_error("FONT_RESOLUTION_FAILED", "Failed to resolve font paths.", details=str(e))

        outdir = Path(args.outdir).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)

        safe_title = sanitize_filename(title)
        out_path = outdir / f"{safe_title}.mp4"
        thumb_path = outdir / f"{safe_title}_youtube_thumb.jpg"
        graph_path = outdir / f"{safe_title}_filter.txt"
        stats_path = outdir / f"{safe_title}_astats.txt"
        quality = QUALITY_PRESETS[args.quality]

        need_analysis = args.reactive_level > 0.0 or args.beat_cut == "strict" or (args.beat_cut == "auto" and ((args.broll_dir or "").strip() or (args.broll_video or "").strip()))
        beat = {"cut_times": [], "pulse_times": []}
        if need_analysis:
            try:
                beat = analyze_audio_reactivity(
                    ffmpeg_path=ffp.ffmpeg,
                    song_path=song,
                    stats_path=stats_path,
                    sensitivity=args.beat_sensitivity,
                    min_cut_interval=args.min_cut_interval,
                    max_cut_interval=args.max_cut_interval,
                    duration=dur,
                )
            except Exception as e:
                if args.beat_cut == "strict":
                    raise_user_error("E_BEAT_ANALYSIS_FAILED", "Beat analysis failed in strict mode.", details=str(e))
                print(f"[WARN] E_BEAT_ANALYSIS_FAILED: {e}")

        cut_times_for_broll = []
        if args.beat_cut == "strict":
            cut_times_for_broll = beat.get("cut_times", [])
        elif args.beat_cut == "auto" and ((args.broll_dir or "").strip() or (args.broll_video or "").strip()):
            cut_times_for_broll = beat.get("cut_times", [])

        broll_prepared = None
        broll_concat = None
        if (args.broll_dir or "").strip():
            try:
                broll_prepared, broll_concat = _prepare_broll_from_dir(
                    ffp=ffp,
                    broll_dir=Path(args.broll_dir).expanduser(),
                    outdir=outdir,
                    safe_title=safe_title,
                    duration=dur,
                    cut_times=cut_times_for_broll,
                    min_cut_interval=args.min_cut_interval,
                    max_cut_interval=args.max_cut_interval,
                )
            except UserFacingError as e:
                print(f"[WARN] {e}\n[WARN] Falling back to cover-only background.")

        if broll_prepared is None and (args.broll_video or "").strip():
            try:
                broll_prepared = _prepare_broll_from_video(
                    ffp=ffp,
                    broll_video=Path(args.broll_video).expanduser(),
                    outdir=outdir,
                    safe_title=safe_title,
                    duration=dur,
                )
            except UserFacingError as e:
                print(f"[WARN] {e}\n[WARN] Falling back to cover-only background.")

        preset = resolve_color_preset(args.color_preset, cover)
        color_filter = preset_filter_chain(preset)

        lut_path = None
        if (args.lut or "").strip():
            cand = Path(args.lut).expanduser()
            if not cand.exists() or cand.suffix.lower() != ".cube":
                print(f"[WARN] E_LUT_INVALID: {cand} (fallback to preset-only)")
            else:
                lut_path = cand.resolve()

        graph = build_filter_graph(
            title=title,
            artist=artist,
            accent_hex=scheme.accent_hex,
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
            broll_enabled=(broll_prepared is not None),
            broll_blend=args.broll_blend,
            broll_opacity=args.broll_opacity,
            camera_motion=args.camera_motion,
            camera_strength=args.camera_strength,
            reactive_level=args.reactive_level,
            reactive_glow=args.reactive_glow,
            reactive_blur=args.reactive_blur,
            reactive_shake=args.reactive_shake,
            pulse_times=beat.get("pulse_times", []),
            color_filter_chain=color_filter,
            lut_path=lut_path,
            lut_intensity=args.lut_intensity,
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
        print("music-to-video (Python v14)")
        print(f"Song        : {song}")
        print(f"Cover       : {cover}")
        print(f"Title       : {title}")
        print(f"Artist      : {artist}")
        print(f"Style       : {args.style}")
        print(f"Quality     : {args.quality}")
        print(f"Dur(s)      : {dur:.3f}   Total: {total}")
        print(f"B-roll      : {broll_prepared if broll_prepared else '(disabled)'}")
        if broll_concat:
            print(f"B-roll map  : {broll_concat}")
        print(f"Beat cut    : {args.beat_cut} (cuts={len(beat.get('cut_times', []))})")
        print(f"Camera      : {args.camera_motion} ({args.camera_strength:.2f})")
        print(f"Reactive    : level={args.reactive_level:.2f} glow={args.reactive_glow:.2f} blur={args.reactive_blur:.2f} shake={args.reactive_shake:.2f}")
        print(f"Color       : {preset} (lut={lut_path if lut_path else 'none'})")
        print(f"Graph       : {graph_path}")
        print(f"Thumb       : {thumb_path} ({'generated' if thumb_generated else 'skipped'})")
        print(f"Out         : {out_path}")
        print("=====================================================")

        cmd = [ffp.ffmpeg, "-hide_banner", "-y", "-loop", "1", "-i", str(cover), "-i", str(song)]
        if broll_prepared is not None:
            cmd.extend(["-stream_loop", "-1", "-i", str(broll_prepared)])
        level_v = "4.2" if int(quality["fps"]) >= 60 else "4.1"
        cmd.extend([
            "-filter_complex", graph,
            "-map", "[vout]",
            "-map", "1:a",
            "-shortest",
            "-r", quality["fps"],
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            "-profile:v", "high",
            "-level:v", level_v,
            "-crf", quality["crf"],
            "-preset", quality["preset"],
            "-maxrate", quality["maxrate"],
            "-bufsize", quality["bufsize"],
            "-movflags", "+faststart",
            "-c:a", "aac",
            "-b:a", quality["audio_bitrate"],
            "-ar", "48000",
            str(out_path),
        ])

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

