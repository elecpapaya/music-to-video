#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a glass-style YouTube-ready player video from audio + cover image."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from color_presets import preset_filter_chain, resolve_color_preset
from font_resolver import FontResolutionError, resolve_ffmpeg_font_paths, resolve_pillow_font_paths
from make_thumbnail_glass import make_thumbnail
from player_core import (
    AUDIO_EXTS,
    IMAGE_EXTS,
    QUALITY_PRESETS,
    UserFacingError,
    cover_color_scheme,
    ff_paths,
    ffmpeg_escape_drawtext_text,
    ffmpeg_escape_path,
    format_mmss,
    get_duration_seconds,
    raise_user_error,
    resolve_media_path,
    sanitize_filename,
)
from text_layout import fit_title_block, truncate_with_ellipsis


TITLE_MAX_LINES = 3
TITLE_MIN_FONT = 44
TITLE_MAX_FONT = 86
THUMB_TITLE_MAX_LINES = 5
THUMB_TITLE_MIN_FONT = 40
THUMB_TITLE_MAX_FONT = 102


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _load_pillow_font(font_path: str | None, size: int) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            return ImageFont.load_default()
    return ImageFont.load_default()


def _append_motion_chain(chains: list[str], *, input_label: str, output_label: str, mode: str, strength: float) -> None:
    s = _clamp01(strength)
    if mode == "none" or s <= 0.001:
        chains.append(f"[{input_label}]null[{output_label}]")
        return
    if mode == "drift":
        ax, ay = max(1, int(round(26.0 * s))), max(1, int(round(16.0 * s)))
        chains.append(
            f"[{input_label}]scale=2060:1168:flags=lanczos,crop=1920:1080:x='(iw-1920)/2+{ax}*sin(t*0.07)':y='(ih-1080)/2+{ay}*cos(t*0.05)'[{output_label}]"
        )
        return
    if mode == "zoom":
        amp = max(0.01, 0.085 * s)
        cw = f"1920-240*{amp:.5f}*(0.5+0.5*sin(t*0.06))"
        ch = f"1080-136*{amp:.5f}*(0.5+0.5*sin(t*0.06))"
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
    total_mmss: str,
    dur: float,
    font_title_path: str,
    font_body_path: str,
    motion_smooth: float,
    camera_motion: str,
    camera_strength: float,
    reactive_level: float,
    reactive_glow: float,
    color_filter_chain: str,
    lut_path: Path | None,
    lut_intensity: float,
) -> str:
    panel_x, panel_y, panel_w, panel_h = 84, 84, 752, 912
    panel_pad = 46
    title_x, title_y = panel_x + panel_pad, panel_y + 146
    title_area_w = panel_w - panel_pad * 2
    title_area_h = 336

    cover_x, cover_y, cover_size = 930, 138, 804

    spectrum_x, spectrum_y, spectrum_w, spectrum_h = title_x, panel_y + 560, title_area_w, 124
    hbar_x, hbar_y, hbar_w = title_x, panel_y + 820, title_area_w
    controls_y = hbar_y + 44

    draw = ImageDraw.Draw(Image.new("RGB", (4, 4), (0, 0, 0)))
    title_size, title_lines, _ = fit_title_block(
        draw=draw,
        text=title,
        font_loader=lambda size: _load_pillow_font(font_title_path, size),
        min_font=TITLE_MIN_FONT,
        max_font=TITLE_MAX_FONT,
        max_width=title_area_w,
        max_height=title_area_h,
        max_lines=TITLE_MAX_LINES,
        line_gap=10,
        stroke_width=2,
    )
    font_title_ff = ffmpeg_escape_path(font_title_path)
    font_body_ff = ffmpeg_escape_path(font_body_path)
    title_line_spacing = max(8, int(title_size * 0.17))
    escaped_title_lines = [ffmpeg_escape_drawtext_text(line) for line in title_lines if line]
    if not escaped_title_lines:
        escaped_title_lines = [ffmpeg_escape_drawtext_text(title)]

    # Avoid embedded "\n" in drawtext text; render each title line as a separate drawtext step.
    title_draw_filters: list[str] = []
    title_in = "p6"
    for idx, line in enumerate(escaped_title_lines):
        out_label = "p7" if idx == len(escaped_title_lines) - 1 else f"pt{idx}"
        y_line = title_y + idx * (title_size + title_line_spacing)
        title_draw_filters.append(
            f"[{title_in}]drawtext=fontfile='{font_title_ff}':text='{line}':x={title_x}:y={y_line}:"
            f"fontsize={title_size}:fontcolor=white@0.99:borderw=2:bordercolor=black@0.28[{out_label}]"
        )
        title_in = out_label
    title_draw_chain = ";".join(title_draw_filters)

    artist_size = 42
    while artist_size > 22:
        if draw.textlength(artist, font=_load_pillow_font(font_body_path, artist_size)) <= title_area_w:
            break
        artist_size -= 1
    artist_text = truncate_with_ellipsis(draw, artist, _load_pillow_font(font_body_path, artist_size), title_area_w)
    artist_e = ffmpeg_escape_drawtext_text(artist_text)
    artist_y = title_y + 356

    elapsed_expr = r"%{eif\:t/60\:d\:2}\:%{eif\:mod(t\,60)\:d\:2}"
    total_e = ffmpeg_escape_drawtext_text(total_mmss)

    p_expr = f"(min(t\\,{dur:.6f})/{dur:.6f})"
    s_expr = f"(3*pow({p_expr}\\,2)-2*pow({p_expr}\\,3))"
    k = _clamp01(float(motion_smooth))
    ps_expr = f"((1-{k:.6f})*{p_expr}+{k:.6f}*{s_expr})"
    progress_x_expr = f"({hbar_x}+{hbar_w}*{ps_expr})"
    rl = _clamp01(reactive_level)
    rg = _clamp01(reactive_glow)
    eq_alpha = min(0.48, 0.18 + rl * 0.22 + rg * 0.08)

    chains = [
        "[0:v]split=2[cvsrc][bgsrc]",
        "[bgsrc]scale=1920:1080:force_original_aspect_ratio=increase:flags=lanczos,crop=1920:1080,format=rgba[bg0]",
        "[bg0]gblur=sigma=40,eq=brightness=-0.58:contrast=1.14:saturation=0.66[bg1]",
        f"[bg1]drawbox=x=0:y=0:w=1920:h=1080:color=0x{mood_tint_hex}@0.14:t=fill[bg2]",
        f"[bg2]drawbox=x=0:y=0:w=1920:h=1080:color=black@0.28:t=fill[bg3]",
        f"[bg3]drawbox=x=0:y=0:w=690:h=1080:color=black@0.26:t=fill[bg4]",
        f"[bg4]drawbox=x=740:y=0:w=8:h=1080:color=0x{accent_hex}@0.22:t=fill[bg5]",
        f"[bg5]drawbox=x=0:y=0:w=1920:h=210:color=0x{mood_bg_hex}@0.55:t=fill[bg6]",
        f"[bg6]drawbox=x=0:y=842:w=1920:h=238:color=0x{mood_bg_hex}@0.60:t=fill[bg7]",
        "[bg7]drawgrid=width=96:height=96:thickness=1:color=white@0.05[bg8]",
        "[bg8]vignette=PI/3:0.72[bg9]",
    ]

    _append_motion_chain(chains, input_label="bg9", output_label="bg_motion", mode=camera_motion, strength=camera_strength)

    if color_filter_chain:
        chains.append(f"[bg_motion]{color_filter_chain}[bg_color]")
    else:
        chains.append("[bg_motion]null[bg_color]")

    if lut_path:
        lut_ff = ffmpeg_escape_path(str(lut_path))
        chains.extend(
            [
                "[bg_color]split=2[bg_o][bg_l]",
                f"[bg_l]lut3d=file='{lut_ff}'[bg_lut]",
                f"[bg_o][bg_lut]blend=all_mode=normal:all_opacity={_clamp01(lut_intensity):.3f}[bg_grade]",
            ]
        )
    else:
        chains.append("[bg_color]null[bg_grade]")

    if rl > 0.0001:
        alpha = min(0.22, 0.02 + rl * rg * 0.18)
        chains.extend(
            [
                f"[1:a]showspectrum=s=1920x1080:mode=separate:slide=scroll:color=rainbow:scale=lin,format=rgba,colorchannelmixer=aa={alpha:.3f}[rx0]",
                "[rx0]gblur=sigma=24[rx1]",
                "[bg_grade][rx1]overlay=0:0:format=auto[bg_react]",
            ]
        )
    else:
        chains.append("[bg_grade]null[bg_react]")

    chains.extend(
        [
            f"[cvsrc]scale={cover_size}:{cover_size}:force_original_aspect_ratio=increase:flags=lanczos,crop={cover_size}:{cover_size},format=rgba[cv0]",
            "[cv0]eq=contrast=1.06:saturation=1.08[cv]",
            f"[bg_react]drawbox=x={cover_x-34}:y={cover_y-34}:w={cover_size+68}:h={cover_size+68}:color=black@0.46:t=fill[s0]",
            "[s0]boxblur=20:2[s1]",
            f"[s1][cv]overlay={cover_x}:{cover_y}:format=auto[s2]",
            f"[s2]drawbox=x={cover_x-4}:y={cover_y-4}:w={cover_size+8}:h={cover_size+8}:color=white@0.22:t=2[s3]",
            f"[s3]drawbox=x={cover_x-22}:y={cover_y-22}:w={cover_size+44}:h={cover_size+44}:color=0x{accent_hex}@0.34:t=2[s4]",
            f"[s4]drawbox=x={panel_x}:y={panel_y}:w={panel_w}:h={panel_h}:color=black@0.42:t=fill[p0]",
            f"[p0]drawbox=x={panel_x}:y={panel_y}:w={panel_w}:h=2:color=0x{accent_hex}@0.90:t=fill[p1]",
            f"[p1]drawbox=x={panel_x}:y={panel_y}:w=2:h={panel_h}:color=0x{accent_hex}@0.76:t=fill[p2]",
            f"[p2]drawbox=x={panel_x}:y={panel_y+96}:w={panel_w}:h=1:color=white@0.20:t=fill[p3]",
            f"[p3]drawbox=x={panel_x+24}:y={panel_y+25}:w=14:h=14:color=0x{accent_hex}@0.94:t=fill[p4]",
            f"[p4]drawtext=fontfile='{font_body_ff}':text='LIVE SESSION':x={panel_x+50}:y={panel_y+22}:fontsize=28:fontcolor=white@0.84[p5]",
            f"[p5]drawtext=fontfile='{font_body_ff}':text='SYSTEM 02':x={panel_x+512}:y={panel_y+26}:fontsize=20:fontcolor=white@0.48[p6]",
            title_draw_chain,
            f"[p7]drawtext=fontfile='{font_body_ff}':text='{artist_e}':x={title_x}:y={artist_y}:fontsize={artist_size}:fontcolor=white@0.84[p8]",
            f"[p8]drawbox=x={title_x}:y={artist_y+76}:w=278:h=4:color=0x{accent_hex}@0.88:t=fill[p9]",
            f"color=c=black@0.0:s={spectrum_w}x{spectrum_h},format=rgba,drawbox=x=0:y=0:w={spectrum_w}:h={spectrum_h}:color=white@0.06:t=fill[sp0]",
            f"[1:a]showfreqs=s={spectrum_w}x{spectrum_h}:mode=bar:fscale=log:ascale=sqrt:win_func=hann:colors=white,format=rgba,colorkey=0x000000:0.10:0.0[sp1]",
            f"[sp1]gblur=sigma=1.2,colorchannelmixer=aa={eq_alpha:.3f}[sp2]",
            f"[sp0][sp2]overlay=0:0:format=auto[sp3]",
            f"[sp3]drawbox=x=0:y=0:w={spectrum_w}:h=1:color=white@0.16:t=fill[sp4]",
            f"[sp4]drawbox=x=0:y={spectrum_h-1}:w={spectrum_w}:h=1:color=white@0.12:t=fill[sp5]",
            f"[p9][sp5]overlay={spectrum_x}:{spectrum_y}:format=auto[c0]",
            f"[c0]drawbox=x={hbar_x}:y={hbar_y}:w={hbar_w}:h=6:color=white@0.18:t=fill[c1]",
            f"color=c=black@0.0:s={hbar_w}x6,format=rgba[pbar0]",
            f"color=c=0x{accent_hex}@0.86:s={hbar_w}x6,format=rgba[pbar1]",
            f"[pbar0][pbar1]overlay=x='({hbar_w}*{ps_expr}-{hbar_w}+1)':y=0:eval=frame[pbar2]",
            f"[c1][pbar2]overlay=x={hbar_x}:y={hbar_y}:eval=frame[c3]",
            f"color=c=0x{accent_hex}@0.32:s=20x20[hkg0]",
            "[hkg0]gblur=sigma=2.3[hkg]",
            f"[c3][hkg]overlay=x='{progress_x_expr}-10':y={hbar_y-8}:eval=frame[c4]",
            "color=c=white@0.96:s=10x10[hkb]",
            f"[c4][hkb]overlay=x='{progress_x_expr}-5':y={hbar_y-2}:eval=frame[c5]",
            f"[c5]drawtext=fontfile='{font_body_ff}':text='{elapsed_expr}':x={hbar_x}:y={hbar_y+16}:fontsize=24:fontcolor=white@0.84[d0]",
            f"[d0]drawtext=fontfile='{font_body_ff}':text='{total_e}':x='{hbar_x}+{hbar_w}-tw':y={hbar_y+16}:fontsize=24:fontcolor=white@0.74[d1]",
            f"[d1]drawtext=fontfile='{font_body_ff}':text='PREV':x={hbar_x}:y={controls_y}:fontsize=20:fontcolor=white@0.56[d2]",
            f"[d2]drawtext=fontfile='{font_body_ff}':text='PLAY':x='{hbar_x}+{hbar_w}/2-tw/2':y={controls_y}:fontsize=22:fontcolor=white@0.92[d3]",
            f"[d3]drawtext=fontfile='{font_body_ff}':text='NEXT':x='{hbar_x}+{hbar_w}-tw':y={controls_y}:fontsize=20:fontcolor=white@0.56[d4]",
            "[d4]scale=1920:1080:flags=lanczos,setsar=1[vout]",
        ]
    )

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
    ap.add_argument("--quality", choices=tuple(QUALITY_PRESETS), default="high")
    ap.add_argument("--motion-smooth", type=float, default=0.20)
    ap.add_argument("--font-title", default="")
    ap.add_argument("--font-body", default="")
    ap.add_argument("--camera-motion", choices=("none", "drift", "zoom"), default="drift")
    ap.add_argument("--camera-strength", type=float, default=0.18)
    ap.add_argument("--reactive-level", type=float, default=0.24)
    ap.add_argument("--reactive-glow", type=float, default=0.34)
    ap.add_argument("--color-preset", choices=("auto", "neutral", "cinema", "neon", "warm", "cool", "mono"), default="auto")
    ap.add_argument("--lut", default="")
    ap.add_argument("--lut-intensity", type=float, default=0.60)
    ap.add_argument("--outdir", default="output")
    args = ap.parse_args()

    try:
        for name in ("motion_smooth", "camera_strength", "reactive_level", "reactive_glow", "lut_intensity"):
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
        out_path = outdir / f"{safe_title}_glass.mp4"
        thumb_path = outdir / f"{safe_title}_glass_thumb.jpg"
        graph_path = outdir / f"{safe_title}_glass_filter.txt"
        quality = QUALITY_PRESETS[args.quality]

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
            total_mmss=total,
            dur=dur,
            font_title_path=ff_fonts["title"],
            font_body_path=ff_fonts["body"],
            motion_smooth=args.motion_smooth,
            camera_motion=args.camera_motion,
            camera_strength=args.camera_strength,
            reactive_level=args.reactive_level,
            reactive_glow=args.reactive_glow,
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
                    title_max_lines=THUMB_TITLE_MAX_LINES,
                    title_min_font=THUMB_TITLE_MIN_FONT,
                    title_max_font=THUMB_TITLE_MAX_FONT,
                    font_title_path=pillow_fonts["title"],
                    font_body_path=pillow_fonts["body"],
                )
                thumb_generated = True
            except Exception as e:
                print(f"[WARN] thumbnail generation failed: {e}")

        print("=====================================================")
        print("music-to-video (Glass)")
        print(f"Song        : {song}")
        print(f"Cover       : {cover}")
        print(f"Title       : {title}")
        print(f"Artist      : {artist}")
        print(f"Quality     : {args.quality}")
        print(f"Dur(s)      : {dur:.3f}   Total: {total}")
        print(f"Camera      : {args.camera_motion} ({args.camera_strength:.2f})")
        print(f"Reactive    : level={args.reactive_level:.2f} glow={args.reactive_glow:.2f}")
        print(f"Color       : {preset} (lut={lut_path if lut_path else 'none'})")
        print(f"Graph       : {graph_path}")
        print(f"Thumb       : {thumb_path} ({'generated' if thumb_generated else 'skipped'})")
        print(f"Out         : {out_path}")
        print("=====================================================")

        level_v = "4.2" if int(quality["fps"]) >= 60 else "4.1"
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
            level_v,
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
