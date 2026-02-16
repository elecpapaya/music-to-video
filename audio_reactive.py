#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio reactivity helpers:
- astats RMS extraction
- beat/cut time estimation
- pulse point extraction for visual effects
"""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path


RMS_KEY = "lavfi.astats.Overall.RMS_level"


def _db_to_linear(db: float) -> float:
    if math.isinf(db) and db < 0:
        return 0.0
    return 10.0 ** (db / 20.0)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    q = max(0.0, min(1.0, q))
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * q))
    return arr[idx]


def _ema(values: list[float], alpha: float = 0.26) -> list[float]:
    if not values:
        return []
    out: list[float] = []
    cur = values[0]
    for v in values:
        cur = alpha * v + (1.0 - alpha) * cur
        out.append(cur)
    return out


def _parse_astats_file(path: Path) -> tuple[list[float], list[float]]:
    pts_re = re.compile(r"pts_time[:=](?P<t>-?\d+(?:\.\d+)?)")
    rms_re = re.compile(rf"{re.escape(RMS_KEY)}[=:](?P<db>-?(?:inf|\d+(?:\.\d+)?))", re.IGNORECASE)

    times: list[float] = []
    energy: list[float] = []
    current_t: float | None = None

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m_t = pts_re.search(raw)
        if m_t:
            try:
                current_t = float(m_t.group("t"))
            except ValueError:
                current_t = None
            continue

        m_r = rms_re.search(raw)
        if not m_r:
            continue

        db_txt = m_r.group("db").strip().lower()
        db = float("-inf") if db_txt == "-inf" else float(db_txt)
        if current_t is None:
            continue

        times.append(max(0.0, current_t))
        energy.append(_db_to_linear(db))

    return times, energy


def _escape_ffmpeg_filter_path(path: Path) -> str:
    p = str(path.resolve()).replace("\\", "/")
    p = p.replace(":", r"\:")
    p = p.replace("'", r"\'")
    return p


def _extract_cut_times(
    times: list[float],
    energy: list[float],
    *,
    sensitivity: float,
    min_cut_interval: float,
    max_cut_interval: float,
    duration: float,
) -> list[float]:
    if not times or not energy:
        return []

    e_smooth = _ema(energy, alpha=0.24)
    diff: list[float] = [0.0]
    for i in range(1, len(e_smooth)):
        diff.append(max(0.0, e_smooth[i] - e_smooth[i - 1]))

    pos = [d for d in diff if d > 0.0]
    if not pos:
        return []

    sensitivity = max(0.2, min(1.2, sensitivity))
    q = max(0.40, min(0.90, 0.84 - (sensitivity - 0.2) * 0.38))
    thr = _quantile(pos, q)

    chosen: list[float] = []
    last_t = 0.0
    for i, d in enumerate(diff):
        t = times[i]
        if t < 0.2 or t > duration - 0.2:
            continue
        if d < thr:
            continue
        if t - last_t < min_cut_interval:
            continue
        chosen.append(t)
        last_t = t

    # Fill long gaps.
    out: list[float] = []
    prev = 0.0
    for t in chosen:
        while t - prev > max_cut_interval:
            prev += max_cut_interval
            out.append(prev)
        out.append(t)
        prev = t
    while duration - prev > max_cut_interval:
        prev += max_cut_interval
        if prev < duration - 0.2:
            out.append(prev)
        else:
            break

    return sorted(set(round(x, 3) for x in out if 0.2 < x < duration - 0.2))


def _extract_pulse_times(times: list[float], energy: list[float], duration: float) -> list[float]:
    if not times or not energy:
        return []

    e_smooth = _ema(energy, alpha=0.20)
    candidates: list[tuple[float, float]] = []
    for i in range(1, len(e_smooth) - 1):
        if e_smooth[i] > e_smooth[i - 1] and e_smooth[i] >= e_smooth[i + 1]:
            t = times[i]
            if 0.1 < t < duration - 0.1:
                candidates.append((e_smooth[i], t))

    if not candidates:
        return []

    candidates.sort(reverse=True)
    max_pulses = min(56, max(18, int(duration / 3.0)))
    pulse_times: list[float] = []
    for _, t in candidates:
        if any(abs(t - p) < 0.20 for p in pulse_times):
            continue
        pulse_times.append(round(t, 3))
        if len(pulse_times) >= max_pulses:
            break
    pulse_times.sort()
    return pulse_times


def analyze_audio_reactivity(
    *,
    ffmpeg_path: str,
    song_path: Path,
    stats_path: Path,
    sensitivity: float,
    min_cut_interval: float,
    max_cut_interval: float,
    duration: float,
) -> dict[str, list[float]]:
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-y",
        "-i",
        str(song_path),
        "-af",
        f"astats=metadata=1:reset=1,ametadata=print:file='{_escape_ffmpeg_filter_path(stats_path)}'",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg astats pass failed")
    if not stats_path.exists():
        raise RuntimeError("astats metadata file was not produced")

    times, energy = _parse_astats_file(stats_path)
    if not times:
        raise RuntimeError("could not parse RMS timeline from astats output")

    cuts = _extract_cut_times(
        times,
        energy,
        sensitivity=sensitivity,
        min_cut_interval=min_cut_interval,
        max_cut_interval=max_cut_interval,
        duration=duration,
    )
    pulses = _extract_pulse_times(times, energy, duration)
    return {"cut_times": cuts, "pulse_times": pulses}
