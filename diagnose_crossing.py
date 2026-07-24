#!/usr/bin/env python3
"""Mode-1 test: count crossing / pass-in-front ID swaps, main vs v2.

Read-only, diagnostic. A crossing swap is an ID teleporting onto the person it just crossed:
the tracker keeps the ID alive but reassigns it to the adjacent detection. From per-frame
boxes (render_frames.yaml) that shows up as a track's centroid JUMPING far in one frame while
another track was right next to it the frame before (the crossing context). No per-frame GT
identity is needed, and the same rule runs on both pipelines' artifacts, so the counts are
comparable. It is a proxy (a fast walker beside a crossing can trip it) but symmetric across
pipelines, which is what the comparison needs.

    python diagnose_crossing.py --run outputs/window_runs/win_old_crowded \
        --run outputs/window_runs/win_v2_crowded --jump 60 --near 110

Reports crossing-swap events per track-minute for each run.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import yaml


def center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def analyse(run_dir: Path, jump: float, near: float):
    art = yaml.safe_load((run_dir / "render_frames.yaml").read_text())
    fps = art["fps"]
    # frame -> {id: center}, customers only (kind 0); staff excluded from the count.
    frames: dict[int, dict[int, tuple]] = {}
    track_frames: dict[int, int] = defaultdict(int)
    for fi, rows in art["frames"].items():
        d = {}
        for tid, x1, y1, x2, y2, kind in rows:
            if kind == 0:
                d[tid] = center((x1, y1, x2, y2))
                track_frames[tid] += 1
        frames[fi] = d

    ordered = sorted(frames)
    swaps = 0
    events = []
    for a, b in zip(ordered, ordered[1:]):
        if b - a != 1:
            continue  # only consecutive frames
        prev, cur = frames[a], frames[b]
        for tid, (cx, cy) in cur.items():
            if tid not in prev:
                continue
            px, py = prev[tid]
            moved = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if moved < jump:
                continue
            # crossing context: another track was within `near` of this track's PREVIOUS
            # position the frame before (i.e. someone was beside it when the id jumped).
            neighbour = any(
                otid != tid and ((ox - px) ** 2 + (oy - py) ** 2) ** 0.5 < near
                for otid, (ox, oy) in prev.items()
            )
            if neighbour:
                swaps += 1
                events.append((b, tid, round(moved)))

    track_seconds = sum(track_frames.values()) / fps
    per_min = swaps / (track_seconds / 60) if track_seconds else 0.0
    return {
        "run": run_dir.name,
        "tracks": len(track_frames),
        "track_minutes": round(track_seconds / 60, 1),
        "crossing_swaps": swaps,
        "swaps_per_track_min": round(per_min, 2),
        "examples": events[:8],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--jump", type=float, default=60.0, help="min centroid jump (px)")
    parser.add_argument("--near", type=float, default=110.0, help="neighbour distance (px)")
    args = parser.parse_args()

    print(f"crossing-swap proxy (jump>{args.jump}px with a neighbour within {args.near}px)\n")
    print(f"{'run':24} {'tracks':>7} {'trk_min':>8} {'swaps':>6} {'per_trk_min':>12}")
    print("-" * 62)
    for run in args.run:
        r = analyse(run, args.jump, args.near)
        print(f"{r['run']:24} {r['tracks']:>7} {r['track_minutes']:>8} "
              f"{r['crossing_swaps']:>6} {r['swaps_per_track_min']:>12}")
        if r["examples"]:
            print(f"    e.g. (frame, id, jump_px): {r['examples'][:5]}")


if __name__ == "__main__":
    main()
