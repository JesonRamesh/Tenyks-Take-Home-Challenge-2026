#!/usr/bin/env python3
"""Test a candidate ROI polygon against a real frame and a real window, before adopting it.

Read-only, diagnostic. define_roi.py draws a polygon; this answers the question that
actually matters — *who changes status if I adopt it*. It renders one frame with the
current and candidate polygons overlaid and every box marked IN or OUT under the
candidate, then reports, across the whole window, how many in-zone frames each track
would gain or lose. A polygon that looks right on one frame can still delete a track
elsewhere, which is the mistake this is here to catch.

    python preview_roi.py --run-dir outputs/staff_thresh/staff05_crowded --frame 108637

Edit configs/roi_candidate.yaml between runs. Nothing here writes to a pipeline config:
adopting a polygon is a deliberate copy you make yourself.

Scope caveat, stated because it changes how the output should be read: boxes come from
the run's render artifact, which contains only detections that already *passed* the
current gate. So this shows what a candidate polygon would REMOVE, and cannot show what a
tighter current polygon is wrongly excluding today. Widening a polygon therefore reads as
"no change" here even when it would admit new boxes in a real run.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.overlay import style
from src.zones.roi import point_in_polygon

_CURRENT = (120, 120, 120)   # dim grey, the polygon in force today
_CANDIDATE = style.ROI_COLOR  # cyan, the polygon under test
_IN = (120, 220, 120)
_OUT = (60, 60, 235)


def passes(box: tuple[int, int, int, int], polygon: list, depth: float, aspect: float) -> bool:
    """The production zone test from src/zones/roi.py, evaluated against any polygon."""
    x1, y1, x2, y2 = box
    height = y2 - y1
    if height <= 0 or (x2 - x1) / height < aspect:
        return False
    cx = (x1 + x2) / 2
    return point_in_polygon(cx, y2, polygon) and point_in_polygon(cx, y2 - depth * height, polygon)


def max_depth_inside(box: tuple[int, int, int, int], polygon: list) -> float:
    """Largest box_depth_frac at which this box still passes — its margin to the edge."""
    x1, y1, x2, y2 = box
    height = y2 - y1
    cx = (x1 + x2) / 2
    if height <= 0 or not point_in_polygon(cx, y2, polygon):
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(30):
        mid = (low + high) / 2
        if point_in_polygon(cx, y2 - mid * height, polygon):
            low = mid
        else:
            high = mid
    return low


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("digital_kiosk.mp4"))
    parser.add_argument("--candidate", type=Path, default=Path("configs/roi_candidate.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/cam1_v2.yaml"),
                        help="config holding the polygon currently in force, for comparison")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--out", type=Path, default=Path("outputs/roi_candidate_preview.png"))
    args = parser.parse_args()

    candidate = yaml.safe_load(args.candidate.read_text())
    cand_poly = [tuple(p) for p in candidate["roi_polygon"]]
    depth = candidate["box_depth_frac"]
    aspect = candidate["min_box_aspect"]
    current = yaml.safe_load(args.config.read_text())["kiosk_roi"]
    curr_poly = [tuple(p) for p in current["roi_polygon"]]
    curr_depth = current["box_depth_frac"]

    artifact = yaml.safe_load((args.run_dir / "render_frames.yaml").read_text())
    frames = artifact["frames"]
    if args.frame not in frames:
        raise SystemExit(f"frame {args.frame} has no boxes in {args.run_dir}/render_frames.yaml")

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {args.frame}")

    cv2.polylines(frame, [np.array(curr_poly, np.int32)], True, _CURRENT, 2, cv2.LINE_AA)
    style.fill_polygon(frame, np.array(cand_poly, np.int32), _CANDIDATE, alpha=0.12)
    cv2.polylines(frame, [np.array(cand_poly, np.int32)], True, _CANDIDATE, 2, cv2.LINE_AA)

    text = style.TextLayer()
    font = style.load_font(15, bold=True)
    small = style.load_font(13, bold=True)
    print(f"\nframe {args.frame}: box status under the candidate polygon")
    print(f"{'track':>7} {'kind':>7} {'now':>5} {'cand':>5} {'max_depth':>10}")
    for track_id, x1, y1, x2, y2, kind in sorted(frames[args.frame]):
        box = (x1, y1, x2, y2)
        now = passes(box, curr_poly, curr_depth, aspect)
        cand = passes(box, cand_poly, depth, aspect)
        color = _IN if cand else _OUT
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        # The two points the gate actually tests: the feet anchor and the depth point.
        cx = (x1 + x2) // 2
        cv2.circle(frame, (cx, y2), 4, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, int(y2 - depth * (y2 - y1))), 4, color, -1, cv2.LINE_AA)
        cv2.line(frame, (cx, y2), (cx, int(y2 - depth * (y2 - y1))), color, 1, cv2.LINE_AA)
        tag = "STAFF" if kind == 1 else f"ID {track_id}"
        label = f"{tag}  {'IN' if cand else 'OUT'}"
        size = style.text_size(label, font)
        style.rounded_rect(frame, (x1, y1 - size[1] - 10), (x1 + size[0] + 14, y1 - 2),
                           color, radius=5, thickness=-1, alpha=0.85)
        text.add((x1 + 7, y1 - size[1] - 6), label, font, (255, 255, 255))
        print(f"{tag:>7} {'staff' if kind == 1 else 'cust':>7} "
              f"{'IN' if now else 'OUT':>5} {'IN' if cand else 'OUT':>5} "
              f"{max_depth_inside(box, cand_poly):>10.3f}")

    text.add((16, 16), "grey = current polygon    cyan = candidate", small, (235, 235, 235))
    frame = text.flush(frame)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), frame)

    # Window-wide impact: a polygon that looks right on one frame can still starve a track
    # of the frames it needs to survive the stationarity gate.
    kept: dict[int, int] = defaultdict(int)
    lost: dict[int, int] = defaultdict(int)
    for rows in frames.values():
        for track_id, x1, y1, x2, y2, _kind in rows:
            if passes((x1, y1, x2, y2), cand_poly, depth, aspect):
                kept[track_id] += 1
            else:
                lost[track_id] += 1
    print(f"\nacross the whole window ({artifact['start_frame']}-{artifact['end_frame']}):")
    print(f"{'track':>7} {'kept':>7} {'lost':>7} {'kept_s':>8}  note")
    fps = artifact["fps"]
    for track_id in sorted(set(kept) | set(lost)):
        k, l = kept[track_id], lost[track_id]
        note = "TRACK DISAPPEARS" if k == 0 else ("under 3s: stationarity gate may drop it"
                                                 if k / fps < 3.0 else "")
        print(f"{track_id:>7} {k:>7} {l:>7} {k / fps:>8.1f}  {note}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
