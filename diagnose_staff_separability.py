#!/usr/bin/env python3
"""Is the staff heuristic separable at all? Score every GT customer and every staff sighting.

Read-only, diagnostic. The staff filter's threshold question ("which min_staff_frame_frac
works") is only meaningful if the two populations separate in the first place. This scores
the per-frame staff heuristic across every GT customer and every known staff sighting and
prints the two distributions side by side, so a threshold choice is made against real
overlap rather than against one hand-picked track.

Attribution problem: GT is temporal only (person_id, enter, exit) with no boxes, so a crop
cannot be tied to a person in a crowd. Solved by scoring only **solo frames** — frames where
exactly one subject is present and the detector returns exactly one in-zone box. That box is
unambiguously the subject. Subjects without enough solo frames are reported as such rather
than guessed at.

    python diagnose_staff_separability.py --config configs/cam1_v2.yaml --samples 120
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.detect.build import build_detector
from src.device import resolve_device
from src.staff.filter import StaffClassifier
from src.zones.roi import in_zone
from src.track.base import Track


def _intervals(path: Path, key: str) -> dict[str, list[tuple[int, int]]]:
    rows = yaml.safe_load(path.read_text())
    rows = rows["events"] if isinstance(rows, dict) else rows
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        out[str(row[key])].append((row["enter_frame"], row["exit_frame"]))
    return dict(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("digital_kiosk.mp4"))
    parser.add_argument("--config", type=Path, default=Path("configs/cam1_v2.yaml"))
    parser.add_argument("--gt", type=Path, default=Path("eval/label/kiosk_gt.yaml"))
    parser.add_argument("--staff-intervals", type=Path, default=Path("outputs/staff.yaml"))
    # Extra staff sightings not in staff.yaml, as name=start:end (e.g. staff-680's window).
    parser.add_argument("--extra-staff", action="append", default=[])
    parser.add_argument("--samples", type=int, default=120, help="frames sampled per subject")
    parser.add_argument("--out", type=Path, default=Path("outputs/staff_separability.json"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    roi = config["kiosk_roi"]
    polygon = [tuple(p) for p in roi["roi_polygon"]]
    depth, aspect = roi["box_depth_frac"], roi["min_box_aspect"]
    staff_clf = StaffClassifier(config["staff"])
    detector = build_detector(config["detector"], resolve_device(config["detector"]["device"]))

    customers = _intervals(args.gt, "person_id")
    staff = _intervals(args.staff_intervals, "track_id") if args.staff_intervals.exists() else {}
    staff = {f"staff-{k}": v for k, v in staff.items()}
    for spec in args.extra_staff:
        name, span = spec.split("=", 1)
        start, end = span.split(":")
        staff[name] = [(int(start), int(end))]

    # A frame is "solo" for a subject if no OTHER subject's interval covers it. Staff and
    # customers are pooled here on purpose: a customer standing beside staff is exactly the
    # case that makes attribution ambiguous, so those frames are excluded from both.
    everyone = {**{f"cust-{k}": v for k, v in customers.items()}, **staff}

    def covers(spans: list[tuple[int, int]], frame: int) -> bool:
        return any(a <= frame < b for a, b in spans)

    cap = cv2.VideoCapture(str(args.video))
    results: dict[str, dict] = {}
    for name, spans in sorted(everyone.items()):
        total = sum(b - a for a, b in spans)
        # Sample evenly across the subject's own frames, then keep only the solo ones.
        candidates: list[int] = []
        step = max(1, total // max(args.samples * 3, 1))
        for a, b in spans:
            candidates.extend(range(a, b, step))
        others = [s for other, s in everyone.items() if other != name]
        solo = [f for f in candidates if not any(covers(s, f) for s in others)]

        scored = hits = 0
        for frame_index in solo:
            if scored >= args.samples:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            boxes = [
                (d.x1, d.y1, d.x2, d.y2)
                for d in detector.detect(frame)
                if in_zone(Track(0, d.x1, d.y1, d.x2, d.y2, d.confidence), polygon, depth, aspect)
            ]
            # Exactly one in-zone box => unambiguously this subject.
            if len(boxes) != 1:
                continue
            scored += 1
            hits += int(staff_clf.staff_frames(frame, boxes)[0])

        results[name] = {
            "kind": "staff" if name in staff else "customer",
            "gt_frames": total,
            "solo_candidates": len(solo),
            "scored_frames": scored,
            "staff_frames": hits,
            "staff_frac": round(hits / scored, 4) if scored else None,
        }
        print(f"{name:16} {results[name]['kind']:9} scored {scored:>4} "
              f"frac {results[name]['staff_frac']}", flush=True)
    cap.release()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    usable = {k: v for k, v in results.items() if v["scored_frames"] >= 10}
    cust = sorted((v["staff_frac"], k) for k, v in usable.items() if v["kind"] == "customer")
    stf = sorted((v["staff_frac"], k) for k, v in usable.items() if v["kind"] == "staff")
    print("\n" + "=" * 70)
    print(f"SEPARABILITY  ({len(cust)} customers, {len(stf)} staff sightings with >=10 scored frames)")
    print("=" * 70)
    print("customers (staff_frac ascending):")
    for frac, name in cust:
        print(f"   {name:16} {frac:.3f}")
    print("staff sightings (staff_frac ascending):")
    for frac, name in stf:
        print(f"   {name:16} {frac:.3f}")
    if cust and stf:
        print(f"\nhighest customer {cust[-1][1]} {cust[-1][0]:.3f} | lowest staff {stf[0][1]} {stf[0][0]:.3f}")
        if cust[-1][0] < stf[0][0]:
            print(f"SEPARABLE: any threshold in ({cust[-1][0]:.3f}, {stf[0][0]:.3f}] splits them cleanly.")
        else:
            inverted = [(f, n) for f, n in cust if f >= stf[0][0]]
            print(f"NOT SEPARABLE: {len(inverted)} customer(s) score >= the lowest staff sighting: "
                  f"{[n for _, n in inverted]}")


if __name__ == "__main__":
    main()
