#!/usr/bin/env python3
"""Phase A: measure same-vs-different appearance separation for candidate ReID backbones.

Read-only, diagnostic. The ID-switching / wrong-merge problem is precision-bounded by one
quantity: can a backbone tell two crops of the same person apart from two of different
people, on THIS top-down camera? This measures exactly that (the Phase 5 figure of merit),
sweeping the backbone instead of the threshold.

The first crowded-window pass gave a flat ~0 margin for every backbone, which is a red flag
for the INSTRUMENT: in a crowd, temporal-only GT cannot say which co-present person a track
is (label noise), and crops are occluded (garbage in). Both collapse the measured margin
toward zero regardless of backbone quality. So this version adds the decisive control:

  --solo-frac: restrict same/different pairs to SOLO-present tracks — tracks whose frames are
  mostly frames where GT says exactly one person is in the window. Solo => the label is
  unambiguous AND the crop is unoccluded, i.e. the fair test for ReID. Pool several windows
  (each contributing a different solo person) so there are cross-person pairs to compare.

    python diagnose_reid.py --config configs/cam1.yaml \
        --window 85000 91000 --window 135404 155000 --window 26700 41960 \
        --backbones weights/osnet_x0_25_msmt17.pt weights/osnet_x1_0_msmt17.pt \
        --solo-frac 0.8 --out outputs/reid_sep_yolo_solo.json

Reports, per backbone, the margin (same_p10 - diff_p90) on ALL labelled pairs and on
SOLO-only pairs. If the solo margin flips clearly positive while the all-pairs one is
negative, the crowded null was the confound and a better backbone can help. If the solo
margin is ALSO ~0/negative, the limit is the viewpoint, not compute — and we stop.
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
from src.reid.embed import Embedder
from src.zones.roi import in_zone
from src.track.base import Track


def _build_tracker(tracker_cfg: dict, device: str):
    if tracker_cfg["type"] == "bytetrack":
        from src.track.bytetrack import ByteTrackTracker

        return ByteTrackTracker(**{k: tracker_cfg[k] for k in (
            "track_high_thresh", "track_low_thresh", "new_track_thresh",
            "match_thresh", "track_buffer", "fuse_score")})
    from src.track.boxmot_tracker import BoxmotTracker

    return BoxmotTracker(tracker_cfg["type"], device, tracker_cfg["half"],
                         tracker_cfg.get("reid_weights"), tracker_cfg.get("params"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("digital_kiosk.mp4"))
    parser.add_argument("--config", required=True, type=Path)
    # Repeatable: pool several windows so different solo people give cross-person pairs.
    parser.add_argument("--window", nargs=2, type=int, action="append", required=True,
                        metavar=("START", "END"))
    parser.add_argument("--gt", type=Path, default=Path("eval/label/kiosk_gt.yaml"))
    parser.add_argument("--backbones", nargs="+", required=True)
    parser.add_argument("--stride", type=int, default=4, help="embed every Nth in-zone frame")
    parser.add_argument("--min-track-frames", type=int, default=20)
    parser.add_argument("--solo-frac", type=float, default=0.8,
                        help="a track is 'solo' if >= this share of its frames are solo-present")
    parser.add_argument("--out", type=Path, default=Path("outputs/reid_separation.json"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    roi = config["kiosk_roi"]
    polygon = [tuple(p) for p in roi["roi_polygon"]]
    depth, aspect = roi["box_depth_frac"], roi["min_box_aspect"]
    device = resolve_device(config["detector"]["device"])

    raw_gt = yaml.safe_load(args.gt.read_text())["events"]

    def present(f: int) -> list[str]:
        return [e["person_id"] for e in raw_gt if e["enter_frame"] <= f < e["exit_frame"]]

    embedders = {Path(w).stem: Embedder(w, device) for w in args.backbones}

    # Per GLOBAL track key (window_idx, track_id): frames, solo-frame count, and per-backbone
    # running embedding sum + count. One detect+track+embed pass per window.
    track_frames: dict[tuple, list[int]] = defaultdict(list)
    track_solo: dict[tuple, int] = defaultdict(int)
    emb_sum: dict[str, dict[tuple, np.ndarray]] = {n: {} for n in embedders}
    emb_cnt: dict[tuple, int] = defaultdict(int)

    for wi, (start, end) in enumerate(args.window):
        detector = build_detector(config["detector"], device)
        tracker = _build_tracker(config["tracker"], device)
        cap = cv2.VideoCapture(str(args.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        fi = start
        while fi < end:
            ok, frame = cap.read()
            if not ok:
                break
            n_present = len(present(fi))
            zone = [t for t in tracker.update(detector.detect(frame), frame, fi)
                    if in_zone(t, polygon, depth, aspect)]
            for t in zone:
                key = (wi, t.track_id)
                track_frames[key].append(fi)
                if n_present == 1:
                    track_solo[key] += 1
            # embed on a stride to bound cost; all backbones on the same crops.
            if zone and fi % args.stride == 0:
                boxes = [(t.x1, t.y1, t.x2, t.y2) for t in zone]
                for name, e in embedders.items():
                    feats = e.embed(frame, boxes)
                    for t, f in zip(zone, feats):
                        key = (wi, t.track_id)
                        emb_sum[name][key] = emb_sum[name].get(key, 0.0) + f
                for t in zone:
                    emb_cnt[(wi, t.track_id)] += 1
            fi += 1
        cap.release()
        print(f"window {start}-{end}: {len(zone) if 'zone' in dir() else 0} tracks last frame, "
              f"pass done", flush=True)

    # GT assignment per track (within its window), and solo fraction.
    def assign(key) -> str | None:
        wi = key[0]
        s, e = args.window[wi]
        fs = set(track_frames[key])
        gtf = {}
        for ev in raw_gt:
            if ev["enter_frame"] < e and ev["exit_frame"] > s:
                gtf.setdefault(ev["person_id"], set()).update(
                    range(max(ev["enter_frame"], s), min(ev["exit_frame"], e)))
        best = max(gtf, key=lambda p: len(fs & gtf[p]), default=None)
        return best if best is not None and (fs & gtf.get(best, set())) else None

    kept = [k for k, fr in track_frames.items()
            if len(fr) >= args.min_track_frames and emb_cnt[k] > 0]
    person = {k: assign(k) for k in kept}
    solo_frac = {k: track_solo[k] / len(track_frames[k]) for k in kept}
    kept = [k for k in kept if person[k] is not None]

    def span(k):
        return track_frames[k][0], track_frames[k][-1]

    def disjoint(a, b):
        # tracks from different windows are always disjoint in time; same window: check spans
        if a[0] != b[0]:
            return True
        sa, sb = span(a), span(b)
        return not (sa[0] <= sb[1] and sb[0] <= sa[1])

    results = {}
    for name in embedders:
        emb = {k: emb_sum[name][k] / emb_cnt[k] for k in kept if k in emb_sum[name]}
        emb = {k: v / (np.linalg.norm(v) or 1.0) for k, v in emb.items()}

        def pairs(solo_only: bool):
            same, diff = [], []
            ids = [k for k in kept if k in emb and (not solo_only or solo_frac[k] >= args.solo_frac)]
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    if not disjoint(a, b):
                        continue
                    cos = float(emb[a] @ emb[b])
                    (same if person[a] == person[b] else diff).append(cos)
            return same, diff

        row = {}
        for label, solo_only in (("all", False), ("solo", True)):
            same, diff = pairs(solo_only)
            if same and diff:
                s10 = float(np.percentile(same, 10))
                d90 = float(np.percentile(diff, 90))
                row[label] = {
                    "same_pairs": len(same), "diff_pairs": len(diff),
                    "same_median": round(float(np.median(same)), 3), "same_p10": round(s10, 3),
                    "diff_median": round(float(np.median(diff)), 3), "diff_p90": round(d90, 3),
                    "margin": round(s10 - d90, 3),
                }
            else:
                row[label] = {"same_pairs": len(same), "diff_pairs": len(diff), "margin": None}
        results[name] = row
        a, s = row["all"], row["solo"]
        print(f"{name:26} ALL margin {str(a.get('margin')):>7} ({a['same_pairs']}/{a['diff_pairs']})  "
              f"|  SOLO margin {str(s.get('margin')):>7} "
              f"(same_p10 {s.get('same_p10','-')}, diff_p90 {s.get('diff_p90','-')}; "
              f"{s['same_pairs']}/{s['diff_pairs']} pairs)", flush=True)

    n_solo = sum(1 for k in kept if solo_frac[k] >= args.solo_frac)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": str(args.config), "windows": args.window,
        "tracks_kept": len(kept), "solo_tracks": n_solo,
        "solo_frac_threshold": args.solo_frac, "results": results}, indent=2))
    print(f"\n{len(kept)} tracks kept, {n_solo} solo. wrote {args.out}")
    print("SOLO margin > 0 => on clean unoccluded crops a threshold separates identities "
          "=> the crowded null was a confound and a better backbone can help.")


if __name__ == "__main__":
    main()
