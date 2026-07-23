#!/usr/bin/env python3
"""Phase A: measure same-vs-different appearance separation for candidate ReID backbones.

Read-only, diagnostic. The ID-switching / wrong-merge problem is precision-bounded by one
quantity: can a backbone tell two crops of the same person apart from two crops of
different people, on THIS top-down camera? This measures exactly that, reusing the Phase 5
methodology (tune_stitch's separation analysis) so results are comparable, but sweeping the
backbone instead of the threshold.

Method:
  1. Run detect -> track on a window (per the given config, so the crops match that
     pipeline's boxes) and collect each raw track's in-zone boxes.
  2. Assign each raw track to the GT person it most overlaps in time (argmax).
  3. For each backbone: embed sampled crops per track, mean + L2-normalize.
  4. same-person pairs  = two non-time-overlapping tracks of the SAME GT person -> cosine.
     different-person   = two tracks of DIFFERENT GT people                    -> cosine.
  5. Report same 10th pct and different 90th pct. The margin (same_p10 - diff_p90) is the
     figure of merit: positive means a threshold exists that cleanly separates them, i.e.
     the merge can be made both complete (no switches) and pure (no wrong merges). The
     current osnet_x0_25 is NEGATIVE here (0.492 vs 0.693), which is the whole problem.

    python diagnose_reid.py --config configs/cam1.yaml --slice 103000 109000 \
        --backbones weights/osnet_x0_25_msmt17.pt weights/osnet_x1_0_msmt17.pt

Temporal-only GT cannot perfectly separate co-present people, so per-pair labels are noisy
in a crowd; the 10th/90th percentiles are used (not min/max) precisely to be robust to that.
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
    parser.add_argument("--slice", nargs=2, type=int, required=True)
    parser.add_argument("--gt", type=Path, default=Path("eval/label/kiosk_gt.yaml"))
    parser.add_argument("--backbones", nargs="+", required=True)
    parser.add_argument("--samples-per-track", type=int, default=32)
    parser.add_argument("--min-track-frames", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("outputs/reid_separation.json"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    roi = config["kiosk_roi"]
    polygon = [tuple(p) for p in roi["roi_polygon"]]
    depth, aspect = roi["box_depth_frac"], roi["min_box_aspect"]
    device = resolve_device(config["detector"]["device"])
    start, end = args.slice

    # ---- Pass 1: track, and reservoir-sample (frame, box) per raw track ----
    detector = build_detector(config["detector"], device)
    tracker = _build_tracker(config["tracker"], device)
    rng = np.random.default_rng(0)
    track_frames: dict[int, list[int]] = defaultdict(list)      # for GT assignment
    samples: dict[int, list[tuple[int, tuple]]] = defaultdict(list)  # (frame, box) to embed
    seen: dict[int, int] = defaultdict(int)

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_index = start
    while frame_index < end:
        ok, frame = cap.read()
        if not ok:
            break
        tracks = tracker.update(detector.detect(frame), frame, frame_index)
        for t in tracks:
            if not in_zone(t, polygon, depth, aspect):
                continue
            tid = t.track_id
            track_frames[tid].append(frame_index)
            box = (t.x1, t.y1, t.x2, t.y2)
            # reservoir sampling: an unbiased K-sample of each track's boxes.
            seen[tid] += 1
            if len(samples[tid]) < args.samples_per_track:
                samples[tid].append((frame_index, box))
            elif rng.random() < args.samples_per_track / seen[tid]:
                samples[tid][rng.integers(args.samples_per_track)] = (frame_index, box)
        frame_index += 1
    cap.release()

    tracks_kept = [t for t, fr in track_frames.items() if len(fr) >= args.min_track_frames]
    frame_boxes: dict[int, list[tuple[int, tuple]]] = defaultdict(list)
    for tid in tracks_kept:
        for f, box in samples[tid]:
            frame_boxes[f].append((tid, box))

    # ---- GT assignment: each track -> the GT person it overlaps most in time ----
    raw_gt = yaml.safe_load(args.gt.read_text())
    gt_frames: dict[str, set[int]] = defaultdict(set)
    for e in raw_gt["events"]:
        if e["enter_frame"] < end and e["exit_frame"] > start:
            gt_frames[e["person_id"]].update(range(max(e["enter_frame"], start), min(e["exit_frame"], end)))
    assign: dict[int, str] = {}
    for tid in tracks_kept:
        fs = set(track_frames[tid])
        best = max(gt_frames, key=lambda p: len(fs & gt_frames[p]), default=None)
        if best is not None and fs & gt_frames[best]:
            assign[tid] = best

    def span(tid: int) -> tuple[int, int]:
        return track_frames[tid][0], track_frames[tid][-1]

    results = {}
    for weight in args.backbones:
        embedder = Embedder(weight, device)
        emb: dict[int, np.ndarray] = {}
        acc: dict[int, list[np.ndarray]] = defaultdict(list)
        cap = cv2.VideoCapture(str(args.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        fi = start
        while fi < end:
            ok, frame = cap.read()
            if not ok:
                break
            if fi in frame_boxes:
                entries = frame_boxes[fi]
                feats = embedder.embed(frame, [b for _, b in entries])
                for (tid, _b), f in zip(entries, feats):
                    acc[tid].append(f)
            fi += 1
        cap.release()
        for tid, fs in acc.items():
            v = np.mean(fs, axis=0)
            emb[tid] = v / (np.linalg.norm(v) or 1.0)

        same, diff = [], []
        ids = [t for t in tracks_kept if t in assign and t in emb]
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                sa, sb = span(a), span(b)
                overlap = sa[0] <= sb[1] and sb[0] <= sa[1]
                cos = float(emb[a] @ emb[b])
                if assign[a] == assign[b]:
                    if not overlap:      # same person, disjoint in time = a real re-link case
                        same.append(cos)
                elif not overlap:        # different people, disjoint = a must-not-merge case
                    diff.append(cos)

        name = Path(weight).stem
        if same and diff:
            same_p10 = float(np.percentile(same, 10))
            diff_p90 = float(np.percentile(diff, 90))
            results[name] = {
                "same_pairs": len(same), "diff_pairs": len(diff),
                "same_median": round(float(np.median(same)), 3),
                "same_p10": round(same_p10, 3),
                "diff_median": round(float(np.median(diff)), 3),
                "diff_p90": round(diff_p90, 3),
                "margin": round(same_p10 - diff_p90, 3),
            }
        else:
            results[name] = {"same_pairs": len(same), "diff_pairs": len(diff), "margin": None}
        r = results[name]
        print(f"{name:28} same_p10 {r.get('same_p10','-'):>6}  diff_p90 {r.get('diff_p90','-'):>6}  "
              f"margin {r.get('margin','-'):>6}  (same {r['same_pairs']}, diff {r['diff_pairs']})",
              flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"config": str(args.config), "slice": [start, end],
                                    "tracks_kept": len(tracks_kept), "results": results}, indent=2))
    print(f"\nwrote {args.out}")
    print("margin > 0 => a threshold cleanly separates same from different (the ID problem becomes solvable)")


if __name__ == "__main__":
    main()
