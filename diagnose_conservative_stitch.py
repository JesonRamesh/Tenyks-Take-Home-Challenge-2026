#!/usr/bin/env python3
"""Conservative-stitch test: does merging LESS aggressively improve dwell + purity?

Read-only, diagnostic. The full-run evidence showed ~84% of long-gap re-link candidates that
clear the appearance gate are actually different people, so the stitch may be net-harmful to
DWELL accuracy (a wrong merge dumps a new person's frames onto an old id). This sweeps the
stitch's aggressiveness -- gap_frames (how far back it re-links; gap 0 = stitch effectively
off) x min_similarity -- and reports count_error, matched, DWELL MAE, and purity, so the
count-vs-dwell-vs-purity trade is visible instead of assumed.

Reuses diagnose_stitch_ab's single-pass state (detection+tracking+embedding once), replaying
the exact post-stitch pipeline per (gap, sim). Backbone fixed to the production osnet_x0_25 --
this tests stitch POLICY, not the backbone (Phase B settled the backbone).

    python diagnose_conservative_stitch.py --config configs/cam1_roifix.yaml --slice 26700 71000
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from eval.label.schema import TrackInterval
from eval.metrics import coverage_report, evaluate
from evaluate_baseline import load_kiosk_ground_truth
from src.detect.build import build_detector
from src.device import resolve_device
from src.dwell.aggregate import aggregate
from src.reid.embed import Embedder
from src.reid.stitch import TrackAppearance, stitch
from src.staff.filter import StaffClassifier, is_staff_track
from src.track.bytetrack import ByteTrackTracker
from src.zones.roi import anchor, in_zone
from src.zones.stationarity import is_visit

BACKBONE = "weights/osnet_x0_25_msmt17.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("digital_kiosk.mp4"))
    parser.add_argument("--config", type=Path, default=Path("configs/cam1_roifix.yaml"))
    parser.add_argument("--slice", nargs=2, type=int, default=[26700, 71000])
    parser.add_argument("--gt", type=Path, default=Path("eval/label/kiosk_gt.yaml"))
    # gap 0 disables long-gap re-linking; the current production value is 3000.
    parser.add_argument("--gaps", nargs="+", type=int, default=[0, 90, 300, 900, 3000])
    parser.add_argument("--sims", nargs="+", type=float, default=[0.6, 0.7])
    parser.add_argument("--embed-stride", type=int, default=8)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    roi = config["kiosk_roi"]
    polygon = [tuple(p) for p in roi["roi_polygon"]]
    depth, aspect = roi["box_depth_frac"], roi["min_box_aspect"]
    device = resolve_device(config["detector"]["device"])
    start, end = args.slice
    tcfg = config["tracker"]

    detector = build_detector(config["detector"], device)
    tracker = ByteTrackTracker(**{k: tcfg[k] for k in (
        "track_high_thresh", "track_low_thresh", "new_track_thresh",
        "match_thresh", "track_buffer", "fuse_score")})
    embedder = Embedder(BACKBONE, device)
    staff_clf = StaffClassifier(config["staff"])

    in_zone_frames: dict[int, list[int]] = defaultdict(list)
    track_anchors: dict[int, list] = defaultdict(list)
    staff_hits: dict[int, int] = defaultdict(int)
    first_anchor: dict[int, tuple] = {}
    last_anchor: dict[int, tuple] = {}
    esum: dict[int, np.ndarray] = {}
    ecnt: dict[int, int] = defaultdict(int)

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    fi = start
    fps = cap.get(cv2.CAP_PROP_FPS)
    while fi < end:
        ok, frame = cap.read()
        if not ok:
            break
        zone = [t for t in tracker.update(detector.detect(frame), frame, fi)
                if in_zone(t, polygon, depth, aspect)]
        if zone:
            boxes = [(t.x1, t.y1, t.x2, t.y2) for t in zone]
            flags = staff_clf.staff_frames(frame, boxes)
            do_embed = fi % args.embed_stride == 0
            feats = embedder.embed(frame, boxes) if do_embed else None
            for i, t in enumerate(zone):
                tid = t.track_id
                pt = anchor(t)
                in_zone_frames[tid].append(fi)
                track_anchors[tid].append(pt)
                staff_hits[tid] += int(flags[i])
                first_anchor.setdefault(tid, pt)
                last_anchor[tid] = pt
                if do_embed:
                    esum[tid] = esum.get(tid, 0.0) + feats[i]
                    ecnt[tid] += 1
        fi += 1
    cap.release()

    for tid in [t for t in list(in_zone_frames) if ecnt[t] == 0]:
        for d in (in_zone_frames, track_anchors, staff_hits, first_anchor, last_anchor):
            d.pop(tid, None)

    ground_truth, _fps, _ = load_kiosk_ground_truth(args.gt, (start, end))
    raw_gt = yaml.safe_load(args.gt.read_text())
    gt_person_frames: dict[str, set] = defaultdict(set)
    for e in raw_gt["events"]:
        if e["enter_frame"] < end and e["exit_frame"] > start:
            gt_person_frames[e["person_id"]].update(range(max(e["enter_frame"], start), min(e["exit_frame"], end)))
    min_iou = config["eval"]["min_iou"]
    seg_gap = config["dwell"]["segment_gap_frames"]
    st = config["stationarity"]
    min_staff = config["staff"]["min_staff_frame_frac"]

    def unit(v):
        return v / (np.linalg.norm(v) or 1.0)

    def replay(gap: int, sim: float):
        app = {tid: TrackAppearance(fr[0], fr[-1], first_anchor[tid], last_anchor[tid],
                                    unit(esum[tid] / ecnt[tid]))
               for tid, fr in in_zone_frames.items()}
        # gap 0 => no track can re-link (a track's end is never within 0 frames of a later
        # start), i.e. the long-gap stitch is off; each raw fragment stands alone.
        id_map = stitch(app, gap, 400, sim) if gap > 0 else {t: t for t in app}
        merged_f: dict[int, list] = defaultdict(list)
        merged_a: dict[int, list] = defaultdict(list)
        staff_const: dict[int, list] = defaultdict(list)
        for tid, fr in in_zone_frames.items():
            c = id_map[tid]
            merged_f[c].extend(fr)
            merged_a[c].extend(track_anchors[tid])
            staff_const[c].append((staff_hits[tid], len(fr)))
        visit = {}
        for c, fr in merged_f.items():
            order = sorted(range(len(fr)), key=fr.__getitem__)
            sf = [fr[i] for i in order]
            sa = [merged_a[c][i] for i in order]
            if is_visit(sf, sa, fps, st["min_dwell_s"], st["max_step_px"], st["min_still_frames"]):
                visit[c] = fr
        cust = {c: fr for c, fr in visit.items() if not is_staff_track(staff_const[c], min_staff)}
        records = aggregate(cust, fps, seg_gap)
        tracks = [TrackInterval(r.track_id, r.enter_frame, r.exit_frame) for r in records]
        result = evaluate(tracks, ground_truth, fps=fps, min_iou=min_iou)
        _, purity = coverage_report({c: set(f) for c, f in cust.items()}, gt_person_frames)
        mp = sum(purity.values()) / len(purity) if purity else 1.0
        return result, mp

    print(f"\ndense slice {start}-{end}, {len(ground_truth)} GT people, "
          f"{len(in_zone_frames)} raw tracks. backbone osnet_x0_25, anchor 400.")
    print("gap 0 = long-gap stitch OFF (each fragment separate).\n")
    print(f"{'gap':>5} {'sim':>4} {'pred':>5} {'cnt_err':>8} {'matched':>8} {'dwell_MAE':>10} "
          f"{'dwell_MAPE':>11} {'mean_pur':>9}")
    print("-" * 70)
    for gap in args.gaps:
        for sim in (args.sims if gap > 0 else [args.sims[0]]):
            r, mp = replay(gap, sim)
            print(f"{gap:>5} {sim:>4} {r.pred_count:>5} {r.count_error:>+8} "
                  f"{r.num_matched:>4}/{r.gt_count:<3} {r.dwell_mae_s:>9.1f}s {r.dwell_mape_pct:>10.1f}% "
                  f"{mp:>9.3f}", flush=True)
        print()


if __name__ == "__main__":
    main()
