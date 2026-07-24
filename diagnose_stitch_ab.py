#!/usr/bin/env python3
"""Phase B (local): does swapping the stitch backbone OSNet -> CLIP reduce wrong re-links?

Read-only, diagnostic. Runs detection + tracking ONCE on a slice, embeds every in-zone crop
with BOTH backbones in the same pass, then replays the exact post-stitch pipeline (stitch ->
stationarity -> staff -> aggregate -> eval) for each backbone across a min_similarity sweep.
Only the appearance embeddings differ, so any change in count_error / purity is attributable
to the backbone. CLIP's cosine distribution differs from OSNet's (diff_median 0.28 vs 0.58),
so its threshold must be swept, not inherited.

Purity is the key metric: a wrong re-link (a new person merged onto an old id) shows up as a
merged track spanning two GT people, i.e. purity < 1. Coverage/matched guard against the
threshold merely becoming so conservative it stops merging real fragments.

    python diagnose_stitch_ab.py --config configs/cam1_roifix.yaml --slice 26700 71000

Mirrors run.py's post-stitch stages exactly (dominant-segment staff verdict included).
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
from src.staff.filter import is_staff_track
from src.track.bytetrack import ByteTrackTracker
from src.zones.roi import anchor, in_zone
from src.zones.stationarity import is_visit

_DEFAULT = ["weights/osnet_x0_25_msmt17.pt", "weights/osnet_x1_0_msmt17.pt",
            "weights/osnet_ain_x1_0_msmt17.pt"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("digital_kiosk.mp4"))
    parser.add_argument("--config", type=Path, default=Path("configs/cam1_roifix.yaml"))
    parser.add_argument("--slice", nargs=2, type=int, default=[26700, 71000])
    parser.add_argument("--gt", type=Path, default=Path("eval/label/kiosk_gt.yaml"))
    parser.add_argument("--sims", nargs="+", type=float, default=[0.4, 0.5, 0.6, 0.7, 0.8])
    # The stitch uses each track's MEAN embedding, so embedding every Nth in-zone frame is
    # equivalent and makes CLIP (heavy per crop) tractable. All in-zone frames are still
    # recorded for the stitch's frame/anchor/staff bookkeeping; only embedding is strided.
    parser.add_argument("--embed-stride", type=int, default=8)
    parser.add_argument("--backbones", nargs="+", default=_DEFAULT)
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
    from pathlib import Path as _P
    embedders = {_P(w).stem: Embedder(w, device) for w in args.backbones}

    in_zone_frames: dict[int, list[int]] = defaultdict(list)
    track_anchors: dict[int, list] = defaultdict(list)
    staff_hits: dict[int, int] = defaultdict(int)
    first_anchor: dict[int, tuple] = {}
    last_anchor: dict[int, tuple] = {}
    esum = {n: {} for n in embedders}
    ecnt: dict[int, int] = defaultdict(int)

    from src.staff.filter import StaffClassifier
    staff_clf = StaffClassifier(config["staff"])

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
            feats = {n: e.embed(frame, boxes) for n, e in embedders.items()} if do_embed else None
            for i, t in enumerate(zone):
                tid = t.track_id
                pt = anchor(t)
                in_zone_frames[tid].append(fi)
                track_anchors[tid].append(pt)
                staff_hits[tid] += int(flags[i])
                if tid not in first_anchor:
                    first_anchor[tid] = pt
                last_anchor[tid] = pt
                if do_embed:
                    for n in embedders:
                        esum[n][tid] = esum[n].get(tid, 0.0) + feats[n][i]
                    ecnt[tid] += 1
        fi += 1
    cap.release()

    # Drop tracks that never landed on a stride frame (no embedding). These are <embed_stride
    # frames long (<~0.3s), removed by the stationarity gate regardless, so excluding them
    # here does not change the result.
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

    def replay(backbone: str, gap: int, anchor_dist: int, sim: float):
        app = {tid: TrackAppearance(fr[0], fr[-1], first_anchor[tid], last_anchor[tid],
                                    unit(esum[backbone][tid] / ecnt[tid]))
               for tid, fr in in_zone_frames.items()}
        id_map = stitch(app, gap, anchor_dist, sim)
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
        low = min(purity.values()) if purity else 1.0
        return result, mp, low

    print(f"\ndense slice {start}-{end}, {len(ground_truth)} GT people, "
          f"{len(in_zone_frames)} raw tracks. gap 3000 / anchor 400.\n")
    print(f"{'backbone':8} {'sim':>4} {'pred':>5} {'cnt_err':>8} {'matched':>8} {'mean_pur':>9} {'min_pur':>8}")
    print("-" * 56)
    for backbone in embedders:
        for sim in args.sims:
            r, mp, low = replay(backbone, 3000, 400, sim)
            print(f"{backbone:8} {sim:>4} {r.pred_count:>5} {r.count_error:>+8} "
                  f"{r.num_matched:>4}/{r.gt_count:<3} {mp:>9.3f} {low:>8.3f}", flush=True)
        print()


if __name__ == "__main__":
    main()
