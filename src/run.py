"""Single entrypoint: video + config -> outputs.

    python -m src.run --video digital_kiosk.mp4 --config configs/cam1.yaml

Pipeline per frame: detect -> track -> ROI gate. The tracker is config-driven
(tracker.type): our ByteTrack wrapper, or a boxmot tracker with appearance
association built in. When reid.post_hoc_stitch is set (the ByteTrack baseline),
each in-zone box is embedded and appearance re-association stitches the motion-only
fragments back into one identity; a boxmot tracker does that association internally,
so the post-hoc stitch is bypassed to avoid double-applying it. Dwell is aggregated
over the whole video. Writes predictions to outputs/tracks.yaml and peak VRAM +
measured FPS to outputs/perf.yaml.

--slice runs only a frame range (the eval_slice in the config, or an explicit
start/end pair) for fast iteration; the full video is the default.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from src.detect.build import build_detector
from src.device import resolve_device
from src.dwell.aggregate import aggregate
from src.reid.embed import Embedder
from src.reid.stitch import TrackAppearance, stitch
from src.staff.filter import StaffClassifier, is_staff_track
from src.zones.roi import anchor, in_zone
from src.zones.stationarity import is_visit


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    # No values: use the config's eval_slice. Two values: an explicit [start, end).
    # Absent: process the full video.
    parser.add_argument("--slice", nargs="*", type=int)
    # Save up to N frames annotated with any box the staff heuristic fired on, for
    # eyeballing what it flags. 0 (default) writes none.
    parser.add_argument("--staff-debug", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    polygon = [tuple(point) for point in config["kiosk_roi"]["roi_polygon"]]
    box_depth_frac = config["kiosk_roi"]["box_depth_frac"]
    min_box_aspect = config["kiosk_roi"]["min_box_aspect"]
    det_cfg = config["detector"]
    device = resolve_device(det_cfg["device"])

    if args.slice is None:
        start_frame, end_frame = 0, None
    else:
        start_frame, end_frame = args.slice or config["eval_slice"]

    # Detector is config-driven (detector.type): yolo, the NMS-free rtdetr/rfdetr, or
    # cached (replaying a dump_detections.py .npz). All emit the same xyxy pixel
    # Detection, so nothing downstream changes.
    detector = build_detector(det_cfg, device)
    staff_clf = StaffClassifier(config["staff"])

    # Tracker is config-driven. ByteTrack is motion-only and relies on the post-hoc
    # appearance stitch below; the boxmot trackers associate on ReID internally, so
    # that stitch is switched off for them (post_hoc_stitch) rather than run twice.
    tracker_cfg = config["tracker"]
    post_hoc_stitch = config["reid"]["post_hoc_stitch"]
    # Optional per-frame render artifact for the overlay renderer: the surviving in-zone
    # boxes (post stitch / stationarity / staff) on each frame, which tracks.yaml's
    # interval form can't carry. Off unless a config asks, so plain analytics runs and
    # the full-video pass aren't burdened with a large per-frame file.
    overlay_cfg = config.get("overlay")
    emit_render_frames = bool(overlay_cfg) and overlay_cfg.get("emit_render_frames", False)
    # Imported per branch, like the detector: ultralytics (ByteTrack) and boxmot need
    # incompatible pins against some detector backends, so only the selected one is loaded.
    if tracker_cfg["type"] == "bytetrack":
        from src.track.bytetrack import ByteTrackTracker

        tracker = ByteTrackTracker(
            **{
                key: tracker_cfg[key]
                for key in (
                    "track_high_thresh",
                    "track_low_thresh",
                    "new_track_thresh",
                    "match_thresh",
                    "track_buffer",
                    "fuse_score",
                )
            }
        )
    else:
        from src.track.boxmot_tracker import BoxmotTracker

        tracker = BoxmotTracker(
            tracker_cfg["type"],
            device,
            tracker_cfg["half"],
            tracker_cfg.get("reid_weights"),
            tracker_cfg.get("params"),
        )
    # The post-hoc embedder is only needed for the stitch path; a boxmot tracker
    # carries its own ReID, so skip loading a second appearance model for it.
    embedder = Embedder(config["reid"]["model"], device) if post_hoc_stitch else None

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    in_zone_frames: dict[int, list[int]] = {}
    # Per-track accumulators resolved after the loop: running embedding sum + count
    # for re-association, and the anchor on every in-zone frame (aligned with
    # in_zone_frames) for the stationarity gate. first/last anchor are just the ends
    # of that sequence, kept separately as the endpoints stitching re-associates on.
    embedding_sum: dict[int, np.ndarray] = {}
    embedding_count: dict[int, int] = {}
    first_anchor: dict[int, tuple[float, float]] = {}
    last_anchor: dict[int, tuple[float, float]] = {}
    track_anchors: dict[int, list[tuple[float, float]]] = {}
    # Frames of each track whose crop matched the staff uniform; the flag is a track
    # majority, so a fraction of these over the track's total in-zone frames decides.
    staff_hits: dict[int, int] = {}
    # frame -> [(raw_track_id, box), ...] for every in-zone box, populated only when a
    # render artifact was requested; resolved to canonical id + kind after the loop.
    frame_render: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    staff_debug_dir = args.out_dir / "staff_debug"
    if args.staff_debug:
        staff_debug_dir.mkdir(parents=True, exist_ok=True)
    staff_debug_saved = 0

    frame_index = start_frame
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok or (end_frame is not None and frame_index >= end_frame):
            break
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame, frame_index)
        zone_tracks = [track for track in tracks if in_zone(track, polygon, box_depth_frac, min_box_aspect)]
        if zone_tracks:
            boxes = [(t.x1, t.y1, t.x2, t.y2) for t in zone_tracks]
            staff_flags = staff_clf.staff_frames(frame, boxes)
            embeddings = embedder.embed(frame, boxes) if post_hoc_stitch else [None] * len(boxes)
            for track, embedding, staff_flag, box in zip(zone_tracks, embeddings, staff_flags, boxes):
                point = anchor(track)
                in_zone_frames.setdefault(track.track_id, []).append(frame_index)
                track_anchors.setdefault(track.track_id, []).append(point)
                staff_hits[track.track_id] = staff_hits.get(track.track_id, 0) + staff_flag
                if emit_render_frames:
                    frame_render.setdefault(frame_index, []).append((track.track_id, box))
                if staff_flag and staff_debug_saved < args.staff_debug:
                    vis = frame.copy()
                    x1b, y1b, x2b, y2b = (int(v) for v in box)
                    cv2.rectangle(vis, (x1b, y1b), (x2b, y2b), (0, 0, 255), 2)
                    cv2.putText(vis, f"staff? t{track.track_id} f{frame_index}", (x1b, max(0, y1b - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imwrite(str(staff_debug_dir / f"f{frame_index}_t{track.track_id}.png"), vis)
                    staff_debug_saved += 1
                # Appearance accumulation only feeds the post-hoc stitch; boxmot
                # trackers already carry a persistent id, so skip it for them.
                if post_hoc_stitch:
                    if track.track_id in embedding_sum:
                        embedding_sum[track.track_id] += embedding
                        embedding_count[track.track_id] += 1
                    else:
                        embedding_sum[track.track_id] = embedding.copy()
                        embedding_count[track.track_id] = 1
                        first_anchor[track.track_id] = point
                    last_anchor[track.track_id] = point
        frame_index += 1
    elapsed = time.time() - start
    cap.release()

    if post_hoc_stitch:
        appearances = {
            track_id: TrackAppearance(
                first_frame=frames[0],
                last_frame=frames[-1],
                first_anchor=first_anchor[track_id],
                last_anchor=last_anchor[track_id],
                embedding=_unit(embedding_sum[track_id] / embedding_count[track_id]),
            )
            for track_id, frames in in_zone_frames.items()
        }
        reid_cfg = config["reid"]
        id_map = stitch(
            appearances, reid_cfg["gap_frames"], reid_cfg["max_anchor_dist"], reid_cfg["min_similarity"]
        )
    else:
        # boxmot tracker ids are already persistent across the video; no re-association.
        id_map = {track_id: track_id for track_id in in_zone_frames}
    merged_frames: dict[int, list[int]] = {}
    merged_anchors: dict[int, list[tuple[float, float]]] = {}
    # Per merged id, its constituent raw tracks as (staff_frames, total_frames), so the
    # staff verdict can judge the dominant segment instead of the diluted pooled fraction.
    staff_constituents: dict[int, list[tuple[int, int]]] = {}
    for track_id, frames in in_zone_frames.items():
        canonical = id_map[track_id]
        merged_frames.setdefault(canonical, []).extend(frames)
        merged_anchors.setdefault(canonical, []).extend(track_anchors[track_id])
        staff_constituents.setdefault(canonical, []).append((staff_hits[track_id], len(frames)))

    # Stationarity gate: drop merged tracks that only walked through the ROI, keeping
    # those that dwelt or held still. Sort each track's samples by frame first.
    st_cfg = config["stationarity"]
    visit_frames: dict[int, list[int]] = {}
    for canonical, frames in merged_frames.items():
        order = sorted(range(len(frames)), key=frames.__getitem__)
        sorted_frames = [frames[i] for i in order]
        sorted_anchors = [merged_anchors[canonical][i] for i in order]
        if is_visit(
            sorted_frames,
            sorted_anchors,
            fps,
            st_cfg["min_dwell_s"],
            st_cfg["max_step_px"],
            st_cfg["min_still_frames"],
        ):
            visit_frames[canonical] = frames

    # Staff filter: split the visits into customers and staff by the uniform-frame
    # fraction. Staff are written separately so the eval can check none of them
    # actually match a GT customer (a false positive).
    min_staff_frac = config["staff"]["min_staff_frame_frac"]
    customer_frames: dict[int, list[int]] = {}
    staff_frames: dict[int, list[int]] = {}
    for canonical, frames in visit_frames.items():
        target = staff_frames if is_staff_track(staff_constituents[canonical], min_staff_frac) else customer_frames
        target[canonical] = frames

    segment_gap = config["dwell"]["segment_gap_frames"]
    records = aggregate(customer_frames, fps, segment_gap)
    records.sort(key=lambda record: record.enter_frame)
    staff_records = sorted(
        aggregate(staff_frames, fps, segment_gap), key=lambda record: record.enter_frame
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def _intervals(rows: list) -> list[dict]:
        return [
            {"track_id": r.track_id, "enter_frame": r.enter_frame, "exit_frame": r.exit_frame}
            for r in rows
        ]

    (args.out_dir / "tracks.yaml").write_text(yaml.safe_dump(_intervals(records), sort_keys=False))
    (args.out_dir / "staff.yaml").write_text(yaml.safe_dump(_intervals(staff_records), sort_keys=False))

    if emit_render_frames:
        # Resolve every collected in-zone box to its canonical id and kind (0 customer,
        # 1 staff), dropping boxes whose track the stationarity gate removed as a
        # walk-through. Rows are [canonical_id, x1, y1, x2, y2, kind] per frame.
        customer_ids, staff_ids = set(customer_frames), set(staff_frames)
        rendered: dict[int, list[list[int]]] = {}
        for index, entries in frame_render.items():
            rows = []
            for raw_id, box in entries:
                canonical = id_map[raw_id]
                if canonical in customer_ids:
                    kind = 0
                elif canonical in staff_ids:
                    kind = 1
                else:
                    continue
                rows.append([canonical, int(box[0]), int(box[1]), int(box[2]), int(box[3]), kind])
            if rows:
                rendered[index] = rows
        artifact = {"fps": fps, "start_frame": start_frame, "end_frame": frame_index, "frames": rendered}
        (args.out_dir / "render_frames.yaml").write_text(yaml.safe_dump(artifact, sort_keys=False))

        # Dump the raw per-track state (before stitch) so the merge + downstream gates can
        # be replayed and tuned offline with different thresholds, without re-running
        # detection. Post-hoc path only; pickled because frame lists are ragged.
        if post_hoc_stitch:
            import pickle

            state = {
                "fps": fps,
                "appearances": {
                    tid: (a.first_frame, a.last_frame, a.first_anchor, a.last_anchor, a.embedding)
                    for tid, a in appearances.items()
                },
                "in_zone_frames": in_zone_frames,
                "track_anchors": track_anchors,
                "staff_hits": staff_hits,
            }
            (args.out_dir / "stitch_state.pkl").write_bytes(pickle.dumps(state))

    # VRAM peaks only report on CUDA (0 on mps/cpu), so peak VRAM must be measured on
    # the T4 target. reset_peak_memory_stats ran before the loop; synchronize so any
    # async kernels finish before the peaks are read. Log both allocated (live tensor
    # bytes) and reserved (allocator's total device reservation, incl. freed-but-cached
    # blocks). reserved is the number reported against the 16 GB budget: it is the
    # closer proxy to what nvidia-smi shows live, since PyTorch's caching allocator
    # holds freed memory rather than returning it to the driver.
    if device == "cuda":
        torch.cuda.synchronize()
        peak_vram_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
        peak_vram_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
    else:
        peak_vram_allocated_gb = peak_vram_reserved_gb = 0.0
    # frame_index ends at the last frame read; subtract the seek offset so a slice
    # run reports its processed-frame count and true throughput, not the whole video.
    processed = frame_index - start_frame
    perf = {
        "device": device,
        "tracker": tracker_cfg["type"],
        "frames": processed,
        "elapsed_s": round(elapsed, 2),
        "fps": round(processed / elapsed, 2),
        "peak_vram_allocated_gb": round(peak_vram_allocated_gb, 3),
        "peak_vram_reserved_gb": round(peak_vram_reserved_gb, 3),
    }
    (args.out_dir / "perf.yaml").write_text(yaml.safe_dump(perf, sort_keys=False))


if __name__ == "__main__":
    main()
