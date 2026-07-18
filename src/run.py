"""Single entrypoint: video + config -> outputs.

    python -m src.run --video digital_kiosk.mp4 --config configs/cam1.yaml

Pipeline per frame: detect -> track -> ROI gate, embedding each in-zone box.
Then appearance re-association stitches ByteTrack fragments back into one
identity before dwell aggregation over the whole video. Writes predictions to
outputs/tracks.yaml and peak VRAM + measured FPS to outputs/perf.yaml.

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

from src.detect.yolo import YoloDetector
from src.dwell.aggregate import aggregate
from src.reid.embed import Embedder
from src.reid.stitch import TrackAppearance, stitch
from src.staff.filter import StaffClassifier
from src.track.bytetrack import ByteTrackTracker
from src.zones.roi import anchor, in_zone
from src.zones.stationarity import is_visit


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    det_cfg = config["detector"]
    device = resolve_device(det_cfg["device"])

    if args.slice is None:
        start_frame, end_frame = 0, None
    else:
        start_frame, end_frame = args.slice or config["eval_slice"]

    detector = YoloDetector(
        det_cfg["model"], det_cfg["confidence"], det_cfg["classes"], det_cfg["imgsz"], device
    )
    embedder = Embedder(config["reid"]["model"], device)
    staff_clf = StaffClassifier(config["staff"])
    tracker = ByteTrackTracker(
        **{
            key: config["tracker"][key]
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
        tracks = tracker.update(detections, frame_index)
        zone_tracks = [track for track in tracks if in_zone(track, polygon, box_depth_frac)]
        if zone_tracks:
            boxes = [(t.x1, t.y1, t.x2, t.y2) for t in zone_tracks]
            embeddings = embedder.embed(frame, boxes)
            staff_flags = staff_clf.staff_frames(frame, boxes)
            for track, embedding, staff_flag, box in zip(zone_tracks, embeddings, staff_flags, boxes):
                point = anchor(track)
                in_zone_frames.setdefault(track.track_id, []).append(frame_index)
                track_anchors.setdefault(track.track_id, []).append(point)
                staff_hits[track.track_id] = staff_hits.get(track.track_id, 0) + staff_flag
                if staff_flag and staff_debug_saved < args.staff_debug:
                    vis = frame.copy()
                    x1b, y1b, x2b, y2b = (int(v) for v in box)
                    cv2.rectangle(vis, (x1b, y1b), (x2b, y2b), (0, 0, 255), 2)
                    cv2.putText(vis, f"staff? t{track.track_id} f{frame_index}", (x1b, max(0, y1b - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imwrite(str(staff_debug_dir / f"f{frame_index}_t{track.track_id}.png"), vis)
                    staff_debug_saved += 1
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
    merged_frames: dict[int, list[int]] = {}
    merged_anchors: dict[int, list[tuple[float, float]]] = {}
    merged_staff_hits: dict[int, int] = {}
    for track_id, frames in in_zone_frames.items():
        canonical = id_map[track_id]
        merged_frames.setdefault(canonical, []).extend(frames)
        merged_anchors.setdefault(canonical, []).extend(track_anchors[track_id])
        merged_staff_hits[canonical] = merged_staff_hits.get(canonical, 0) + staff_hits[track_id]

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
        target = staff_frames if merged_staff_hits[canonical] / len(frames) >= min_staff_frac else customer_frames
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

    # torch.cuda.max_memory_allocated only reports on CUDA; on mps/cpu it is 0 and
    # peak VRAM must be re-measured on the T4 target.
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    # frame_index ends at the last frame read; subtract the seek offset so a slice
    # run reports its processed-frame count and true throughput, not the whole video.
    processed = frame_index - start_frame
    perf = {
        "device": device,
        "frames": processed,
        "elapsed_s": round(elapsed, 2),
        "fps": round(processed / elapsed, 2),
        "peak_vram_gb": round(peak_vram_gb, 3),
    }
    (args.out_dir / "perf.yaml").write_text(yaml.safe_dump(perf, sort_keys=False))


if __name__ == "__main__":
    main()
