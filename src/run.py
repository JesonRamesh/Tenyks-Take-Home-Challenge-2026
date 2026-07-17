"""Single entrypoint: video + config -> outputs.

    python -m src.run --video digital_kiosk.mp4 --config configs/cam1.yaml

Pipeline per frame: detect -> track -> ROI gate. Then dwell aggregation over the
whole video. Writes predictions to outputs/tracks.yaml and peak VRAM + measured
FPS to outputs/perf.yaml.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
import yaml

from src.detect.yolo import YoloDetector
from src.dwell.aggregate import aggregate
from src.track.bytetrack import ByteTrackTracker
from src.zones.roi import in_zone


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
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    polygon = [tuple(point) for point in config["kiosk_roi"]["roi_polygon"]]
    det_cfg = config["detector"]
    device = resolve_device(det_cfg["device"])

    detector = YoloDetector(
        det_cfg["model"], det_cfg["confidence"], det_cfg["classes"], det_cfg["imgsz"], device
    )
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

    in_zone_frames: dict[int, list[int]] = {}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    frame_index = 0
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame_index)
        for track in tracks:
            if in_zone(track, polygon):
                in_zone_frames.setdefault(track.track_id, []).append(frame_index)
        frame_index += 1
    elapsed = time.time() - start
    cap.release()

    records = aggregate(in_zone_frames, fps, config["dwell"]["segment_gap_frames"])
    records.sort(key=lambda record: record.enter_frame)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracks_yaml = [
        {"track_id": record.track_id, "enter_frame": record.enter_frame, "exit_frame": record.exit_frame}
        for record in records
    ]
    (args.out_dir / "tracks.yaml").write_text(yaml.safe_dump(tracks_yaml, sort_keys=False))

    # torch.cuda.max_memory_allocated only reports on CUDA; on mps/cpu it is 0 and
    # peak VRAM must be re-measured on the T4 target.
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    perf = {
        "device": device,
        "frames": frame_index,
        "elapsed_s": round(elapsed, 2),
        "fps": round(frame_index / elapsed, 2),
        "peak_vram_gb": round(peak_vram_gb, 3),
    }
    (args.out_dir / "perf.yaml").write_text(yaml.safe_dump(perf, sort_keys=False))


if __name__ == "__main__":
    main()
