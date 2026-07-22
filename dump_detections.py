#!/usr/bin/env python3
"""Dump one detector's raw output over a frame window to an .npz the pipeline can replay.

Detection dominates the pipeline's cost (RT-DETR ~8 FPS on MPS), and both the detector
bake-off and the tracker sweep re-run the same frames many times. Dumping once per
(detector, window) makes every later run a cache replay, and lets detectors whose
dependencies conflict (rfdetr vs the pinned transformers) be compared at all, by running
each in its own environment and meeting at the cache file.

    python dump_detections.py --config configs/cam1.yaml --slice 26700 30000 \
        --out outputs/det_cache/rtdetr_sliceb.npz

Also records detector-only throughput, which is the honest per-detector cost signal
(a full run.py FPS mixes in tracking and ReID).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.detect.build import build_detector
from src.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("digital_kiosk.mp4"))
    parser.add_argument("--config", required=True, type=Path)
    # No values: the config's eval_slice. Two values: an explicit [start, end).
    parser.add_argument("--slice", nargs="*", type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    det_cfg = config["detector"]
    device = resolve_device(det_cfg["device"])
    start_frame, end_frame = args.slice or config["eval_slice"]

    detector = build_detector(det_cfg, device)

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: list[int] = []
    boxes: list[tuple[float, float, float, float, float]] = []
    frame_index = start_frame
    start = time.time()
    while frame_index < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        for d in detector.detect(frame):
            frames.append(frame_index)
            boxes.append((d.x1, d.y1, d.x2, d.y2, d.confidence))
        frame_index += 1
    elapsed = time.time() - start
    cap.release()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        frames=np.array(frames, dtype=np.int32),
        boxes=np.array(boxes, dtype=np.float32).reshape(-1, 5),
        start_frame=start_frame,
        # The true end: a window running past the video's last frame stops early, and the
        # replaying CachedDetector must know where the cache actually runs out.
        end_frame=frame_index,
        detector_name=f"{det_cfg.get('type', 'yolo')}:{det_cfg['model']}",
        device=device,
        detect_fps=round((frame_index - start_frame) / elapsed, 2),
    )
    processed = frame_index - start_frame
    print(
        f"{det_cfg.get('type', 'yolo')}:{det_cfg['model']} [{start_frame},{frame_index}) "
        f"{processed} frames, {len(boxes)} detections, "
        f"{processed / elapsed:.2f} detect-FPS on {device} -> {args.out}"
    )


if __name__ == "__main__":
    main()
