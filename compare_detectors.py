#!/usr/bin/env python3
"""Detector-level comparison on a single frame: box separation, latency, model size.

Read-only, diagnostic. Answers the Step 1 question — which detector separates two
adjacent people into two boxes — at the detector's own output, before any tracking can
mask or compound the result. Reports every person box, plus how many fall inside a
region of interest (the couple), so a merge shows up as one box where two are expected.

Detectors with conflicting dependency pins run in their own environments, so this emits
JSON for one detector per invocation and the rows are combined afterwards:

    python compare_detectors.py --config configs/cam1.yaml --frame 26800 \
        --region 740 300 915 615 --out outputs/detector_v2/yolo_26800.json

Latency is measured after warm-up iterations, since a first call includes lazy weight
load and kernel compilation and is not representative.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import yaml

from src.detect.build import build_detector
from src.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("digital_kiosk.mp4"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--frame", required=True, type=int)
    # x1 y1 x2 y2 of the area under test; a box counts as inside when its center is.
    parser.add_argument("--region", nargs=4, type=float, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    det_cfg = config["detector"]
    device = resolve_device(det_cfg["device"])
    detector = build_detector(det_cfg, device)

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {args.frame} from {args.video}")

    for _ in range(args.warmup):
        detector.detect(frame)
    start = time.time()
    for _ in range(args.repeats):
        detections = detector.detect(frame)
    latency_ms = (time.time() - start) / args.repeats * 1000

    rx1, ry1, rx2, ry2 = args.region
    boxes = []
    for d in detections:
        cx, cy = (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2
        # ultralytics hands back numpy float32, which json can't encode; cast at the
        # boundary so every detector's record serializes the same way.
        boxes.append(
            {
                "box": [round(float(v), 1) for v in (d.x1, d.y1, d.x2, d.y2)],
                "confidence": round(float(d.confidence), 3),
                "in_region": bool(rx1 <= cx <= rx2 and ry1 <= cy <= ry2),
                "aspect": round(float(d.x2 - d.x1) / max(float(d.y2 - d.y1), 1e-6), 3),
            }
        )
    boxes.sort(key=lambda b: -b["confidence"])

    # Parameter count is the honest cost signal alongside latency: MPS timings are a
    # weak proxy for the T4 target, but parameter count is hardware-independent.
    model = getattr(detector, "model", None)
    inner = getattr(model, "model", model)
    params = sum(p.numel() for p in inner.parameters()) if hasattr(inner, "parameters") else None

    record = {
        "detector": f"{det_cfg.get('type', 'yolo')}:{det_cfg['model']}",
        "device": device,
        "frame": args.frame,
        "confidence": det_cfg["confidence"],
        "imgsz": det_cfg["imgsz"],
        "latency_ms": round(latency_ms, 1),
        "params_m": round(params / 1e6, 1) if params else None,
        "num_boxes": len(boxes),
        "num_in_region": sum(b["in_region"] for b in boxes),
        "boxes": boxes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2))
    print(
        f"{record['detector']:32s} boxes={record['num_boxes']:3d} "
        f"in_region={record['num_in_region']} latency={record['latency_ms']:.0f}ms "
        f"params={record['params_m']}M"
    )
    for b in boxes:
        if b["in_region"]:
            print(f"    region box conf={b['confidence']:.2f} {b['box']} aspect={b['aspect']}")


if __name__ == "__main__":
    main()
