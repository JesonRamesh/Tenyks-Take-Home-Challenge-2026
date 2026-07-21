"""Overlay video renderer: draw the pipeline's per-frame result on the source video.

    python -m src.overlay.renderer --video digital_kiosk.mp4 --config configs/cam1.yaml

Consumes the render artifact run.py emits (render_frames.yaml in --out-dir: the
surviving in-zone boxes on each frame, post stitch / stationarity / staff, each tagged
with its canonical id and whether the track is a customer or staff) plus the source
video, and writes an annotated mp4. tracks.yaml carries only per-visit intervals, not
per-frame boxes, so the artifact is the box source — it is the same full-pipeline
result with the same provenance as tracks.yaml.

Each active track gets a box coloured consistently by id, labelled with its id and its
running dwell so far; a corner shows the live count of in-zone non-staff people; staff
tracks render red with a STAFF tag rather than being hidden, so where the staff filter
fires stays visible for review. Output path, codec, and the rendered span (inherited
from the artifact's frame range, i.e. run.py --slice) are config-driven.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

_STAFF_COLOR = (0, 0, 255)  # BGR red; staff render distinct from customers


def track_color(track_id: int) -> tuple[int, int, int]:
    # Stable, well-separated hue per id (golden-angle step) at full saturation/value, so
    # each track keeps one distinct colour across the whole clip.
    hue = int((track_id * 137) % 180)
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    overlay_cfg = config["overlay"]
    polygon = np.array(config["kiosk_roi"]["roi_polygon"], dtype=np.int32)

    artifact = yaml.safe_load((args.out_dir / "render_frames.yaml").read_text())
    fps = artifact["fps"]
    start_frame, end_frame = artifact["start_frame"], artifact["end_frame"]
    frames = artifact["frames"]

    cap = cv2.VideoCapture(str(args.video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = cv2.VideoWriter(
        overlay_cfg["output_path"],
        cv2.VideoWriter_fourcc(*overlay_cfg["codec"]),
        fps,
        (width, height),
    )

    # Running dwell per canonical id: in-zone frames seen so far, which lands on the same
    # value the pipeline aggregates once a visit ends.
    dwell_frames: dict[int, int] = defaultdict(int)
    for frame_index in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        cv2.polylines(frame, [polygon], isClosed=True, color=(90, 90, 90), thickness=1)
        current_count = 0
        for track_id, x1, y1, x2, y2, kind in frames.get(frame_index, []):
            dwell_frames[track_id] += 1
            if kind == 1:
                color, label = _STAFF_COLOR, f"STAFF {track_id}"
            else:
                current_count += 1
                color = track_color(track_id)
                label = f"ID {track_id}  {dwell_frames[track_id] / fps:.1f}s"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(frame, (10, 12), (360, 52), (0, 0, 0), -1)
        cv2.putText(frame, f"Current count: {current_count}", (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    cap.release()


if __name__ == "__main__":
    main()
