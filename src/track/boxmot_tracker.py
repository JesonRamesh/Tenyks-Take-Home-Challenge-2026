"""boxmot trackers implementing the Tracker Protocol.

Wraps boxmot's `create_tracker` so an appearance-aware tracker (StrongSORT,
BoT-SORT, DeepOCSORT) or a motion-only one (OC-SORT) can stand in for
ByteTrackTracker behind the same interface. Unlike our ByteTrack path, the
appearance trackers associate on ReID features *inside* the tracker, so the
post-hoc src/reid stitch is bypassed when one is used (run.py, driven by
reid.post_hoc_stitch) — otherwise appearance matching would be double-applied.

The tracker type and ReID backbone are config-driven, never hardcoded. boxmot
consumes the raw frame for appearance crops and camera-motion compensation, so
update takes it alongside the detections. Tracker hyperparameters are boxmot's
own defaults (from its bundled per-tracker yaml) so the bake-off compares each
tracker out of the box, not a version tuned to this camera.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from boxmot import create_tracker, get_tracker_config

from src.detect.base import Detection
from src.track.base import Track


class BoxmotTracker:
    def __init__(self, tracker_type: str, device: str, half: bool, reid_weights: str | None):
        # boxmot's select_device wants a device *index* ("0"), not torch's "cuda"
        # string, which it parses as a 4-GPU request and rejects. Translate our
        # resolved device name to boxmot's convention; "cpu"/"mps"/"cuda:N" pass through.
        if device == "cuda":
            device = "0"
        elif device.startswith("cuda:"):
            device = device.split(":", 1)[1]
        # OC-SORT is motion-only and takes no ReID weights; create_tracker ignores
        # the argument for it, so a None weight path is fine there.
        weights = Path(reid_weights) if reid_weights else None
        # boxmot auto-downloads the backbone but does not create the target directory,
        # and gdown lists it before writing, so make it here (like yolov8n.pt's dir).
        if weights is not None:
            weights.parent.mkdir(parents=True, exist_ok=True)
        self._tracker = create_tracker(
            tracker_type,
            get_tracker_config(tracker_type),
            reid_weights=weights,
            device=device,
            half=half,
            per_class=False,
        )

    def update(self, detections: list[Detection], frame: np.ndarray, frame_index: int) -> list[Track]:
        # boxmot consumes dets as [x1, y1, x2, y2, conf, cls]; a single person class.
        height, width = frame.shape[:2]
        rows = []
        for d in detections:
            # boxmot crops every detection for ReID and cv2.resize raises on a zero-area
            # crop, so drop boxes that collapse to nothing once clamped to the frame (one
            # fully past an edge, or a sub-pixel sliver). Coordinates are otherwise passed
            # through untouched, leaving boxmot's own motion model unaffected.
            if min(d.x2, float(width)) - max(d.x1, 0.0) < 1.0:
                continue
            if min(d.y2, float(height)) - max(d.y1, 0.0) < 1.0:
                continue
            rows.append((d.x1, d.y1, d.x2, d.y2, d.confidence, 0.0))
        dets = np.array(rows, dtype=np.float32).reshape(-1, 6)

        out = self._tracker.update(dets, frame)
        # boxmot rows: [x1, y1, x2, y2, track_id, conf, cls, det_ind].
        return [Track(int(r[4]), r[0], r[1], r[2], r[3], r[5]) for r in out]
