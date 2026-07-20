"""ByteTrack tracker implementing the Tracker Protocol.

Wraps ultralytics' BYTETracker: pure motion/IoU association through a Kalman
filter, no appearance/ReID. BYTETracker reads a results-like object, so we adapt
the Detection list into the minimal (.xywh/.conf/.cls) view it consumes.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from src.detect.base import Detection
from src.track.base import Track


class _Dets:
    # Minimal results-like view BYTETracker expects: center-form xywh boxes,
    # scores, class ids, len(), and boolean-mask indexing that stays the same view.
    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: np.ndarray) -> "_Dets":
        return _Dets(self.xywh[mask], self.conf[mask], self.cls[mask])


class ByteTrackTracker:
    def __init__(
        self,
        track_high_thresh: float,
        track_low_thresh: float,
        new_track_thresh: float,
        match_thresh: float,
        track_buffer: int,
        fuse_score: bool,
    ):
        args = SimpleNamespace(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            fuse_score=fuse_score,
        )
        self._tracker = BYTETracker(args)

    def update(self, detections: list[Detection], frame: np.ndarray, frame_index: int) -> list[Track]:
        # frame and frame_index are part of the interface; ByteTrack is motion-only
        # so it ignores the pixels, and BYTETracker keeps its own frame counter and
        # only needs to be called once per frame in order.
        n = len(detections)
        xywh = np.empty((n, 4), dtype=np.float32)
        conf = np.empty(n, dtype=np.float32)
        for i, d in enumerate(detections):
            xywh[i] = ((d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2, d.x2 - d.x1, d.y2 - d.y1)
            conf[i] = d.confidence
        cls = np.zeros(n, dtype=np.float32)

        out = self._tracker.update(_Dets(xywh, conf, cls))
        # BYTETracker rows: [x1, y1, x2, y2, track_id, score, cls, idx].
        return [Track(int(r[4]), r[0], r[1], r[2], r[3], r[5]) for r in out]
