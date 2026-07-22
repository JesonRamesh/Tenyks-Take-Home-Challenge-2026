"""Replays detections dumped by dump_detections.py, implementing the Detector Protocol.

Detection is by far the most expensive stage (RT-DETR runs ~8 FPS on MPS), and tracker
tuning re-runs the pipeline dozens of times over the same frames with the same detector.
Caching the detector's output once per window makes a sweep replay in seconds instead of
hours, exactly as stitch_state.pkl lets tune_stitch.py replay the merge offline. The
cached boxes are the detector's real output at the config's feed confidence, so a cached
run is bit-identical to the live one.

The cache is keyed by absolute frame index; the cursor starts at the dumped window's
start and advances once per detect() call, matching run.py's strictly sequential read.
"""

from __future__ import annotations

import numpy as np

from src.detect.base import Detection


class CachedDetector:
    def __init__(self, cache_path: str):
        cache = np.load(cache_path)
        frames = cache["frames"]
        boxes = cache["boxes"]
        self.start_frame = int(cache["start_frame"])
        self.end_frame = int(cache["end_frame"])
        self.detector_name = str(cache["detector_name"])
        # frame index -> its detection rows, so replay is a dict lookup per frame.
        self._by_frame: dict[int, list[Detection]] = {}
        for frame_index, box in zip(frames, boxes):
            self._by_frame.setdefault(int(frame_index), []).append(
                Detection(float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4]))
            )
        self._cursor = self.start_frame

    def detect(self, frame: np.ndarray) -> list[Detection]:
        # The frame itself is unused: the cache already holds this frame's detections.
        # Reading past the dumped window is a bug in the caller's slice, not an empty
        # frame, so fail loudly rather than silently returning no detections.
        if self._cursor >= self.end_frame:
            raise IndexError(
                f"cache covers [{self.start_frame}, {self.end_frame}); asked for {self._cursor}"
            )
        detections = self._by_frame.get(self._cursor, [])
        self._cursor += 1
        return detections
