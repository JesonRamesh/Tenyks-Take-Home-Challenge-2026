"""Staff-exclusion heuristic (no trained classifier).

Staff wear a specific uniform: a horizontal green / red / white chest stripe
across the upper-mid torso on an otherwise all-black outfit with a dark head
covering. The stripe is the distinguishing signal — plain black clothing is common
on customers, so darkness alone would over-flag. A frame reads as staff only if a
narrow chest band holds BOTH a saturated-green and a saturated-red cluster AND the
rest of the body (head, lower torso, legs) is consistently dark. A track is flagged
staff only if that pattern holds across a high fraction of its frames, so a single
lucky/unlucky frame can't decide it.

HSV ranges are placeholders pending calibration against real staff crops on the
GPU box; the region/logic structure is the fixed part.
"""

from __future__ import annotations

import cv2
import numpy as np

# Horizontal center strip of the box used for every region test, to keep the
# left/right background out of the color and darkness measurements.
_CORE_X = (0.2, 0.8)


class StaffClassifier:
    def __init__(self, cfg: dict):
        self.stripe_lo, self.stripe_hi = cfg["stripe_band"]
        self.green_low = np.array(cfg["green_hsv_low"], dtype=np.uint8)
        self.green_high = np.array(cfg["green_hsv_high"], dtype=np.uint8)
        # Red straddles the hue wraparound, so it needs two ranges.
        self.red_low1 = np.array(cfg["red_hsv_low1"], dtype=np.uint8)
        self.red_high1 = np.array(cfg["red_hsv_high1"], dtype=np.uint8)
        self.red_low2 = np.array(cfg["red_hsv_low2"], dtype=np.uint8)
        self.red_high2 = np.array(cfg["red_hsv_high2"], dtype=np.uint8)
        self.min_cluster_frac = cfg["min_cluster_frac"]
        self.dark_v_max = cfg["dark_v_max"]
        self.min_dark_frac = cfg["min_dark_frac"]

    def _is_staff_frame(self, crop: np.ndarray) -> bool:
        h, w = crop.shape[:2]
        core = crop[:, int(_CORE_X[0] * w) : int(_CORE_X[1] * w)]
        lo, hi = int(self.stripe_lo * h), int(self.stripe_hi * h)
        if hi <= lo or core.shape[1] == 0:
            return False

        hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
        band = hsv[lo:hi]
        band_px = band.shape[0] * band.shape[1]
        green = cv2.inRange(band, self.green_low, self.green_high).sum() / 255
        red = (
            cv2.inRange(band, self.red_low1, self.red_high1).sum()
            + cv2.inRange(band, self.red_low2, self.red_high2).sum()
        ) / 255
        if green / band_px < self.min_cluster_frac or red / band_px < self.min_cluster_frac:
            return False

        outside = np.concatenate([hsv[:lo], hsv[hi:]], axis=0)
        dark_frac = float((outside[:, :, 2] <= self.dark_v_max).mean())
        return dark_frac >= self.min_dark_frac

    def staff_frames(self, frame: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> list[bool]:
        """Per-box staff verdict for one frame."""
        height, width = frame.shape[:2]
        flags = []
        for x1, y1, x2, y2 in boxes:
            xi1, yi1 = max(0, int(x1)), max(0, int(y1))
            xi2, yi2 = min(width, int(x2)), min(height, int(y2))
            flags.append(self._is_staff_frame(frame[yi1:yi2, xi1:xi2]))
        return flags
