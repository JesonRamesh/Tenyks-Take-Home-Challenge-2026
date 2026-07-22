"""Staff-exclusion heuristic (no trained classifier).

Staff wear a specific uniform: a horizontal green-over-red-over-white chest stripe
on an otherwise all-black outfit with a dark head covering. The stripe is the
distinguishing signal — plain black clothing is common on customers, so darkness
alone would over-flag. A frame reads as staff only if a chest band holds BOTH a
saturated-green and a saturated-red cluster AND the green cluster sits above the
red one (the uniform's actual layout).

Two calibration facts drive the ranges, both from real staff crops:
- The black outfit renders as mid-gray on this camera, so a low-brightness test is
  unreliable; the green-above-red ordering is used as the confirming signal instead.
- The staff are dark-skinned and skin falls at hue ~0-15, overlapping pure red, so
  red is matched on the high wraparound side (hue ~155-179) only — where the stripe
  red actually sits — to keep skin out of the red cluster.

Ranges stay config-driven for re-tuning against the full video.
"""

from __future__ import annotations

import cv2
import numpy as np

# Horizontal center strip of the box used for the band test, to keep the left/right
# background out of the color measurement.
_CORE_X = (0.2, 0.8)


class StaffClassifier:
    def __init__(self, cfg: dict):
        self.stripe_lo, self.stripe_hi = cfg["stripe_band"]
        self.green_low = np.array(cfg["green_hsv_low"], dtype=np.uint8)
        self.green_high = np.array(cfg["green_hsv_high"], dtype=np.uint8)
        self.red_low = np.array(cfg["red_hsv_low"], dtype=np.uint8)
        self.red_high = np.array(cfg["red_hsv_high"], dtype=np.uint8)
        self.min_cluster_frac = cfg["min_cluster_frac"]

    def _is_staff_frame(self, crop: np.ndarray) -> bool:
        h, w = crop.shape[:2]
        core = crop[:, int(_CORE_X[0] * w) : int(_CORE_X[1] * w)]
        lo, hi = int(self.stripe_lo * h), int(self.stripe_hi * h)
        if hi <= lo or core.shape[1] == 0:
            return False

        band = cv2.cvtColor(core[lo:hi], cv2.COLOR_BGR2HSV)
        band_px = band.shape[0] * band.shape[1]
        green_per_row = cv2.inRange(band, self.green_low, self.green_high).sum(axis=1) / 255
        red_per_row = cv2.inRange(band, self.red_low, self.red_high).sum(axis=1) / 255
        if green_per_row.sum() / band_px < self.min_cluster_frac:
            return False
        if red_per_row.sum() / band_px < self.min_cluster_frac:
            return False

        rows = np.arange(band.shape[0])
        green_center = (rows * green_per_row).sum() / green_per_row.sum()
        red_center = (rows * red_per_row).sum() / red_per_row.sum()
        return green_center < red_center

    def staff_frames(self, frame: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> list[bool]:
        """Per-box staff verdict for one frame."""
        height, width = frame.shape[:2]
        flags = []
        for x1, y1, x2, y2 in boxes:
            xi1, yi1 = max(0, int(x1)), max(0, int(y1))
            xi2, yi2 = min(width, int(x2)), min(height, int(y2))
            flags.append(self._is_staff_frame(frame[yi1:yi2, xi1:xi2]))
        return flags


def is_staff_track(constituents: list[tuple[int, int]], min_staff_frame_frac: float) -> bool:
    """Track-level staff verdict from its merged segments' (staff_frames, total_frames).

    A merged identity is staff if its dominant segment — the constituent raw track with
    the most frames — shows the uniform in at least min_staff_frame_frac of its frames.
    Judging the dominant segment rather than the pooled fraction over the whole merged
    track keeps a real staff member flagged when the stitch merges in segments where the
    uniform isn't camera-facing: those would otherwise dilute the pooled fraction below
    the threshold (the staff-680 dilution). For an unmerged track it is identical to the
    old whole-track fraction.
    """
    staff_hits, frames = max(constituents, key=lambda segment: segment[1])
    return frames > 0 and staff_hits / frames >= min_staff_frame_frac
