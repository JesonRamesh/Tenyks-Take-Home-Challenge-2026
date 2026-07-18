"""Stationarity / min-dwell gate.

The kiosk ROI unavoidably spans the shop entrance walkway, so people cross its
interior at walking pace without ever using the kiosk. Zone hardening only catches
those who clip the polygon edge; someone who walks straight through the middle
still passes the membership test. This gate keeps a track as a real visit only if
it lingers: its total in-ROI dwell reaches min_dwell_s, OR it holds nearly still
(centroid step <= max_step_px) for a sustained run of frames while inside. A
pass-through satisfies neither and is dropped before counting.
"""

from __future__ import annotations

import numpy as np

Point = tuple[float, float]


def is_visit(
    frames: list[int],
    anchors: list[Point],
    fps: float,
    min_dwell_s: float,
    max_step_px: float,
    min_still_frames: int,
) -> bool:
    """frames/anchors are the track's in-ROI samples, sorted by frame."""
    if (frames[-1] - frames[0]) / fps >= min_dwell_s:
        return True
    # Longest run of consecutive frames whose frame-to-frame anchor step stays
    # under max_step_px. A gap (person briefly out of ROI) resets the run, so only
    # genuinely held-still stretches count.
    still_run = 0
    for i in range(1, len(frames)):
        if frames[i] - frames[i - 1] != 1:
            still_run = 0
            continue
        step = np.hypot(anchors[i][0] - anchors[i - 1][0], anchors[i][1] - anchors[i - 1][1])
        if step <= max_step_px:
            still_run += 1
            if still_run >= min_still_frames:
                return True
        else:
            still_run = 0
    return False
