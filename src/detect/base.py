"""Detector interface. Concrete model wrappers implement this and stay swappable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One person detected in a single frame, box in pixel coordinates (x1,y1)-(x2,y2)."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class Detector(Protocol):
    """A per-frame person detector. Implementations wrap one concrete model."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return person detections for one BGR frame."""
        ...
