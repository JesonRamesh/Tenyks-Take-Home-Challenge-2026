"""Tracker interface. Concrete tracker wrappers implement this and stay swappable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.detect.base import Detection


@dataclass(frozen=True)
class Track:
    """A detection bound to a persistent identity for one frame."""

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class Tracker(Protocol):
    """A multi-object tracker. Maps per-frame detections to stable identities."""

    def update(self, detections: list[Detection], frame_index: int) -> list[Track]:
        """Advance the tracker by one frame and return the tracks active in it."""
        ...
