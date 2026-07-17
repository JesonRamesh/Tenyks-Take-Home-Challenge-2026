"""Ground-truth and prediction interval formats, plus loaders.

A person's or track's presence in the kiosk ROI is a single inclusive frame span.
We keep the two types distinct so a mismatch between "a labelled person" and "a
predicted track" is impossible to introduce by accident during matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PersonInterval:
    """Hand-labelled presence of one person in the kiosk ROI."""

    person_id: int
    enter_frame: int
    exit_frame: int

    @property
    def dwell_frames(self) -> int:
        return self.exit_frame - self.enter_frame


@dataclass(frozen=True)
class TrackInterval:
    """Predicted presence of one track in the kiosk ROI."""

    track_id: int
    enter_frame: int
    exit_frame: int

    @property
    def dwell_frames(self) -> int:
        return self.exit_frame - self.enter_frame


def load_ground_truth(path: str | Path) -> list[PersonInterval]:
    rows = yaml.safe_load(Path(path).read_text())
    return [PersonInterval(**row) for row in rows]


def load_predictions(path: str | Path) -> list[TrackInterval]:
    rows = yaml.safe_load(Path(path).read_text())
    return [TrackInterval(**row) for row in rows]
