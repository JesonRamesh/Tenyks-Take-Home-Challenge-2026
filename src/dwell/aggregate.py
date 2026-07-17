"""Per-track dwell aggregation.

Each track_id has the frames on which its anchor sat inside the ROI. Consecutive
in-ROI frames form a segment; a gap longer than segment_gap_frames means the track
left the zone for good and a later return is a new segment. Per-track dwell is the
sum of segment durations, mirroring how GT sums repeated visits under one
person_id. Output matches the interval format eval/ loads.
"""

from __future__ import annotations

from dataclasses import dataclass

Segment = tuple[int, int]


@dataclass(frozen=True)
class DwellRecord:
    track_id: int
    enter_frame: int
    # enter_frame + summed active frames. For a single-segment visit this is the
    # real last in-ROI frame; with re-entries it compresses the gaps out so that
    # (exit_frame - enter_frame) equals the summed dwell eval reads back.
    exit_frame: int
    dwell_s: float


def collapse_segments(segments: list[Segment]) -> Segment:
    """First enter frame and (first enter + summed segment durations) for a set of
    (enter, exit) segments under one identity. Shared with GT so both sides sum
    durations the same way."""
    enter = min(start for start, _ in segments)
    active = sum(end - start for start, end in segments)
    return enter, enter + active


def _segments(frames: list[int], gap: int) -> list[Segment]:
    segments = []
    start = prev = frames[0]
    for frame in frames[1:]:
        if frame - prev > gap:
            segments.append((start, prev))
            start = frame
        prev = frame
    segments.append((start, prev))
    return segments


def aggregate(
    in_zone_frames: dict[int, list[int]],
    fps: float,
    segment_gap_frames: int,
) -> list[DwellRecord]:
    records = []
    for track_id, frames in in_zone_frames.items():
        segments = _segments(sorted(frames), segment_gap_frames)
        enter, exit_frame = collapse_segments(segments)
        records.append(DwellRecord(track_id, enter, exit_frame, (exit_frame - enter) / fps))
    return records
