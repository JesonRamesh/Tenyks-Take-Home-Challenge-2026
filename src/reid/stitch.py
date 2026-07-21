"""Gap-stitching: merge track fragments that are the same person.

ByteTrack is motion-only, so a person occluded for longer than its track buffer
returns as a fresh track_id. This post-process re-links such fragments: a track
that ends is merged with a later one that starts soon after (within gap_frames),
near where the first left off (within max_anchor_dist), and looks the same
(appearance cosine >= min_similarity). All three gates must hold, which keeps the
weaker ImageNet appearance signal from over-merging adjacent different people. Above
all three sits a hard invariant: two identities never merge if their segments were ever
simultaneously active (two people on screen at once cannot be one person), enforced
across whole merged groups so a transitive A-B, A-C chain can't collapse overlapping
B and C into one id.

Scope is within-visit occlusion breaks, not multi-visit returns: a person who
leaves and comes back minutes later exceeds gap_frames and stays a separate
identity, matching how GT keeps distinct visits under one person only for genuine
short step-outs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Point = tuple[float, float]


@dataclass(frozen=True)
class TrackAppearance:
    first_frame: int
    last_frame: int
    first_anchor: Point
    last_anchor: Point
    embedding: np.ndarray  # mean box embedding over the track, L2-normalized


def _distance(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def stitch(
    appearances: dict[int, TrackAppearance],
    gap_frames: int,
    max_anchor_dist: float,
    min_similarity: float,
) -> dict[int, int]:
    """Map each track_id to a canonical id (the earliest id in its merged group)."""
    # Union-find keyed so the representative is always the smallest track_id in a
    # group, giving stable, interpretable merged ids.
    parent = {track_id: track_id for track_id in appearances}
    # Per current root: the frame intervals of every member of that group, maintained
    # incrementally on union. The simultaneity check then scans only the two groups'
    # intervals instead of rescanning all tracks to rebuild the member lists, which
    # keeps stitch near O(N^2) rather than O(N^3) -- it matters on the full video, where
    # the raw-track count is large.
    intervals: dict[int, list[tuple[int, int]]] = {
        track_id: [(app.first_frame, app.last_frame)] for track_id, app in appearances.items()
    }

    def find(track_id: int) -> int:
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return
        root, other = (root_a, root_b) if root_a < root_b else (root_b, root_a)
        parent[other] = root
        intervals[root].extend(intervals.pop(other))

    def ever_simultaneous(a: int, b: int) -> bool:
        # True if any segment in a's group was active at the same time as any in b's group
        # (each starts on or before the other ends). Whole-group, not just the a/b pair, so
        # a transitive A-B, A-C chain can't collapse overlapping B and C into one identity.
        for start_a, end_a in intervals[find(a)]:
            for start_b, end_b in intervals[find(b)]:
                if start_a <= end_b and start_b <= end_a:
                    return True
        return False

    order = sorted(appearances, key=lambda track_id: appearances[track_id].first_frame)
    for later_pos, later_id in enumerate(order):
        later = appearances[later_id]
        for earlier_id in order[:later_pos]:
            earlier = appearances[earlier_id]
            gap = later.first_frame - earlier.last_frame
            if gap < 0 or gap > gap_frames:
                continue
            # Hard invariant before any spatial/appearance gate: two identities can never
            # merge if their segments were ever simultaneously active. Two people on screen
            # at once cannot be the same person, whatever their appearance/anchor score.
            if ever_simultaneous(earlier_id, later_id):
                continue
            if _distance(earlier.last_anchor, later.first_anchor) > max_anchor_dist:
                continue
            if float(earlier.embedding @ later.embedding) < min_similarity:
                continue
            union(earlier_id, later_id)

    return {track_id: find(track_id) for track_id in appearances}
