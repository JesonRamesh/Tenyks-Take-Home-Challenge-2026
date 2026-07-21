"""Pin the stitch invariants, especially the no-simultaneous-merge correctness rule."""

import numpy as np

from src.reid.stitch import TrackAppearance, stitch


def _appearance(first: int, last: int, anchor=(0.0, 0.0)) -> TrackAppearance:
    # Unit embedding so appearance never blocks a merge; the temporal/anchor gates are
    # what these tests exercise.
    embedding = np.ones(4) / np.linalg.norm(np.ones(4))
    return TrackAppearance(first, last, anchor, anchor, embedding)


def test_merges_clean_sequential_fragments():
    # Non-overlapping, within the gap, same spot, same look -> one identity.
    appearances = {1: _appearance(0, 100), 2: _appearance(120, 200)}
    id_map = stitch(appearances, gap_frames=90, max_anchor_dist=1e9, min_similarity=-1.0)
    assert id_map[1] == id_map[2]


def test_never_merges_simultaneous_segments_transitively():
    # The crowding-collapse case: A (ends 100) is individually mergeable with both B
    # (starts 120) and C (starts 130) within the gap, but B and C overlap each other,
    # so they are two people on screen at once and must not collapse into one identity
    # via A. min_similarity/max_anchor_dist are wide open so only the temporal invariant
    # can keep them apart.
    appearances = {
        1: _appearance(0, 100),
        2: _appearance(120, 300),
        3: _appearance(130, 310),
    }
    id_map = stitch(appearances, gap_frames=90, max_anchor_dist=1e9, min_similarity=-1.0)
    assert id_map[2] != id_map[3]


def test_directly_overlapping_segments_stay_separate():
    appearances = {1: _appearance(0, 200), 2: _appearance(150, 400)}
    id_map = stitch(appearances, gap_frames=90, max_anchor_dist=1e9, min_similarity=-1.0)
    assert id_map[1] != id_map[2]
