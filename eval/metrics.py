"""Count and dwell metrics.

Predicted tracks are matched to ground-truth persons by temporal IoU under a
Hungarian (optimal one-to-one) assignment. Dwell error is then reported only over
matched pairs; unmatched tracks/persons surface as count error instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from eval.label.schema import PersonInterval, TrackInterval


def temporal_iou(a: TrackInterval, b: PersonInterval) -> float:
    """IoU of two frame spans, treating dwell as (exit - enter) frames."""
    inter = max(0, min(a.exit_frame, b.exit_frame) - max(a.enter_frame, b.enter_frame))
    if inter == 0:
        return 0.0
    union = a.dwell_frames + b.dwell_frames - inter
    return inter / union


def match(
    predictions: list[TrackInterval],
    ground_truth: list[PersonInterval],
    min_iou: float,
) -> list[tuple[TrackInterval, PersonInterval]]:
    """Optimal one-to-one track<->person pairs whose temporal IoU clears min_iou."""
    if not predictions or not ground_truth:
        return []
    iou = np.zeros((len(predictions), len(ground_truth)))
    for i, track in enumerate(predictions):
        for j, person in enumerate(ground_truth):
            iou[i, j] = temporal_iou(track, person)
    # Maximise total IoU: linear_sum_assignment minimises, so negate.
    rows, cols = linear_sum_assignment(-iou)
    return [
        (predictions[i], ground_truth[j])
        for i, j in zip(rows, cols)
        if iou[i, j] >= min_iou
    ]


@dataclass
class EvalResult:
    gt_count: int
    pred_count: int
    count_error: int  # signed: positive => over-counting
    num_matched: int
    dwell_mae_s: float
    dwell_mape_pct: float
    fps: float


@dataclass
class PersonCoverage:
    person_id: str | int
    gt_frames: int
    fragment_count: int  # distinct predicted tracks that share any frame with this person
    coverage: float  # fraction of the person's GT frames with >= 1 predicted box on them


def coverage_report(
    pred_track_frames: dict[int, set[int]],
    gt_person_frames: dict[str | int, set[int]],
) -> tuple[list[PersonCoverage], dict[int, float]]:
    """Partial-credit companions to the strict IoU matcher, computed on per-frame sets.

    The binary IoU>=0.5 match can't tell "captured most of a person but split across
    fragments" from "missed them". These do:

    - coverage (per GT person): fraction of their real frames on which any predicted
      track has a box. Recall-like partial credit the binary matcher discards.
    - fragment_count (per GT person): how many distinct tracks that presence is split
      across (the same temporal-overlap notion the diagnostic uses).
    - purity (per predicted track): the largest share of a track's frames that fall on a
      single GT person. 1.0 means the track never spans two people -- the invariant the
      crowding-collapse fix enforces, so this should read ~1.0 everywhere post-fix.
    """
    all_pred_frames: set[int] = set().union(*pred_track_frames.values()) if pred_track_frames else set()
    people = [
        PersonCoverage(
            person_id=person_id,
            gt_frames=len(person_frames),
            fragment_count=sum(
                1 for track_frames in pred_track_frames.values() if not track_frames.isdisjoint(person_frames)
            ),
            coverage=len(person_frames & all_pred_frames) / len(person_frames) if person_frames else 0.0,
        )
        for person_id, person_frames in gt_person_frames.items()
    ]
    purity = {
        track_id: (
            max((len(track_frames & pf) for pf in gt_person_frames.values()), default=0) / len(track_frames)
            if track_frames
            else 0.0
        )
        for track_id, track_frames in pred_track_frames.items()
    }
    return people, purity


def evaluate(
    predictions: list[TrackInterval],
    ground_truth: list[PersonInterval],
    fps: float,
    min_iou: float = 0.1,
) -> EvalResult:
    matches = match(predictions, ground_truth, min_iou)

    abs_errors_s: list[float] = []
    abs_pct_errors: list[float] = []
    for track, person in matches:
        pred_s = track.dwell_frames / fps
        gt_s = person.dwell_frames / fps
        abs_errors_s.append(abs(pred_s - gt_s))
        # GT dwell is always > 0 for a labelled person, so no zero-division guard.
        abs_pct_errors.append(abs(pred_s - gt_s) / gt_s)

    return EvalResult(
        gt_count=len(ground_truth),
        pred_count=len(predictions),
        count_error=len(predictions) - len(ground_truth),
        num_matched=len(matches),
        dwell_mae_s=float(np.mean(abs_errors_s)) if abs_errors_s else 0.0,
        dwell_mape_pct=float(np.mean(abs_pct_errors) * 100.0) if abs_pct_errors else 0.0,
        fps=fps,
    )
