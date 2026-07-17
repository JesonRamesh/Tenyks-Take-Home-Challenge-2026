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
