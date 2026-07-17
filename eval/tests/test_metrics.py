"""Pin the metrics to numbers computed by hand from the fixtures."""

from pathlib import Path

import pytest

from eval.label.schema import PersonInterval, TrackInterval, load_ground_truth, load_predictions
from eval.metrics import evaluate, match, temporal_iou

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FPS = 10.0


def test_temporal_iou_matches_hand_calc():
    # T10 [0,90] vs P1 [0,100]: inter=90, union=90+100-90=100 -> 0.9
    t = TrackInterval(track_id=10, enter_frame=0, exit_frame=90)
    p = PersonInterval(person_id=1, enter_frame=0, exit_frame=100)
    assert temporal_iou(t, p) == pytest.approx(0.9)


def test_disjoint_spans_have_zero_iou():
    t = TrackInterval(track_id=13, enter_frame=300, exit_frame=310)
    p = PersonInterval(person_id=1, enter_frame=0, exit_frame=100)
    assert temporal_iou(t, p) == 0.0


def test_fixture_matching_pairs_correct_ids():
    gt = load_ground_truth(FIXTURES / "gt.yaml")
    pred = load_predictions(FIXTURES / "pred.yaml")
    pairs = {(t.track_id, p.person_id) for t, p in match(pred, gt, min_iou=0.1)}
    assert pairs == {(10, 1), (11, 2), (12, 3)}  # T13 left unmatched


def test_fixture_metrics_match_hand_calc():
    gt = load_ground_truth(FIXTURES / "gt.yaml")
    pred = load_predictions(FIXTURES / "pred.yaml")
    result = evaluate(pred, gt, fps=FPS)

    assert result.gt_count == 3
    assert result.pred_count == 4
    assert result.count_error == 1  # one spurious track over-counted
    assert result.num_matched == 3
    # MAE = mean(|1.0|, |0.5|, |1.0|) = 2.5/3
    assert result.dwell_mae_s == pytest.approx(2.5 / 3)
    # MAPE = mean(1.0/10, 0.5/10, 1.0/6) * 100
    assert result.dwell_mape_pct == pytest.approx((0.1 + 0.05 + 1 / 6) / 3 * 100)
