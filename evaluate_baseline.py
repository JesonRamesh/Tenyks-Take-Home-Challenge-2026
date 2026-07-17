"""Score the baseline pipeline's predictions against the hand-labelled kiosk GT.

    python evaluate_baseline.py

kiosk_gt.yaml is a {video, fps, events, ...} structure where one person_id can
hold several (enter, exit) visits, so we sum each person's durations with the same
collapse_segments the pipeline uses, then hand flat interval lists to the eval
harness unchanged.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import yaml

from eval.label.schema import PersonInterval, load_predictions
from eval.metrics import evaluate
from eval.report import report
from src.dwell.aggregate import collapse_segments


def load_kiosk_ground_truth(path: Path) -> tuple[list[PersonInterval], float]:
    raw = yaml.safe_load(path.read_text())
    visits: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for event in raw["events"]:
        visits[event["person_id"]].append((event["enter_frame"], event["exit_frame"]))
    ground_truth = [
        PersonInterval(person_id, *collapse_segments(segments))
        for person_id, segments in visits.items()
    ]
    return ground_truth, raw["fps"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, default=Path("eval/label/kiosk_gt.yaml"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/tracks.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/cam1.yaml"))
    parser.add_argument("--csv", type=Path, default=Path("outputs/eval_report.csv"))
    args = parser.parse_args()

    ground_truth, fps = load_kiosk_ground_truth(args.gt)
    predictions = load_predictions(args.predictions)
    min_iou = yaml.safe_load(args.config.read_text())["eval"]["min_iou"]

    result = evaluate(predictions, ground_truth, fps=fps, min_iou=min_iou)
    report({"baseline": result}, csv_path=args.csv)


if __name__ == "__main__":
    main()
