"""Run the metrics over a GT/prediction pair and print + save the comparison table.

Standalone so the eval harness can be exercised before any pipeline exists:
    python -m eval.run_eval --gt eval/fixtures/gt.yaml --pred eval/fixtures/pred.yaml --fps 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.label.schema import load_ground_truth, load_predictions
from eval.metrics import evaluate
from eval.report import report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--min-iou", type=float, default=0.1)
    parser.add_argument("--name", default="fixture")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    result = evaluate(
        load_predictions(args.pred),
        load_ground_truth(args.gt),
        fps=args.fps,
        min_iou=args.min_iou,
    )
    report({args.name: result}, csv_path=args.csv)


if __name__ == "__main__":
    main()
