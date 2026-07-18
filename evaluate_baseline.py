"""Score the baseline pipeline's predictions against the hand-labelled kiosk GT.

    python evaluate_baseline.py
    python evaluate_baseline.py --slice           # score only the config eval_slice
    python evaluate_baseline.py --slice 26700 71000

kiosk_gt.yaml is a {video, fps, events, ...} structure where one person_id can
hold several (enter, exit) visits, so we sum each person's durations with the same
collapse_segments the pipeline uses, then hand flat interval lists to the eval
harness unchanged.

--slice scores a run.py --slice output against only the GT inside that window,
using the same [start, end] so the two stay in sync. GT segments crossing the
window boundary are CLIPPED to it before any duration metric, so a segment that
runs past the slice end (e.g. P6's 60520-71338 over a 71000 end) doesn't inflate
dwell error for reasons unrelated to the change being tested.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import yaml

from eval.label.schema import PersonInterval, load_predictions
from eval.metrics import evaluate, temporal_iou
from eval.report import report
from src.dwell.aggregate import collapse_segments

Window = tuple[int, int]


def load_kiosk_ground_truth(
    path: Path, window: Window | None = None
) -> tuple[list[PersonInterval], float, tuple[int, int, int, int]]:
    raw = yaml.safe_load(path.read_text())
    visits: dict[str, list[tuple[int, int]]] = defaultdict(list)
    excluded_segments = 0
    for event in raw["events"]:
        segment = (event["enter_frame"], event["exit_frame"])
        if window is not None:
            start, end = window
            enter, exit_frame = segment
            # Overlap test on the open window, then clip to it so boundary-crossing
            # segments are scored on their in-window duration only.
            if not (enter < end and exit_frame > start):
                excluded_segments += 1
                continue
            segment = (max(enter, start), min(exit_frame, end))
        visits[event["person_id"]].append(segment)

    ground_truth = [
        PersonInterval(person_id, *collapse_segments(segments))
        for person_id, segments in visits.items()
    ]
    people = {event["person_id"] for event in raw["events"]}
    stats = (len(visits), len(people) - len(visits), len(raw["events"]) - excluded_segments, excluded_segments)
    return ground_truth, raw["fps"], stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, default=Path("eval/label/kiosk_gt.yaml"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/tracks.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/cam1.yaml"))
    parser.add_argument("--csv", type=Path, default=Path("outputs/eval_report.csv"))
    parser.add_argument("--staff", type=Path, default=Path("outputs/staff.yaml"))
    # No values: use the config's eval_slice. Two values: an explicit [start, end).
    # Absent: score the full video against all GT.
    parser.add_argument("--slice", nargs="*", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    min_iou = config["eval"]["min_iou"]
    if args.slice is None:
        window, name = None, "baseline"
    else:
        window = tuple(args.slice) if args.slice else tuple(config["eval_slice"])
        name = "slice"

    ground_truth, fps, stats = load_kiosk_ground_truth(args.gt, window)
    predictions = load_predictions(args.predictions)

    if window is not None:
        included_people, excluded_people, included_segments, excluded_segments = stats
        print(
            f"slice [{window[0]}, {window[1]}]: "
            f"{included_people} GT people included ({excluded_people} excluded), "
            f"{included_segments} segments included ({excluded_segments} excluded)"
        )

    # Staff false-positive check: a track the pipeline flagged as staff should never
    # temporally match a real GT customer. Report any that do — they are the ones the
    # heuristic would wrongly exclude.
    if args.staff.exists():
        staff = load_predictions(args.staff)
        false_positives = [
            (track.track_id, person.person_id)
            for track in staff
            for person in ground_truth
            if temporal_iou(track, person) >= min_iou
        ]
        print(
            f"staff filter: {len(staff)} tracks flagged, "
            f"{len(false_positives)} matching a GT customer (false positives): {false_positives}"
        )

    result = evaluate(predictions, ground_truth, fps=fps, min_iou=min_iou)
    report({name: result}, csv_path=args.csv)


if __name__ == "__main__":
    main()
