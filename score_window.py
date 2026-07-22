#!/usr/bin/env python3
"""Score one pipeline run over one validation window: the full comparison row.

Read-only. Collects into a single row every number the detector/tracker comparison is
judged on — strict count_error / matched (eval/metrics.evaluate), partial-credit
coverage / purity (eval/metrics.coverage_report), the staff false-positive check, and
the run's own throughput / peak VRAM — so configs are compared on identical footing
across windows instead of by re-reading several tools' prose output.

    python score_window.py --run-dir outputs/v2_sliceb --window 26700 30000 --name v2

Emits the row as JSON next to the run (row.json) and prints it. --quiet prints only the
one-line summary, which is what the parameter sweep consumes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from eval.label.schema import load_predictions
from eval.metrics import coverage_report, evaluate, temporal_iou
from evaluate_baseline import load_kiosk_ground_truth


def score(run_dir: Path, window: tuple[int, int], gt_path: Path, min_iou: float) -> dict:
    start, end = window
    ground_truth, fps, _stats = load_kiosk_ground_truth(gt_path, window)

    predictions = load_predictions(run_dir / "tracks.yaml")
    result = evaluate(predictions, ground_truth, fps=fps, min_iou=min_iou)

    # Staff false positives: a track flagged staff must never match a real GT customer.
    staff_path = run_dir / "staff.yaml"
    staff = load_predictions(staff_path) if staff_path.exists() else []
    staff_fp = [
        (track.track_id, person.person_id)
        for track in staff
        for person in ground_truth
        if temporal_iou(track, person) >= min_iou
    ]

    row = {
        "run_dir": str(run_dir),
        "window": [start, end],
        "gt_count": result.gt_count,
        "pred_count": result.pred_count,
        "count_error": result.count_error,
        "num_matched": result.num_matched,
        "dwell_mae_s": round(result.dwell_mae_s, 2),
        "dwell_mape_pct": round(result.dwell_mape_pct, 2),
        "staff_flagged": len(staff),
        "staff_false_positives": len(staff_fp),
        "staff_fp_pairs": staff_fp,
    }

    # Coverage/purity need per-frame boxes, which only the render artifact carries; a run
    # without it still yields the strict metrics, so report them as absent rather than fail.
    render_path = run_dir / "render_frames.yaml"
    if render_path.exists():
        artifact = yaml.safe_load(render_path.read_text())
        pred_track_frames: dict[int, set[int]] = defaultdict(set)
        for frame_index, rows in artifact["frames"].items():
            for track_id, _x1, _y1, _x2, _y2, kind in rows:
                if kind == 0:
                    pred_track_frames[track_id].add(frame_index)

        raw_gt = yaml.safe_load(gt_path.read_text())
        gt_person_frames: dict[str, set[int]] = defaultdict(set)
        for event in raw_gt["events"]:
            enter, exit_frame = event["enter_frame"], event["exit_frame"]
            if enter < end and exit_frame > start:
                gt_person_frames[event["person_id"]].update(
                    range(max(enter, start), min(exit_frame, end))
                )

        if pred_track_frames and gt_person_frames:
            people, purity = coverage_report(dict(pred_track_frames), dict(gt_person_frames))
            row["mean_coverage"] = round(sum(p.coverage for p in people) / len(people), 4)
            row["total_fragments"] = sum(p.fragment_count for p in people)
            row["mean_purity"] = round(sum(purity.values()) / len(purity), 4)
            row["min_purity"] = round(min(purity.values()), 4)
            row["per_person_coverage"] = {
                p.person_id: round(p.coverage, 4) for p in sorted(people, key=lambda x: x.person_id)
            }
        else:
            row["mean_coverage"] = 0.0
            row["total_fragments"] = 0
            row["mean_purity"] = None
            row["min_purity"] = None

    perf_path = run_dir / "perf.yaml"
    if perf_path.exists():
        perf = yaml.safe_load(perf_path.read_text())
        row["fps"] = perf["fps"]
        row["device"] = perf["device"]
        row["peak_vram_reserved_gb"] = perf["peak_vram_reserved_gb"]
        row["peak_vram_allocated_gb"] = perf["peak_vram_allocated_gb"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--window", nargs=2, type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/cam1.yaml"))
    parser.add_argument("--gt", type=Path, default=Path("eval/label/kiosk_gt.yaml"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    row = score(args.run_dir, tuple(args.window), args.gt, config["eval"]["min_iou"])
    row["name"] = args.name or args.run_dir.name
    (args.run_dir / "row.json").write_text(json.dumps(row, indent=2))

    coverage = row.get("mean_coverage")
    purity = row.get("mean_purity")
    print(
        f"{row['name']:28s} count_err {row['count_error']:>+4d}  "
        f"matched {row['num_matched']}/{row['gt_count']}  "
        f"cov {coverage if coverage is None else f'{coverage:.1%}':>6}  "
        f"pur {purity if purity is None else f'{purity:.3f}':>5}  "
        f"staffFP {row['staff_false_positives']}  "
        f"dwellMAE {row['dwell_mae_s']:.1f}s  fps {row.get('fps', '-')}"
    )
    if not args.quiet:
        print(json.dumps({k: v for k, v in row.items() if k != "per_person_coverage"}, indent=2))


if __name__ == "__main__":
    main()
