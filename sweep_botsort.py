#!/usr/bin/env python3
"""Sweep BoT-SORT's own hyperparameters over cached detections, scoring every window.

Same methodology as the Phase 5 stitch sweep (tune_stitch.py): replay a fixed detector
output so only the association parameters vary, and score each point with the harness
rather than by eye. Detection is replayed from a dump_detections.py cache, so a sweep
point costs only tracking + ReID, and every point sees byte-identical detector input.

Points are given as a JSON list of {name, params} objects, so a staged sweep (vary one
axis, then combine the winners) is driven from the command line instead of a hardcoded
grid:

    python sweep_botsort.py --base-config configs/cam1_v2.yaml \
        --window sliceb 26700 30000 --cache outputs/det_cache/rtdetr_sliceb.npz \
        --points sweeps/stage1.json --out-root outputs/sweep

Every point is scored on every window it is given, because tuning against a single
window already produced a misleading result once in this project.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

from score_window import score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--video", type=Path, default=Path("digital_kiosk.mp4"))
    # Repeatable: --window NAME START END --cache PATH, paired in order given.
    parser.add_argument("--window", nargs=3, action="append", required=True,
                        metavar=("NAME", "START", "END"))
    parser.add_argument("--cache", action="append", required=True)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--out-root", type=Path, default=Path("outputs/sweep"))
    parser.add_argument("--gt", type=Path, default=Path("eval/label/kiosk_gt.yaml"))
    args = parser.parse_args()

    base = yaml.safe_load(args.base_config.read_text())
    min_iou = base["eval"]["min_iou"]
    points = json.loads(args.points.read_text())
    windows = [(name, int(start), int(end)) for name, start, end in args.window]
    if len(windows) != len(args.cache):
        raise ValueError(f"{len(windows)} windows but {len(args.cache)} caches")

    rows: list[dict] = []
    for point in points:
        for (window_name, start, end), cache in zip(windows, args.cache):
            run_dir = args.out_root / f"{point['name']}_{window_name}"
            run_dir.mkdir(parents=True, exist_ok=True)

            config = copy.deepcopy(base)
            # Replay the dumped detections: the detector is held fixed so the sweep
            # measures association only.
            config["detector"] = {**base["detector"], "type": "cached", "cache": cache}
            config["tracker"] = {**base["tracker"], "params": point["params"]}
            # A point may also re-tune the post-hoc stitch, so "is the stitch still needed
            # on top of this tracker, and at what thresholds" is answerable in the same
            # sweep, against the real pipeline rather than a replay of an older one.
            if "reid" in point:
                config["reid"] = {**base["reid"], **point["reid"]}
            # ...and the zone gate, so the ROI depth can be swept on exactly the same
            # cached detections and scored by the same harness as everything else.
            if "kiosk_roi" in point:
                config["kiosk_roi"] = {**base["kiosk_roi"], **point["kiosk_roi"]}
            # Coverage/purity need the per-frame artifact, so force it on for scoring.
            config["overlay"] = {**base.get("overlay", {}), "emit_render_frames": True}
            config_path = run_dir / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))

            subprocess.run(
                [sys.executable, "-m", "src.run", "--video", str(args.video),
                 "--config", str(config_path), "--out-dir", str(run_dir),
                 "--slice", str(start), str(end)],
                check=True,
            )
            row = score(run_dir, (start, end), args.gt, min_iou)
            row["name"] = point["name"]
            row["window"] = window_name
            row["params"] = point["params"]
            rows.append(row)
            (run_dir / "row.json").write_text(json.dumps(row, indent=2))
            print(
                f"[{point['name']:>22} | {window_name:8}] "
                f"count_err {row['count_error']:>+4d}  matched {row['num_matched']}/{row['gt_count']}  "
                f"cov {row.get('mean_coverage', 0):.1%}  pur {row.get('mean_purity') or 0:.3f}  "
                f"staffFP {row['staff_false_positives']}  fps {row.get('fps', '-')}",
                flush=True,
            )

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "sweep_rows.json").write_text(json.dumps(rows, indent=2))

    print("\n" + "=" * 100)
    print(f"{'point':24} {'window':9} {'cnt_err':>8} {'matched':>8} {'cov':>7} {'pur':>6} {'sFP':>4} {'fps':>7}")
    print("=" * 100)
    for row in rows:
        print(f"{row['name']:24} {row['window']:9} {row['count_error']:>+8d} "
              f"{row['num_matched']:>4}/{row['gt_count']:<3} {row.get('mean_coverage', 0):>6.1%} "
              f"{row.get('mean_purity') or 0:>6.3f} {row['staff_false_positives']:>4} "
              f"{row.get('fps', 0):>7.1f}")


if __name__ == "__main__":
    main()
