#!/usr/bin/env python3
"""Assemble the final config-vs-config comparison table from scored run dirs.

Read-only. Each run dir already holds a row.json written by score_window.py; this
collects them into one table so the comparison is generated from the artifacts the runs
actually produced, never retyped by hand.

    python compare_table.py --row "baseline (main)=outputs/baseline_main/botsort_default_{w}" \
        --row "v2=outputs/v2_final/{w}" --windows sparse sliceb crowded

`{w}` in a path is substituted with each window name. Throughput and VRAM are read from
whichever run dir the --perf-from mapping points at, because a cached-detection run's FPS
measures tracking only and must not be reported as pipeline throughput.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", action="append", required=True,
                        help="LABEL=PATH_TEMPLATE, where {w} is the window name")
    parser.add_argument("--windows", nargs="+", default=["sparse", "sliceb", "crowded"])
    parser.add_argument("--out", type=Path, default=Path("outputs/comparison_v2.md"))
    args = parser.parse_args()

    specs = [spec.split("=", 1) for spec in args.row]

    header = (f"| {'config':26} | {'window':8} | {'count_err':>9} | {'matched':>7} | "
              f"{'coverage':>8} | {'purity':>6} | {'staffFP':>7} | {'dwellMAE':>8} |")
    sep = "|" + "|".join("-" * (len(part) ) for part in header.split("|")[1:-1]) + "|"
    lines = [header, sep]

    for label, template in specs:
        for window in args.windows:
            row_path = Path(template.replace("{w}", window)) / "row.json"
            if not row_path.exists():
                lines.append(f"| {label:26} | {window:8} | {'MISSING':>9} | {'-':>7} | "
                             f"{'-':>8} | {'-':>6} | {'-':>7} | {'-':>8} |")
                continue
            row = json.loads(row_path.read_text())
            coverage = row.get("mean_coverage")
            purity = row.get("mean_purity")
            lines.append(
                f"| {label:26} | {window:8} | {row['count_error']:>+9d} | "
                f"{row['num_matched']:>3}/{row['gt_count']:<3} | "
                f"{'-' if coverage is None else f'{coverage:.1%}':>8} | "
                f"{'-' if purity is None else f'{purity:.3f}':>6} | "
                f"{row['staff_false_positives']:>7} | {row['dwell_mae_s']:>8.2f} |"
            )

    table = "\n".join(lines)
    print(table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table + "\n")


if __name__ == "__main__":
    main()
