"""Comparison table across evaluated configurations: print to stdout and save CSV."""

from __future__ import annotations

import csv
from pathlib import Path

from eval.metrics import EvalResult

_COLUMNS = [
    ("config", "{}"),
    ("gt_count", "{}"),
    ("pred_count", "{}"),
    ("count_error", "{:+d}"),
    ("num_matched", "{}"),
    ("dwell_mae_s", "{:.3f}"),
    ("dwell_mape_pct", "{:.2f}"),
    ("fps", "{:.1f}"),
]


def _row(name: str, result: EvalResult) -> list[str]:
    values = {"config": name, **result.__dict__}
    return [fmt.format(values[key]) for key, fmt in _COLUMNS]


def comparison_table(results: dict[str, EvalResult]) -> str:
    headers = [key for key, _ in _COLUMNS]
    rows = [_row(name, result) for name, result in results.items()]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    lines += [fmt.format(*row) for row in rows]
    return "\n".join(lines)


def report(results: dict[str, EvalResult], csv_path: str | Path | None = None) -> None:
    print(comparison_table(results))
    if csv_path is not None:
        with Path(csv_path).open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([key for key, _ in _COLUMNS])
            for name, result in results.items():
                writer.writerow(_row(name, result))
