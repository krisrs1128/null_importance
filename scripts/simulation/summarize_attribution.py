"""Summarize simulation attribution metric CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulations.attribution.summarize import summarize_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=ROOT / "results" / "simulation_attribution" / "metrics",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "simulation_attribution" / "summaries" / "summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = summarize_metrics(args.metrics_dir, args.output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
