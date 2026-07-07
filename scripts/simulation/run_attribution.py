"""Run attribution benchmarks over generated simulation CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from simulations.attribution.run import run_sweep  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "simulation_attribution" / "minimal.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open() as file:
        config = yaml.safe_load(file)
    outputs = run_sweep(config, overwrite=True if args.overwrite else None)
    for score_path, metric_path in outputs:
        print(f"Wrote {score_path}")
        print(f"Wrote {metric_path}")


if __name__ == "__main__":
    main()
