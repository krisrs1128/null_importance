"""Generate the main synthetic simulation data sweep.

Run the minimal smoke-test sweep:
    python scripts/generate_simulation_data.py --minimal

Run the full Section 8 sweep:
    python scripts/generate_simulation_data.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation_data import (  # noqa: E402
    build_filename,
    generate_linear_additive_data,
    generate_product_interaction_data,
    generate_xor_data,
    save_npz_dataset,
)

SEED = 0
DATASET_TYPES = ["linear_additive", "xor", "product_interaction"]
SAMPLE_SIZES = [200, 500, 1000, 5000]
DIMENSIONS = [10, 20, 50, 100]

GENERATORS = {
    "linear_additive": generate_linear_additive_data,
    "xor": generate_xor_data,
    "product_interaction": generate_product_interaction_data,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "simulations",
        help="Root directory for generated simulation files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Global seed used with random.seed and np.random.seed.",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Generate one small n/p config per dataset type.",
    )
    parser.add_argument(
        "--dataset-types",
        nargs="+",
        choices=DATASET_TYPES,
        default=None,
        help="Dataset types to generate, in the requested order.",
    )
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Sample sizes to generate, in the requested order.",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=None,
        help="Feature dimensions to generate, in the requested order.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NPZ files. By default existing outputs are skipped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataset_types = args.dataset_types or DATASET_TYPES
    sample_sizes = args.sample_sizes or ([20] if args.minimal else SAMPLE_SIZES)
    dimensions = args.dimensions or ([10] if args.minimal else DIMENSIONS)

    generated, skipped = generate_sweep(
        output_dir=args.output_dir,
        seed=args.seed,
        dataset_types=dataset_types,
        sample_sizes=sample_sizes,
        dimensions=dimensions,
        overwrite=args.overwrite,
    )
    logging.info("Done: generated=%s skipped=%s", generated, skipped)


def generate_sweep(
    *,
    output_dir: Path,
    seed: int,
    dataset_types: list[str],
    sample_sizes: list[int],
    dimensions: list[int],
    overwrite: bool = False,
) -> tuple[int, int]:
    random.seed(int(seed))
    np.random.seed(int(seed))
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_config = {
        "seed": int(seed),
        "dataset_types": list(dataset_types),
        "sample_sizes": [int(n) for n in sample_sizes],
        "dimensions": [int(p) for p in dimensions],
        "loop_order": "dataset_type -> n -> p",
    }
    with (output_dir / "sweep_config.json").open("w") as file:
        json.dump(sweep_config, file, indent=2)
        file.write("\n")

    generated = 0
    skipped = 0
    for dataset_type in dataset_types:
        out_dir = output_dir / dataset_type
        out_dir.mkdir(parents=True, exist_ok=True)
        gen_fn = GENERATORS[dataset_type]

        for n in sample_sizes:
            for p in dimensions:
                filename = build_filename(dataset_type, n, p, seed)
                output_path = out_dir / filename

                try:
                    X, y, y_mean, metadata = gen_fn(n=n, p=p, seed=None)
                except ValueError as exc:
                    logging.warning(
                        "Skipping %s n=%s p=%s: %s",
                        dataset_type,
                        n,
                        p,
                        exc,
                    )
                    skipped += 1
                    continue

                metadata["seed"] = int(seed)
                if output_path.exists() and not overwrite:
                    logging.info("Skipping existing %s", output_path)
                    skipped += 1
                    continue

                save_npz_dataset(X, y, y_mean, metadata, output_path)
                generated += 1
                logging.info("Wrote %s", output_path)

    return generated, skipped


if __name__ == "__main__":
    main()
