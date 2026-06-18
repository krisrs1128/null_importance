"""Generate the main synthetic simulation data sweep.

Run the minimal smoke-test sweep:
    python scripts/generate_simulation_data.py --minimal

Run the full Section 8 sweep:
    python scripts/generate_simulation_data.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation_data import (  # noqa: E402
    build_filename,
    derive_seed,
    generate_linear_additive_data,
    generate_product_interaction_data,
    generate_xor_data,
    save_npz_dataset,
)

MASTER_SEED = 20260618
DATASET_TYPES = ["linear_additive", "xor", "product_interaction"]
SAMPLE_SIZES = [200, 500, 1000, 5000]
DIMENSIONS = [10, 20, 50, 100]
REP_INDICES = [0, 1, 2, 3, 4]

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
        "--master-seed",
        type=int,
        default=MASTER_SEED,
        help="Single seed controlling the entire sweep.",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Generate one small n/p/rep config per dataset type.",
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

    sample_sizes = [20] if args.minimal else SAMPLE_SIZES
    dimensions = [10] if args.minimal else DIMENSIONS
    rep_indices = [0] if args.minimal else REP_INDICES

    generated = 0
    skipped = 0
    for dataset_type in DATASET_TYPES:
        out_dir = args.output_dir / dataset_type / f"master{args.master_seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        gen_fn = GENERATORS[dataset_type]

        for n in sample_sizes:
            for p in dimensions:
                for rep_index in rep_indices:
                    derived_seed = derive_seed(
                        master_seed=args.master_seed,
                        dataset_type=dataset_type,
                        n=n,
                        p=p,
                        rep_index=rep_index,
                    )
                    filename = build_filename(dataset_type, n, p, rep_index)
                    output_path = out_dir / filename
                    if output_path.exists() and not args.overwrite:
                        logging.info("Skipping existing %s", output_path)
                        skipped += 1
                        continue

                    try:
                        X, y, y_mean, metadata = gen_fn(n=n, p=p, seed=derived_seed)
                    except ValueError as exc:
                        logging.warning(
                            "Skipping %s n=%s p=%s rep=%s: %s",
                            dataset_type,
                            n,
                            p,
                            rep_index,
                            exc,
                        )
                        skipped += 1
                        continue

                    metadata["master_seed"] = int(args.master_seed)
                    metadata["rep_index"] = int(rep_index)
                    metadata["seed"] = int(derived_seed)
                    save_npz_dataset(X, y, y_mean, metadata, output_path)
                    generated += 1
                    logging.info("Wrote %s", output_path)

    logging.info("Done: generated=%s skipped=%s", generated, skipped)


if __name__ == "__main__":
    main()
