"""Creates the empirical analogue of table one, giving properties of methods
across different types of null importance.

The definitions of the different types of NOA and which data sets are used to
test them are given in config.yaml. That config also describes the power and
false discovery thresholds needed to earn different labels.
"""

from pathlib import Path
import pandas as pd
import yaml

_script_dir = Path(__file__).resolve().parent

def main():
    # Read config
    config_path = _script_dir / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # types of null importance
    notions = ["marginal", "conditional", "risk", "functional", "causal"]
    categories = cfg["categories"]
    theoretical_rows = cfg["theoretical_rows"]

    rows = []
    for method, row_name in theoretical_rows.items():
        if row_name is None:
            continue

        targets = set(categories.get(method, []))
        row_data = {"method": row_name}

        for notion in notions:
            row_data[notion] = "check" if notion in targets else "cross"
        rows.append(row_data)

    # format the results
    df = pd.DataFrame(rows)
    output_path = _script_dir / "table1_theoretical.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
