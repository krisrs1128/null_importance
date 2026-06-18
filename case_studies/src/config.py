"""Case study configs."""

from pathlib import Path

import yaml


def load_config(study_dir):
    """Read config.yaml from a case study directory.

    Parameters
    ----------
    study_dir : Path
        Directory containing config.yaml.

    Returns
    -------
    dict
    """
    with open(Path(study_dir) / "config.yaml") as f:
        return yaml.safe_load(f)
