"""Reproducibility for case study runs."""

import subprocess
from pathlib import Path
from hydra.core.hydra_config import HydraConfig


def capture_run_metadata(script_dir: Path) -> dict:
    """What was the git commit and hydra directory used in this run?

    Args:
        script_dir: Path to the directory where the main() hydra script lives

    Returns:
        A dictionary with keys:
            - git_commit: Current git commit hash
            - hydra_output_dir: Path to the hydra output directory for this run
    """
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=script_dir
    ).decode().strip()
    hydra_output_dir = HydraConfig.get().runtime.output_dir

    return {
        "git_commit": git_commit,
        "hydra_output_dir": hydra_output_dir,
    }
