from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from PipelineManager import PipelineManager


def run_framework(csv_path: str, output_dir: str | None = None) -> Dict[str, Any]:
    """
    Run the crisis-management framework on a prepared CSV input.

    Parameters
    ----------
    csv_path : str
        Path to the normalized input CSV.
    output_dir : str | None
        Directory where all pipeline outputs should be written.

    Returns
    -------
    dict
        Framework results returned by PipelineManager.run().
    """
    config = {
        "data_source": csv_path,
        "output_dir": output_dir,
    }
    pipeline = PipelineManager(config)
    return pipeline.run()