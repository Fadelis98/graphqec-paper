"""Lazy lookup of the experimental-data root used by the Sycamore loader.

Only the classes that read recorded hardware data need ``$QEC_DATA_PATH``; the
check runs when such an object is constructed, never at import time, so every
simulation-only code path works without the variable.
"""

import os

ENV_VAR = "QEC_DATA_PATH"


class MissingExperimentDataError(RuntimeError):
    """Raised when an experimental-data loader is used without ``$QEC_DATA_PATH``."""


def qec_data_path(purpose: str = "experimental QEC data") -> str:
    path = os.environ.get(ENV_VAR)
    if not path:
        raise MissingExperimentDataError(
            f"Loading {purpose} requires the environment variable ${ENV_VAR} to point at the "
            "directory that holds the downloaded experiment data (e.g. the Sycamore release, "
            "doi:10.5281/zenodo.6804040). Simulated codes (BB, color, SHYPS, 4D toric, rotated "
            "surface) do not need it."
        )
    return os.path.abspath(path)
