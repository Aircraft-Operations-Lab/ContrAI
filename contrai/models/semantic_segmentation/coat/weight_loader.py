"""
Utility functions for downloading and managing pretrained and final model weights.

This module ensures that required model weight files are accessible to the
user without being bundled inside the installed package. If a weight file
is not found in the user's current working directory, it is automatically
downloaded from Zenodo (or another public repository).

Weights are stored under the user's current working directory:

    ./weights/pretrained/   -> pretrained CoaT backbone
    ./weights/final/        -> trained contrail detection model

This design allows:
- lightweight pip installations
- decoupled distribution of large model files
- reproducible results regardless of environment
"""

from __future__ import annotations

import requests
from pathlib import Path
from tqdm import tqdm


# -------------------------------------------------------------------------
# Filenames and URLs (replace with your actual Zenodo records)
# -------------------------------------------------------------------------

PRETRAINED_NAME = "coat_small_7479cf9b_checkpoint.pth"
FINAL_NAME = "coat_model.pth"

ZENODO_URL_PRETRAINED = (
    "https://zenodo.org/records/17599045/files/coat_small_7479cf9b_checkpoint.pth"
)

ZENODO_URL_FINAL = (
    "https://zenodo.org/records/17599045/files/coat_model.pth"
)


# -------------------------------------------------------------------------
# Helper: download a file with progress bar
# -------------------------------------------------------------------------

def _download_with_progress(url: str, target: Path) -> None:
    """
    Download a file from a URL into the given target path.

    :param url: Direct download URL.
    :type url: str
    :param target: Local path to save the downloaded file.
    :type target: pathlib.Path
    :raises requests.HTTPError: If the request fails.
    """
    target.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    with open(target, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {target.name}",
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


# -------------------------------------------------------------------------
# Public API: get pretrained or final weight
# -------------------------------------------------------------------------

def ensure_local_weight(weight_type: str = "pretrained") -> Path:
    """
    Ensure that the requested weight file exists locally.

    Depending on the ``weight_type`` argument, this function will return either:
    - the pretrained CoaT backbone weights
    - the final trained contrail detection model weights

    If the weight file does not exist in the user's working directory, it is
    automatically downloaded from Zenodo.

    :param weight_type: Either ``"pretrained"`` or ``"final"``.
    :type weight_type: str
    :return: Local filesystem path to the requested weight file.
    :rtype: pathlib.Path
    :raises ValueError: If an unknown ``weight_type`` is provided.
    """
    if weight_type == "pretrained":
        filename = PRETRAINED_NAME
        url = ZENODO_URL_PRETRAINED
        folder = "pretrained"
    elif weight_type == "final":
        filename = FINAL_NAME
        url = ZENODO_URL_FINAL
        folder = "final"
    else:
        raise ValueError(
            "weight_type must be either 'pretrained' or 'final', "
            f"but got: {weight_type}"
        )

    root = Path.cwd()
    target = root / "weights" / folder / filename

    if not target.exists():
        _download_with_progress(url, target)

    return target
