import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# -----------------------------------------------------------------------------
# Paths & imports for local modules
# -----------------------------------------------------------------------------

def add_repo_paths() -> None:
    """
    Add the repo's relevant directories to sys.path.
    Assumes this file lives somewhere inside the repo.
    """
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3] if len(script_dir.parents) > 3 else script_dir.parents[1]

    # Adjust these to match your layout if needed
    model_root = repo_root / "models" / "semantic_segmentation" / "econtrail_detection"
    utils_root = model_root

    for p in (model_root, utils_root, repo_root):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.append(p_str)

add_repo_paths()

from .CoaT_U import CoaT_U
from .utils import Full_Scene_Probability_Mask
from .vis import overlay_mask_on_image

# -----------------------------------------------------------------------------
# CONFIG (edit these for your setup)
# -----------------------------------------------------------------------------

# Path to model weights
MODEL_PATH = Path(
    "/home/irortiza/Documents/PUBLIC_GITHUB/ECONTRAIL_detection/"
    "models/semantic_segmentation/econtrail_detection/weights/contrail/model.pth"
)

# Path to input Ash-RGB image
IMAGE_PATH = Path(
    "/home/irortiza/Documents/PUBLIC_GITHUB/ECONTRAIL_detection/"
    "data/ash_rgb_goes16/2025/01/05/1030/ash_rgb_0p02deg.png"
)

# Where to save overlay; set to None to disable saving
OUTPUT_PATH = Path(
    "/home/irortiza/Documents/PUBLIC_GITHUB/ECONTRAIL_detection/"
    "data/ash_rgb_goes16/2025/01/05/1030/ash_rgb_0p02deg_overlay.png"
)

# Whether to show overlay in a window
SHOW = False

# Log level
LOG_LEVEL = "INFO"

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    model = CoaT_U(num_classes=1)

    state = torch.load(model_path, map_location=device, weights_only=True)

    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def ensure_large_image_ok() -> None:
    Image.MAX_IMAGE_PIXELS = None


def run_inference(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    tile_h: int,
    tile_w: int,
    stride: int,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    return Full_Scene_Probability_Mask(
        model,
        str(image_path),
        device,
        tile_h,
        tile_w,
        stride,
        threshold
    )


def save_overlay(overlay: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(30, 30))
    plt.imshow(overlay)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close()

# -----------------------------------------------------------------------------
# CLI (only tiling + threshold)
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Contrail segmentation inference & overlay "
                    "(paths and options configured inside the script)."
    )
    p.add_argument("--tile-h", type=int, default=256, help="Tile height")
    p.add_argument("--tile-w", type=int, default=256, help="Tile width")
    p.add_argument("--stride", type=int, default=128, help="Stride for tiling")
    p.add_argument("--threshold", type=float, default=0.1, help="Probability threshold")
    return p.parse_args()

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    ensure_large_image_ok()

    device = get_device()
    logging.info("Using device: %s", device)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Input image not found: {IMAGE_PATH}")

    logging.info("Loading model from %s", MODEL_PATH)
    model = load_model(MODEL_PATH, device)

    logging.info(
        "Running inference: tile_h=%d tile_w=%d stride=%d threshold=%.3f",
        args.tile_h, args.tile_w, args.stride, args.threshold
    )

    mask, image = run_inference(
        model=model,
        image_path=IMAGE_PATH,
        device=device,
        tile_h=args.tile_h,
        tile_w=args.tile_w,
        stride=args.stride,
        threshold=args.threshold
    )

    logging.info("Creating overlay")
    overlayed_image = overlay_mask_on_image(image, mask)

    if OUTPUT_PATH is not None:
        save_overlay(overlayed_image, OUTPUT_PATH)
        logging.info("Saved overlay to %s", OUTPUT_PATH)

    if SHOW:
        plt.figure(figsize=(12, 12))
        plt.imshow(overlayed_image)
        plt.title("Overlay")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()


""" python -m models.semantic_segmentation.econtrail_detection.predict \
  --tile-h 256 \
  --tile-w 256 \
  --stride 128 \
  --threshold 0.1
 """