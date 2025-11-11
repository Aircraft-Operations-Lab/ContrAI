
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
    Add the repo's model and predict_functions directories to sys.path.
    Assumes this script lives in .../3_Make_predictions/.
    """
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent  # .../Satellite_Detection_UC3M
    create_model_dir = repo_root / "2_Create_Model"
    make_predictions_dir = repo_root / "3_Make_predictions"

    for p in (create_model_dir, make_predictions_dir):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.append(p_str)

add_repo_paths()

# Now that paths are set, import project modules
from .CoaT_U import CoaT_U
from .utils import Full_Scene_Probability_Mask
from .vis import overlay_mask_on_image

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """
    Load NeXtViT_U with pretrained weights.

    Parameters
    ----------
    model_path : Path
        Path to the checkpoint file (state_dict or checkpoint with 'state_dict').
    device : torch.device
        Device to load the model on.

    Returns
    -------
    torch.nn.Module
        The model in eval mode on the given device.
    """
    model = CoaT_U(num_classes=1)

    # Prefer safe loads (tensors only)
    state = torch.load(model_path, map_location=device, weights_only=True)

    # Some checkpoints store under 'state_dict'
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def ensure_large_image_ok() -> None:
    """Allow PIL to open very large images."""
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
    """
    Run tiled inference to get probability mask and original image array.

    Returns
    -------
    mask : np.ndarray
        Probability (or thresholded) mask.
    image : np.ndarray
        Original image as ndarray suitable for overlay.
    """
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
    """Save the overlayed image with matplotlib (preserves large dims)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(30, 30))
    plt.imshow(overlay)
    plt.axis("off")

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Contrail segmentation inference & overlay")
    p.add_argument("--model-path", type=Path, required=True, help="Path to the trained weights file")
    p.add_argument("--image", type=Path, required=True, help="Path to the input image")
    p.add_argument("--tile-h", type=int, default=250, help="Tile height")
    p.add_argument("--tile-w", type=int, default=500, help="Tile width")
    p.add_argument("--stride", type=int, default=100, help="Stride for tiling")
    p.add_argument("--threshold", type=float, default=0.09, help="Probability threshold")
    p.add_argument("--output", type=Path, default=None, help="Output path for the overlayed image (PNG recommended)")
    p.add_argument("--show", action="store_true", help="Show the overlay in a window")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    ensure_large_image_ok()

    device = get_device()
    logging.info("Using device: %s", device)

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")

    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    logging.info("Loading model from %s", args.model_path)
    model = load_model(args.model_path, device)

    logging.info(
        "Running inference: tile_h=%d tile_w=%d stride=%d threshold=%.3f",
        args.tile_h, args.tile_w, args.stride, args.threshold
    )
    mask, image = run_inference(
        model=model,
        image_path=args.image,
        device=device,
        tile_h=args.tile_h,
        tile_w=args.tile_w,
        stride=args.stride,
        threshold=args.threshold
    )

    logging.info("Creating overlay")
    overlayed_image = overlay_mask_on_image(image, mask)

    # Save if requested
    if args.output is not None:
        save_overlay(overlayed_image, args.output)
        logging.info("Saved overlay to %s", args.output)

    # Optionally visualize
    if args.show:
        plt.figure(figsize=(12, 12))
        plt.imshow(overlayed_image)
        plt.title("Overlay")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()


""" example:
    python -m models.semantic_segmentation.coat.predict_old \
  --model-path ~/Documents/PUBLIC_GITHUB/coat/models/semantic_segmentation/coat/weights/contrail/model.pth \
  --image /home/irortiza/Documents/PUBLIC_GITHUB/coat/data/GOES16/goes16_ash_rgb/2025/01/05/1030/ash_rgb_0p02deg.png \
  --tile-h 256 \
  --tile-w 256 \
  --stride 128 \
  --threshold 0.1 \
  --output None \
  --show
 """