"""
Prediction utilities and CLI for contrail segmentation.

Provides a library-style :func:`predict` function for use in Python code
and notebooks, and a command-line interface when executed as a module.

Example
-------
From Python::

    from models.semantic_segmentation.coat.predict import predict

    overlay, mask, image = predict(
        model_path="path/to/model.pth",
        image_path="path/to/image.png",
        tile_h=256,
        tile_w=256,
        stride=128,
        threshold=0.1,
        output=None,
        show=True,
        log_level="INFO",
    )
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .CoaT_U import CoaT_U
from .utils import Full_Scene_Probability_Mask
from .vis import overlay_mask_on_image


def get_device() -> torch.device:
    """
    Select the device to run inference on.

    Returns
    -------
    torch.device
        ``"cuda"`` if available, otherwise ``"cpu"``.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """
    Load a :class:`CoaT_U` model with pretrained weights.

    Parameters
    ----------
    model_path : Path
        Path to the checkpoint file.
    device : torch.device
        Device on which to load the model.

    Returns
    -------
    torch.nn.Module
        Model in evaluation mode on the specified device.

    Raises
    ------
    FileNotFoundError
        If ``model_path`` does not exist.
    RuntimeError
        If the state dictionary cannot be loaded.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = CoaT_U(num_classes=1)

    state = torch.load(model_path, map_location=device, weights_only=True)

    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def ensure_large_image_ok() -> None:
    """
    Configure PIL to support very large input images.
    """
    Image.MAX_IMAGE_PIXELS = None


def run_inference(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    tile_h: int,
    tile_w: int,
    stride: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run tiled inference and return probability mask and input image.

    Parameters
    ----------
    model : torch.nn.Module
        Segmentation model in evaluation mode.
    image_path : Path
        Path to the input image.
    device : torch.device
        Device used for inference.
    tile_h : int
        Tile height in pixels.
    tile_w : int
        Tile width in pixels.
    stride : int
        Stride between tiles in pixels.
    threshold : float
        Probability threshold to apply.

    Returns
    -------
    mask : numpy.ndarray
        Predicted mask for the full image.
    image : numpy.ndarray
        Original image as an array.

    Raises
    ------
    FileNotFoundError
        If ``image_path`` does not exist.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    mask, image = Full_Scene_Probability_Mask(
        model,
        str(image_path),
        device,
        tile_h,
        tile_w,
        stride,
        threshold,
    )
    return mask, image


def save_overlay(overlay: np.ndarray, output_path: Path) -> None:
    """
    Save an overlay image to disk.

    Parameters
    ----------
    overlay : numpy.ndarray
        RGB overlay image of shape ``(H, W, 3)``.
    output_path : Path
        Destination file path. Parent directories are created if needed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(30, 30))
    plt.imshow(overlay)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close()


def predict(
    model_path: Path,
    image_path: Path,
    tile_h: int = 250,
    tile_w: int = 500,
    stride: int = 100,
    threshold: float = 0.09,
    output: Optional[Path] = None,
    show: bool = False,
    log_level: str = "INFO",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run contrail segmentation and optionally save/show the overlay.

    Parameters
    ----------
    model_path : Path or str
        Path to the trained model weights.
    image_path : Path or str
        Path to the input image.
    tile_h : int, optional
        Tile height in pixels, by default ``250``.
    tile_w : int, optional
        Tile width in pixels, by default ``500``.
    stride : int, optional
        Stride between tiles in pixels, by default ``100``.
    threshold : float, optional
        Probability threshold, by default ``0.09``.
    output : Path or str or None, optional
        Output path for the overlay image. If ``None``, nothing is saved.
    show : bool, optional
        If ``True``, display the overlay. Default is ``False``.
    log_level : {"DEBUG", "INFO", "WARNING", "ERROR"}, optional
        Logging level, by default ``"INFO"``.

    Returns
    -------
    overlay : numpy.ndarray
        RGB overlay image.
    mask : numpy.ndarray
        Predicted mask.
    image : numpy.ndarray
        Original input image.

    Raises
    ------
    FileNotFoundError
        If ``model_path`` or ``image_path`` do not exist.
    RuntimeError
        If model loading fails.
    """
    model_path = Path(model_path)
    image_path = Path(image_path)

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    ensure_large_image_ok()
    device = get_device()

    logging.info("Using device: %s", device)
    logging.info("Loading model from %s", model_path)

    model = load_model(model_path, device)

    logging.info(
        "Running inference: tile_h=%d tile_w=%d stride=%d threshold=%.3f",
        tile_h,
        tile_w,
        stride,
        threshold,
    )

    mask, image = run_inference(
        model=model,
        image_path=image_path,
        device=device,
        tile_h=tile_h,
        tile_w=tile_w,
        stride=stride,
        threshold=threshold,
    )

    logging.info("Creating overlay")
    overlay = overlay_mask_on_image(image, mask)

    if output is not None:
        output_path = Path(output)
        save_overlay(overlay, output_path)
        logging.info("Saved overlay to %s", output_path)

    if show:
        plt.figure(figsize=(12, 12))
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis("off")
        plt.show()

    return overlay, mask, image


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Contrail segmentation inference and overlay generation."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the trained weights file.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--tile-h",
        type=int,
        default=250,
        help="Tile height in pixels.",
    )
    parser.add_argument(
        "--tile-w",
        type=int,
        default=500,
        help="Tile width in pixels.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=100,
        help="Stride between tiles in pixels.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.09,
        help="Probability threshold.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the overlay image (PNG recommended).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the overlay image.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """
    args = parse_args()
    predict(
        model_path=args.model_path,
        image_path=args.image,
        tile_h=args.tile_h,
        tile_w=args.tile_w,
        stride=args.stride,
        threshold=args.threshold,
        output=args.output,
        show=args.show,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
