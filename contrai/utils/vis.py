import sys
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import torch
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import os
from .tools import preprocess_tile2, Full_Scene_Probability_Mask, predict_single_tile


def overlay_predict_images(images_paths, model, output_root: str | None = None):
    """
    Generate and save overlay visualizations for full-scene predictions.

    For each image path, this function computes the full-scene mask using
    ``Full_Scene_Probability_Mask`` and overlays the predicted mask on top of
    the original image.

    If ``output_root`` is not provided, results are saved under an automatically
    created ``outputs/`` directory parallel to the image folders, preserving the
    same subdirectory structure. If provided, ``output_root`` acts as the base
    folder for saving all outputs.

    :param images_paths: List of file paths to input images.
    :type images_paths: list[str]
    :param model: Trained model used to generate predictions.
    :type model: torch.nn.Module
    :param output_root: Optional root directory where results are saved.
                        If None, defaults to ``<parent>/outputs`` beside input images.
    :type output_root: str | None
    :return: List of saved overlay image paths.
    :rtype: list[str]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved_paths = []

    for image_path in tqdm(images_paths):
        mask, image = Full_Scene_Probability_Mask(model, image_path, device)
        overlayed_image = overlay_mask_on_image(image, mask)

        # Compute output path dynamically
        output_path = _make_output_path(image_path, output_root)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        overlayed_image.save(output_path)
        saved_paths.append(output_path)

    return saved_paths


def overlay_predict_tiles(images_paths, model, output_root: str | None = None):
    """
    Generate and save overlay visualizations for single-tile predictions.

    The output is saved either in a user-specified directory (``output_root``)
    or under a default ``outputs/`` folder beside the image directories.

    :param images_paths: List of file paths to input images.
    :type images_paths: list[str]
    :param model: Trained model used to generate predictions.
    :type model: torch.nn.Module
    :param output_root: Optional root directory for saving overlays.
    :type output_root: str | None
    :return: List of saved overlay image paths.
    :rtype: list[str]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved_paths = []

    for image_path in tqdm(images_paths):
        image = Image.open(image_path)
        mask = predict_single_tile(model, image, device, tile_size=256)
        overlayed_image = overlay_mask_on_image(image, mask)

        output_path = _make_output_path(image_path, output_root)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        overlayed_image.save(output_path)
        saved_paths.append(output_path)

    return saved_paths


def plot_images_grid(images, n_rows=2, n_cols=8):
    """
    Plot a list of images in a grid layout.

    :param images: Collection of images to display.
    :type images: list[PIL.Image.Image | np.ndarray]
    :param n_rows: Number of rows in the grid.
    :type n_rows: int
    :param n_cols: Number of columns in the grid.
    :type n_cols: int
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    for idx, ax in enumerate(axes.flatten()):
        if idx < len(images):
            image = images[idx]
            if isinstance(image, np.ndarray):
                ax.imshow(image)
            elif isinstance(image, Image.Image):
                ax.imshow(np.array(image))
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()


def overlay_mask_on_image(image, mask):
    """
    Overlay a segmentation mask on an image using a custom colormap.

    :param image: Input image (PIL Image or NumPy array).
    :type image: PIL.Image.Image | np.ndarray
    :param mask: Segmentation or probability mask.
    :type mask: np.ndarray
    :return: RGB image with the mask overlayed.
    :rtype: PIL.Image.Image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    colors = [
        (0.0, (0, 0, 0, 0)),            # Transparent
        (0.2, (255, 255, 255, 255)),    # White
        (0.4, (255, 255, 0, 255)),      # Yellow
        (0.6, (255, 0, 0, 255)),        # Red
        (1.0, (185, 82, 174, 255)),     # Magenta
    ]
    custom_cmap = LinearSegmentedColormap.from_list('custom_jet', colors, N=256)

    mask_rgba = (custom_cmap(mask) * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_rgba).convert("RGBA")

    new_img = image.copy().convert("RGBA")
    new_img = Image.alpha_composite(new_img, mask_img)
    return new_img.convert("RGB")


def _make_output_path(image_path: str, output_root: str | None = None) -> str:
    """
    Construct an output path for the overlayed image.

    If ``output_root`` is not provided, it defaults to an ``outputs/`` folder
    in the same parent directory as the input images.

    Example:
        >>> _make_output_path('/data/train/img001.png')
        '/data/outputs/train/img001.png'

        >>> _make_output_path('/data/train/img001.png', '/custom/output')
        '/custom/output/train/img001.png'

    :param image_path: Original image path.
    :type image_path: str
    :param output_root: Optional base directory for outputs.
    :type output_root: str | None
    :return: Full path where the overlay should be saved.
    :rtype: str
    """
    image_path = os.path.abspath(image_path)
    parent_dir = os.path.dirname(image_path)
    grandparent = os.path.dirname(parent_dir)
    relative_subdir = os.path.relpath(parent_dir, grandparent)

    # Default base folder if user didn't provide one
    if output_root is None:
        output_root = os.path.join(grandparent, "outputs")

    output_dir = os.path.join(output_root, relative_subdir)
    filename = os.path.basename(image_path)
    return os.path.join(output_dir, filename)
