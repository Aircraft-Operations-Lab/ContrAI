from PIL import Image, ImageDraw
import os
import io
import numpy as np
import torch
import cv2
from tqdm import tqdm
import sys
from pathlib import Path
import xarray as xr

def predict_single_tile(model, image, device, tile_size: int, th: float):
    """
    Predict a segmentation mask for a single image tile.

    This function processes a single tile of the input image through a trained model,
    applies postprocessing, and returns the predicted binary mask for that tile.

    :param model: Trained PyTorch model used for prediction.
    :type model: torch.nn.Module
    :param image: Input image (PIL Image or NumPy array).
    :type image: PIL.Image.Image | np.ndarray
    :param device: Torch device (e.g., 'cpu' or 'cuda').
    :type device: torch.device
    :param tile_size: Size of the tile (in pixels) to extract and process.
    :type tile_size: int
    :param th: Threshold value (0–1) to binarize the output probabilities.
    :type th: float
    :return: Predicted binary mask for the specified tile.
    :rtype: np.ndarray
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    width, height = image.size
    x, y = 0, 0
    x_end = min(x + tile_size, width)
    y_end = min(y + tile_size, height)

    tile = image.crop((x, y, x_end, y_end))

    if tile_size != 256:
        tile = cv2.resize(np.array(tile), (256, 256), interpolation=cv2.INTER_LINEAR)
    else:
        tile = np.array(tile)

    tile_tensor = preprocess_tile2(tile).to(device)

    with torch.no_grad():
        output = model(tile_tensor)

    predicted_tile_mask = postprocess_mask(output, th)

    if tile_size != 256:
        predicted_tile_mask = cv2.resize(predicted_tile_mask, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)

    return predicted_tile_mask


def create_img_masks_lists(images_path: str, masks_path: str):
    """
    Create sorted lists of images and their corresponding masks.

    :param images_path: Directory containing input images.
    :type images_path: str
    :param masks_path: Directory containing corresponding mask images.
    :type masks_path: str
    :return: Sorted lists of loaded images and masks.
    :rtype: tuple[list[PIL.Image.Image], list[np.ndarray]]
    """
    images_list = sorted([Image.open(os.path.join(images_path, im)) for im in os.listdir(images_path)])
    masks_list = sorted([cv2.imread(os.path.join(masks_path, im)) for im in os.listdir(masks_path)])
    return images_list, masks_list


def preprocess_tile2(tile: np.ndarray) -> torch.Tensor:
    """
    Preprocess an image tile for model inference.

    Normalizes pixel values to [0, 1] and converts the image into a 4D PyTorch tensor.

    :param tile: Input RGB image tile.
    :type tile: np.ndarray
    :return: Normalized tensor of shape (1, 3, H, W).
    :rtype: torch.Tensor
    """
    tile = np.array(tile)
    tile = tile / 255.0
    tile = torch.from_numpy(tile).float().permute(2, 0, 1).unsqueeze(0)
    return tile


def postprocess_mask(mask: torch.Tensor, th: float) -> np.ndarray:
    """
    Postprocess model output to generate a binary mask.

    :param mask: Raw output tensor from the model.
    :type mask: torch.Tensor
    :param th: Threshold value (0–1) for binarization.
    :type th: float
    :return: Binary mask (values 0 or 1) as NumPy array.
    :rtype: np.ndarray
    """
    mask = torch.sigmoid(mask)
    mask = mask.squeeze().cpu().detach().numpy()
    mask = (mask > th).astype(np.float32)
    return mask


def predict_tiles(images_paths: list[str], model: torch.nn.Module):
    """
    Predict segmentation masks for multiple image files.

    :param images_paths: List of file paths to input images.
    :type images_paths: list[str]
    :param model: Trained PyTorch model for inference.
    :type model: torch.nn.Module
    :return: List of predicted masks for each input image.
    :rtype: list[np.ndarray]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    masks = []
    for image_path in tqdm(images_paths):
        image = Image.open(image_path)
        mask = predict_single_tile(model, image, device, tile_size=256, th=0.5)
        masks.append(mask)
    return masks


def Sliding_Window(model: torch.nn.Module, image: Image.Image, device: torch.device,
                   tile_h: int, tile_w: int, stride: int, th: float) -> np.ndarray:
    """
    Perform tiled inference with overlap (sliding window).

    Processes an entire image by dividing it into overlapping tiles,
    predicting each tile, and averaging overlapping regions.

    :param model: Trained PyTorch model.
    :type model: torch.nn.Module
    :param image: Input PIL image.
    :type image: PIL.Image.Image
    :param device: Device to run inference on ('cpu' or 'cuda').
    :type device: torch.device
    :param tile_h: Tile height in pixels.
    :type tile_h: int
    :param tile_w: Tile width in pixels.
    :type tile_w: int
    :param stride: Step size for moving the sliding window.
    :type stride: int
    :param th: Threshold to apply sigmoid binarization.
    :type th: float
    :return: Full-size probability map (averaged over overlapping tiles).
    :rtype: np.ndarray
    """
    width, height = image.size
    accum = np.zeros((height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    ys = list(range(0, max(height - tile_h + 1, 1), stride))
    xs = list(range(0, max(width - tile_w + 1, 1), stride))
    if ys[-1] != height - tile_h:
        ys.append(max(height - tile_h, 0))
    if xs[-1] != width - tile_w:
        xs.append(max(width - tile_w, 0))

    for y in ys:
        for x in xs:
            tile = image.crop((x, y, x + tile_w, y + tile_h))

            if (tile_h, tile_w) != (256, 256):
                tile_np = cv2.resize(np.array(tile), (256, 256), interpolation=cv2.INTER_LINEAR)
            else:
                tile_np = np.array(tile)

            tile_tensor = preprocess_tile2(tile_np).to(device)

            with torch.no_grad():
                output = model(tile_tensor)

            prob_tile = postprocess_mask(output, th)

            if (tile_h, tile_w) != (256, 256):
                prob_tile = cv2.resize(prob_tile, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)

            accum[y:y + tile_h, x:x + tile_w] += prob_tile
            weight[y:y + tile_h, x:x + tile_w] += 1.0

    weight = np.maximum(weight, 1e-6)
    full_prob = accum / weight
    return full_prob


def Full_Scene_Probability_Mask(model: torch.nn.Module, image_path: str, device: torch.device,
                                tile_h: int, tile_w: int, stride: int, th: float = None):
    """
    Generate a full-scene probability or binary mask using tiled inference.

    Uses a sliding-window approach to process the entire image efficiently.
    Optionally thresholds the probability map into a binary segmentation mask.

    :param model: Trained segmentation model.
    :type model: torch.nn.Module
    :param image_path: Path to the input image file.
    :type image_path: str
    :param device: Device to perform inference on.
    :type device: torch.device
    :param tile_h: Tile height in pixels.
    :type tile_h: int
    :param tile_w: Tile width in pixels.
    :type tile_w: int
    :param stride: Stride value controlling tile overlap.
    :type stride: int
    :param th: Optional threshold for binarization of the averaged probabilities.
    :type th: float, optional
    :return: Tuple of (mask/probability map, original image).
    :rtype: tuple[np.ndarray, PIL.Image.Image]
    """
    image_path = Path(image_path)

    if image_path.suffix == ".nc":
        # NetCDF Ash RGB case: read bands and build an RGB image
        with xr.open_dataset(image_path) as ds:
            for var_name in ("ash_red", "ash_green", "ash_blue"):
                if var_name not in ds:
                    raise ValueError(
                        f"Expected variable '{var_name}' in {image_path}, "
                        "but it was not found."
                    )

            ash_red = ds["ash_red"].values
            ash_green = ds["ash_green"].values
            ash_blue = ds["ash_blue"].values


        # Stack into RGB (H, W, 3)
        rgb = np.stack([ash_red, ash_green, ash_blue], axis=-1)
        image = Image.fromarray(rgb, mode="RGB")

    else:
        # Regular image file
        image = Image.open(image_path).convert("RGB")

    prob = Sliding_Window(
        model, image, device,
        tile_h=tile_h, tile_w=tile_w,
        stride=stride, th=th
    )

    if th is not None:
        mask = (prob >= th).astype(np.float32)
        return mask, image
    else:
        return prob, image
