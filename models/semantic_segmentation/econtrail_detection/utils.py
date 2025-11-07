from PIL import Image, ImageDraw

import os
import io
import numpy as np
import torch
import cv2
from tqdm import tqdm


import sys


current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
create_model_dir = os.path.join(parent_dir, '2_Create_Model/')

# Add the create_model_dir to the system path
sys.path.append(create_model_dir)



def predict_single_tile(model, image, device, tile_size,th):
    """
    Predicts a single tile from the given image.

    Parameters:
    - model: The trained model for prediction.
    - image: Input image (PIL.Image format or NumPy array).
    - device: PyTorch device (e.g., 'cpu' or 'cuda').
    - tile_size: Size of the tile for processing.

    Returns:
    - predicted_tile_mask: Predicted mask for the specified tile.
    """
    # Ensure image is in PIL format
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    width, height = image.size

    # Extract the first tile from the top-left corner
    x, y = 0, 0
    x_end = min(x + tile_size, width)
    y_end = min(y + tile_size, height)

    # Crop and process the tile
    tile = image.crop((x, y, x_end, y_end))

    # Resize the tile to 256x256 for the model if necessary
    if tile_size != 256:
        tile = cv2.resize(np.array(tile), (256, 256), interpolation=cv2.INTER_LINEAR)
    else:
        tile = np.array(tile)

    tile_tensor = preprocess_tile2(tile).to(device)

    with torch.no_grad():
        output = model(tile_tensor)

    # Postprocess the model output to generate a mask
    predicted_tile_mask = postprocess_mask(output,th)

    # Resize back to the original tile size if resized earlier
    if tile_size != 256:
        predicted_tile_mask = cv2.resize(predicted_tile_mask, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)

    return predicted_tile_mask



def create_img_masks_lists(images_path,masks_path):
    images_list=sorted([Image.open(os.path.join(images_path,im)for im in os.listdir(images_path))])
    masks_list=sorted([cv2.imread(os.path.join(masks_path,im)for im in os.listdir(masks_path))])
    return images_list,masks_list

def preprocess_tile2(tile):
    tile = np.array(tile)
    tile = tile / 255.0  # Normalize to [0, 1]
    tile = torch.from_numpy(tile).float().permute(2, 0, 1).unsqueeze(0)  # Convert to torch tensor
    return tile

def postprocess_mask(mask,th):
    mask = torch.sigmoid(mask)  # Apply sigmoid to get probabilities
    mask = mask.squeeze().cpu().detach().numpy()  # Remove batch dimension and move to CPU
    mask = (mask > th).astype(np.float32)  # Convert to binary mask
    return mask


def predict_single_tile(model, image, device, tile_size,th):
    """
    Predicts a single tile from the given image.

    Parameters:
    - model: The trained model for prediction.
    - image: Input image (PIL.Image format or NumPy array).
    - device: PyTorch device (e.g., 'cpu' or 'cuda').
    - tile_size: Size of the tile for processing.

    Returns:
    - predicted_tile_mask: Predicted mask for the specified tile.
    """
    # Ensure image is in PIL format
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    width, height = image.size

    # Extract the first tile from the top-left corner
    x, y = 0, 0
    x_end = min(x + tile_size, width)
    y_end = min(y + tile_size, height)

    # Crop and process the tile
    tile = image.crop((x, y, x_end, y_end))

    # Resize the tile to 256x256 for the model if necessary
    if tile_size != 256:
        tile = cv2.resize(np.array(tile), (256, 256), interpolation=cv2.INTER_LINEAR)
    else:
        tile = np.array(tile)

    tile_tensor = preprocess_tile2(tile).to(device)

    with torch.no_grad():
        output = model(tile_tensor)

    # Postprocess the model output to generate a mask
    predicted_tile_mask = postprocess_mask(output,th)

    # Resize back to the original tile size if resized earlier
    if tile_size != 256:
        predicted_tile_mask = cv2.resize(predicted_tile_mask, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)

    return predicted_tile_mask



def create_img_masks_lists(images_path,masks_path):
    images_list=sorted([Image.open(os.path.join(images_path,im)for im in os.listdir(images_path))])
    masks_list=sorted([cv2.imread(os.path.join(masks_path,im)for im in os.listdir(masks_path))])
    return images_list,masks_list



def predict_tiles(images_paths,model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    masks = []
    for image_path in tqdm(images_paths):
        image=Image.open(image_path)
        mask=predict_single_tile(model, image, device, tile_size=256)   

        masks.append(mask)
    return masks

def sliding_window_predict2(model, image, device, tile_size, stride, th):
    """
    Tiled inference with overlap (stride) and edge handling.
    Returns a probability mask in [0,1], averaged over overlaps.
    """
    width, height = image.size

    # Accumulators to average overlapping tiles
    accum = np.zeros((height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    # Step over the image with given stride; ensure we also cover the right/bottom edges
    ys = list(range(0, max(height - tile_size + 1, 1), stride))
    xs = list(range(0, max(width  - tile_size + 1, 1), stride))
    if ys[-1] != height - tile_size:
        ys.append(max(height - tile_size, 0))
    if xs[-1] != width - tile_size:
        xs.append(max(width - tile_size, 0))

    for y in ys:
        for x in xs:
            # Crop tile (tile_size x tile_size)
            tile = image.crop((x, y, x + tile_size, y + tile_size))

            # Resize to model's input if needed (256 here)
            if tile_size != 256:
                tile_np = cv2.resize(np.array(tile), (256, 256), interpolation=cv2.INTER_LINEAR)
            else:
                tile_np = np.array(tile)

            tile_tensor = preprocess_tile2(tile_np).to(device)

            with torch.no_grad():
                output = model(tile_tensor)

            # IMPORTANT: postprocess should return PROBABILITIES in [0,1].
            # If yours returns a binary mask already, consider adding a `postprocess_prob`
            # so we can average probabilities and threshold once at the end.
            prob_tile = postprocess_mask(output, th)  # expects float probs

            # Back to tile_size if we resized earlier
            if tile_size != 256:
                prob_tile = cv2.resize(prob_tile, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)

            # Accumulate and count overlaps
            accum[y:y + tile_size, x:x + tile_size] += prob_tile
            weight[y:y + tile_size, x:x + tile_size] += 1.0

    # Avoid division by zero
    weight = np.maximum(weight, 1e-6)
    full_prob = accum / weight
    return full_prob


def Full_scene_probab_mask1(model, image_path, device, tile_sizes, stride, th=None):
    """
    Run multi-scale tiled inference and average the probability maps.

    tile_sizes: list/tuple of ints, e.g., [250, 500, 100]
    stride: int, overlap step in pixels
    th: optional threshold to binarize final averaged probs
    """
    image = Image.open(image_path).convert('RGB')

    prob_maps = []
    for ts in tile_sizes:
        prob = sliding_window_predict2(model, image, device, tile_size=ts, stride=stride, th=th)
        prob_maps.append(prob)

    prob_avg = np.mean(prob_maps, axis=0)

    if th is not None:
        mask = (prob_avg >= th).astype(np.float32)
        return mask, image
    else:
        return prob_avg, image





