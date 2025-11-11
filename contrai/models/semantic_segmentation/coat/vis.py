import sys
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from IPython.display import display, Image as IPImage
import cv2
import numpy as np
from tqdm import tqdm
from .utils import preprocess_tile2, Full_Scene_Probability_Mask, predict_single_tile




def overlay_predict_images(images_paths,model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    overlayed_images = []
    for image_path in tqdm(images_paths):
        mask, image = Full_Scene_Probability_Mask(model, image_path, device)
        overlayed_image = overlay_mask_on_image(image, mask)
        overlayed_images.append(overlayed_image)
    return overlayed_images




def overlay_predict_tiles(images_paths,model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    overlayed_images = []
    for image_path in tqdm(images_paths):
        image=Image.open(image_path)
        mask=predict_single_tile(model, image, device, tile_size=256)   

        overlayed_image = overlay_mask_on_image(image, mask)
        overlayed_images.append(overlayed_image)
    return overlayed_images




def plot_images_grid(images, n_rows=2, n_cols=8):
    """
    Plots a grid of images with specified number of rows and columns.

    Parameters:
    - images: List of images to plot (PIL.Image or NumPy arrays).
    - n_rows: Number of rows in the grid.
    - n_cols: Number of columns in the grid.

    Returns:
    - None
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
    Draws only the segmented elements of the mask on top of the image.

    Parameters:
    - image: Input image (NumPy array or PIL.Image).
    - mask: Mask to overlay (NumPy array).

    Returns:
    - new_img: The image with only the segmented elements of the mask drawn on top.
    """
    # Ensure image is in PIL format
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Define custom colormap with transparency
    colors = [(0, (0, 0, 0, 0)),    # Transparent for black (value 0)
              (0.2, (255, 255, 255, 255)),  # White
              (0.4, (255, 255, 0, 255)), 
                (0.6, (255, 0, 0, 255)),     # Yellow
              (1, (185, 82, 174, 255)) ]        # Red
    custom_cmap = LinearSegmentedColormap.from_list('custom_jet', colors, N=256)

    # Apply the colormap to the mask
    mask_rgba = (custom_cmap(mask) * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_rgba).convert("RGBA")

    # Create a new image to draw only the segmented elements
    new_img = image.copy().convert("RGBA")
    mask_data = np.array(mask_img)

    # Add the mask overlay
    mask_overlay = Image.fromarray(mask_data)
    new_img = Image.alpha_composite(new_img, mask_overlay)

    return new_img.convert("RGB")



