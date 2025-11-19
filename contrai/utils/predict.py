"""
Prediction utilities and CLI for contrail segmentation.

"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Type, Union, Dict, Any, Literal
import cv2
import geojson
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from .tools import Full_Scene_Probability_Mask
from .vis import overlay_mask_on_image


def get_device() -> torch.device:
    """
    Select the best available device for model inference.

    Returns
    -------
    torch.device
        Returns a CUDA device if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_cls: Type[torch.nn.Module], model_path: Path, device: torch.device) -> torch.nn.Module:
    """
    Load a model with pretrained weights from a checkpoint.

    Parameters
    ----------
    model_cls : Type[torch.nn.Module]
        The model class to instantiate.
    model_path : Path
        Path to the checkpoint file.
    device : torch.device
        Device to load the model onto.

    Returns
    -------
    torch.nn.Module
        The model instance in evaluation mode on the specified device.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    RuntimeError
        If the state dictionary fails to load.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    # Instantiate model
    model = model_cls(num_classes=1)

    # Load weights
    state = torch.load(model_path, map_location=device)

    # Handle checkpoints saved with wrappers like PyTorch Lightning
    if isinstance(state, dict) and "state_dict" in state:
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
from typing import List, Tuple, Dict, Any, Optional

import cv2
import geojson
import numpy as np


def _find_contours(mask: np.ndarray) -> List[List[List[int]]]:
    """
    Find contours in a binary mask and return them as lists of [x, y] coordinates.

    Parameters
    ----------
    mask : np.ndarray
        2D uint8 image where non-zero pixels belong to the foreground.

    Returns
    -------
    list of list of list of int
        A list of contours, each contour being a list of [x, y] pixel coordinates.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_list: List[List[List[int]]] = []
    for contour in contours:
        contour_coords = contour.squeeze().tolist()
        # Handle case where contour has a single point
        if isinstance(contour_coords[0], int):
            contour_coords = [contour_coords]
        contours_list.append(contour_coords)

    return contours_list


from typing import List, Tuple, Optional
import numpy as np

from typing import List, Tuple, Optional
import numpy as np
def fix_geojson(fc):
    fixed = {"type": "FeatureCollection", "features": []}

    for feat in fc.get("features", []):
        geom = feat.get("geometry")
        if not geom:
            continue

        if geom.get("type") != "Polygon":
            continue  # ignore non-polygons

        coords = geom.get("coordinates", [])
        if not coords or not coords[0]:
            continue

        ring = coords[0]

        # Skip polygons with fewer than 4 coordinates
        if len(ring) < 4:
           
            continue

        # Close ring if needed
        if ring[0] != ring[-1]:
            ring = ring + [ring[0]]
            geom["coordinates"][0] = ring

        fixed["features"].append(feat)

    return fixed


def _coords_to_latlon(
    contours: List[List[List[int]]],
    coords: Optional[Tuple[int, int, int, int]],
    lat: np.ndarray,
    lon: np.ndarray,
) -> List[List[Tuple[float, float]]]:
    """
    Map pixel coordinates to (lon, lat) using provided grids.

    Parameters
    ----------
    contours : list of list of list of int
        Contours in pixel coordinates [[x, y], ...].
        OpenCV gives (x, y) = (col, row).
    coords : tuple of int, optional
        (x1, x2, y1, y2) indices if the mask is a sub-window of the full
        lat/lon grid. If None, pixel indices are taken as absolute.
    lat : np.ndarray
        Latitude grid (2D or 1D).
    lon : np.ndarray
        Longitude grid (2D or 1D).

    Returns
    -------
    list of list of tuple of float
        Contours in (lon, lat) coordinates.
    """
    georef_contours: List[List[Tuple[float, float]]] = []

    lat_ndim = lat.ndim
    lon_ndim = lon.ndim

    if not (lat_ndim in (1, 2) and lon_ndim in (1, 2)):
        raise ValueError(
            f"lat and lon must be 1D or 2D arrays, got lat.ndim={lat_ndim}, "
            f"lon.ndim={lon_ndim}"
        )
    if lat_ndim != lon_ndim:
        raise ValueError(
            "lat and lon must both be 1D or both be 2D. "
            f"Got lat.ndim={lat_ndim}, lon.ndim={lon_ndim}"
        )

    # --- Pre-compute orientation & size for y flip ---
    if lat_ndim == 2:
        nrows = lat.shape[0]
        # True if lat increases going downwards (south -> north),
        # i.e. row 0 is southernmost; image row 0 is northernmost.
        flip_y = lat[0, 0] < lat[-1, 0]
        print('flip_y:', flip_y)
    else:  # 1D
        nrows = lat.size
        flip_y = lat[0] < lat[-1]

    for contour in contours:
        georef_contour: List[Tuple[float, float]] = []

        for x_pix, y_pix in contour:
            # x_pix = column, y_pix = row in the *image/mask*
            if coords is not None:
                # coords are (x1, x2, y1, y2); use upper-left as origin
                x1, x2, y1, y2 = coords
                x0 = min(x1, x2)  # left-most col
                y0 = min(y1, y2)  # top-most row
                x = x0 + x_pix
                y = y0 + y_pix
            else:
                x = x_pix
                y = y_pix

            # Map image row to lat grid row (flip if needed)
            if flip_y:
                y_idx = (nrows - 1) - y
            else:
                y_idx = y

            # Now map to lat/lon
            if lat_ndim == 2:
                lon_val = float(lon[y_idx, x])
                lat_val = float(lat[y_idx, x])
            else:  # 1D lat/lon
                lon_val = float(lon[x])
                lat_val = float(lat[y_idx])

            georef_contour.append((lon_val, lat_val))

        georef_contours.append(georef_contour)

    return georef_contours



def _contours_to_geojson(
    contours_latlon: List[List[Tuple[float, float]]],
    feature_id: str = "contrail_polygon",
) -> Dict[str, Any]:
    """
    Build a GeoJSON FeatureCollection from a list of lon/lat contours.

    Parameters
    ----------
    contours_latlon : list of list of tuple of float
        Each element is a contour in (lon, lat) coordinates.
    feature_id : str, optional
        Base feature identifier used in the ``id`` property.

    Returns
    -------
    dict
        GeoJSON FeatureCollection with CRS EPSG:4326.
    """
    features: List[geojson.Feature] = []

    for idx, coords in enumerate(contours_latlon):
        # Close polygon if needed
        if coords and coords[0] != coords[-1]:
            coords = coords + [coords[0]]

        feature = geojson.Feature(
            geometry=geojson.Polygon([coords]),
            properties={"id": f"{feature_id}_{idx}"},
        )
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features,
        "crs": {
            "type": "name",
            "properties": {"name": "EPSG:4326"},
        },
    }


def mask_to_geojson(
    mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    coords: Optional[Tuple[int, int, int, int]] = None,
    feature_id: str = "CONTRAIL",
) -> Dict[str, Any]:
    """
    Convert a single probability mask to a GeoJSON FeatureCollection.

    Parameters
    ----------
    mask : np.ndarray
        Probability mask in [0, 1] or [0, 255].
    lat : np.ndarray
        Latitude grid.
    lon : np.ndarray
        Longitude grid.
    coords : tuple of int, optional
        (x1, x2, y1, y2) if the mask is a sub-window of the full lat/lon grid.
    feature_id : str, optional
        Base ID to use in the GeoJSON feature properties.

    Returns
    -------
    dict
        GeoJSON FeatureCollection.
    """
    # Ensure 0–255 uint8 mask for OpenCV
    if mask.dtype != np.uint8:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask

    contours = _find_contours(mask_uint8)
    contours_latlon = _coords_to_latlon(contours, coords, lat, lon)
    geojson_fc = fix_geojson(_contours_to_geojson(contours_latlon, feature_id=feature_id))
    return geojson_fc



def predict(
    model_cls: Type[torch.nn.Module],
    model_path: Path,
    image_path: Path,
    tile_h: int = 250,
    tile_w: int = 250,
    stride: int = 200,
    threshold: float = 0.15,
    output: Optional[Path] = None,
    show: bool = False,
    log_level: str = "INFO",
    product: Literal["overlay", "mask", "geojson"] = "overlay",
    lat: Optional[np.ndarray] = None,
    lon: Optional[np.ndarray] = None,
    coords: Optional[tuple[int, int, int, int]] = None,
    feature_id: str = "CONTRAIL",
) -> Union[np.ndarray, Dict[str, Any]]:
    """
    Run contrail segmentation and optionally return overlay, mask, or GeoJSON.

    Parameters
    ----------
    model_cls : Type[torch.nn.Module]
        The model class used to instantiate the network.
    model_path : Path
        Path to the trained model weights.
    image_path : Path
        Path to the input image.
    tile_h, tile_w : int, optional
        Tile size used for sliding-window inference.
    stride : int, optional
        Stride (in pixels) between tiles.
    threshold : float, optional
        Probability threshold for mask binarization.
    output : Path, optional
        If given, save the overlay image to this path.
    show : bool, optional
        If True, display the overlay using matplotlib.
    log_level : {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}, optional
        Logging level.
    product : {"overlay", "mask", "geojson"}, optional
        Output format.
    lat, lon : np.ndarray, optional
        Required for GeoJSON output.
    coords : tuple[int, int, int, int], optional
        Mask location within larger georeferenced window.
    feature_id : str, optional
        Base ID used in GeoJSON properties.

    Returns
    -------
    np.ndarray or dict
        Output depending on `product`.
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

    model = load_model(model_cls=model_cls, model_path=model_path, device=device)

    logging.info(
        "Running inference: tile_h=%d tile_w=%d stride=%d threshold=%.3f",
        tile_h, tile_w, stride, threshold,
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

    overlay = None
    if product == "overlay" or output is not None or show:
        overlay = overlay_mask_on_image(image, mask)

    if output is not None and overlay is not None:
        save_overlay(overlay, Path(output))
        logging.info("Saved overlay to %s", output)

    if show and overlay is not None:
        plt.figure(figsize=(12, 12))
        plt.imshow(overlay)
        plt.axis("off")
        plt.title("Overlay")
        plt.show()

    if product == "overlay":
        return overlay if overlay is not None else overlay_mask_on_image(image, mask)

    if product == "mask":
        return mask

    if product == "geojson":
        if lat is None or lon is None:
            raise ValueError("lat and lon must be provided for product='geojson'")
        return mask_to_geojson(
            mask=mask,
            lat=lat,
            lon=lon,
            coords=coords,
            feature_id=feature_id,
        )

    raise ValueError(f"Unknown product: {product}")




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
        default=250,
        help="Tile width in pixels.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=200,
        help="Stride between tiles in pixels.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
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
