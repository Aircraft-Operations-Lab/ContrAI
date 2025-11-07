"""
Utility Functions Module

This module provides helper functions for image processing, data loading,
and post-processing of predictions.
"""

import numpy as np
from typing import Union, Tuple, Optional
from pathlib import Path


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Image as numpy array (H, W, C)
    
    Example:
        >>> from econtrail_detection.utils import load_image
        >>> image = load_image('test_images/sample.png')
        >>> print(image.shape)
    """
    try:
        from PIL import Image
        img = Image.open(image_path)
        return np.array(img)
    except ImportError:
        # Fallback if PIL is not available
        import cv2
        img = cv2.imread(str(image_path))
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raise FileNotFoundError(f"Could not load image: {image_path}")


def preprocess_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess an image for model input.
    
    Args:
        image: Input image as numpy array
        target_size: Target size (height, width) for resizing. If None, no resizing.
        normalize: Whether to normalize pixel values to [0, 1]
    
    Returns:
        Preprocessed image
    
    Example:
        >>> from econtrail_detection.utils import preprocess_image
        >>> import numpy as np
        >>> image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        >>> preprocessed = preprocess_image(image, target_size=(256, 256))
    """
    processed = image.copy()
    
    # Resize if target size is specified
    if target_size is not None:
        try:
            from PIL import Image
            pil_img = Image.fromarray(processed)
            pil_img = pil_img.resize((target_size[1], target_size[0]))
            processed = np.array(pil_img)
        except ImportError:
            import cv2
            processed = cv2.resize(processed, (target_size[1], target_size[0]))
    
    # Normalize to [0, 1] if requested
    if normalize and processed.dtype == np.uint8:
        processed = processed.astype(np.float32) / 255.0
    
    return processed


def postprocess_prediction(
    prediction: np.ndarray,
    threshold: Optional[float] = None,
    min_area: int = 0
) -> np.ndarray:
    """
    Post-process prediction masks.
    
    Args:
        prediction: Raw prediction mask
        threshold: Optional threshold to apply (if prediction is continuous)
        min_area: Minimum area for detected regions (pixels)
    
    Returns:
        Post-processed prediction mask
    
    Example:
        >>> from econtrail_detection.utils import postprocess_prediction
        >>> import numpy as np
        >>> prediction = np.random.rand(256, 256)
        >>> processed = postprocess_prediction(prediction, threshold=0.5, min_area=100)
    """
    result = prediction.copy()
    
    # Apply threshold if provided and prediction is continuous
    if threshold is not None and result.dtype in [np.float32, np.float64]:
        result = (result > threshold).astype(np.uint8)
    
    # Filter small regions if min_area is specified
    if min_area > 0:
        try:
            import cv2
            # Find connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                result.astype(np.uint8), connectivity=8
            )
            
            # Keep only regions larger than min_area
            filtered = np.zeros_like(result)
            for label in range(1, num_labels):
                if stats[label, cv2.CC_STAT_AREA] >= min_area:
                    filtered[labels == label] = 1
            
            result = filtered
        except ImportError:
            # If OpenCV is not available, skip filtering
            pass
    
    return result


def save_prediction(
    prediction: np.ndarray,
    output_path: Union[str, Path],
    colormap: bool = True
) -> None:
    """
    Save a prediction mask to file.
    
    Args:
        prediction: Prediction mask to save
        output_path: Path where to save the prediction
        colormap: Whether to apply a colormap for visualization
    
    Example:
        >>> from econtrail_detection.utils import save_prediction
        >>> import numpy as np
        >>> prediction = np.random.randint(0, 2, (256, 256))
        >>> save_prediction(prediction, 'output/prediction.png')
    """
    try:
        from PIL import Image
        
        if colormap:
            # Apply colormap for better visualization
            colored = (prediction * 255).astype(np.uint8)
            img = Image.fromarray(colored)
            if img.mode != 'RGB':
                img = img.convert('RGB')
        else:
            img = Image.fromarray((prediction * 255).astype(np.uint8))
        
        img.save(output_path)
    except ImportError:
        import cv2
        cv2.imwrite(str(output_path), (prediction * 255).astype(np.uint8))


def calculate_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray
) -> dict:
    """
    Calculate evaluation metrics for predictions.
    
    Args:
        prediction: Predicted mask
        ground_truth: Ground truth mask
    
    Returns:
        Dictionary with metrics (accuracy, precision, recall, f1_score, iou)
    
    Example:
        >>> from econtrail_detection.utils import calculate_metrics
        >>> import numpy as np
        >>> pred = np.random.randint(0, 2, (256, 256))
        >>> gt = np.random.randint(0, 2, (256, 256))
        >>> metrics = calculate_metrics(pred, gt)
        >>> print(f"IoU: {metrics['iou']:.3f}")
    """
    # Ensure binary masks
    pred = (prediction > 0).astype(np.uint8).flatten()
    gt = (ground_truth > 0).astype(np.uint8).flatten()
    
    # Calculate confusion matrix elements
    tp = np.sum((pred == 1) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'iou': iou
    }


if __name__ == "__main__":
    print("ECONTRAIL Utility Functions Module")
    print("Available functions: load_image, preprocess_image, postprocess_prediction, save_prediction, calculate_metrics")
