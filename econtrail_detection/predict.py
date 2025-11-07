"""
Prediction Module

This module provides functions for running contrail detection predictions
using trained neural network models.
"""

import numpy as np
from typing import Union, Optional
from pathlib import Path


def predict_contrails(
    image: Union[np.ndarray, str, Path],
    model_path: Optional[str] = None,
    threshold: float = 0.5,
    device: str = "cpu"
) -> np.ndarray:
    """
    Predict contrails in satellite imagery.
    
    Args:
        image: Input image as numpy array or path to image file
        model_path: Path to the trained model weights. If None, uses default model.
        threshold: Confidence threshold for detection (0.0 to 1.0)
        device: Device to run inference on ('cpu' or 'cuda')
    
    Returns:
        Prediction mask as numpy array with contrail detections
    
    Example:
        >>> from econtrail_detection import predict_contrails
        >>> prediction = predict_contrails('test_images/sample.png')
        >>> # Or with numpy array
        >>> import numpy as np
        >>> image = np.random.rand(256, 256, 3)
        >>> prediction = predict_contrails(image, threshold=0.7)
    """
    # Import utility functions
    from .utils import load_image, preprocess_image, postprocess_prediction
    
    # Load image if path is provided
    if isinstance(image, (str, Path)):
        image = load_image(image)
    
    # Preprocess the image
    preprocessed = preprocess_image(image)
    
    # TODO: Load model and run inference
    # For now, return a dummy prediction
    # In a real implementation, you would:
    # 1. Load the model from model_path
    # 2. Run inference on preprocessed image
    # 3. Apply threshold
    
    # Placeholder: return dummy prediction mask
    prediction = np.random.rand(*image.shape[:2]) > threshold
    
    # Postprocess the prediction
    result = postprocess_prediction(prediction)
    
    return result


def predict_batch(
    images: list,
    model_path: Optional[str] = None,
    threshold: float = 0.5,
    device: str = "cpu",
    batch_size: int = 8
) -> list:
    """
    Predict contrails for a batch of images.
    
    Args:
        images: List of images (numpy arrays or file paths)
        model_path: Path to the trained model weights
        threshold: Confidence threshold for detection
        device: Device to run inference on
        batch_size: Number of images to process at once
    
    Returns:
        List of prediction masks
    
    Example:
        >>> from econtrail_detection import predict_batch
        >>> images = ['img1.png', 'img2.png', 'img3.png']
        >>> predictions = predict_batch(images, batch_size=2)
    """
    predictions = []
    
    for image in images:
        pred = predict_contrails(image, model_path, threshold, device)
        predictions.append(pred)
    
    return predictions


def main():
    """
    Command-line interface for ECONTRAIL prediction.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ECONTRAIL Contrail Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "image",
        type=str,
        help="Path to input image or directory of images"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Path to model weights (optional)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold (default: 0.5)"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference (default: cpu)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output",
        help="Output directory for predictions (default: output)"
    )
    
    args = parser.parse_args()
    
    # Import here to avoid circular imports
    from pathlib import Path
    from .utils import save_prediction
    
    input_path = Path(args.image)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        # Single image
        print(f"Processing: {input_path}")
        prediction = predict_contrails(
            str(input_path),
            model_path=args.model,
            threshold=args.threshold,
            device=args.device
        )
        
        output_path = output_dir / f"pred_{input_path.stem}.png"
        save_prediction(prediction, output_path)
        print(f"Saved: {output_path}")
        
    elif input_path.is_dir():
        # Directory of images
        image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
        print(f"Found {len(image_files)} images")
        
        for img_file in image_files:
            print(f"Processing: {img_file.name}")
            prediction = predict_contrails(
                str(img_file),
                model_path=args.model,
                threshold=args.threshold,
                device=args.device
            )
            
            output_path = output_dir / f"pred_{img_file.stem}.png"
            save_prediction(prediction, output_path)
            print(f"  Saved: {output_path}")
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return 1
    
    print("\nPrediction complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
