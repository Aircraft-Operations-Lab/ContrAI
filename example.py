#!/usr/bin/env python
"""
Example script demonstrating how to use the ECONTRAIL detection package.

This script shows:
1. How to import the package
2. How to create dummy test data
3. How to run predictions
4. How to calculate metrics
5. How to save results
"""

import numpy as np
from pathlib import Path

# Import ECONTRAIL detection functions
from econtrail_detection import predict_contrails
from econtrail_detection.utils import (
    preprocess_image,
    calculate_metrics,
    save_prediction
)


def main():
    print("="*60)
    print("ECONTRAIL Detection - Example Usage")
    print("="*60)
    print()
    
    # 1. Create example data
    print("1. Creating example data...")
    # Create a dummy image (256x256 RGB)
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    print(f"   Created image with shape: {image.shape}")
    
    # 2. Preprocess the image
    print("\n2. Preprocessing image...")
    preprocessed = preprocess_image(image, target_size=(256, 256), normalize=True)
    print(f"   Preprocessed shape: {preprocessed.shape}")
    print(f"   Value range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
    
    # 3. Run prediction
    print("\n3. Running prediction...")
    prediction = predict_contrails(image, threshold=0.5, device='cpu')
    print(f"   Prediction shape: {prediction.shape}")
    print(f"   Unique values: {np.unique(prediction)}")
    
    # 4. Create dummy ground truth and calculate metrics
    print("\n4. Calculating metrics (with dummy ground truth)...")
    ground_truth = np.random.randint(0, 2, prediction.shape)
    metrics = calculate_metrics(prediction, ground_truth)
    
    print("   Metrics:")
    for key, value in metrics.items():
        print(f"     {key:12s}: {value:.3f}")
    
    # 5. Save prediction
    print("\n5. Saving prediction...")
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'example_prediction.png'
    save_prediction(prediction, output_path, colormap=True)
    print(f"   Saved to: {output_path}")
    
    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)
    print("\nNext steps:")
    print("  - Add your own images to 'test_images/' directory")
    print("  - Run 'jupyter notebook evaluation.ipynb' for interactive evaluation")
    print("  - Run 'jupyter notebook dataset_testing.ipynb' to explore datasets")
    print("  - Use 'econtrail-predict' command for batch processing")


if __name__ == "__main__":
    main()
