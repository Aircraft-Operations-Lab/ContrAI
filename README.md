# ECONTRAIL Detection

This repository provides an easy-to-use tool for running contrail detection using one of the neural network models evaluated in our research paper: https://doi.org/10.1109/TGRS.2025.3629628

## Features

- **Prediction Module**: Easy-to-use functions for contrail detection in satellite imagery
- **Utility Functions**: Tools for image processing, evaluation, and metrics calculation
- **Evaluation Notebook**: Interactive notebook for model evaluation and visualization
- **Dataset Testing Notebook**: Tools for loading, exploring, and testing datasets
- **Structured Package**: Properly organized Python package with `__init__.py` files

## Repository Structure

```
ECONTRAIL_detection/
├── econtrail_detection/      # Main package
│   ├── __init__.py           # Package initialization
│   ├── predict.py            # Prediction functions
│   └── utils.py              # Utility functions
├── test_images/              # Test images for evaluation
│   └── README.md
├── data/                     # Modified dataset
│   ├── images/               # Input images
│   ├── masks/                # Segmentation masks
│   ├── ground_truth/         # Ground truth for evaluation
│   └── README.md
├── evaluation.ipynb          # Model evaluation notebook
├── dataset_testing.ipynb     # Dataset loading and testing notebook
├── setup.py                  # Package installation script
├── .gitignore               # Git ignore rules
├── LICENSE                   # License file
└── README.md                # This file
```

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/irortiza/ECONTRAIL_detection.git
cd ECONTRAIL_detection

# Install the package
pip install -e .
```

### Development Installation

For development with notebooks and visualization tools:

```bash
pip install -e ".[dev]"
```

### GPU Support

For GPU acceleration (requires CUDA):

```bash
pip install -e ".[gpu]"
```

### OpenCV Support

For advanced image processing features (optional):

```bash
pip install -e ".[opencv]"
```

## Quick Start

### 1. Making Predictions

```python
from econtrail_detection import predict_contrails

# Predict on a single image
prediction = predict_contrails('test_images/sample.png', threshold=0.5)

# Predict on multiple images
from econtrail_detection.predict import predict_batch
predictions = predict_batch(['img1.png', 'img2.png', 'img3.png'])
```

### 2. Using Utility Functions

```python
from econtrail_detection.utils import (
    load_image,
    preprocess_image,
    calculate_metrics,
    save_prediction
)

# Load and preprocess an image
image = load_image('test_images/sample.png')
processed = preprocess_image(image, target_size=(256, 256), normalize=True)

# Calculate metrics
metrics = calculate_metrics(prediction, ground_truth)
print(f"IoU: {metrics['iou']:.3f}")

# Save prediction
save_prediction(prediction, 'output/result.png')
```

### 3. Running Evaluation

Open and run the evaluation notebook:

```bash
jupyter notebook evaluation.ipynb
```

This notebook allows you to:
- Load test images
- Run predictions
- Visualize results
- Calculate metrics (if ground truth is available)
- Save predictions

### 4. Testing the Dataset

Open and run the dataset testing notebook:

```bash
jupyter notebook dataset_testing.ipynb
```

This notebook helps you:
- Explore the dataset structure
- Visualize samples
- Verify image-mask pairs
- Test preprocessing pipelines
- Calculate dataset statistics

## Dataset Structure

Place your data in the appropriate directories:

- **test_images/**: Test satellite images for evaluation
- **data/images/**: Training/validation images
- **data/masks/**: Corresponding segmentation masks
- **data/ground_truth/**: Ground truth annotations for evaluation

See `data/README.md` and `test_images/README.md` for more details.

## API Reference

### Prediction Functions

- `predict_contrails(image, model_path, threshold, device)`: Predict contrails in a single image
- `predict_batch(images, model_path, threshold, device, batch_size)`: Batch prediction

### Utility Functions

- `load_image(image_path)`: Load an image from file
- `preprocess_image(image, target_size, normalize)`: Preprocess image for model input
- `postprocess_prediction(prediction, threshold, min_area)`: Post-process prediction masks
- `save_prediction(prediction, output_path, colormap)`: Save prediction to file
- `calculate_metrics(prediction, ground_truth)`: Calculate evaluation metrics

## Examples

See the notebooks for comprehensive examples:
- `evaluation.ipynb`: Model evaluation workflow
- `dataset_testing.ipynb`: Dataset loading and exploration

## Citation

If you use this tool in your research, please cite our paper:

```bibtex
@article{econtrail2025,
  title={ECONTRAIL: Easy Contrail Detection},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  doi={10.1109/TGRS.2025.3629628}
}
```

## License

This project is licensed under the terms specified in the LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions or issues, please open an issue on GitHub or refer to the research paper: https://doi.org/10.1109/TGRS.2025.3629628