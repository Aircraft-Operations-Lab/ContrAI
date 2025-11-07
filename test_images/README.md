# Test Images

This directory contains test images for evaluating the ECONTRAIL contrail detection model.

## Structure

Place your test satellite images in this directory. Supported formats:
- PNG (.png)
- JPEG (.jpg, .jpeg)
- TIFF (.tif, .tiff)

## Usage

The test images in this directory will be automatically discovered by:
- `evaluation.ipynb` notebook for running predictions
- Prediction scripts for batch processing

## Example

```python
from econtrail_detection import predict_contrails

# Predict on a single test image
prediction = predict_contrails('test_images/sample.png')
```

## Adding Test Images

1. Copy your satellite images to this directory
2. Run the evaluation notebook: `jupyter notebook evaluation.ipynb`
3. View predictions and metrics

For more information, see the main README and the research paper.
