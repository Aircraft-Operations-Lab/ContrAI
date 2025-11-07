# Modified Dataset

This directory contains the modified dataset for ECONTRAIL contrail detection.

## Structure

```
data/
├── images/          # Input satellite images
├── masks/           # Corresponding segmentation masks
├── ground_truth/    # Ground truth annotations for evaluation
└── README.md        # This file
```

## Dataset Organization

### images/
Place your satellite images here. These are the input images for the model.

### masks/
Place the corresponding segmentation masks here. Each mask should have the same filename as its corresponding image.

### ground_truth/
Place ground truth annotations here for model evaluation and metric calculation.

## Usage

### Loading the Dataset

Use the `dataset_testing.ipynb` notebook to:
- Explore the dataset structure
- Visualize samples
- Verify image-mask pairs
- Test preprocessing pipelines

### Evaluation

Use the `evaluation.ipynb` notebook to:
- Run predictions on test images
- Calculate metrics against ground truth
- Visualize results

## Example

```python
from econtrail_detection.utils import load_image

# Load an image from the dataset
image = load_image('data/images/sample.png')

# Load corresponding mask
mask = load_image('data/masks/sample.png')
```

## Data Format

- **Images**: RGB or grayscale satellite imagery
- **Masks**: Binary masks (0 = no contrail, 255 = contrail)
- **Ground Truth**: Same format as masks

## Adding Data

1. Place images in `data/images/`
2. Place masks in `data/masks/` (same filenames as images)
3. Place ground truth in `data/ground_truth/` (for evaluation)
4. Run `dataset_testing.ipynb` to verify

For more information, see the main README and the research paper: https://doi.org/10.1109/TGRS.2025.3629628
