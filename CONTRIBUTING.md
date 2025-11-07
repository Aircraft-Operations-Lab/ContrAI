# Contributing to ECONTRAIL Detection

Thank you for your interest in contributing to ECONTRAIL Detection!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ECONTRAIL_detection.git
   cd ECONTRAIL_detection
   ```
3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Setup

### Required Dependencies
- Python >= 3.8
- numpy >= 1.19.0
- Pillow >= 8.0.0

### Optional Dependencies
- jupyter (for notebooks)
- matplotlib (for visualization)
- pytest (for testing)

## Project Structure

```
ECONTRAIL_detection/
├── econtrail_detection/      # Main package
│   ├── __init__.py           # Package initialization
│   ├── predict.py            # Prediction functions
│   └── utils.py              # Utility functions
├── test_images/              # Test images
├── data/                     # Dataset directory
├── evaluation.ipynb          # Evaluation notebook
├── dataset_testing.ipynb     # Dataset testing notebook
└── example.py                # Example usage script
```

## Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the coding style:
   - Follow PEP 8 style guide
   - Add docstrings to all functions
   - Include type hints where appropriate
   - Keep functions focused and modular

3. Test your changes:
   ```bash
   python example.py
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request

## Coding Standards

- Use descriptive variable and function names
- Add comments for complex logic
- Keep functions under 50 lines when possible
- Write docstrings in NumPy style

### Example Function

```python
def process_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Process an image for model input.
    
    Args:
        image: Input image as numpy array
        size: Target size as (height, width)
    
    Returns:
        Processed image
    
    Example:
        >>> img = np.random.rand(512, 512, 3)
        >>> processed = process_image(img, (256, 256))
    """
    # Implementation here
    pass
```

## Adding Features

When adding new features:

1. **Prediction Functions**: Add to `econtrail_detection/predict.py`
2. **Utility Functions**: Add to `econtrail_detection/utils.py`
3. **Documentation**: Update README.md and docstrings
4. **Examples**: Add usage examples to notebooks or `example.py`

## Testing

Before submitting a PR, ensure:
- [ ] Code runs without errors
- [ ] Example script works: `python example.py`
- [ ] Package imports correctly
- [ ] Docstrings are complete
- [ ] README is updated if needed

## Questions?

If you have questions or need help:
- Open an issue on GitHub
- Refer to the research paper: https://doi.org/10.1109/TGRS.2025.3629628

Thank you for contributing!
