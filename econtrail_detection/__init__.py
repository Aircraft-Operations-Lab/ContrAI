"""
ECONTRAIL Detection Package

This package provides tools for contrail detection using neural network models
as evaluated in the research paper: https://doi.org/10.1109/TGRS.2025.3629628
"""

__version__ = "0.1.0"
__author__ = "ECONTRAIL Team"

from .predict import predict_contrails
from .utils import load_image, preprocess_image, postprocess_prediction

__all__ = [
    "predict_contrails",
    "load_image",
    "preprocess_image",
    "postprocess_prediction",
]
