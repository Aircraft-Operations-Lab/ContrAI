"""
Setup script for ECONTRAIL Detection package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="econtrail-detection",
    version="0.1.0",
    author="ECONTRAIL Team",
    description="Tool for running contrail detection using neural network models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/irortiza/ECONTRAIL_detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "Pillow>=8.0.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "matplotlib>=3.3.0",
            "pytest>=6.0.0",
        ],
        "gpu": [
            "torch>=1.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "econtrail-predict=econtrail_detection.predict:main",
        ],
    },
)
