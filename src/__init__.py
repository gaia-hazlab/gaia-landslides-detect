"""
Gaia Landslides Detection Package

A PyTorch-based framework for seismic data analysis and landslide detection.
"""

__version__ = "0.1.0"
__author__ = "Gaia Hazlab"

from . import models
from . import data
from . import utils

__all__ = ["models", "data", "utils"]
