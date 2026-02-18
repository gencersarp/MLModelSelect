"""
MLModelSelect - Vast range of ML Models for plug and use with advanced model comparison.

This library provides:
- Easy-to-use ML models for classification, regression, and clustering
- Advanced model comparison functionality
- Dataset analysis utilities
- Plug-and-play interface for rapid prototyping
"""

__version__ = "0.1.0"

from .models import classification, regression, clustering
from .model_compare import ModelCompare, compare_models
from .base import BaseModel

__all__ = [
    'classification',
    'regression', 
    'clustering',
    'ModelCompare',
    'compare_models',
    'BaseModel',
]
