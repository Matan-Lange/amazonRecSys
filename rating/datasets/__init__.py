"""
Datasets for recommendation systems.

This package contains dataset implementations for various recommendation tasks:
- Base dataset for general recommendation
- Dataset factory for creating datasets
- Cold start dataset for handling cold start scenarios
"""

from rating.datasets.base import AmazonDataset
from rating.datasets.factory import DatasetFactory
from rating.datasets.cold_start import TimeBasedColdStartDatasetFactory

__all__ = [
    'AmazonDataset',
    'DatasetFactory',
    'TimeBasedColdStartDatasetFactory',
]