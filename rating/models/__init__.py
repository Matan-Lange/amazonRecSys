"""
Models for recommendation systems.

This package contains model implementations for various recommendation algorithms:
- Matrix Factorization (MF)
- Neural Collaborative Filtering (NCF)
"""

from rating.models.base import BaseRecommenderModel, BiasedEmbeddingModel
from rating.models.mf import MFModel
from rating.models.ncf import NCFModel

__all__ = [
    'BaseRecommenderModel',
    'BiasedEmbeddingModel',
    'MFModel',
    'NCFModel',
]