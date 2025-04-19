"""
Trainers for recommendation models.

This package contains trainer implementations for various recommendation algorithms:
- Base trainer for general models
- Weighted trainer for models with weighted training
- NCF trainer for Neural Collaborative Filtering models
"""

from rating.trainers.base import BaseTrainer
from rating.trainers.weighted import WeightedTrainer
from rating.trainers.ncf import NCFTrainer

__all__ = [
    'BaseTrainer',
    'WeightedTrainer',
    'NCFTrainer',
]