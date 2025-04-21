"""
Trainer module for recommendation models.

This module contains trainers for recommendation models:
- BaseTrainer: Generic trainer for recommendation models with modular design for easy extension
"""

from rating_refactor.trainers.base import BaseTrainer

__all__ = ['BaseTrainer']