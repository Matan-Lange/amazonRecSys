"""
Matrix Factorization models for recommendation systems.

This module contains implementations of Matrix Factorization models:
- MFModel: Basic Matrix Factorization model with optional biases
"""

import torch
from torch import nn
import torch.nn.init as init
from typing import Dict, List, Optional, Union, Any
from rating.models.base import BiasedEmbeddingModel


class MFModel(BiasedEmbeddingModel):
    """Matrix Factorization model with configurable biases."""

    def __init__(
        self, 
        num_users: int, 
        num_items: int,
        embed_dim: int = 32, 
        biases: Optional[List[str]] = None,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the MF model with configurable biases.

        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            embed_dim: Dimension of the embedding vectors
            biases: List of bias types to include ['user', 'item', 'category', 'store']
            dropout_rate: Dropout rate for embeddings (default: 0.2)
        """
        # Initialize base class with user and item biases
        base_biases = [b for b in biases if b in ['user', 'item']] if biases else []
        super().__init__(num_users, num_items, embed_dim, base_biases)
        self.biases = biases if biases else []
        self.dropout = nn.Dropout(dropout_rate)



    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass incorporating selected biases.

        Args:
            batch: Dictionary containing input tensors including:
                user_idx: User indices
                item_idx: Item indices

        Returns:
            Tensor of predictions
        """
        user_idx = batch['user_idx']
        item_idx = batch['item_idx']
        user_emb = self.user_emb(user_idx)
        item_emb = self.item_emb(item_idx)

        # Apply dropout to embeddings during training
        if self.training:
            user_emb = self.dropout(user_emb)
            item_emb = self.dropout(item_emb)

        element_product = (user_emb * item_emb).sum(1)
        bias_sum = self.get_bias_sum(batch)
        logit = element_product + bias_sum + self.global_mean
        return logit
