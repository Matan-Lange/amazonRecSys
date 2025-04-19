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
        num_categories: int,
        num_stores: int,
        embed_dim: int = 32, 
        biases: Optional[List[str]] = None,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the MF model with configurable biases.

        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            num_categories: Number of unique categories
            num_stores: Number of unique stores
            embed_dim: Dimension of the embedding vectors
            biases: List of bias types to include ['user', 'item', 'category', 'store']
            dropout_rate: Dropout rate for embeddings (default: 0.2)
        """
        # Initialize base class with user and item biases
        base_biases = [b for b in biases if b in ['user', 'item']] if biases else []
        super().__init__(num_users, num_items, embed_dim, base_biases)

        # Store full biases list for additional biases
        self.biases = biases if biases else []

        # Add dropout for embeddings
        self.dropout = nn.Dropout(dropout_rate)

        # Additional bias embeddings for category and store
        if 'category' in self.biases:
            self.bias_layers['category'] = nn.Embedding(num_categories, 1)
            init.zeros_(self.bias_layers['category'].weight)
        if 'store' in self.biases:
            self.bias_layers['store'] = nn.Embedding(num_stores, 1)
            init.zeros_(self.bias_layers['store'].weight)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass incorporating selected biases.

        Args:
            batch: Dictionary containing input tensors including:
                user_idx: User indices
                item_idx: Item indices
                category_idx (optional): Category indices
                store_idx (optional): Store indices

        Returns:
            Tensor of predictions
        """
        user_idx = batch['user_idx']
        item_idx = batch['item_idx']
        category_idx = batch.get('category_idx')
        store_idx = batch.get('store_idx')

        # Base matrix factorization with dropout
        user_emb = self.user_emb(user_idx)
        item_emb = self.item_emb(item_idx)

        # Apply dropout to embeddings during training
        if self.training:
            user_emb = self.dropout(user_emb)
            item_emb = self.dropout(item_emb)

        element_product = (user_emb * item_emb).sum(1)

        # Get base biases (user and item)
        bias_sum = self.get_bias_sum(batch)

        # Add additional biases
        if 'category' in self.biases and category_idx is not None:
            bias_sum += self.bias_layers['category'](category_idx).squeeze()
        if 'store' in self.biases and store_idx is not None:
            bias_sum += self.bias_layers['store'](store_idx).squeeze()

        # Final prediction
        logit = element_product + bias_sum + self.global_mean
        return logit
