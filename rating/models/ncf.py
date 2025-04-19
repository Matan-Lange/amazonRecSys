"""
Neural Collaborative Filtering models for recommendation systems.

This module contains implementations of Neural Collaborative Filtering models:
- NCFModel: Neural Collaborative Filtering model with pre-trained item embeddings
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from rating.models.base import BaseRecommenderModel


class NCFModel(BaseRecommenderModel):
    """Neural Collaborative Filtering model with pre-trained item embeddings."""
    
    def __init__(
        self,
        num_users: int,
        item_embeddings: nn.Embedding,
        embed_dim: int = 32,
        mlp_dims: List[int] = [16, 8],
        dropout_rate: float = 0.5
    ):
        """
        Initialize the NCF model.
        
        Args:
            num_users: Number of unique users
            item_embeddings: Pre-trained item embeddings
            embed_dim: Dimension of the user embedding vectors
            mlp_dims: Dimensions of the MLP layers
            dropout_rate: Dropout rate for regularization
        """
        super().__init__(num_users, num_users, embed_dim)  # num_items is not used directly

        # User embeddings and normalization
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.user_ln = nn.LayerNorm(embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)

        # User bias
        self.user_bias = nn.Embedding(num_users, 1)
        nn.init.zeros_(self.user_bias.weight)

        # Item embeddings (pre-trained and frozen)
        self.item_emb = item_embeddings
        self.item_ln = nn.LayerNorm(self.item_emb.weight.shape[1])
        
        # Freeze item embeddings
        for param in self.item_emb.parameters():
            param.requires_grad = False

        # Project item embeddings to same dimension as user embeddings
        self.item_projection = nn.Linear(self.item_emb.weight.shape[1], embed_dim)
        nn.init.xavier_uniform_(self.item_projection.weight)

        # MLP for combining user and item embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, mlp_dims[0]),
            nn.LayerNorm(mlp_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dims[0], 1)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the NCF model.
        
        Args:
            batch: Dictionary containing input tensors including:
                user_idx: User indices
                item_idx: Item indices
                
        Returns:
            Tensor of predictions
        """
        # Get user and item embeddings
        user_e = self.user_ln(self.user_emb(batch['user_idx']))
        item_e = self.item_ln(self.item_emb(batch['item_idx']))
        
        # Get user bias
        user_b = self.user_bias(batch['user_idx']).squeeze()
        
        # Project item embeddings to match user embedding dimensions
        item_e = self.item_projection(item_e)
        
        # Concatenate user and item embeddings
        x = torch.cat([user_e, item_e], dim=1)
        
        # Pass through MLP
        raw_output = self.mlp(x)
        
        # Add global mean and user bias
        output = self.global_mean + user_b + raw_output.squeeze()
        
        return output