"""
Base model classes for recommendation systems.

This module contains base classes for recommendation models:
- BaseRecommenderModel: Abstract base class for all recommender models
- BiasedEmbeddingModel: Base class for models with user/item embeddings and biases
"""

import torch
from torch import nn
import torch.nn.init as init
from typing import Dict, List, Optional, Union, Any


class BaseRecommenderModel(nn.Module):
    """Base class for recommender models."""
    
    def __init__(self, num_users: int, num_items: int, embed_dim: int = 32):
        """
        Initialize the base recommender model.
        
        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            embed_dim: Dimension of the embedding vectors
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        
        # Global mean rating (common in both MF and NCF)
        self.global_mean = nn.Parameter(torch.tensor(4.321553826471724), requires_grad=True)
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass to be implemented by subclasses.
        
        Args:
            batch: Dictionary containing input tensors
            
        Returns:
            Tensor of predictions
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement forward method")


class BiasedEmbeddingModel(BaseRecommenderModel):
    """Base class for models with user/item embeddings and optional biases."""
    
    def __init__(
        self, 
        num_users: int, 
        num_items: int, 
        embed_dim: int = 32, 
        biases: Optional[List[str]] = None
    ):
        """
        Initialize the model with configurable biases.
        
        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            embed_dim: Dimension of the embedding vectors
            biases: List of bias types to include ['user', 'item']
        """
        super().__init__(num_users, num_items, embed_dim)
        
        # User and item embeddings
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        
        # Initialize embeddings
        init.xavier_uniform_(self.user_emb.weight)
        init.xavier_uniform_(self.item_emb.weight)
        
        # Bias configuration
        self.biases = biases if biases else []
        
        # Bias embeddings based on configuration
        self.bias_layers = nn.ModuleDict()
        if 'user' in self.biases:
            self.bias_layers['user'] = nn.Embedding(num_users, 1)
            init.zeros_(self.bias_layers['user'].weight)
        if 'item' in self.biases:
            self.bias_layers['item'] = nn.Embedding(num_items, 1)
            init.zeros_(self.bias_layers['item'].weight)
    
    def get_bias_sum(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the sum of biases for the batch.
        
        Args:
            batch: Dictionary containing input tensors
        
        Returns:
            Sum of biases as a tensor
        """
        user_idx = batch['user_idx']
        item_idx = batch['item_idx']
        
        bias_sum = torch.zeros_like(user_idx, dtype=torch.float).to(self.device)
        
        if 'user' in self.biases:
            bias_sum += self.bias_layers['user'](user_idx).squeeze()
        if 'item' in self.biases:
            bias_sum += self.bias_layers['item'](item_idx).squeeze()
            
        return bias_sum