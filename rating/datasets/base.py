"""
Base dataset for recommendation systems.

This module contains the base dataset class for recommendation tasks:
- AmazonDataset: PyTorch Dataset for Amazon product ratings with metadata
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Union, Any, List


class AmazonDataset(Dataset):
    """PyTorch Dataset for Amazon product ratings with metadata."""

    def __init__(self, df_events: pd.DataFrame, df_metadata: pd.DataFrame, hashmaps: Dict[str, Dict[Any, int]]):
        """
        Initialize the dataset.

        Args:
            df_events: DataFrame containing user-item interactions
            df_metadata: DataFrame containing item metadata
            hashmaps: Dictionary of id mappings from factory
        """
        super().__init__()
        self.hashmaps = hashmaps
        self.df = df_events.merge(df_metadata, on='parent_asin', how='left')

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing tensors for user_idx, item_idx, category_idx, store_idx, and rating
        """
        row = self.df.iloc[idx]
        return {
            'user_idx': torch.tensor(self.hashmaps['user'].get(row.user_id), dtype=torch.long),
            'item_idx': torch.tensor(self.hashmaps['item'].get(row.parent_asin), dtype=torch.long),
            'category_idx': torch.tensor(self.hashmaps['category'].get(row.categories), dtype=torch.long),
            'store_idx': torch.tensor(self.hashmaps['store'].get(row.store), dtype=torch.long),
            'rating': torch.tensor(row.rating, dtype=torch.float32)
        }

    @property
    def num_users(self) -> int:
        """Return the number of unique users in the dataset."""
        return len(self.hashmaps['user'])

    @property
    def num_items(self) -> int:
        """Return the number of unique items in the dataset."""
        return len(self.hashmaps['item'])

    @property
    def num_categories(self) -> int:
        """Return the number of unique categories in the dataset."""
        return len(self.hashmaps['category'])

    @property
    def num_stores(self) -> int:
        """Return the number of unique stores in the dataset."""
        return len(self.hashmaps['store'])

    def user_to_row_indices(self) -> Dict[int, List[int]]:
        """
        Helper for the sampler.
        """
        user_idx_series = self.df["user_id"].map(self.hashmaps["user"])
        groups =  user_idx_series.groupby(user_idx_series).groups
        return {k: list(v) for k, v in groups.items()}



