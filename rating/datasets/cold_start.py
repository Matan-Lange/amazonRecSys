"""
Cold start dataset handling for recommendation systems.

This module contains dataset factory classes for cold start scenarios:
- TimeBasedColdStartDatasetFactory: Factory for creating cold start datasets based on timestamp
"""

from typing import Tuple
import pandas as pd
from rating.datasets.factory import DatasetFactory

class TimeBasedColdStartDatasetFactory(DatasetFactory):
    """Factory for creating cold start datasets based on timestamp."""
    
    def create_cold_start_splits(self, test_ratio=0.2) -> Tuple['AmazonDataset', 'AmazonDataset']:
        """
        Create train/test splits for cold-start scenario based on timestamp.
        
        Args:
            test_ratio: Ratio of data to use for testing
            
        Returns:
            Tuple containing (train_dataset, test_dataset)
        """
        # Sort by timestamp
        df_sorted = self.df_train_events.sort_values('timestamp')

        # Calculate split point
        split_idx = int(len(df_sorted) * (1 - test_ratio))
        split_timestamp = df_sorted.iloc[split_idx]['timestamp']

        # Split data
        df_train = df_sorted[df_sorted['timestamp'] < split_timestamp]
        df_test = df_sorted[df_sorted['timestamp'] >= split_timestamp]

        # Get items that appear only in test set
        train_items = set(df_train['parent_asin'].unique())
        test_items = set(df_test['parent_asin'].unique())
        cold_start_items = test_items - train_items
        print(f"Cold start items: {len(cold_start_items)}")
        print(f"Train items: {len(train_items)}")

        # Filter test set to include only cold start items
        df_test_cold = df_test[df_test['parent_asin'].isin(cold_start_items)]

        # Create hashmaps from training data
        df_merged = df_train.merge(self.df_metadata, on='parent_asin', how='left')
        self.hashmaps = {
            'user': {val: idx for idx, val in enumerate(df_merged.user_id.unique())},
            'item': {val: idx for idx, val in enumerate(df_merged.parent_asin.unique())},
            'category': {val: idx for idx, val in enumerate(df_merged.categories.unique())},
            'store': {val: idx for idx, val in enumerate(df_merged.store.unique())}
        }

        # Import here to avoid circular imports
        from rating.datasets.base import AmazonDataset
        
        train_dataset = AmazonDataset(df_train, self.df_metadata, self.hashmaps)
        test_dataset = AmazonDataset(df_test_cold, self.df_metadata, self.hashmaps)

        return train_dataset, test_dataset