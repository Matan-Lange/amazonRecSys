"""
Factory classes for creating datasets.

This module contains factory classes for creating datasets:
- DatasetFactory: Factory for creating train/val/test datasets with consistent mappings
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple, Union

class DatasetFactory:
    """Factory class to create train/val/test datasets with consistent mappings."""

    def __init__(self, train_events_path: Union[str, Path], test_events_path: Union[str, Path],
                 metadata_path: Union[str, Path]):
        self.train_events_path = Path(train_events_path)
        self.test_events_path = Path(test_events_path)
        self.metadata_path = Path(metadata_path)

        # Load datasets
        self.df_train_events = self._load_train_events()
        self.df_test_events = self._load_test_events()
        self.df_metadata = self._load_metadata()

        # Create global hashmaps from training data only
        self.hashmaps = self._create_global_hashmaps()

    def _load_train_events(self) -> pd.DataFrame:
        """Load training events data with timestamp."""
        df = pd.read_csv(self.train_events_path)
        return df[['user_id', 'parent_asin', 'rating', 'timestamp']]

    def _load_test_events(self) -> pd.DataFrame:
        """Load test events data without timestamp."""
        df = pd.read_csv(self.test_events_path)
        return df[['user_id', 'parent_asin', 'rating']]

    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata."""
        with open(self.metadata_path, "r") as file:
            data = [json.loads(line) for line in file]
        df = pd.DataFrame.from_records(data)
        df['categories'] = df['categories'].astype(str)
        return df[['parent_asin', 'categories', 'store']]

    def _create_global_hashmaps(self) -> Dict:
        """Create global hashmaps from training dataset only."""
        df_merged = self.df_train_events.merge(self.df_metadata, on='parent_asin', how='left')
        return {
            'user': {val: idx for idx, val in enumerate(df_merged.user_id.unique())},
            'item': {val: idx for idx, val in enumerate(df_merged.parent_asin.unique())},
            'category': {val: idx for idx, val in enumerate(df_merged.categories.unique())},
            'store': {val: idx for idx, val in enumerate(df_merged.store.unique())}
        }

    def create_datasets(self) -> Tuple['AmazonDataset', 'AmazonDataset', 'AmazonDataset']:
        """Create train/val/test datasets using regression split for train/val."""
        # Split train into train and validation
        from rating.utils.common import regression_split_train_validation
        df_train, df_val = regression_split_train_validation(self.df_train_events)

        # Import here to avoid circular imports
        from rating.datasets.base import AmazonDataset
        
        train_dataset = AmazonDataset(df_train, self.df_metadata, self.hashmaps)
        val_dataset = AmazonDataset(df_val, self.df_metadata, self.hashmaps)
        test_dataset = AmazonDataset(self.df_test_events, self.df_metadata, self.hashmaps)

        return train_dataset, val_dataset, test_dataset