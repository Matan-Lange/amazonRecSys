import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Union
from pathlib import Path
import json
from PIL import Image
from utils import regression_split_train_validation


class DatasetFactory:
    """Factory class to create train/val/test datasets with consistent mappings."""

    def __init__(self, train_events_path: Union[str, Path],
                 test_events_path: Union[str, Path],
                 metadata_path: Union[str, Path],
                 images_path: str = None):
        self.train_events_path = Path(train_events_path)
        self.test_events_path = Path(test_events_path)
        self.metadata_path = Path(metadata_path)
        self.images_path = images_path

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
        df_train, df_val = regression_split_train_validation(self.df_train_events)

        train_dataset = AmazonDataset(df_train, self.df_metadata, self.hashmaps, self.images_path)
        val_dataset = AmazonDataset(df_val, self.df_metadata, self.hashmaps, self.images_path)
        test_dataset = AmazonDataset(self.df_test_events, self.df_metadata, self.hashmaps, self.images_path)

        return train_dataset, val_dataset, test_dataset


class AmazonDataset(Dataset):
    """PyTorch Dataset for Amazon product ratings with metadata."""

    def __init__(self, df_events: pd.DataFrame, df_metadata: pd.DataFrame, hashmaps: Dict, images_path=None):
        """
        Initialize the dataset.
        """
        super().__init__()
        self.hashmaps = hashmaps
        self.images_path = images_path
        self.df = df_events.merge(df_metadata, on='parent_asin', how='left')

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]

        data = {
            'user_idx': torch.tensor(self.hashmaps['user'].get(row.user_id, -1), dtype=torch.long),
            'item_idx': torch.tensor(self.hashmaps['item'].get(row.parent_asin, -1), dtype=torch.long),
            'category_idx': torch.tensor(self.hashmaps['category'].get(row.categories, -1), dtype=torch.long),
            'store_idx': torch.tensor(self.hashmaps['store'].get(row.store, -1), dtype=torch.long),
            'text':torch.tensor(0), #Tdo - fix this
            'rating': torch.tensor(row.rating, dtype=torch.float32)
        }
        # only add images if path is provided
        if self.images_path:
            image_path = self.images_path / f"{row.parent_asin}.jpg"
            image = Image.open(image_path)
            data['image'] = image

        return data

    @property
    def num_users(self) -> int:
        return len(self.hashmaps['user'])

    @property
    def num_items(self) -> int:
        return len(self.hashmaps['item'])

    @property
    def num_categories(self) -> int:
        return len(self.hashmaps['category'])

    @property
    def num_stores(self) -> int:
        return len(self.hashmaps['store'])
