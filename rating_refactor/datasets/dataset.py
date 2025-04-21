import torch
from sympy.codegen.ast import Raise
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple, Dict, Any


class InteractionDataset(Dataset):
    """PyTorch Dataset for warm items with metadata."""

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
        self.compute_item_user_frequency()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'user_idx': torch.tensor(row.user_idx, dtype=torch.long),
            'item_idx': torch.tensor(row.item_idx, dtype=torch.long),
            'rating': torch.tensor(row.rating, dtype=torch.float32),
            'user_id': row.user_id,
            'parent_asin': row.parent_asin
        }

    def compute_item_user_frequency(self):
        """Compute the frequency of each item and user in the dataset."""
        print('computing item user frequency...')
        item_frequency = self.df['item_idx'].value_counts().to_dict()
        user_frequency = self.df['user_idx'].value_counts().to_dict()

        return item_frequency, user_frequency

    @property
    def num_users(self):
        return self.df.user_idx.max() + 1  # +1 for 0-based indexing

    @property
    def num_items(self):
        return self.df.item_idx.max() + 1  # +1 for 0-based indexing


class DatasetFactory:
    @staticmethod
    def create_datasets(train_path: str,
                        test_path: str,
                        scenario: str) -> tuple[InteractionDataset, InteractionDataset, InteractionDataset]:
        train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)

        if scenario == 'warm':
            train, val = DatasetFactory.warm_item_split_train_validation(train)
            return InteractionDataset(train), InteractionDataset(val), InteractionDataset(test)
        elif scenario == 'cold':
            train, val = DatasetFactory.cold_item_split_train_validation(train)
            return InteractionDataset(train), InteractionDataset(val), InteractionDataset(test)
        else:
            raise ValueError(f"Invalid scenario: {scenario}. Expected 'warm' or 'cold'")

    @staticmethod
    def warm_item_split_train_validation(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the input DataFrame into training and validation sets.
        The validation set contains the last rating of each user based on the timestamp.

        Parameters:
            df (pd.DataFrame): Input DataFrame with columns ['user_id', 'parent_asin', 'rating', 'timestamp'].

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training set DataFrame and validation set DataFrame.
        """
        df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
        last_interactions = df.groupby('user_id').tail(1)
        val_df = df.loc[last_interactions.index]
        train_df = df.drop(last_interactions.index)
        return train_df, val_df

    @staticmethod
    def cold_item_split_train_validation(df: pd.DataFrame, test_ratio: int = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = df.sort_values('timestamp')

        # Calculate split point
        split_idx = int(len(df) * (1 - test_ratio))
        split_timestamp = df.iloc[split_idx]['timestamp']

        df_train = df[df['timestamp'] < split_timestamp]
        df_val = df[df['timestamp'] >= split_timestamp]

        # Get items that appear only in test set
        train_items = set(df_train['parent_asin'].unique())
        val_items = set(df_val['parent_asin'].unique())
        cold_start_items = val_items - train_items
        print(f"Cold start items: {len(cold_start_items)}")
        print(f"Train items: {len(train_items)}")

        df_val_cold = df_val[df_val['parent_asin'].isin(cold_start_items)]

        return df_train, df_val_cold


if __name__ == '__main__':
    train_path = '/tmp/pycharm_project_190/rating_refactor/proc/warm_items.parquet'
    test_path = '/tmp/pycharm_project_190/rating_refactor/proc/warm_items_test.parquet'

    train, val, test = DatasetFactory.create_datasets(train_path, test_path, scenario='warm')
    print(test[0])
