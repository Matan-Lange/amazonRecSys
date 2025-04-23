from torch.utils.data import Dataset
import torch


class PairwiseDataset(Dataset):
    def __init__(self, data):
        self.data = data.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'user_idx': torch.tensor(row['user_idx'], dtype=torch.long),
            'positive_item_idx': torch.tensor(row['positive_item_idx'], dtype=torch.long),
            'negative_item_idx': torch.tensor(row['negative_item_idx'], dtype=torch.long),
            'parent_asin': row['parent_asin'],
            'user_id': row['user_id'],
        }


class PairwiseTestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'user_idx': torch.tensor(row['user_idx'], dtype=torch.long),
            'item_0_idx': torch.tensor(row['item_0_idx'], dtype=torch.long),
            'item_1_idx': torch.tensor(row['item_1_idx'], dtype=torch.long),
            'item_0': row['item_0'],
            'item_1': row['item_1'],
            'user_id': row['user_id'],

        }
