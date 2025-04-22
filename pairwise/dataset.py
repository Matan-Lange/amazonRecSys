from torch.utils.data import Dataset
import torch


class PairwiseDataset(Dataset):
    def __init__(self, data):
        self.data = data

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
