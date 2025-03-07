import torch
from torch.utils.data import Dataset
import pandas as pd


class AmazonDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the dataset with user, item, and rating data.

        Args:
            df (pd.DataFrame): DataFrame containing user_id, parent_asin, and rating columns.
        """
        super().__init__()
        self.df = df[['user_id', 'parent_asin', 'rating']]
        self.users = df.user_id.values
        self.items = df.parent_asin.values
        self.y_rating = self.df.rating.values

        # Convert user and item strings to ids & keep map for inference
        self.user_hashmap = self.create_id_map(self.df.user_id)
        self.item_hashmap = self.create_id_map(self.df.parent_asin)

    @staticmethod
    def create_id_map(ids: pd.Series) -> dict:
        """
        Creates a mapping from unique ids to integer indices.

        Args:
            ids (pd.Series): Series of ids.

        Returns:
            dict: Mapping from id to index.
        """
        ids = ids.unique()
        return {id: i for i, id in enumerate(ids)}

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.df)

    def get_num_users(self) -> int:
        """
        Returns the number of unique users.
        """
        return len(self.user_hashmap)

    def get_num_items(self) -> int:
        """
        Returns the number of unique items.
        """
        return len(self.item_hashmap)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves the user_id, item_id, and rating for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (user_id, item_id, rating) for the given index.
        """
        user_id = self.user_hashmap[self.users[idx]]
        item_id = self.item_hashmap[self.items[idx]]
        rating = torch.tensor(self.y_rating[idx], dtype=torch.float32)

        return user_id, item_id, rating


if __name__ == "__main__":
    # Test
    path = "insert/path/here"
    df = pd.read_csv(path)
    dataset = AmazonDataset(df)
    print(dataset.get_num_items())
    print(dataset.get_num_users())
    print(dataset[0])
