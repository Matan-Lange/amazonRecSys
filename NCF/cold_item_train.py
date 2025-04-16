import wandb
import torch
from typing import  Tuple
from train import Trainer
from dataset import DatasetFactory, AmazonDataset
from NCF_model import NCF

class TimeBasedColdStartDatasetFactory(DatasetFactory):
    def create_cold_start_splits(self, test_ratio=0.2) -> Tuple['AmazonDataset', 'AmazonDataset']:
        """Create train/test splits for cold-start scenario based on timestamp"""
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

        train_dataset = AmazonDataset(df_train, self.df_metadata, self.hashmaps)
        test_dataset = AmazonDataset(df_test_cold, self.df_metadata, self.hashmaps)

        return train_dataset, test_dataset


class Config:
    emb_dim = 16
    batch_size = 1024
    learning_rate = 5e-4 #,0.0001
    weight_decay = 0.1
    epochs = 15
    image_model = 'fashion_clip'  # 'dinov2' or 'fashion_clip'
    def to_dict(self):
        return self.__dict__


def train_cold_start():
    config = Config()
    wandb.init(project='NCF_cold_start', config=config.to_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device



    train_path = '/tmp/pycharm_project_760/data/user_item_rating_table_train.csv'
    test_path = '/tmp/pycharm_project_760/data/warm_items_rating_prediction_test_format.csv'
    metadata_path = '/tmp/pycharm_project_760/data/items_metadata.jsonl'

    # Create cold start datasets
    factory = TimeBasedColdStartDatasetFactory(train_path, test_path, metadata_path)
    train_dataset, test_dataset = factory.create_cold_start_splits(test_ratio=0.2)

    # Load item embeddings
    if config.image_model == 'fashion_clip':
        item_embeddings = torch.nn.Embedding(198771, 512)
        item_embeddings.load_state_dict(torch.load('/tmp/pycharm_project_760/NCF/fashion_clip_embeddings.pt'))

    # Initialize model
    model = NCF(
        num_users=train_dataset.num_users,
        item_embbeings=item_embeddings
    )

    # Train model
    trainer = Trainer(model, train_dataset, test_dataset, config)
    final_rmse = trainer.train()
    wandb.log({"final_cold_start_rmse": final_rmse})

train_cold_start()