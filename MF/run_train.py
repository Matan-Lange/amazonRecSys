import wandb
import torch
from MF_model import MfModel
from train import Trainer
from dataset import DatasetFactory


class Config:
    emb_dim = 8
    batch_size = 16384
    biases = ['user', 'item', 'category', 'store']
    learning_rate = 0.005
    weight_decay = 0.001
    epochs = 20

    def to_dict(self):
        return self.__dict__


def train():
    config = Config()

    wandb.init(project='MF', config=config.to_dict())

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    # Setup paths
    train_path = "/tmp/data/recsys_data_and_test_files/user_item_rating_train.csv"
    test_path = "/tmp/data/recsys_data_and_test_files/warm_items_rating_prediction_test_format.csv"
    metadata_path = "/tmp/data/recsys_data_and_test_files/items_metadata.jsonl"

    # Create datasets using factory
    factory = DatasetFactory(train_path, test_path, metadata_path)
    train_dataset, val_dataset, test_dataset = factory.create_datasets()

    model = MfModel(
        num_users=train_dataset.num_users,
        num_items=train_dataset.num_items,
        num_categories=train_dataset.num_categories,
        num_stores=train_dataset.num_stores,
        emb_dim=config.emb_dim,
        biases=config.biases
    )

    trainer = Trainer(model, train_dataset, val_dataset, config)
    final_rmse = trainer.train()
    wandb.log({"final_val_rmse": final_rmse})


if __name__ == "__main__":
    train()
