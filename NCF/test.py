import wandb
import torch
from NCF_model import NCF
from train import Trainer
from dataset import DatasetFactory


class Config:
    emb_dim = 16
    batch_size = 1024
    learning_rate = 5e-4 #,0.0001
    weight_decay = 0.5
    epochs = 15
    image_model = 'fashion_clip'  # 'dinov2' or 'fashion_clip'
    def to_dict(self):
        return self.__dict__


def train():
    config = Config()

    wandb.init(project='NCF', config=config.to_dict())

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    train_path = '/tmp/pycharm_project_760/data/user_item_rating_table_train.csv'
    test_path = '/tmp/pycharm_project_760/data/warm_items_rating_prediction_test_format.csv'
    metadata_path = '/tmp/pycharm_project_760/data/items_metadata.jsonl'

    # Create datasets using factory
    factory = DatasetFactory(train_path, test_path, metadata_path)
    train_dataset, val_dataset, test_dataset = factory.create_datasets()

    #load item embeddings
    if config.image_model == 'dinov2':
        item_embeddings = torch.nn.Embedding(198771, 768)
        item_embeddings.load_state_dict(torch.load('/tmp/pycharm_project_760/NCF/dinov2_embeddings.pt'))
    elif config.image_model == 'fashion_clip':
        item_embeddings = torch.nn.Embedding(198771, 512)
        item_embeddings.load_state_dict(torch.load('/tmp/pycharm_project_760/NCF/fashion_clip_embeddings.pt'))

    model = NCF(
        num_users=train_dataset.num_users,
        item_embbeings=item_embeddings
    )

    trainer = Trainer(model, train_dataset, val_dataset, config)
    final_rmse = trainer.train()
    wandb.log({"final_val_rmse": final_rmse})


if __name__ == "__main__":
    train()
