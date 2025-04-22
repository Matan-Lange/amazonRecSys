import os
import argparse
import torch
import wandb
import pandas as pd
from dotenv import load_dotenv
from torch.nn.functional import dropout

# Load environment variables
load_dotenv()

# Import the NCF model from rating_refactor
from rating_refactor.models import NCFModel

# Import the ContrastiveTrainer and PairwiseDataset
from pairwise.trainer import ContrastiveTrainer
from pairwise.dataset import PairwiseDataset


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    wandb.init(project=f"pairwise_ncf", config=vars(args))

    # Load the datasets
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)

    # Create datasets
    train_dataset = PairwiseDataset(train_df)
    val_dataset = PairwiseDataset(val_df)

    # Get the number of users from the dataset
    num_users = train_df['user_idx'].max() + 1
    print(f'Number of users: {num_users}')

    # # Initialize the NCF model
    # from rating_refactor.proc import emb_layers
    # model = NCFModel(
    #     num_users=num_users,
    #     embed_dim=args.embed_dim,
    #     embeddings=emb_layers,  # Pass the embeddings explicitly
    #     dropout_rate = 0.2
    # )
    from rating_refactor.models import MFModel
    print(f'Number of items: {train_df["positive_item_idx"].max() + 1}')
    model = MFModel(
        num_users=num_users,
        num_items=int(train_df['positive_item_idx'].max() + 1) , #make sure this is int
        embed_dim=args.embed_dim,
        dropout_rate=0
    )
    # Create a config object for the trainer
    class Config:
        def __init__(self, args, device):
            self.batch_size = args.batch_size
            self.learning_rate = args.learning_rate
            self.weight_decay = args.weight_decay
            self.epochs = args.epochs
            self.device = device
            self.loss_type = args.loss_type

    config = Config(args, device)

    # Initialize the trainer
    trainer = ContrastiveTrainer(model, train_dataset, val_dataset, config)

    # Train the model
    best_acc = trainer.train()

    print(f"Training completed with best validation accuracy: {best_acc:.4f}")
    print(f"Best model saved to: {trainer.best_model_path}")


def main():
    parser = argparse.ArgumentParser(description='Train a pairwise recommendation model')

    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of epochs')
    parser.add_argument('--loss_type', type=str, choices=['cross_entropy', 'bce'], default='cross_entropy',
                        help='Loss function type')

    # Paths
    parser.add_argument('--train_path', type=str,
                        default='/tmp/pycharm_project_190/pairwise/train_pairwise.csv',
                        help='/tmp/pycharm_project_190/pairwise/val_pairwise.csv')
    parser.add_argument('--val_path', type=str,
                        default='val_pairwise.csv',
                        help='Path to validation data')

    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == "__main__":
    main()
