import os
import argparse
import torch
import wandb
import yaml
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Tuple, Dict, Any, Optional

# Load environment variables
load_dotenv()

# Import models
from rating_refactor.models import MFModel, NCFModel
# Import trainers
from rating_refactor.trainers.base import BaseTrainer
# Import datasets
from rating_refactor.datasets import DatasetFactory


def get_trainer_class(trainer_type):
    """Get trainer class based on trainer type."""
    trainers = {
        'base': BaseTrainer,

    }
    return trainers.get(trainer_type, BaseTrainer)


def get_model_class(model_type):
    """Get model class based on model type."""
    models = {
        'mf': MFModel,
        'ncf': NCFModel
    }
    return models.get(model_type)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # init wandb
    wandb.init(project=f"{args.model_type}_{args.scenario}", config=vars(args))

    train_path = args.train_path or os.getenv('TRAIN_PATH')
    test_path = args.test_path or os.getenv('TEST_PATH')
    metadata_path = args.metadata_path or os.getenv('METADATA_PATH')

    train_dataset, val_dataset, test_dataset = DatasetFactory.create_datasets(train_path, test_path, args.scenario)

    model_class = get_model_class(args.model_type)
    if args.model_type == 'mf':
        model = model_class(
            num_users=train_dataset.num_users,
            num_items=train_dataset.num_items,
            embed_dim=args.embed_dim,
            biases=args.biases.split(',') if args.biases else [],
            dropout_rate=args.dropout_rate
        )

    elif args.model_type == 'ncf':
        model = model_class(
            num_users=train_dataset.num_users,
            embed_dim=args.embed_dim,
        )


    class Config:
        def __init__(self, args, device):
            self.batch_size = args.batch_size
            self.learning_rate = args.learning_rate
            self.weight_decay = args.weight_decay
            self.epochs = args.epochs
            self.device = device
            self.dropout_rate = args.dropout_rate

    config = Config(args, device)
    trainer_class = get_trainer_class(args.trainer_type)
    trainer = trainer_class(model, train_dataset, val_dataset, config, test_dataset)

    trainer.train()
    trainer.test()



def main():
    parser = argparse.ArgumentParser(description='Train a recommendation model')

    # Configuration file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file (YAML)')

    # Model parameters
    parser.add_argument('--model_type', type=str, choices=['mf', 'ncf'], default='mf',
                        help='Type of model to train (mf or ncf)')
    parser.add_argument('--trainer_type', type=str, choices=['base', 'weighted', 'ncf', 'adaptive'], default='base',
                        help='Type of trainer to use (base, weighted, ncf, or adaptive)')
    parser.add_argument('--scenario', type=str, choices=['warm', 'cold_start'], default='warm',
                        help='Training scenario (warm or cold_start)')

    # Hyperparameters
    parser.add_argument('--embed_dim', type=int, default=8,
                        help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of epochs')
    parser.add_argument('--biases', type=str, default='user,item,category,store',
                        help='Comma-separated list of biases to use (for MF model)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Test ratio for cold start scenario')
    parser.add_argument('--image_model', type=str, choices=['fashion_clip', 'dinov2'], default='fashion_clip',
                        help='Image model to use for NCF')
    parser.add_argument('--huber_delta', type=float, default=1.0,
                        help='Delta parameter for Huber loss (for adaptive trainer)')
    parser.add_argument('--l2_reg', type=float, default=0.01,
                        help='L2 regularization strength (for adaptive trainer)')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate for embeddings (for MF model)')

    # Paths
    parser.add_argument('--train_path', type=str, default=None,
                        help='Path to training data (defaults to TRAIN_PATH in .env)')
    parser.add_argument('--test_path', type=str, default=None,
                        help='Path to test data (defaults to TEST_PATH in .env)')
    parser.add_argument('--metadata_path', type=str, default=None,
                        help='Path to metadata (defaults to METADATA_PATH in .env)')
    parser.add_argument('--embeddings_path', type=str, default=None,
                        help='Path to embeddings (defaults to FASHION_CLIP_EMBEDDINGS in .env)')

    args = parser.parse_args()

    def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.

        Parameters:
            config_path (str): Path to the YAML configuration file.

        Returns:
            Dict[str, Any]: Dictionary containing configuration parameters.
        """
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    # Load configuration from file if specified
    if args.config:
        config = load_config_from_yaml(args.config)
        # Update args with values from config file (command-line args take precedence)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)

    train(args)


if __name__ == "__main__":
    main()
