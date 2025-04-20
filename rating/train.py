import os
import argparse
import torch
import wandb
import yaml
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Import models
from rating.models.mf import MFModel
from rating.models.ncf import NCFModel

# Import trainers
from rating.trainers.base import BaseTrainer
from rating.trainers.weighted import WeightedTrainer
from rating.trainers.ncf import NCFTrainer
from rating.trainers.adaptive import FrequencyAdaptiveTrainer
from rating.trainers.sampling import SamplingTrainer
# Import datasets
from rating.datasets import DatasetFactory, TimeBasedColdStartDatasetFactory


def get_trainer_class(trainer_type):
    """Get trainer class based on trainer type."""
    trainers = {
        'base': BaseTrainer,
        'weighted': WeightedTrainer,
        'ncf': NCFTrainer,
        'adaptive': FrequencyAdaptiveTrainer,
        'sampling': SamplingTrainer
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
    """Train a model with the specified configuration."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    wandb.init(project=f"{args.model_type}_{args.scenario}", config=vars(args))

    # Setup paths
    train_path = args.train_path or os.getenv('TRAIN_PATH')
    test_path = args.test_path or os.getenv('TEST_PATH')
    metadata_path = args.metadata_path or os.getenv('METADATA_PATH')

    # Create datasets
    if args.scenario == 'cold_start':
        factory = TimeBasedColdStartDatasetFactory(train_path, test_path, metadata_path)
        train_dataset, test_dataset = factory.create_cold_start_splits(test_ratio=args.test_ratio)
        val_dataset = test_dataset  # Use test dataset as validation for cold start
    else:
        factory = DatasetFactory(train_path, test_path, metadata_path)
        train_dataset, val_dataset, test_dataset = factory.create_datasets()

    # Create model
    model_class = get_model_class(args.model_type)

    if args.model_type == 'mf':
        model = model_class(
            num_users=train_dataset.num_users,
            num_items=train_dataset.num_items,
            num_categories=train_dataset.num_categories,
            num_stores=train_dataset.num_stores,
            embed_dim=args.embed_dim,
            biases=args.biases.split(',') if args.biases else [],
            dropout_rate=args.dropout_rate
        )
    elif args.model_type == 'ncf':
        # Load item embeddings for NCF
        if args.image_model:
            embeddings_path = args.embeddings_path or os.getenv('FASHION_CLIP_EMBEDDINGS')
            item_embeddings = torch.nn.Embedding(198771, 512)
            item_embeddings.load_state_dict(torch.load(embeddings_path))
            model = model_class(
                num_users=train_dataset.num_users,
                item_embeddings=item_embeddings
            )
        else:
            model = model_class(
                num_users=train_dataset.num_users
            )

    # Create trainer
    trainer_class = get_trainer_class(args.trainer_type)

    # Create config object for trainer
    class Config:
        def __init__(self, args, device):
            self.batch_size = args.batch_size
            self.learning_rate = args.learning_rate
            self.weight_decay = args.weight_decay
            self.epochs = args.epochs
            self.device = device
            self.image_model = args.image_model
            self.huber_delta = args.huber_delta
            self.l2_reg = args.l2_reg
            self.dropout_rate = args.dropout_rate

    config = Config(args, device)

    # Train model
    trainer = trainer_class(model, train_dataset, val_dataset, config)
    final_rmse = trainer.train()

    # Log final results
    wandb.log({"final_val_rmse": final_rmse})

    # ===== PREDICTION STEP =====
    # Load the best model weights
    best_model_path = trainer.best_model_path
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.eval()

    # Create dataloader for test set
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=trainer.collate_fn
    )

    # Make predictions
    all_predictions = []
    all_user_ids = []
    all_item_ids = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Predicting on test set"):
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            predictions = model(batch)

            # Get original user and item IDs
            user_indices = batch['user_idx'].cpu().numpy()
            item_indices = batch['item_idx'].cpu().numpy()

            # Convert indices back to original IDs
            user_id_map = {idx: id for id, idx in train_dataset.hashmaps['user'].items()}
            item_id_map = {idx: id for id, idx in train_dataset.hashmaps['item'].items()}

            user_ids = [user_id_map[idx] for idx in user_indices]
            item_ids = [item_id_map[idx] for idx in item_indices]

            all_predictions.extend(predictions.cpu().numpy())
            all_user_ids.extend(user_ids)
            all_item_ids.extend(item_ids)

    # Create DataFrame with predictions

    df_predictions = pd.DataFrame({
        'user_id': all_user_ids,
        'parent_asin': all_item_ids,
        'rating': all_predictions
    })

    # Save predictions to CSV
    output_path = f"{args.model_type}_{args.scenario}_predictions.csv"
    df_predictions.to_csv(output_path, index=False)

    # Upload predictions file to wandb as an artifact
    predictions_artifact = wandb.Artifact(
        name=f"{args.model_type}_{args.scenario}_predictions_{wandb.run.id}",
        type="predictions",
        description=f"Predictions for {args.model_type} model on {args.scenario} scenario"
    )
    predictions_artifact.add_file(output_path)
    wandb.log_artifact(predictions_artifact)

    print(f"Predictions saved to {output_path} and uploaded to wandb")

    print(f"Training completed with final RMSE: {final_rmse:.4f}")
    return final_rmse


def load_config_from_yaml(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
