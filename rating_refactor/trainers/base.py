"""
Base trainer for recommendation models.

This module contains the base trainer class for training recommendation models:
- BaseTrainer: Generic trainer for recommendation models with modular design for easy extension
"""

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from collections import defaultdict


class BaseTrainer:
    """Base trainer for recommendation models with modular design."""

    def __init__(
            self,
            model: Any,
            train_dataset: Any,
            val_dataset: Any,
            config: Any,
            test_dataset: Any = None
    ):
        """
        Initialize trainer with model, datasets, and configuration.

        Args:
            model: PyTorch model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Configuration object with training parameters
            test_dataset: Test dataset for final evaluation (optional)
        """
        self.model = model.to(config.device)
        self.batch_size = config.batch_size
        self.lr = config.learning_rate
        self.weight_decay = config.weight_decay
        self.train_dataloader = self._create_dataloader(train_dataset, shuffle=True)
        self.val_dataloader = self._create_dataloader(val_dataset, shuffle=False)
        self.test_dataset = test_dataset
        if test_dataset is not None:
            self.test_dataloader = self._create_dataloader(test_dataset, shuffle=False)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler(config)
        self.num_epochs = config.epochs
        self.device = config.device
        self.step = 0

        # Set model name based on model class
        self.model_name = model.__class__.__name__
        self.best_model_path = f'best_model_{self.model_name.lower()}.pt'

        # Compute user and item frequencies for head/tail analysis
        self.item_freq, self.user_freq = train_dataset.compute_item_user_frequency()

        # Define head and tail thresholds
        self.head_item_threshold = getattr(config, 'head_item_threshold', 1000)
        self.tail_item_threshold = getattr(config, 'tail_item_threshold', 5)
        self.head_user_threshold = getattr(config, 'head_user_threshold', 50)
        self.tail_user_threshold = getattr(config, 'tail_user_threshold', 5)

        # Identify head and tail users and items
        self._identify_head_tail_entities()

    def _create_dataloader(self, dataset: Any, shuffle: bool = True) -> DataLoader:
        """
        Create a DataLoader for the given dataset.

        Args:
            dataset: Dataset to create loader for
            shuffle: Whether to shuffle the data

        Returns:
            DataLoader for the dataset
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=5
        )

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer for the model.

        Returns:
            Optimizer instance
        """
        return AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def _create_scheduler(self, config: Any):
        """
        Create learning rate scheduler.

        Args:
            config: Configuration object with training parameters

        Returns:
            Learning rate scheduler
        """
        return OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            total_steps=len(self.train_dataloader) * config.epochs,
            pct_start=getattr(config, 'warmup_pct', 0.05),
            div_factor=getattr(config, 'div_factor', 25),
            final_div_factor=getattr(config, 'final_div_factor', 1e4),
        )

    def _identify_head_tail_entities(self) -> None:
        """
        Identify head and tail users and items based on frequency thresholds.
        """
        self.head_items = set(idx for idx, freq in self.item_freq.items()
                              if freq >= self.head_item_threshold)
        self.tail_items = set(idx for idx, freq in self.item_freq.items()
                              if freq <= self.tail_item_threshold)
        self.head_users = set(idx for idx, freq in self.user_freq.items()
                              if freq >= self.head_user_threshold)
        self.tail_users = set(idx for idx, freq in self.user_freq.items()
                              if freq <= self.tail_user_threshold)

        print(f"Head items: {len(self.head_items)}, Tail items: {len(self.tail_items)}")
        print(f"Head users: {len(self.head_users)}, Tail users: {len(self.tail_users)}")

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Generic collate function to handle dictionary batch data.

        Args:
            batch: List of dictionaries containing batch data

        Returns:
            Dictionary with batched tensors or lists
        """
        batch_dict = {}
        for key in batch[0]:
            if isinstance(batch[0][key], torch.Tensor):
                batch_dict[key] = torch.stack([item[key] for item in batch])
            else:
                batch_dict[key] = [item[key] for item in batch]
        return batch_dict

    def calculate_rmse(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate RMSE between predictions and targets.

        Args:
            predictions: Predicted values
            targets: Target values

        Returns:
            RMSE value
        """
        return torch.sqrt(torch.mean((predictions - targets) ** 2))

    def process_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Process batch data before model forward pass insert to device.
        """
        return {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    def forward_pass(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform forward pass through the model.
        """
        return self.model(batch)

    def calculate_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss between predictions and targets. - can be extended
        """
        return self.criterion(predictions, targets)

    def backward_pass(self, loss: torch.Tensor) -> None:
        """
        Perform backward pass to update model parameters.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def log_step_metrics(self, loss: torch.Tensor) -> None:
        """
        Log metrics for the current training step.
        """
        current_lr = self.scheduler.get_last_lr()[0]
        wandb.log({
            "train_step_loss": loss.item(),
            "learning_rate": current_lr,
            "step": self.step
        })
        self.step += 1

    def collect_segment_metrics(
            self,
            user_ids: torch.Tensor,
            item_ids: torch.Tensor,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Collect predictions and targets for different user/item segments.

        Args:
            user_ids: User IDs in the batch
            item_ids: Item IDs in the batch
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Dictionary with segment predictions and targets
        """
        segments = {
            "head_items": {"preds": [], "targets": []},
            "tail_items": {"preds": [], "targets": []},
            "head_users": {"preds": [], "targets": []},
            "tail_users": {"preds": [], "targets": []}
        }

        # Collect metrics for different item segments
        for i, item_id in enumerate(item_ids):
            item_id = item_id.item()
            if item_id in self.head_items:
                segments["head_items"]["preds"].append(predictions[i].detach())
                segments["head_items"]["targets"].append(targets[i].detach())
            elif item_id in self.tail_items:
                segments["tail_items"]["preds"].append(predictions[i].detach())
                segments["tail_items"]["targets"].append(targets[i].detach())

        # Collect metrics for different user segments
        for i, user_id in enumerate(user_ids):
            user_id = user_id.item()
            if user_id in self.head_users:
                segments["head_users"]["preds"].append(predictions[i].detach())
                segments["head_users"]["targets"].append(targets[i].detach())
            elif user_id in self.tail_users:
                segments["tail_users"]["preds"].append(predictions[i].detach())
                segments["tail_users"]["targets"].append(targets[i].detach())

        return segments

    def calculate_segment_metrics(
            self,
            segments: Dict[str, Dict[str, List[torch.Tensor]]],
            prefix: str = ""
    ) -> Dict[str, float]:
        """
        Calculate metrics for different user/item segments.

        Args:
            segments: Dictionary with segment predictions and targets
            prefix: Prefix for metric names (e.g., 'train_' or 'val_')

        Returns:
            Dictionary with segment metrics
        """
        metrics = {}

        for segment_name, segment_data in segments.items():
            if segment_data["preds"]:
                segment_rmse = self.calculate_rmse(
                    torch.tensor(segment_data["preds"]),
                    torch.tensor(segment_data["targets"])
                ).item()

                metric_name = f"{prefix}{segment_name}_rmse"
                metrics[metric_name] = segment_rmse
                print(f"{metric_name.capitalize()}: {segment_rmse:.4f}")

                # Log to wandb
                wandb.log({metric_name: segment_rmse})

        return metrics

    def train_epoch(self, epoch: int) -> Tuple[float, float, Dict[str, float]]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average training loss, training RMSE, segment metrics)
        """
        self.model.train()
        total_train_loss = 0
        all_train_preds = []
        all_train_targets = []

        # Segment metrics collection
        segments = {
            "head_items": {"preds": [], "targets": []},
            "tail_items": {"preds": [], "targets": []},
            "head_users": {"preds": [], "targets": []},
            "tail_users": {"preds": [], "targets": []}
        }

        for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"):
            # Process batch
            batch = self.process_batch(batch)
            user_ids = batch['user_idx']
            item_ids = batch['item_idx']
            rating = batch['rating']

            # Forward pass
            prediction = self.forward_pass(batch)

            # Calculate loss
            loss = self.calculate_loss(prediction, rating)

            # Backward pass
            self.backward_pass(loss)

            # Log step metrics
            self.log_step_metrics(loss)

            # Collect metrics
            total_train_loss += loss.item()
            all_train_preds.extend(prediction.detach())
            all_train_targets.extend(rating.detach())

            # Collect segment metrics
            batch_segments = self.collect_segment_metrics(user_ids, item_ids, prediction, rating)
            for segment_name, segment_data in batch_segments.items():
                segments[segment_name]["preds"].extend(segment_data["preds"])
                segments[segment_name]["targets"].extend(segment_data["targets"])

        # Calculate overall training metrics
        avg_train_loss = total_train_loss / len(self.train_dataloader)
        train_rmse = self.calculate_rmse(
            torch.tensor(all_train_preds),
            torch.tensor(all_train_targets)
        ).item()

        # Calculate segment metrics
        segment_metrics = self.calculate_segment_metrics(segments, prefix="train_")

        return avg_train_loss, train_rmse, segment_metrics

    def validate(self) -> Tuple[float, float, Dict[str, float]]:
        """
        Validate the model.

        Returns:
            Tuple of (average validation loss, validation RMSE, segment metrics)
        """
        self.model.eval()
        total_val_loss = 0
        all_val_preds = []
        all_val_targets = []

        # Segment metrics collection
        segments = {
            "head_items": {"preds": [], "targets": []},
            "tail_items": {"preds": [], "targets": []},
            "head_users": {"preds": [], "targets": []},
            "tail_users": {"preds": [], "targets": []}
        }

        # For visualization
        save_predictions = []
        save_ratings = []

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Process batch
                batch = self.process_batch(batch)
                user_ids = batch['user_idx']
                item_ids = batch['item_idx']
                rating = batch['rating']

                # Forward pass
                prediction = self.forward_pass(batch)

                # Calculate loss
                loss = self.calculate_loss(prediction, rating)

                # Collect metrics
                total_val_loss += loss.item()
                all_val_preds.extend(prediction.cpu())
                all_val_targets.extend(rating.cpu())

                # Save predictions for later analysis
                save_predictions.extend(prediction.cpu().numpy())
                save_ratings.extend(rating.cpu().numpy())

                # Collect segment metrics
                batch_segments = self.collect_segment_metrics(user_ids, item_ids, prediction, rating)
                for segment_name, segment_data in batch_segments.items():
                    segments[segment_name]["preds"].extend(segment_data["preds"])
                    segments[segment_name]["targets"].extend(segment_data["targets"])

            # Log predictions to wandb
            wandb.log({
                "predictions": wandb.Histogram(save_predictions),
                "ratings": wandb.Histogram(save_ratings)
            })

        # Calculate overall validation metrics
        avg_val_loss = total_val_loss / len(self.val_dataloader)
        val_rmse = self.calculate_rmse(
            torch.tensor(all_val_preds),
            torch.tensor(all_val_targets)
        ).item()

        # Calculate segment metrics
        segment_metrics = self.calculate_segment_metrics(segments, prefix="val_")

        return avg_val_loss, val_rmse, segment_metrics

    def train(self) -> float:
        """
        Train model with wandb tracking.

        Returns:
            Final validation RMSE
        """
        self.best_val_rmse = float('inf')

        for epoch in range(self.num_epochs):
            # Train for one epoch
            avg_train_loss, train_rmse, train_segment_metrics = self.train_epoch(epoch)

            # Validate
            avg_val_loss, val_rmse, val_segment_metrics = self.validate()

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_rmse": train_rmse,
                "val_loss": avg_val_loss,
                "val_rmse": val_rmse
            })

            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                torch.save(self.model.state_dict(), self.best_model_path)
                wandb.log({"best_val_rmse": val_rmse})

        # Upload best model to wandb
        artifact = wandb.Artifact(
            name=f"best_{self.model_name.lower()}_model_{wandb.run.id}",
            type="model",
            description=f"Best {self.model_name} model with val_rmse: {self.best_val_rmse:.4f}"
        )
        artifact.add_file(self.best_model_path)
        wandb.log_artifact(artifact)

        return val_rmse

    def test(self, output_file: str = None) -> float:
        """
        Test the model on the test dataset and save predictions to CSV.

        Args:
            output_file: Path to save the predictions CSV file (default: predictions_{model_name}.csv)

        Returns:
            Test RMSE
        """
        if not hasattr(self, 'test_dataloader'):
            raise ValueError("Test dataset not provided during initialization")

        # Load best model weights
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path))
            print(f"Loaded best model from {self.best_model_path}")

        self.model.eval()
        all_test_preds = []
        all_user_ids = []
        all_item_ids = []

        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Testing"):
                # Process batch
                batch = self.process_batch(batch)
                user_ids = batch['user_idx']
                item_ids = batch['item_idx']
                rating = batch['rating']

                # Forward pass
                prediction = self.forward_pass(batch)

                # Collect metrics
                all_test_preds.extend(prediction.cpu().numpy())
                all_user_ids.extend(batch['user_id'].cpu().numpy())
                all_item_ids.extend(batch['parent_asin'].cpu().numpy())

        predictions_df = pd.DataFrame({
            'user_id': all_user_ids,
            'parent_asin': all_item_ids,
            'rating': all_test_preds,
        })

        # Save to CSV
        if output_file is None:
            output_file = f"predictions_{self.model_name.lower()}.csv"

        predictions_df.to_csv(output_file, index=False)
        print(f"Saved predictions to {output_file}")

        # log csv to wandb
        predictions_artifact = wandb.Artifact(
            name=f"{self.model_name}_{args.scenario}_predictions_{wandb.run.id}",
            type="predictions",
            description=f"Predictions for {self.model_name}"
        )
        predictions_artifact.add_file(output_file)
        wandb.log_artifact(predictions_artifact)

        return test_rmse
