"""
Base trainer for recommendation models.

This module contains the base trainer class for training recommendation models:
- BaseTrainer: Generic trainer for recommendation models
"""

import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

from rating.models.base import BaseRecommenderModel


class BaseTrainer:
    """Base trainer for recommendation models."""
    
    def __init__(
        self, 
        model: BaseRecommenderModel, 
        train_dataset: Any, 
        val_dataset: Any, 
        config: Any
    ):
        """
        Initialize trainer with model, datasets, and configuration.
        
        Args:
            model: PyTorch model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Configuration object with training parameters
        """
        self.model = model.to(config.device)
        self.batch_size = config.batch_size
        self.lr = config.learning_rate
        self.weight_decay = config.weight_decay
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=getattr(config, 'num_workers', 5)
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=getattr(config, 'num_workers', 5)
        )
        self.criterion = torch.nn.MSELoss()
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )

        # Setup OneCycleLR scheduler with warmup
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            total_steps=len(self.train_dataloader) * config.epochs,
            pct_start=getattr(config, 'warmup_pct', 0.05),  # Default 5% of training for warmup
            div_factor=getattr(config, 'div_factor', 25),  # initial_lr = max_lr/25
            final_div_factor=getattr(config, 'final_div_factor', 1e4),  # final_lr = initial_lr/1e4
        )

        self.num_epochs = config.epochs
        self.device = config.device
        self.step = 0
        
        # Set model name based on model class
        self.model_name = model.__class__.__name__
        self.best_model_path = f'best_model_{self.model_name.lower()}.pt'

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Generic collate function to handle dictionary batch data.
        
        Args:
            batch: List of dictionaries containing batch data
            
        Returns:
            Dictionary with batched tensors
        """
        batch_dict = {}
        for key in batch[0]:
            batch_dict[key] = torch.stack([item[key] for item in batch])
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

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average training loss, training RMSE)
        """
        self.model.train()
        total_train_loss = 0
        all_train_preds = []
        all_train_targets = []

        for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            rating = batch['rating']
            self.optimizer.zero_grad()
            prediction = self.model(batch)
            loss = self.criterion(prediction, rating)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            current_lr = self.scheduler.get_last_lr()[0]
            wandb.log({
                "train_step_loss": loss.item(),
                "learning_rate": current_lr,
                "step": self.step
            })
            self.step += 1

            total_train_loss += loss.item()
            all_train_preds.extend(prediction.detach())
            all_train_targets.extend(rating.detach())

        # Calculate training metrics
        avg_train_loss = total_train_loss / len(self.train_dataloader)
        train_rmse = self.calculate_rmse(
            torch.tensor(all_train_preds),
            torch.tensor(all_train_targets)
        )
        
        return avg_train_loss, train_rmse.item()

    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average validation loss, validation RMSE)
        """
        self.model.eval()
        total_val_loss = 0
        all_val_preds = []
        all_val_targets = []
        
        save_predictions = []
        save_ratings = []

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                rating = batch['rating']
                prediction = self.model(batch)
                # Optional clamping can be implemented in subclasses
                loss = self.criterion(prediction, rating)

                total_val_loss += loss.item()
                all_val_preds.extend(prediction.cpu())
                all_val_targets.extend(rating.cpu())
                
                # Save predictions for later analysis
                save_predictions.extend(prediction.cpu().numpy())
                save_ratings.extend(rating.cpu().numpy())
            
            # Log predictions to wandb
            wandb.log({
                "predictions": wandb.Histogram(save_predictions), 
                "ratings": wandb.Histogram(save_ratings)
            })

        # Calculate validation metrics
        avg_val_loss = total_val_loss / len(self.val_dataloader)
        val_rmse = self.calculate_rmse(
            torch.tensor(all_val_preds),
            torch.tensor(all_val_targets)
        )
        
        return avg_val_loss, val_rmse.item()

    def train(self) -> float:
        """
        Train model with wandb tracking.
        
        Returns:
            Final validation RMSE
        """
        self.best_val_rmse = float('inf')

        for epoch in range(self.num_epochs):
            # Train for one epoch
            avg_train_loss, train_rmse = self.train_epoch(epoch)
            
            # Validate
            avg_val_loss, val_rmse = self.validate()

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

