"""
Weighted trainer for recommendation models.

This module contains the weighted trainer class for training recommendation models:
- WeightedTrainer: Trainer that applies weights based on user and item frequencies
"""

import torch
import numpy as np
from tqdm import tqdm
import wandb
from typing import Dict, List, Tuple, Any, Optional, Union

from rating.trainers.base import BaseTrainer
from rating.models.base import BaseRecommenderModel


class WeightedTrainer(BaseTrainer):
    """Trainer that uses weighted loss based on user and item frequencies."""
    
    def __init__(
        self, 
        model: BaseRecommenderModel, 
        train_dataset: Any, 
        val_dataset: Any, 
        config: Any
    ):
        """
        Initialize weighted trainer with model, datasets, and configuration.
        
        Args:
            model: PyTorch model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Configuration object with training parameters
        """
        super().__init__(model, train_dataset, val_dataset, config)
        
        print("Computing user and item frequencies...")
        self.user_freq, self.item_freq = self._compute_frequencies(train_dataset)
        user_freq_arr = np.array([self.user_freq.get(i, 1) for i in range(model.num_users)])
        item_freq_arr = np.array([self.item_freq.get(i, 1) for i in range(model.num_items)])

        self.user_freq_tensor = torch.tensor(user_freq_arr, dtype=torch.float32, device=config.device)
        self.item_freq_tensor = torch.tensor(item_freq_arr, dtype=torch.float32, device=config.device)
    
    def _compute_frequencies(self, dataset: Any) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Compute user and item frequencies from dataset.
        
        Args:
            dataset: Dataset containing user-item interactions
            
        Returns:
            Tuple of (user frequency dictionary, item frequency dictionary)
        """
        df = dataset.df
        df['user_idx'] = df['user_id'].map(dataset.hashmaps['user'])
        df['item_idx'] = df['parent_asin'].map(dataset.hashmaps['item'])
        user_freq = df['user_idx'].value_counts().to_dict()
        item_freq = df['item_idx'].value_counts().to_dict()
        return user_freq, item_freq
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch with weighted loss.
        
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
            user_ids = batch['user_idx']
            item_ids = batch['item_idx']
            rating = batch['rating']
            self.optimizer.zero_grad()
            prediction = self.model(batch)

            # Weighted loss
            user_freqs = self.user_freq_tensor[user_ids]
            item_freqs = self.item_freq_tensor[item_ids]
            weights = 10 / torch.sqrt(user_freqs * item_freqs + 1e-6)
            loss = torch.mean(weights * (prediction - rating) ** 2)

            wandb.log({
                "avg_sample_weight": weights.mean().item(),
                "max_sample_weight": weights.max().item(),
                "min_sample_weight": weights.min().item(),
            })

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