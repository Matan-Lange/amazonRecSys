"""
Frequency-adaptive trainer for recommendation models.

This module contains the frequency-adaptive trainer class for training recommendation models:
- FrequencyAdaptiveTrainer: Trainer that implements frequency-adaptive L2 regularization,
  uniform-by-user sampler, and Weighted-Huber loss with adaptive-L2
"""

import torch
import numpy as np
from tqdm import tqdm
import wandb
from collections import defaultdict
from torch.utils.data import DataLoader, Sampler
from typing import Dict, List, Tuple, Any, Optional, Union, Iterator

from rating.trainers.weighted import WeightedTrainer
from rating.models.base import BaseRecommenderModel


class UniformUserSampler(Sampler):
    """Sampler that samples uniformly by user."""
    
    def __init__(self, dataset: Any):
        """
        Initialize the uniform user sampler.
        
        Args:
            dataset: Dataset containing user-item interactions
        """
        super().__init__(dataset)
        self.dataset = dataset
        
        # Group indices by user
        self.user_indices = defaultdict(list)
        df = dataset.df
        df['user_idx'] = df['user_id'].map(dataset.hashmaps['user'])
        
        for idx, user_idx in enumerate(df['user_idx']):
            self.user_indices[user_idx.item() if isinstance(user_idx, np.int64) else user_idx].append(idx)
        
        self.users = list(self.user_indices.keys())
        
    def __iter__(self) -> Iterator[int]:
        """
        Return an iterator over the indices.
        
        Returns:
            Iterator over indices
        """
        # Shuffle users
        users = self.users.copy()
        np.random.shuffle(users)
        
        # For each user, sample one interaction
        indices = []
        for user in users:
            user_indices = self.user_indices[user]
            # If user has multiple interactions, sample one randomly
            if len(user_indices) > 0:
                idx = np.random.choice(user_indices)
                indices.append(idx)
        
        # Shuffle the sampled indices
        np.random.shuffle(indices)
        return iter(indices)
    
    def __len__(self) -> int:
        """
        Return the number of samples.
        
        Returns:
            Number of users
        """
        return len(self.users)


class FrequencyAdaptiveTrainer(WeightedTrainer):
    """Trainer that implements frequency-adaptive L2 regularization, uniform-by-user sampler, and Weighted-Huber loss with adaptive-L2."""
    
    def __init__(
        self, 
        model: BaseRecommenderModel, 
        train_dataset: Any, 
        val_dataset: Any, 
        config: Any
    ):
        """
        Initialize frequency-adaptive trainer with model, datasets, and configuration.
        
        Args:
            model: PyTorch model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Configuration object with training parameters
        """
        # Initialize parent class
        super().__init__(model, train_dataset, val_dataset, config)
        
        # Replace the train dataloader with uniform-by-user sampler
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=UniformUserSampler(train_dataset),
            collate_fn=self.collate_fn,
            num_workers=getattr(config, 'num_workers', 5)
        )
        
        # Compute item frequency bins for analysis
        self.head_items = set(idx for idx, freq in self.item_freq.items() if freq >= 1000)
        self.tail_items = set(idx for idx, freq in self.item_freq.items() if freq < 5)
        
        # Huber loss delta parameter
        self.huber_delta = getattr(config, 'huber_delta', 1.0)
        
        # L2 regularization strength
        self.l2_reg = getattr(config, 'l2_reg', 0.01)
        
        print(f"Head items: {len(self.head_items)}, Tail items: {len(self.tail_items)}")
    
    def huber_loss(self, pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        """
        Compute Huber loss.
        
        Args:
            pred: Predictions
            target: Targets
            delta: Huber loss delta parameter
            
        Returns:
            Huber loss
        """
        abs_error = torch.abs(pred - target)
        quadratic = torch.min(abs_error, torch.tensor(delta, device=self.device))
        linear = abs_error - quadratic
        return 0.5 * quadratic.pow(2) + delta * linear
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch with frequency-adaptive L2 regularization and Weighted-Huber loss.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average training loss, training RMSE)
        """
        self.model.train()
        total_train_loss = 0
        all_train_preds = []
        all_train_targets = []
        
        # Track head and tail item performance
        head_preds = []
        head_targets = []
        tail_preds = []
        tail_targets = []

        for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            user_ids = batch['user_idx']
            item_ids = batch['item_idx']
            rating = batch['rating']
            self.optimizer.zero_grad()
            prediction = self.model(batch)

            # Weighted-Huber loss
            user_freqs = self.user_freq_tensor[user_ids]
            item_freqs = self.item_freq_tensor[item_ids]
            weights = 1 / torch.sqrt(item_freqs + 1e-6)
            huber_losses = self.huber_loss(prediction, rating, self.huber_delta)
            loss = torch.mean(weights * huber_losses)
            
            # # Frequency-adaptive L2 regularization
            # l2_reg_loss = 0
            # for name, param in self.model.named_parameters():
            #     if 'emb' in name:  # Only apply to embedding layers
            #         if 'user' in name:
            #             # Get user frequencies for the parameters
            #             param_freqs = self.user_freq_tensor[:param.size(0)]
            #             # Expand to match parameter shape
            #             param_freqs = param_freqs.view(-1, 1).expand_as(param)
            #         elif 'item' in name:
            #             # Get item frequencies for the parameters
            #             param_freqs = self.item_freq_tensor[:param.size(0)]
            #             # Expand to match parameter shape
            #             param_freqs = param_freqs.view(-1, 1).expand_as(param)
            #         else:
            #             # For other embeddings, use a constant frequency
            #             param_freqs = torch.ones_like(param)
            #
            #         # Adaptive L2 regularization: penalize rare items more
            #         adaptive_weight = 1.0 / torch.sqrt(param_freqs + 1e-6)
            #         l2_reg_loss += torch.sum(adaptive_weight * param.pow(2))
            #
            # Add regularization to loss
            #loss += self.l2_reg * l2_reg_loss

            wandb.log({
                "avg_sample_weight": weights.mean().item(),
                "max_sample_weight": weights.max().item(),
                "min_sample_weight": weights.min().item(),
                #"l2_reg_loss": l2_reg_loss.item()
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
            
            # Track head and tail item performance
            for i, item_id in enumerate(item_ids):
                item_id = item_id.item()
                if item_id in self.head_items:
                    head_preds.append(prediction[i].detach())
                    head_targets.append(rating[i].detach())
                elif item_id in self.tail_items:
                    tail_preds.append(prediction[i].detach())
                    tail_targets.append(rating[i].detach())

        # Calculate training metrics
        avg_train_loss = total_train_loss / len(self.train_dataloader)
        train_rmse = self.calculate_rmse(
            torch.tensor(all_train_preds),
            torch.tensor(all_train_targets)
        )
        
        # Calculate head and tail item metrics
        if head_preds:
            head_rmse = self.calculate_rmse(
                torch.tensor(head_preds),
                torch.tensor(head_targets)
            )
            wandb.log({"head_items_rmse": head_rmse.item()})
            print(f"Head items RMSE: {head_rmse.item():.4f}")
        
        if tail_preds:
            tail_rmse = self.calculate_rmse(
                torch.tensor(tail_preds),
                torch.tensor(tail_targets)
            )
            wandb.log({"tail_items_rmse": tail_rmse.item()})
            print(f"Tail items RMSE: {tail_rmse.item():.4f}")
        
        return avg_train_loss, train_rmse.item()
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model with separate metrics for head and tail items.
        
        Returns:
            Tuple of (average validation loss, validation RMSE)
        """
        self.model.eval()
        total_val_loss = 0
        all_val_preds = []
        all_val_targets = []
        
        # Track head and tail item performance
        head_preds = []
        head_targets = []
        tail_preds = []
        tail_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                rating = batch['rating']
                item_ids = batch['item_idx']
                prediction = self.model(batch)
                
                # Huber loss for validation
                loss = torch.mean(self.huber_loss(prediction, rating, self.huber_delta))

                total_val_loss += loss.item()
                all_val_preds.extend(prediction.cpu())
                all_val_targets.extend(rating.cpu())
                
                # Track head and tail item performance
                for i, item_id in enumerate(item_ids):
                    item_id = item_id.item()
                    if item_id in self.head_items:
                        head_preds.append(prediction[i].cpu())
                        head_targets.append(rating[i].cpu())
                    elif item_id in self.tail_items:
                        tail_preds.append(prediction[i].cpu())
                        tail_targets.append(rating[i].cpu())

        # Calculate validation metrics
        avg_val_loss = total_val_loss / len(self.val_dataloader)
        val_rmse = self.calculate_rmse(
            torch.tensor(all_val_preds),
            torch.tensor(all_val_targets)
        )
        
        # Calculate head and tail item metrics
        if head_preds:
            head_rmse = self.calculate_rmse(
                torch.tensor(head_preds),
                torch.tensor(head_targets)
            )
            wandb.log({"val_head_items_rmse": head_rmse.item()})
            print(f"Validation Head items RMSE: {head_rmse.item():.4f}")
        
        if tail_preds:
            tail_rmse = self.calculate_rmse(
                torch.tensor(tail_preds),
                torch.tensor(tail_targets)
            )
            wandb.log({"val_tail_items_rmse": tail_rmse.item()})
            print(f"Validation Tail items RMSE: {tail_rmse.item():.4f}")
        
        return avg_val_loss, val_rmse.item()