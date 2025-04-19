"""
NCF trainer for recommendation models.

This module contains the NCF trainer class for training Neural Collaborative Filtering models:
- NCFTrainer: Trainer specifically for NCF models with prediction clamping
"""

import torch
from tqdm import tqdm
import wandb
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

from rating.trainers.base import BaseTrainer
from rating.models.base import BaseRecommenderModel


class NCFTrainer(BaseTrainer):
    """Trainer specifically for NCF models with prediction clamping."""
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model with prediction clamping.
        
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
                
                # Clamp predictions to valid rating range
                prediction = torch.clamp(prediction, 1.0, 5.0)
                
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