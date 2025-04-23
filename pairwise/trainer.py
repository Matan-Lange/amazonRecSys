"""
Contrastive trainer for Siamese-style recommendation models.
This is a modified version of the original BaseTrainer to handle contrastive training
for models that embed users and items separately, compute similarity scores,
and apply contrastive loss.
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
from typing import Dict, List, Tuple, Any


class ContrastiveTrainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_dataset: torch.utils.data.Dataset,
            val_dataset: torch.utils.data.Dataset,
            test_dataset: torch.utils.data.Dataset,
            config: Any
    ):
        self.model = model.to(config.device)
        self.train_dataloader = self._create_dataloader(train_dataset, config.batch_size, shuffle=False)
        self.val_dataloader = self._create_dataloader(val_dataset, config.batch_size, shuffle=False)
        self.test_dataloader = self._create_dataloader(test_dataset, config.batch_size, shuffle=False)
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=len(self.train_dataloader) * config.epochs,
            pct_start=getattr(config, 'warmup_pct', 0.05),
            div_factor=getattr(config, 'div_factor', 25),
            final_div_factor=getattr(config, 'final_div_factor', 1e4),
        )
        self.device = config.device
        self.epochs = config.epochs
        self.step = 0
        self.model_name = model.__class__.__name__
        self.best_model_path = f"best_model_{self.model_name.lower()}.pt"
        self.loss_type = getattr(config, 'loss_type', 'cross_entropy')  # 'cross_entropy' or 'bce'

    def _create_dataloader(self, dataset, batch_size, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, drop_last=True,
                          collate_fn=self.collate_fn)

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

    def process_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Process batch data before model forward pass insert to device.
        """
        return {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    def contrastive_loss(self, pos_score, neg_score):
        if self.loss_type == 'bce':
            diff = pos_score - neg_score
            return torch.nn.functional.binary_cross_entropy_with_logits(diff, torch.ones_like(diff))
        else:
            logits = torch.stack([pos_score, neg_score], dim=1)  # shape (B, 2)
            labels = torch.zeros(len(pos_score), dtype=torch.long, device=self.device)  # 0 = positive class
            return torch.nn.functional.cross_entropy(logits, labels)

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}"):
            total_correct = 0
            total_samples = 0

            positive_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            negative_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Reassign correct items
            positive_batch['item_idx'] = positive_batch['positive_item_idx']
            negative_batch['item_idx'] = negative_batch['negative_item_idx']

            # Move to device
            positive_batch = self.process_batch(positive_batch)
            negative_batch = self.process_batch(negative_batch)

            pos_score = self.model(positive_batch)  # shape (B,)
            neg_score = self.model(negative_batch)

            loss = self.contrastive_loss(pos_score, neg_score)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            correct = (pos_score > neg_score).sum()

            batch_accurcy = correct / len(pos_score)
            wandb.log({"train_loss": loss.item(), "step": self.step, 'train_accuracy': batch_accurcy})
            self.step += 1
            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss

    def validate(self):
        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                positive_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                negative_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Reassign correct items
                positive_batch['item_idx'] = positive_batch['positive_item_idx']
                negative_batch['item_idx'] = negative_batch['negative_item_idx']

                # Move to device
                positive_batch = self.process_batch(positive_batch)
                negative_batch = self.process_batch(negative_batch)
                pos_score = self.model(positive_batch)  # shape (B,)
                neg_score = self.model(negative_batch)

                correct = (pos_score > neg_score).sum()
                total_correct += correct
                total_samples += len(pos_score)

        acc = total_correct / total_samples
        wandb.log({"val_accuracy": acc})
        return acc

    def train(self):
        best_acc = 0.0
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val_acc = self.validate()

            wandb.log({"epoch": epoch + 1, "train_loss_avg": train_loss})

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), self.best_model_path)
                wandb.log({"best_val_accuracy": best_acc})

        print(f"Best validation accuracy: {best_acc:.4f}")
        return best_acc

    def test(self):

        # Load best model weights
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path))
            print(f"Loaded best model from {self.best_model_path}")

        self.model.eval()
        user_ids = []
        item_0_parent_asin = []
        item_1_parent_asin = []
        user_preferrd_item = []

        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Testing"):
                user_id = batch['user_id']
                item_0 = batch['item_0']
                item_1 = batch['item_1']

                item_0_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                item_1_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Reassign correct items
                item_0_batch['item_idx'] = item_0_batch['item_0_idx']
                item_1_batch['item_idx'] = item_1_batch['item_1_idx']

                # Move to device
                item_0_batch = self.process_batch(item_0_batch)
                item_1_batch = self.process_batch(item_1_batch)

                item_0_score = self.model(item_0_batch)
                item_1_score = self.model(item_1_batch)

                #comapre item_0 and item_1 scores
                item_class = torch.where(item_0_score > item_1_score, 1, 0)

                user_ids.extend(user_id)
                item_0_parent_asin.extend(item_0)
                item_1_parent_asin.extend(item_1)
                user_preferrd_item.extend(item_class.tolist())

        test_results = pd.DataFrame({
            'user_id': user_ids,
            'item_0': item_0_parent_asin,
            'item_1': item_1_parent_asin,
            'class': user_preferrd_item
        })

        test_results.to_csv('pairwise_warm_test_results.csv', index=False)
        # save in wandb artifact
        artifact = wandb.Artifact('pairwise_warm_test_results', type='dataset')
        artifact.add_file('pairwise_warm_test_results.csv')
        wandb.log_artifact(artifact)
        print("Test results saved to test_results.csv")
