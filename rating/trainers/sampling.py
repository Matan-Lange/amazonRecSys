from torch.utils.data import DataLoader
from rating.trainers.base import BaseTrainer
from rating.datasets.sampler import UserUniformBatchSampler
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import wandb


# class SamplingTrainer(BaseTrainer):
#     def __init__(self, model, train_dataset, val_dataset, config):
#         super().__init__(model, train_dataset, val_dataset, config)
#         self.batch_size = getattr(config, 'batch_size', 256)
#         self.k = getattr(config, 'k', 1)
#         self.drop_last = getattr(config, 'drop_last', False)
#
#         sampler = UserUniformBatchSampler(
#             user2idx=train_dataset.user_to_row_indices(),
#             batch_users=self.batch_size,
#             k=self.k,
#             shuffle=True,
#             drop_last=self.drop_last
#         )
#
#         self.train_dataloader = DataLoader(
#             train_dataset,
#             batch_sampler=sampler,
#             collate_fn=self.collate_fn,
#             num_workers=getattr(config, 'num_workers', 5),
#             pin_memory=True
#         )


class SamplingTrainer(BaseTrainer):
    def __init__(self, model, train_dataset, val_dataset, config):
        super().__init__(model, train_dataset, val_dataset, config)
        self.batch_size = getattr(config, 'batch_size', 256)
        self.k = getattr(config, 'k', 1)
        self.drop_last = getattr(config, 'drop_last', False)

        # Compute user and item frequencies
        print("Computing user and item frequencies...")
        self.user_freq, self.item_freq = self._compute_frequencies(train_dataset)
        user_freq_arr = np.array([self.user_freq.get(i, 1) for i in range(model.num_users)])
        item_freq_arr = np.array([self.item_freq.get(i, 1) for i in range(model.num_items)])

        self.user_freq_tensor = torch.tensor(user_freq_arr, dtype=torch.float32, device=config.device)
        self.item_freq_tensor = torch.tensor(item_freq_arr, dtype=torch.float32, device=config.device)

        # Adaptive L2 regularization parameters
        self.lambda0 = getattr(config, 'lambda0', 1)  # base strength
        self.eps = getattr(config, 'eps', 1.0)  # avoids 1/0

        sampler = UserUniformBatchSampler(
            user2idx=train_dataset.user_to_row_indices(),
            batch_users=self.batch_size,
            k=self.k,
            shuffle=True,
            drop_last=self.drop_last
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=getattr(config, 'num_workers', 5),
            pin_memory=True
        )

    def _compute_frequencies(self, dataset):
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

    def train_epoch(self, epoch: int):
        """
        Train for one epoch with adaptive L2 regularization.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average training loss, training RMSE)
        """
        self.model.train()
        total_train_loss = 0
        all_train_preds = []
        all_train_targets = []

        for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"):
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            user = batch['user_idx']
            item = batch['item_idx']
            rating = batch['rating']

            # ---------- forward ----------
            self.optimizer.zero_grad()
            pred = self.model(batch)
            huber = F.mse_loss(pred, rating)  # F.huber_loss(pred, rating, delta=1., reduction='mean')

            # ---------- adaptive L2 ----------
            # 1. unique IDs so we penalize each embedding once per batch
            uniq_u = torch.unique(user)
            uniq_i = torch.unique(item)

            l2_u = (self.lambda0 / torch.sqrt(self.user_freq_tensor[uniq_u] + self.eps)) \
                   * self.model.user_emb(uniq_u).pow(2).sum(dim=1)  # [|U|]
            l2_i = (self.lambda0 / torch.sqrt(self.item_freq_tensor[uniq_i] + self.eps)) \
                   * self.model.item_emb(uniq_i).pow(2).sum(dim=1)  # [|I|]

            denom = uniq_u.numel() + uniq_i.numel()
            adaptive_penalty = (l2_u.sum() + l2_i.sum()) / denom

            if self.step % 500 == 0:
                print(f"step {self.step:>5} | huber={huber.item():.4e} | penalty={adaptive_penalty.item():.4e}")
            wandb.log({"penalty_ratio": adaptive_penalty.item() / huber.item()})
            loss = huber + adaptive_penalty

            # Log metrics
            wandb.log({
                "train_step_loss": loss.item(),
                "huber_loss": huber.item(),
                "adaptive_penalty": adaptive_penalty.item(),
                "learning_rate": self.scheduler.get_last_lr()[0],
                "step": self.step
            })

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1

            total_train_loss += loss.item()
            all_train_preds.extend(pred.detach())
            all_train_targets.extend(rating.detach())

        # Calculate training metrics
        avg_train_loss = total_train_loss / len(self.train_dataloader)
        train_rmse = self.calculate_rmse(
            torch.tensor(all_train_preds),
            torch.tensor(all_train_targets)
        )

        return avg_train_loss, train_rmse.item()
