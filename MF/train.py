import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        """
        Initialize trainer with wandb config
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
            num_workers=5
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=5
        )
        self.criterion = torch.nn.MSELoss()
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


        # Setup OneCycleLR scheduler with warmup
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            total_steps=len(self.train_dataloader) * config.epochs,
            pct_start=0.1,  # 10% of training for warmup
            div_factor=25,  # initial_lr = max_lr/25
            final_div_factor=1e4,  # final_lr = initial_lr/1e4
        )

        self.num_epochs = config.epochs
        self.device = config.device
        self.step = 0

    def collate_fn(self, batch):
        """Generic collate function to handle dictionary batch data"""
        batch_dict = {}
        for key in batch[0]:
            batch_dict[key] = torch.stack([item[key] for item in batch])
        return batch_dict

    def calculate_rmse(self, predictions, targets):
        """Calculate RMSE between predictions and targets"""
        return torch.sqrt(torch.mean((predictions - targets) ** 2))

    def train(self):
        """Train model with wandb sweep"""
        for epoch in range(self.num_epochs):
            self.model.train()
            total_train_loss = 0
            all_train_preds = []
            all_train_targets = []

            for batch in tqdm(self.train_dataloader):
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

            # Validation step
            self.model.eval()
            total_val_loss = 0
            all_val_preds = []
            all_val_targets = []

            with torch.no_grad():
                for batch in tqdm(self.val_dataloader):
                    batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    rating = batch['rating']
                    prediction = self.model(batch)
                    loss = self.criterion(prediction, rating)

                    total_val_loss += loss.item()
                    all_val_preds.extend(prediction.cpu())
                    all_val_targets.extend(rating.cpu())

            # Calculate validation metrics
            avg_val_loss = total_val_loss / len(self.val_dataloader)
            val_rmse = self.calculate_rmse(
                torch.tensor(all_val_preds),
                torch.tensor(all_val_targets)
            )

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_rmse": train_rmse,
                "val_loss": avg_val_loss,
                "val_rmse": val_rmse
            })

        return val_rmse
