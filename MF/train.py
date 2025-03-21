import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        """
        Initialize trainer with wandb config
        """
        self.model = model.to(config.device)
        self.batch_size = config.batch_size
        self.lr = config.learning_rate
        self.weight_decay = config.weight_decay
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.num_epochs = config.epochs
        self.device = config.device
        self.step = 0

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
                user_id, item_id, category_id, store_id, rating = [x.to(self.device) for x in batch]

                self.optimizer.zero_grad()
                prediction = self.model(user_id, item_id, category_id, store_id)
                loss = self.criterion(prediction, rating)
                loss.backward()
                self.optimizer.step()

                step = self.step + 1
                wandb.log({"train_step_loss": loss.item(), "step": step})

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
                    user_id, item_id, category_id, store_id, rating = [x.to(self.device) for x in batch]
                    prediction = self.model(user_id, item_id, category_id, store_id)
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