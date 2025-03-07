import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size=320, lr=0.001, num_epochs=10, weight_decay=1e-5,
                 device='cuda'):
        """
        Initializes the Trainer with model, training and validation datasets, and training parameters.

        Args:
            model (nn.Module): The matrix factorization model
            train_dataset (Dataset): The training dataset.
            val_dataset (Dataset): The validation dataset.
            batch_size (int): Batch size for training.
            lr (float): Learning rate for the optimizer.
            num_epochs (int): Number of epochs to train.
            device (str): Device to run the training on ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.criterion = torch.nn.MSELoss()
        # weight_decay is better then L2 with Adam
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.num_epochs = num_epochs
        self.device = device

    def train(self):
        """
        Trains the model and logs the loss to wandb.
        """
        wandb.init(project="matrix-factorization", config={
            "learning_rate": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs
        })
        for epoch in range(self.num_epochs):
            self.model.train()
            total_train_loss = 0
            for user_id, item_id, rating in tqdm(self.train_dataloader):
                user_id, item_id, rating = user_id.to(self.device), item_id.to(self.device), rating.to(self.device)
                self.optimizer.zero_grad()
                prediction = self.model(user_id, item_id)
                loss = self.criterion(prediction, rating)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(self.train_dataloader)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {avg_train_loss:.4f}")

            # Validation step
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for user_id, item_id, rating in tqdm(self.val_dataloader):
                    user_id, item_id, rating = user_id.to(self.device), item_id.to(self.device), rating.to(self.device)
                    prediction = self.model(user_id, item_id)
                    loss = self.criterion(prediction, rating)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(self.val_dataloader)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Validation Loss: {avg_val_loss:.4f}")

            # Log metrics to wandb
            wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        torch.save(self.model.state_dict(), "mf_model.pth")


if __name__ == "__main__":
    # train model
    from dotenv import load_dotenv

    load_dotenv()
    import pandas as pd
    from utils import regression_split_train_validation
    from dataset import AmazonDataset

    df = pd.read_csv("data/user_item_rating_train.csv")

    df_train, df_val = regression_split_train_validation(df)

    train_dataset = AmazonDataset(df_train)
    val_dataset = AmazonDataset(df_val)

    # validate that all items in val set are in train set
    val_items = set(df_val['parent_asin'].unique())
    train_items = set(df_train['parent_asin'].unique())
    print(val_items.issubset(train_items))
    print(train_dataset.get_num_items())
    print(train_dataset.get_num_users())
    print(val_dataset.get_num_items())
    print(val_dataset.get_num_users())
    from MF_model import MfModel

    model = MfModel(train_dataset.get_num_users(), train_dataset.get_num_items(), emb_dim=50)

    trainer = Trainer(model, train_dataset, val_dataset, batch_size=2048 * 2, lr=0.01, num_epochs=40,
                      weight_decay=1e-5)
    trainer.train()
