import torch
from torch import nn
import pandas as pd
import numpy as np
from utils import regression_split_train_validation
from MF.dataset import AmazonDataset


def calculate_baselines(df_train, df_val):
    # Global statistics
    global_mean = df_train.rating.mean()
    true_ratings = torch.tensor(df_val.rating.values)
    mse_loss_fn = nn.MSELoss()

    # 1. Global Mean Baseline
    global_pred = torch.full((len(df_val),), global_mean)
    global_mse = mse_loss_fn(global_pred, true_ratings)

    # 2. User Bias
    user_means = df_train.groupby('user_id').rating.mean()
    user_bias = user_means - global_mean
    user_pred = torch.tensor(df_val['user_id'].map(user_means).fillna(global_mean).values)
    user_mse = mse_loss_fn(user_pred, true_ratings)

    # 3. Item Bias
    item_means = df_train.groupby('parent_asin').rating.mean()
    item_bias = item_means - global_mean
    item_pred = torch.tensor(df_val['parent_asin'].map(item_means).fillna(global_mean).values)
    item_mse = mse_loss_fn(item_pred, true_ratings)

    # 4. User + Item Bias
    ui_predictions = []
    for _, row in df_val.iterrows():
        pred = global_mean
        pred += user_bias.get(row['user_id'], 0)
        pred += item_bias.get(row['parent_asin'], 0)
        ui_predictions.append(pred)
    ui_pred = torch.tensor(ui_predictions)
    ui_mse = mse_loss_fn(ui_pred, true_ratings)

    # Print results
    print(f"Global Mean RMSE: {torch.sqrt(global_mse).item():.4f}")
    print(f"User Bias RMSE: {torch.sqrt(user_mse).item():.4f}")
    print(f"Item Bias RMSE: {torch.sqrt(item_mse).item():.4f}")
    print(f"User+Item Bias RMSE: {torch.sqrt(ui_mse).item():.4f}")


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/user_item_rating_train.csv")
    df_train, df_val = regression_split_train_validation(df)

    # Calculate all baselines
    calculate_baselines(df_train, df_val)
