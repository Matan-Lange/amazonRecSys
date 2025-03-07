import torch
from torch import nn
import pandas as pd
from itertools import combinations
from utils import regression_split_train_validation, load_metadata, preprocess_metadata


def calculate_bias_combinations(df_train, df_val, df_metadata):
    """
    Calculate RMSE for all possible combinations of biases.
    """
    #statistics
    global_mean = df_train.rating.mean()
    true_ratings = torch.tensor(df_val.rating.values)
    mse_loss_fn = nn.MSELoss()

    #individual biases
    user_means = df_train.groupby('user_id').rating.mean()
    user_bias = user_means - global_mean

    item_means = df_train.groupby('parent_asin').rating.mean()
    item_bias = item_means - global_mean

    df_train_meta = df_train.merge(df_metadata, on='parent_asin', how='left')

    category_means = df_train_meta.groupby('categories_idx').rating.mean()
    category_bias = category_means - global_mean

    store_means = df_train_meta.groupby('store_idx').rating.mean()
    store_bias = store_means - global_mean

    price_means = df_train_meta.groupby('price_idx').rating.mean()
    price_bias = price_means - global_mean


    bias_components = {
        'user': (user_bias, 'user_id'),
        'item': (item_bias, 'parent_asin'),
        'category': (category_bias, 'categories_idx'),
        'store': (store_bias, 'store_idx'),
        'price': (price_bias, 'price_idx')
    }

    results = {}
    df_val_meta = df_val.merge(df_metadata, on='parent_asin', how='left')

    #all possible combinations
    for r in range(1, len(bias_components) + 1):
        for combination in combinations(bias_components.keys(), r):
            predictions = []
            combo_name = '+'.join(combination)

            for _, row in df_val_meta.iterrows():
                pred = global_mean
                for bias_name in combination:
                    bias_values, col_name = bias_components[bias_name]
                    pred += bias_values.get(row.get(col_name), 0)
                pred = max(1, min(5, pred))
                predictions.append(pred)

            pred_tensor = torch.tensor(predictions)
            rmse = torch.sqrt(mse_loss_fn(pred_tensor, true_ratings)).item()
            results[combo_name] = rmse
            print(f"{combo_name} RMSE: {rmse:.4f}")

    # Find best combination
    best_combo = min(results.items(), key=lambda x: x[1])
    print(f"\nBest combination: {best_combo[0]} (RMSE: {best_combo[1]:.4f})")

    return results


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/user_item_rating_train.csv")
    df_train, df_val = regression_split_train_validation(df)

    # Load and preprocess metadata
    df_metadata = load_metadata(r"C:\Users\User\PycharmProjects\amazonRecSys\data\items_metadata.jsonl")
    df_metadata = preprocess_metadata(df_metadata)

    # Calculate all bias combinations
    results = calculate_bias_combinations(df_train, df_val, df_metadata)