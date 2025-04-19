"""
Common utility functions for recommendation systems.
"""

import pandas as pd
import json
import numpy as np
import os
from typing import Tuple, Dict, Any, Optional


def regression_split_train_validation(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the input DataFrame into training and validation sets.
    The validation set contains the last rating of each user based on the timestamp.

    Parameters:
        df (pd.DataFrame): Input DataFrame with columns ['user_id', 'parent_asin', 'rating', 'timestamp'].

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training set DataFrame and validation set DataFrame.
    """
    # Sort the DataFrame by user and timestamp
    df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)

    # Get the last interaction for each user
    last_interactions = df.groupby('user_id').tail(1)

    # Create the validation set
    val_df = df.loc[last_interactions.index]

    # Create the training set by dropping the validation set rows
    train_df = df.drop(last_interactions.index)

    return train_df, val_df


def load_metadata(file_path: str) -> pd.DataFrame:
    """
    Loads metadata from a JSON file and converts it into a pandas DataFrame.

    Parameters:
        file_path (str): The path to the JSON file containing the metadata.

    Returns:
        pd.DataFrame: A DataFrame containing the metadata.
    """
    with open(file_path, "r") as file:
        data = [json.loads(line) for line in file]

    df = pd.DataFrame.from_records(data)
    return df


def preprocess_metadata(df_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the metadata DataFrame by mapping categorical values to unique IDs and binning prices.

    Parameters:
        df_metadata (pd.DataFrame): The input metadata DataFrame with columns 
                                   ['parent_asin', 'categories', 'store', 'price'].

    Returns:
        pd.DataFrame: The preprocessed metadata DataFrame with additional columns for category, store, and price indices.
    """
    df_metadata = df_metadata[['parent_asin', 'categories', 'store', 'price']].copy()

    # map category to unique id
    df_metadata['categories'] = df_metadata['categories'].astype(str)
    categories_map = {cat: idx for idx, cat in enumerate(df_metadata['categories'].unique().tolist())}
    df_metadata['categories_idx'] = df_metadata['categories'].map(categories_map)

    # map store to unique id
    store_map = {store: idx for idx, store in enumerate(df_metadata['store'].unique().tolist())}
    df_metadata['store_idx'] = df_metadata['store'].map(store_map)

    # convert price to float, or to missing value if not a number
    df_metadata['price'] = pd.to_numeric(df_metadata['price'], errors='coerce')
    # bin prices into 100 bins, if null give max price_idx + 1
    df_metadata['price_bin'] = pd.qcut(df_metadata['price'], 100, labels=False, duplicates='drop')
    df_metadata['price_bin'] = df_metadata['price_bin'].astype(str)
    price_map = {bin: idx for idx, bin in enumerate(df_metadata['price_bin'].unique().tolist())}
    # add missing value to, map as len(map)
    price_map[np.nan] = len(price_map)
    df_metadata['price_idx'] = df_metadata['price_bin'].map(price_map)

    return df_metadata


def get_data_paths() -> Dict[str, str]:
    """
    Get paths to data files from environment variables.

    Returns:
        Dict[str, str]: Dictionary containing paths to data files.
    """
    return {
        'train_path': os.getenv('TRAIN_PATH'),
        'test_path': os.getenv('TEST_PATH'),
        'metadata_path': os.getenv('METADATA_PATH'),
        'images_dir': os.getenv('IMAGES_DIR'),
        'images_df_path': os.getenv('IMAGES_DF_PATH'),
        'fashion_clip_embeddings': os.getenv('FASHION_CLIP_EMBEDDINGS'),
    }


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Parameters:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Dictionary containing configuration parameters.
    """
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def calculate_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error (RMSE) between predictions and targets.

    Parameters:
        predictions (np.ndarray): Predicted values.
        targets (np.ndarray): Target values.

    Returns:
        float: RMSE value.
    """
    return np.sqrt(np.mean((predictions - targets) ** 2))


__all__ = [
    'regression_split_train_validation',
    'load_metadata',
    'preprocess_metadata',
    'get_data_paths',
    'load_config_from_yaml',
    'calculate_rmse',
]