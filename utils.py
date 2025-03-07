import pandas as pd
import json
import numpy as np


def regression_split_train_validation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the input DataFrame into training and validation sets.
    The validation set contains the last rating of each user based on the timestamp.

    Parameters:
    df (pd.DataFrame): Input DataFrame with columns ['user', 'item', 'rating', 'timestamp'].

    Returns:
    train_df (pd.DataFrame): Training set DataFrame.
    val_df (pd.DataFrame): Validation set DataFrame.
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


def load_metadata(file_path):
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


def preprocess_metadata(df_metadata: pd.DataFrame):
    """
    Preprocesses the metadata DataFrame by mapping categorical values to unique IDs and binning prices.

    Parameters:
    df_metadata (pd.DataFrame): The input metadata DataFrame with columns ['parent_asin', 'categories', 'store', 'price'].

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
