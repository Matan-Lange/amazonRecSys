import pandas as pd


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
