import os
import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer
from rating_refactor.proc.text_embeddings import create_text_embeddings
from rating_refactor.proc.image_embeddings import create_image_embeddings

from dotenv import load_dotenv

load_dotenv()


def load_metadata(metadata_path) -> pd.DataFrame:
    """Load metadata."""
    with open(metadata_path, "r") as file:
        data = [json.loads(line) for line in file]
    df = pd.DataFrame.from_records(data)
    return df


def category_one_hot_vector(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the categories column in the metadata DataFrame to a one-hot encoded vector.
    """
    # Create a copy to avoid modifying the original dataframe
    df = metadata_df.copy()
    mlb = MultiLabelBinarizer(sparse_output=False)
    embeddings = mlb.fit_transform(df['categories'])
    df['category_vector'] = list(embeddings)  # each item is a numpy.ndarray
    return df[['parent_asin', 'category_vector']]


def preprocess_data():
    image_dir = os.getenv("IMAGES_DIR")
    image_df_path = os.getenv("IMAGES_DF_PATH")
    meta_data_path = os.getenv("METADATA_PATH")

    warm_items_train_path = os.getenv("TRAIN_PATH")
    warm_items_test_path = os.getenv("TEST_PATH")
    cold_items_test_path = os.getenv("COLD_TEST_PATH")

    # Load dataframes
    metadata_df = load_metadata(meta_data_path)

    print(metadata_df['categories'].isna().sum())
    warm_items_df = pd.read_csv(warm_items_train_path)
    warm_items_test_df = pd.read_csv(warm_items_test_path)
    cold_items_test_df = pd.read_csv(cold_items_test_path)

    # Preprocess metadata, use
    # create_text_embeddings(meta_data_path)
    # create_image_embeddings(image_dir, image_df_path)

    text_emb_df = pd.read_parquet('text_embeddings.parquet')
    image_emb_df = pd.read_parquet('image_embeddings.parquet')

    # map items, user, store to ids
    unique_users = set(warm_items_df['user_id']).union(set(warm_items_test_df['user_id'])).union(
        set(cold_items_test_df['user_id']))
    unique_items = set(warm_items_df['parent_asin']).union(set(warm_items_test_df['parent_asin'])).union(
        set(cold_items_test_df['parent_asin']))

    unique_stores = set(metadata_df['store'])

    user_map = {user: i for i, user in enumerate(unique_users)}
    item_map = {item: i for i, item in enumerate(unique_items)}
    store_map = {store: i for i, store in enumerate(unique_stores)}

    # map user, item, store to ids
    warm_items_df['user_idx'] = warm_items_df['user_id'].map(user_map)
    warm_items_df['item_idx'] = warm_items_df['parent_asin'].map(item_map)

    warm_items_test_df['user_idx'] = warm_items_test_df['user_id'].map(user_map)
    warm_items_test_df['item_idx'] = warm_items_test_df['parent_asin'].map(item_map)

    cold_items_test_df['user_idx'] = cold_items_test_df['user_id'].map(user_map)
    cold_items_test_df['item_idx'] = cold_items_test_df['parent_asin'].map(item_map)

    metadata_df['store_idx'] = metadata_df['store'].map(store_map)
    metadata_df['item_idx'] = metadata_df['parent_asin'].map(item_map)

    # metadata build
    metadata_df = metadata_df.merge(text_emb_df, on='parent_asin', how='left')
    metadata_df = metadata_df.merge(image_emb_df, on='parent_asin', how='left')
    one_hot_category = category_one_hot_vector(metadata_df)
    metadata_df = metadata_df.merge(one_hot_category, on='parent_asin', how='left')


    # take relevant columns of metadata
    metadata_df = metadata_df[
        ['item_idx', 'parent_asin', 'store_idx', 'text_embeddings', 'dino_embedding', 'fashion_clip_embedding',
         'category_vector']
    ]

    metadata_df.to_parquet('metadata.parquet', index=False)

    # # merge each event with metadata and save
    warm_items_df.to_parquet('warm_items.parquet', index=False)
    del warm_items_df
    print('saved warm_items.parquet')
    warm_items_test_df.to_parquet('warm_items_test.parquet', index=False)
    del warm_items_test_df
    print('saved warm_items_test.parquet')
    cold_items_test_df.to_parquet('cold_items_test.parquet', index=False)
    del cold_items_test_df
    print('saved cold_items_test.parquet')





if __name__ == "__main__":
    preprocess_data()
