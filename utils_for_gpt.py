import utils
import pandas as pd
import numpy as np
import gpt_predictor as gptp


class DataPreprocessor:
    def __init__(self, metadata_path:str, train_set_path:str, valid_set_path:str, history_limit:int=20):
        self.metadata_path = metadata_path
        self.train_set_path = train_set_path
        self.valid_set_path = valid_set_path
        self.history_limit = history_limit

    def preprocess_product(self, product):
        title = product['title']
        if len(title) > 200:
            title = title[:200].rsplit(' ', 1)[0] + "..." if ' ' in title[:200] else title[:200] + "..."
        store = product['store'] if product['store'] else "N/A"
        price = product['price'] if product['price'] else "Bid"
        categories = " > ".join(product['categories'])

        return f"Title: {title}\nPrice: {price}\nStore: {store}\nCategories: {categories}"

    def process_metadata(self):
        metadata_raw_df = utils.load_metadata(self.metadata_path)
        metadata_for_gpt = metadata_raw_df[["title", "price", "store", "categories", "parent_asin"]]
        metadata_for_gpt_processed = metadata_for_gpt.copy()
        metadata_for_gpt_processed.loc[:, 'processed_info'] = metadata_for_gpt_processed.apply(self.preprocess_product, axis=1)
        metadata_for_gpt_processed = metadata_for_gpt_processed[['parent_asin', 'processed_info']]
        metadata_for_gpt_processed.set_index('parent_asin', inplace=True)
        return metadata_for_gpt_processed

    def process_train_data(self):
        train_df = pd.read_csv(self.train_set_path)
        # Sort by timestamp so the most recent interactions are last
        train_df = train_df.sort_values(by=['user_id', 'timestamp'], ascending=[True, False])
        # Keep only the last 20 interactions per user
        user_itemslist_df = train_df.groupby('user_id').agg({'parent_asin': lambda x: list(x[:self.history_limit])})
        user_itemslist_df = user_itemslist_df.rename(columns={'parent_asin': 'items_list'})
        train_df.set_index(['user_id', 'parent_asin'], inplace=True)
        return train_df, user_itemslist_df

    def process_valid_data(self):
        valid_df = pd.read_csv(self.valid_set_path)
        return valid_df
