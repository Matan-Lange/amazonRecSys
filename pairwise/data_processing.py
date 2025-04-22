import pandas as pd
from tqdm import tqdm
import random


def warm_item_split_train_validation(df: pd.DataFrame):
    df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    last_interactions = df.groupby('user_id').tail(1)
    val_df = df.loc[last_interactions.index]
    train_df = df.drop(last_interactions.index)
    return train_df, val_df


def main(event_dataset, test_dataset, meta_data_df):
    # split data
    train_df, val_df = warm_item_split_train_validation(event_dataset)
    train_df.reset_index(inplace=True)
    val_df.reset_index(inplace=True)

    # get all item ids events in train
    parnet_asins = train_df['parent_asin'].tolist()
    # group for each users items he interacted with in train
    sets = train_df.groupby('user_id')['parent_asin'].agg(set).to_dict()

    # sample for val dataset items by popularity
    random_items_val = []

    rand_list = [random.randint(0, len(parnet_asins)) for _ in range(len(val_df))]

    for index, row in tqdm(val_df.iterrows()):
        user_id = row['user_id']

        negative_sample = parnet_asins[rand_list[index]]

        if negative_sample not in sets.get(user_id, []):
            random_items_val.append(negative_sample)
        else:
            while negative_sample in sets[user_id]:
                negative_sample = parnet_asins[random.randint(0, len(parnet_asins))]
            random_items_val.append(negative_sample)

    val_df['sampled_item'] = random_items_val

    # sample for val train items by popularity
    random_items_train = []

    rand_list = [random.randint(0, len(parnet_asins)) for _ in range(len(train_df))]

    for index, row in tqdm(train_df.iterrows()):

        user_id = row['user_id']

        negative_sample = parnet_asins[rand_list[index]]

        if negative_sample not in sets.get(user_id, []):
            random_items_train.append(negative_sample)
        else:
            while negative_sample in sets[user_id]:
                negative_sample = parnet_asins[random.randint(0, len(parnet_asins))]
            random_items_train.append(negative_sample)

    train_df['sampled_item'] = random_items_train

    # create mapping items to ids and users to ids for train and val test_dataset
    all_items = event_dataset['parent_asin'].unique().tolist()
    all_users = event_dataset['user_id'].unique().tolist()


    #featch parent_asin to idx from meta data
    item_map = meta_data_df.set_index('parent_asin')['item_idx'].to_dict()

    user_map = {user: idx for idx, user in enumerate(all_users)}

    train_df['positive_item_idx'] = train_df['parent_asin'].map(item_map)
    train_df['negative_item_idx'] = train_df['sampled_item'].map(item_map)
    train_df['user_idx'] = train_df['user_id'].map(user_map)

    val_df['positive_item_idx'] = val_df['parent_asin'].map(item_map)
    val_df['negative_item_idx'] = val_df['sampled_item'].map(item_map)
    val_df['user_idx'] = val_df['user_id'].map(user_map)

    test_dataset['item_0_idx'] = test_dataset['item_0'].map(item_map)
    test_dataset['item_1_idx'] = test_dataset['item_1'].map(item_map)
    test_dataset['user_idx'] = test_dataset['user_id'].map(user_map)

    # save the train and val datasets
    train_df.to_csv('train_pairwise.csv', index=False)
    val_df.to_csv('val_pairwise.csv', index=False)
    test_dataset.to_csv('test_pairwise.csv', index=False)


if __name__ == "__main__":
    # Load the datasets
    event_dataset = pd.read_csv('/tmp/pycharm_project_190/data/user_item_rating_table_train.csv')
    test_dataset = pd.read_csv('/tmp/pycharm_project_190/data/warm_items_classification_test_format.csv')
    meta_data_df = pd.read_parquet('/tmp/pycharm_project_190/preproc_data/metadata.parquet')
    main(event_dataset, test_dataset, meta_data_df)
