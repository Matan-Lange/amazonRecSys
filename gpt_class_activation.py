# Initialize predictor
import pandas as pd
import numpy as np
import utils_for_gpt as utils_gpt
import gpt_predictor as gptp

data_preprocessor =  utils_gpt.DataPreprocessor(metadata_path= "items_metadata.jsonl",
                                                train_set_path= "train_data_regression.csv",
                                                valid_set_path= "cold_items_classification_val_1k.csv")

metadata_for_gpt_processed = data_preprocessor.process_metadata()
train_df, user_itemslist_df = data_preprocessor.process_train_data()
val_df = data_preprocessor.process_valid_data()

system_prompt_reg = "You are a helpful assistant that predicts ratings based on user history."
system_prompt_class = "you are helpful assistant that predicts user future interaction with products based on user history."
predictor = gptp.GPTPredictor(system_prompt=system_prompt_reg, classification=True)
val_res_lst = []

size = len(val_df)
batch_size = 5

for i in range((size + batch_size - 1) // batch_size):  # Ensures the last batch is included
    batch_prompt = []
    for j in range(i * batch_size, min(i * batch_size + batch_size, size)):
        single_prompt = []
        user_id = val_df.iloc[j]['user_id']
        user_history = ""
        for item in user_itemslist_df.loc[user_id].tolist()[0]:
            user_history += '*'
            user_history += metadata_for_gpt_processed.loc[item].tolist()[0]
        single_prompt.append(user_history)
        item_0 = val_df.iloc[j]['item_0']
        item_0_meta = metadata_for_gpt_processed.loc[item_0].tolist()[0]
        item_1 = val_df.iloc[j]['item_1']
        item_1_meta = metadata_for_gpt_processed.loc[item_1].tolist()[0]
        items_meta = f"Item 0: {item_0_meta} Item 1: {item_1_meta}"

        single_prompt.append(items_meta)
        batch_prompt.append(single_prompt)
        # Run the prediction
    output = predictor.extract_batch(batch_prompt)
    for k in range(len(output)):
        val_res_lst.append(output[k]['user_item_rating'])

# Print results
print(val_res_lst)
val_df['gpt'] = val_res_lst
val_df.to_csv("gpt_class_1k_output.csv", index=False)