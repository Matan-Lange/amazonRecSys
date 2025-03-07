# # import openai
# # import pandas as pd
# # import numpy as np
# # import json
# #
# # from sqlalchemy.testing.suite.test_reflection import metadata
# #
# # # Set your API key
# #
# # def preprocess_product(product):
# #     title = product['title']
# #     store = product['store'] if product['store'] else "N/A"
# #     price = product['price'] if product['price'] else "Bid"
# #     categories = " > ".join(product['categories'])
# #
# #     # Format for prompt
# #     return f"Title: {title}\nPrice: {price}\nStore: {store}\nCategories: {categories}"
# #
# # metadata_to_prompt_dict = {}
# # texts = []
# # text_ids = []
# # with open("filtered_metadata.jsonl", "r") as infile:
# #     for line in infile:
# #         product = json.loads(line)
# #         texts.append(preprocess_product(product))
# #         text_ids.append(product['parent_asin'])
# #         metadata_to_prompt_dict[product['parent_asin']] = preprocess_product(product)
# #
# #
# # user_item_rating = pd.read_csv()
# #
# #
# # warm_items_validation =
# # cold_items_validation =
# #
# #
# # def predict_rating(user_history, new_item):
# #     prompt = f"""
# #     The user has rated the following items in the past:
# #     {user_history}
# #
# #     Now, the user is about to rate this new item:
# #     {new_item}
# #
# #     Based on the user's past preferences, predict the rating (from 1 to 5). Just return the number.
# #     """
# #
# #     response = openai.chat.completions.create(
# #         model="gpt-4o",
# #         messages=[{"role": "system", "content": "You are a recommendation assistant."},
# #                   {"role": "user", "content": prompt}],
# #         max_tokens=100
# #     )
# #
# #     return response.choices[0].message.content
# #
# # # Example usage
# # user_history = "Brand: Nike, Category: Running Shoes, Color: Black, Price Range: $80-$120"
# # item0 = "Brand: Adidas, Category: Running Shoes, Color: White, Price: $100"
# # item1 = "Brand: Puma, Category: Casual Sneakers, Color: Red, Price: $90"
# #
# # prediction = predict_rating(user_history, item0, item1)
# # print(prediction)
# #
import asyncio
import json
import os
import time
from typing import List
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

class Rating(BaseModel):
    user_item_rating: int = Field(..., description="Rating of the item based on user past interaction (between 0 and 5)")

class GPTRatingPredictor:

    def __init__(self, system_prompt: str):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.system_prompt = system_prompt
        self.schema = Rating

    async def predict_user_rating(self, user_history: str, new_item_desc:str) -> str:
        """
        predict user rating given the user history and new item description
        """
        prompt = f"""
        The user has rated the following items in the past:
        {user_history}

        Now, the user is about to rate this new item:
        {new_item_desc}

        Based on the user's past preferences, predict the rating (from 1 to 5). Just return the number.
        """
        completion = await asyncio.to_thread(
            self.client.beta.chat.completions.parse,
            model=os.getenv('OPENAI_GPT40_API_VERSION'),
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format=self.schema
        )
        return completion.choices[0].message.content

    async def predict_batch_async(self, pred_instances: List[List[str]], limit: int = 10) -> List[str]:
        """
        Asynchronously predicts ratings of list of users.
        Raises an error if the number of users exceeds the limit.
        """
        if len(pred_instances) > limit:
            raise ValueError(f"Number of clinical notes exceeds the limit of {limit}")

        tasks = [self.predict_user_rating(pred_instance[0], pred_instance[1]) for pred_instance in pred_instances]
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        return results

    def extract_batch(self, pred_instances: List[List[str]], limit: int = 10) -> List[dict]:
        """
        wrapper for extracting patient information from a list of clinical notes.
        """
        results = asyncio.run(self.predict_batch_async(pred_instances, limit))
        results = [json.loads(result) for result in results]
        return results

# Initialize predictor
system_prompt = "You are a helpful assistant that predicts ratings based on user history."
predictor = GPTRatingPredictor(system_prompt=system_prompt)

# Example input: list of [user_history, new_item_description]
test_data = [
    ["Jacket (5), Sneakers (4), Hat (3)", "stylish hoodie"],
    ["Phone (2), Laptop (4), Tablet (3)", "high-end gaming laptop."],
]

# Run the prediction
output = predictor.extract_batch(test_data*5)

# Print results
print(output)


