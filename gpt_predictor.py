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

class Classification(BaseModel):
    user_item_rating: int = Field(..., description="Rating of the item based on user past interaction (between 0 and 1)")

class GPTPredictor:

    def __init__(self, system_prompt: str, classification: bool):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.system_prompt = system_prompt
        self.classification = classification
        if self.classification == False:
            self.schema = Rating
        else:
            self.schema = Classification

    async def predict_by_gpt(self, user_history: str, new_item_desc:str) -> str:
        """
        predict user rating given the user history and new item description
        """
        if self.classification == False:
            prompt = f"""
            The user has rated the following items in the past:
            {user_history}

            Now, the user is about to rate this new item:
            {new_item_desc}

            Based on the user's past preferences, predict the rating (from 1 to 5). Just return the number.
            """
        else:
            prompt = f"""
            The user has interacted with the following items:
            {user_history}

            Here are two new items:
            {new_item_desc}

            Which item is the user more likely to interact with? Answer with '0' or '1'.
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

        tasks = [self.predict_by_gpt(pred_instance[0], pred_instance[1]) for pred_instance in pred_instances]
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

# # Initialize predictor
# system_prompt = "You are a helpful assistant that predicts ratings based on user history."
# predictor = GPTRatingPredictor(system_prompt=system_prompt)
#
# # Example input: list of [user_history, new_item_description]
# test_data = [
#     ["Jacket (5), Sneakers (4), Hat (3)", "stylish hoodie"],
#     ["Phone (2), Laptop (4), Tablet (3)", "high-end gaming laptop."],
# ]
#
# # Run the prediction
# output = predictor.extract_batch(test_data*5)
#
# # Print results
# print(output)


