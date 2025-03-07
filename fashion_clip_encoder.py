# pip install fashion-clip

from fashion_clip.fashion_clip import FashionCLIP
import pandas as pd
import numpy as np
import json


def preprocess_product(product):
    title = product['title']
    store = product['store'] if product['store'] else "N/A"
    price = product['price'] if product['price'] else "Bid"
    categories = " > ".join(product['categories'])

    # Format for embedding
    return f"Title: {title}\nPrice: {price}\nStore: {store}\nCategories: {categories}"


texts = []
text_ids = []
with open("filtered_metadata.jsonl", "r") as infile:
    for line in infile:
        product = json.loads(line)
        texts.append(preprocess_product(product))
        text_ids.append(product['parent_asin'])


images_urls_df = pd.read_csv('images_urls.csv')
images_urls_df = images_urls_df[~images_urls_df['parent_asin'].isin(['B07KPNPB5M', 'B0B69RBC1M', 'B07N47FGFC', 'B0728BNRD9', 'B0BM98VVC9'])] #this parent asins arnt in the images folder

base_path = "C:/Users/guest_temp/PycharmProjects/item_image_encoder/items_images/" # images folder path

image_ids = []
images = []
for name, ext in zip(images_urls_df['parent_asin'], images_urls_df['image_format']):
    image_ids.append(name)
    images.append(f"{base_path}{name}.{ext}")

fclip = FashionCLIP('fashion-clip')

# we create image embeddings and text embeddings
image_embeddings = fclip.encode_images(images, batch_size=32)
text_embeddings = fclip.encode_text(texts, batch_size=32)

# we normalize the embeddings to unit norm (so that we can use dot product instead of cosine similarity to do comparisons)
image_embeddings = image_embeddings/np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
text_embeddings = text_embeddings/np.linalg.norm(text_embeddings, ord=2, axis=-1, keepdims=True)

# # Storing image embeddings in a dictionary
data = {
    image_ids[i]: image_embeddings[i].tolist() for i in range(len(image_ids))  # Convert NumPy arrays to lists
}

# Save to JSON file
with open("fashion_embeddings_images.json", "w") as f:
    json.dump(data, f)

# Storing text embeddings in a dictionary
data2 = {
    text_ids[i]: text_embeddings[i].tolist() for i in range(len(text_ids))  # Convert NumPy arrays to lists
}

# Save to JSON file
with open("fashion_embeddings_text.json", "w") as f:
    json.dump(data2, f)