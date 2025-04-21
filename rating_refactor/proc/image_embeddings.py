from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import pandas as pd
import os
from tqdm import tqdm


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir: str, image_df_path: str) -> None:
        self.image_dir = image_dir
        self.image_df = pd.read_csv(image_df_path)
        self.image_df['image_path'] = self.image_df.apply(
            lambda x: os.path.join(image_dir, f"{x['parent_asin']}.{x['image_format']}"), axis=1
        )

    def __len__(self):
        return len(self.image_df)

    def __getitem__(self, idx: int) -> dict:
        row = self.image_df.iloc[idx]
        image_path = row['image_path']

        if not os.path.isfile(image_path):
            print(f"Image file not found: {image_path}")
            # dummy image if image not found
            image = Image.new('RGB', (224, 224), color='black')
        else:
            image = Image.open(image_path).convert('RGB')

        return {
            'image': image,
            'parent_asin': row['parent_asin']
        }

    @staticmethod
    def collate_fn(batch):
        images = [item['image'] for item in batch]
        parent_asin = [item['parent_asin'] for item in batch]

        return {
            'image': images,
            'parent_asin': parent_asin
        }


class ImageEmbedding(ABC):
    @abstractmethod
    def embed(self, images: list[Image.Image]) -> torch.tensor:
        pass

    @abstractmethod
    def get_latent_dim(self) -> int:
        pass


class DinoV2Embedding(ImageEmbedding):
    def __init__(self, model_name: str = 'facebook/dinov2-with-registers-base') -> None:
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def embed(self, images: list[Image.Image]) -> torch.tensor:
        inputs = self.processor(images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu()

    def get_latent_dim(self) -> int:
        return self.model.config.hidden_size


class FashionClipImageEmbedding(ImageEmbedding):
    def __init__(self, model_name: str = 'patrickjohncyh/fashion-clip') -> None:
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def embed(self, images: list[Image.Image]) -> torch.tensor:
        inputs = self.processor(images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
        return embeddings.cpu()

    def get_latent_dim(self) -> int:
        return self.model.config.projection_dim


def create_image_embeddings(image_dir: str, image_df_path: str) -> None:
    dataset = ImageDataset(image_dir, image_df_path)
    dataloader = DataLoader(dataset,
                            batch_size=1024,
                            shuffle=False,
                            num_workers=20,
                            collate_fn=ImageDataset.collate_fn)

    dino = DinoV2Embedding()
    fashion_clip = FashionClipImageEmbedding()

    dino_embeddings = []
    fashion_clip_embeddings = []
    parent_asins = []
    for batch in tqdm(dataloader):
        images = batch['image']
        parent_asin = batch['parent_asin']

        dino_embedding = dino.embed(images)
        fashion_clip_embedding = fashion_clip.embed(images)

        for i in range(len(parent_asin)):
            dino_embeddings.append(dino_embedding[i])
            fashion_clip_embeddings.append(fashion_clip_embedding[i])
            parent_asins.append(parent_asin[i])

    df = pd.DataFrame({
        'parent_asin': parent_asins,
        'dino_embedding': [emb.numpy() for emb in dino_embeddings],
        'fashion_clip_embedding': [emb.numpy() for emb in fashion_clip_embeddings]
    })

    df.to_parquet('image_embeddings.parquet', index=False)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    create_image_embeddings(image_dir=os.getenv("IMAGES_DIR"), image_df_path=os.getenv("IMAGES_DF_PATH"))
