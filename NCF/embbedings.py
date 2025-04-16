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
    def __init__(self, image_dir: str, image_df_path: str, item_map: dict) -> None:
        self.image_dir = image_dir
        self.image_df = pd.read_csv(image_df_path)
        self.image_df['image_path'] = self.image_df.apply(
            lambda x: os.path.join(image_dir, f"{x['parent_asin']}.{x['image_format']}"), axis=1
        )
        self.image_df['item_idx'] = self.image_df['parent_asin'].map(item_map)

        # remove cold items
        self.image_df = self.image_df[self.image_df['item_idx'].notna()]

    def __len__(self):
        return len(self.image_df)

    def __getitem__(self, idx: int) -> dict:
        row = self.image_df.iloc[idx]
        image_path = row['image_path']

        if not os.path.isfile(image_path):
            print(f"Image file not found: {image_path}")
            # dummy image
            # Todo - check if we can find images
            image = Image.new('RGB', (224, 224), color='black')
        else:
            image = Image.open(image_path).convert('RGB')
        return {
            'image': image,
            'item_idx': row['item_idx']
        }

    @staticmethod
    def collate_fn(batch):
        images = [item['image'] for item in batch]
        item_idx = [int(item['item_idx']) for item in batch]

        return {
            'image': images,
            'item_idx': item_idx
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
        return embeddings

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
        return embeddings

    def get_latent_dim(self) -> int:
        return self.model.config.projection_dim


def image_embedding_factory(model_name: str) -> ImageEmbedding:
    if model_name == 'dinov2':
        return DinoV2Embedding()
    elif model_name == 'fashion_clip':
        return FashionClipImageEmbedding()
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def create_image_embeddings(image_dir: str,
                            image_df_path: str,
                            model_name: str,
                            item_map: dict) -> None:
    dataset = ImageDataset(image_dir, image_df_path, item_map)
    dataloader = DataLoader(dataset,
                            batch_size=512,
                            shuffle=False,
                            num_workers=20,
                            collate_fn=ImageDataset.collate_fn)
    embedding_model = image_embedding_factory(model_name)

    emb_store = nn.Embedding(198771, embedding_model.get_latent_dim())
    emb_store.weight.requires_grad = False

    device = next(embedding_model.model.parameters()).device

    for batch in tqdm(dataloader):
        images = batch['image']
        item_idx = batch['item_idx']
        embeddings = embedding_model.embed(images)
        # index tensor and move to correct device
        idx_tensor = torch.tensor([int(idx) for idx in item_idx], dtype=torch.long).to(device)
        emb_store.weight.data[idx_tensor] = embeddings.to(emb_store.weight.device)

    torch.save(emb_store.state_dict(), f"{model_name}_embeddings.pt")
    return emb_store


if __name__ == "__main__":
    from NCF.dataset import DatasetFactory

    # insert paths here
    train_path = '/tmp/pycharm_project_760/data/user_item_rating_table_train.csv'
    test_path = '/tmp/pycharm_project_760/data/warm_items_rating_prediction_test_format.csv'
    metadata_path = '/tmp/pycharm_project_760/data/items_metadata.jsonl'
    image_dir = '/tmp/pycharm_project_760/data/images'
    image_df_path = '/tmp/pycharm_project_760/data/images_urls.csv'

    factory = DatasetFactory(train_path, test_path, metadata_path)
    train_dataset, val_dataset, test_dataset = factory.create_datasets()

    create_image_embeddings(image_dir, image_df_path, 'fashion_clip', train_dataset.hashmaps['item'])
