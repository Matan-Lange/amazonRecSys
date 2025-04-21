from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import json
from tqdm import tqdm


def load_metadata(metadata_path) -> pd.DataFrame:
    """Load metadata."""
    with open(metadata_path, "r") as file:
        data = [json.loads(line) for line in file]
    df = pd.DataFrame.from_records(data)
    return df

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path: str) -> None:
        self.metadata = load_metadata(metadata_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        title = row['title']
        parent_asin = row['parent_asin']

        return {
            'title': title,
            'parent_asin': parent_asin
        }

    @staticmethod
    def collate_fn(batch):
        title = [item['title'] for item in batch]
        parent_asin = [item['parent_asin'] for item in batch]

        return {
            'title': title,
            'parent_asin': parent_asin
        }


class E5Embeddings:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct')
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @staticmethod
    def average_pool(last_hidden_states: Tensor,
                     attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed(self, input_texts: list[str]):
        batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict['input_ids'] = batch_dict['input_ids'].to(self.device)
        batch_dict['attention_mask'] = batch_dict['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return embeddings.cpu()


def create_text_embeddings(metadata_path: str) -> None:
    dataset = TextDataset(metadata_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, collate_fn=TextDataset.collate_fn)

    e5_embeddings = E5Embeddings()

    text_embeddings = []
    parent_asins = []

    for batch in tqdm(dataloader):
        titles = batch['title']
        parent_asin = batch['parent_asin']

        embeddings = e5_embeddings.embed(titles)

        for i in range(len(parent_asin)):
            text_embeddings.append(embeddings[i])
            parent_asins.append(parent_asin[i])


    # Save the embeddings to a parquet file
    df = pd.DataFrame({
        'parent_asin': parent_asins,
        'text_embeddings': [emb.numpy() for emb in text_embeddings],
    })

    df.to_parquet('text_embeddings.parquet', index=False)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    create_text_embeddings(os.getenv('METADATA_PATH'))