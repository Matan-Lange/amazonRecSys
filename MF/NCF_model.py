import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel


class NCFWithFashionCLIP(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64, mlp_dims=[128, 64]):
        super().__init__()

        # Embedding layers
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_collab_emb = nn.Embedding(num_items, embed_dim)

        # FashionCLIP (pretrained)
        self.fclip_model = AutoModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.fclip_processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")

        # Freeze FashionCLIP
        for param in self.fclip_model.parameters():
            param.requires_grad = False

        clip_dim = self.fclip_model.config.projection_dim

        # MLP for regression
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + clip_dim, mlp_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_dims[0], mlp_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_dims[1], 1)
        )

    def forward(self, inputs: dict):
        # Embeddings
        user_e = self.user_emb(inputs['user_idx'])
        item_e = self.item_collab_emb(inputs['item_idx'])

        # Process image + text via FashionCLIP
        inputs = self.fclip_processor(text=inputs['text'], images=inputs['image'], return_tensors="pt", padding=True).to(
            self.device())
        clip_outputs = self.fclip_model(**inputs)
        clip_embedding = clip_outputs.image_embeds

        # Combine
        x = torch.cat([user_e, item_e, clip_embedding], dim=1)
        raw_output = self.mlp(x)

        # Squash to [1, 5]
        rating = 1 + 4 * torch.sigmoid(raw_output.squeeze())
        return rating

    def device(self):
        return next(self.parameters()).device
