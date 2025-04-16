import torch
import torch.nn as nn
from typing import List


class NCF(nn.Module):
    def __init__(self,
                 num_users: int,
                 item_embbeings: nn.Embedding,
                 embed_dim: int = 32,
                 mlp_dims: List[int] = [16, 8],
                 dropout_rate: float = 0.5):
        super().__init__()

        self.global_mean = nn.Parameter(torch.tensor(4.321553826471724), requires_grad=True)

        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.user_ln = nn.LayerNorm(embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)


        # User bias
        self.user_bias = nn.Embedding(num_users, 1)
        nn.init.zeros_(self.user_bias.weight)

        self.item_emb = item_embbeings
        self.item_ln = nn.LayerNorm(self.item_emb.weight.shape[1])
        #freeze item embeddings
        for param in self.item_emb.parameters():
            param.requires_grad = False

        # Project item embeddings to same dimension as user embeddings
        self.item_projection = nn.Linear(self.item_emb.weight.shape[1], embed_dim)
        nn.init.xavier_uniform_(self.item_projection.weight)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, mlp_dims[0]),
            nn.LayerNorm(mlp_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dims[0], 1)
        )

    def forward(self, inputs: dict):
        user_e = self.user_ln(self.user_emb(inputs['user_idx']))
        item_e = self.item_ln(self.item_emb(inputs['item_idx']))
        user_b = self.user_bias(inputs['user_idx']).squeeze()
        item_e = self.item_projection(item_e)
        # Apply layer normalization to item embeddings
        x = torch.cat([user_e, item_e], dim=1)
        raw_output = self.mlp(x)
        output = self.global_mean + user_b + raw_output.squeeze()
        #if we want to clip ratings during training
        #rating = 1 + 4 * torch.sigmoid(raw_output.squeeze())
        #rating = 3 + 2 * torch.tanh(output)
        return output


    def device(self):
        return next(self.parameters()).device
