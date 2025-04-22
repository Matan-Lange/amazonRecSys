import torch
import torch.nn as nn
import torch.nn.functional as F
from rating_refactor.proc import emb_layers
from typing import Dict, List


class NCFModel(nn.Module):
    def __init__(
            self,
            num_users: int,
            embeddings: Dict[str, nn.Embedding] = emb_layers,
            num_categories: int = 1103,
            embed_dim: int = 64,
            cat_embed_dim: int = 32,
            mlp_dims: List[int] = [16, 8],
            dropout_rate: float = 0.5,
    ):
        """
        Neural Collaborative Filtering model with multimodal item embeddings
        and a learnable EmbeddingBag for multi-hot category vectors.

        Args:
            num_users: total number of users
            embeddings: dict of pretrained item embeddings, e.g.
                {
                  "dino_embedding": nn.Embedding(num_items, 768),
                  "text_embeddings": nn.Embedding(num_items, 1024),
                  "fashion_clip_embedding": nn.Embedding(num_items, 512),
                  "category_vector": nn.Embedding(num_items, 1103)
                }
            num_categories: size of the category vocabulary
            embed_dim: target dimension for user & item vectors
            cat_embed_dim: intermediate embedding dim for categories
            mlp_dims: hidden layer sizes for the final MLP
            dropout_rate: dropout probability between MLP layers
        """
        super().__init__()
        # 1. USER EMBEDDING
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.user_ln = nn.LayerNorm(embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)

        # 2. PRECOMPUTED ITEM EMBEDDINGS (frozen)
        self.dino_emb = embeddings["dino_embedding"]
        self.text_emb = embeddings["text_embeddings"]
        self.fclip_emb = embeddings["fashion_clip_embedding"]
        self.one_hot_emb = embeddings["category_vector"]
        for emb in (self.dino_emb, self.text_emb, self.fclip_emb, self.one_hot_emb):
            emb.requires_grad_(False)

        # 3. CATEGORY EmbeddingBag for multi-hot vectors
        self.cat_embedding = nn.EmbeddingBag(
            num_categories,
            cat_embed_dim,
            mode="mean",
            sparse=False
        )
        nn.init.xavier_uniform_(self.cat_embedding.weight)

        # 4. PROJECTION LAYERS into shared embed_dim
        self.proj_dino = nn.Linear(self.dino_emb.embedding_dim, embed_dim)
        self.proj_text = nn.Linear(self.text_emb.embedding_dim, embed_dim)
        self.proj_fclip = nn.Linear(self.fclip_emb.embedding_dim, embed_dim)
        self.proj_cat = nn.Linear(cat_embed_dim, embed_dim)
        self.item_ln = nn.LayerNorm(embed_dim)

        # 5. MLP HEAD
        mlp_input_dim = 2 * embed_dim  # concatenated [user || item]
        dims = [mlp_input_dim] + mlp_dims
        self.mlp_layers = nn.ModuleList(
            nn.Linear(in_d, out_d) for in_d, out_d in zip(dims[:-1], dims[1:])
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(mlp_dims[-1], 1)

    def forward(self,batch) -> torch.FloatTensor:

        user_idx = batch['user_idx']
        item_idx = batch['item_idx']
        cat_multihot = self.one_hot_emb(item_idx)

        # -- USER PATHWAY --
        u = self.user_ln(self.user_emb(user_idx))  # (batch, embed_dim)

        # -- ITEM MODALITIES --
        d = self.proj_dino(self.dino_emb(item_idx))  # DINOv2
        t = self.proj_text(self.text_emb(item_idx))  # Text (E5)
        f = self.proj_fclip(self.fclip_emb(item_idx))  # FashionCLIP

        # -- CATEGORY PATHWAY: multi-hot → indices → EmbeddingBag --
        batch_size, num_categories = cat_multihot.shape
        # 1) find positions of 1s
        row_idx, cat_idx = cat_multihot.nonzero(as_tuple=True)
        # 2) sort by row to group each example's categories
        sorted_row_idx, perm = row_idx.sort()
        sorted_cat_idx = cat_idx[perm]
        # 3) compute counts per example, then offsets
        counts = torch.bincount(sorted_row_idx, minlength=batch_size)
        offsets = torch.cat([sorted_row_idx.new_tensor([0]), counts.cumsum(0)[:-1]])
        # 4) lookup & mean‑reduce
        c_bag = self.cat_embedding(sorted_cat_idx, offsets)  # (batch, cat_embed_dim)
        c = self.proj_cat(c_bag)  # → (batch, embed_dim)

        # -- COMBINE ITEM MODALITIES --
        item_vec = d + t + f + c
        item_vec = self.item_ln(item_vec)  # (batch, embed_dim)

        # -- CONCAT + MLP →
        x = torch.cat([u, item_vec], dim=-1)  # (batch, 2*embed_dim)
        for layer in self.mlp_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        score = self.output(x).squeeze(-1)  # (batch,)

        return score
