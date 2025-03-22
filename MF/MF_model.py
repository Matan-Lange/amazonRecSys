import torch
from torch import nn
import torch.nn.init as init


class MfModel(nn.Module):
    def __init__(self, num_users, num_items, num_categories, num_stores, emb_dim, biases=None):
        """
        Initialize the MF model with configurable biases.

        Args:
            num_users (int): Number of unique users
            num_items (int): Number of unique items
            num_categories (int): Number of unique categories
            num_stores (int): Number of unique stores
            emb_dim (int): Dimension of the embedding vectors
            biases (list): List of bias types to include ['user', 'item', 'category', 'store']
        """
        super().__init__()
        self.biases = biases if biases else []

        # user item embeddings
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        # bias embeddings based on configuration
        self.bias_layers = nn.ModuleDict()
        if 'user' in self.biases:
            self.bias_layers['user'] = nn.Embedding(num_users, 1)
        if 'item' in self.biases:
            self.bias_layers['item'] = nn.Embedding(num_items, 1)
        if 'category' in self.biases:
            self.bias_layers['category'] = nn.Embedding(num_categories, 1)
        if 'store' in self.biases:
            self.bias_layers['store'] = nn.Embedding(num_stores, 1)

        # Initialize weights
        init.xavier_uniform_(self.user_emb.weight)
        init.xavier_uniform_(self.item_emb.weight)
        for layer in self.bias_layers.values():
            init.constant_(layer.weight, 0.0)

    def forward(self, batch):
        """
        Forward pass incorporating selected biases.

        Args:
            user (Tensor): User indices
            item (Tensor): Item indices
            category (Tensor, optional): Category indices
            store (Tensor, optional): Store indices
        """
        user_idx = batch['user_idx']
        item_idx = batch['item_idx']
        category_idx = batch['category_idx']
        store_idx = batch['store_idx']

        # Base matrix factorization
        user_emb = self.user_emb(user_idx)
        item_emb = self.item_emb(item_idx)
        element_product = (user_emb * item_emb).sum(1)

        # Add configured biases
        bias_sum = torch.zeros_like(element_product)
        if 'user' in self.biases:
            bias_sum += self.bias_layers['user'](user_idx).squeeze()
        if 'item' in self.biases:
            bias_sum += self.bias_layers['item'](item_idx).squeeze()
        if 'category' in self.biases and category_idx is not None:
            bias_sum += self.bias_layers['category'](category_idx).squeeze()
        if 'store' in self.biases and store_idx is not None:
            bias_sum += self.bias_layers['store'](store_idx).squeeze()

        logit = element_product + bias_sum
        rating = torch.sigmoid(logit) * 4 + 1
        return rating
