import torch
from torch import nn
import torch.nn.init as init


class MfModel(nn.Module):
    """
    Matrix factorization model with user and item biases.

    Args:
        num_users (int): Number of unique users.
        num_items (int): Number of unique items.
        emb_dim (int): Dimension of the embedding vectors.
    """

    def __init__(self, num_users, num_items, emb_dim):
        """
        Initializes the MfModel with user and item embeddings and biases.

        Args:
            num_users (int): Number of unique users.
            num_items (int): Number of unique items.
            emb_dim (int): Dimension of the embedding vectors.
        """
        super().__init__()
        self.user_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=emb_dim)
        self.item_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=emb_dim)
        self.user_bias = nn.Embedding(num_embeddings=num_users, embedding_dim=1)
        self.item_bias = nn.Embedding(num_embeddings=num_items, embedding_dim=1)

        # wieghts initialization
        init.xavier_uniform_(self.user_emb.weight)
        init.xavier_uniform_(self.item_emb.weight)
        init.constant_(self.user_bias.weight, 0.0)
        init.constant_(self.item_bias.weight, 0.0)

    def forward(self, user, item):
        """
        Forward pass for the model.

        Args:
            user (Tensor): Tensor containing user indices.
            item (Tensor): Tensor containing item indices.

        Returns:
            Tensor: Element-wise product of user and item embeddings summed over the embedding dimension,
                    plus user and item biases.
        """
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        user_bias = self.user_bias(user).squeeze()
        item_bias = self.item_bias(item).squeeze()
        element_product = (user_emb * item_emb).sum(1)
        logit = element_product + user_bias + item_bias
        # clip the rating to be between 1 and 5
        rating = torch.sigmoid(logit) * 4 + 1
        return rating
