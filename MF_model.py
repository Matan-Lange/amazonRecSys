from torch import nn

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
        return element_product + user_bias + item_bias