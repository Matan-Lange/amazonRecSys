import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict


def _df_column_to_tensor(
    df: pd.DataFrame,
    index_col: str,
    vector_col: str
) -> torch.Tensor:
    """
    Sorts df by index_col, fills missing embeddings with zeros,
    stacks vector_col (np.ndarray) into a 2D array,
    and returns it as a float32 torch.Tensor.

    Args:
      df:          DataFrame containing your data.
      index_col:   Column to use as embedding indices (0..N-1).
      vector_col:  Column containing np.ndarray embeddings or nulls.

    Returns:
      A torch.Tensor of shape (N, D), where D is embedding dimension.
    """
    # 1. Reindex & sort by index_col
    df_sorted = df.set_index(index_col).sort_index()

    # 2. Extract list of vectors (some may be None/NaN)
    arrs = df_sorted[vector_col].tolist()

    # 3. Determine embedding dimension from first valid vector
    dim = None
    for x in arrs:
        if isinstance(x, np.ndarray):
            dim = x.shape[0]
            break
    if dim is None:
        raise ValueError(f"No valid embeddings found in column '{vector_col}'")

    # 4. Fill missing with zero vectors
    filled = []
    for x in arrs:
        if isinstance(x, np.ndarray):
            filled.append(x)
        else:
            filled.append(np.zeros(dim, dtype=float))

    # 5. Stack and convert to torch.Tensor
    matrix = np.vstack(filled)
    return torch.tensor(matrix, dtype=torch.float32)


def build_embedding_layer(
    df: pd.DataFrame,
    index_col: str,
    vector_col: str,
    freeze: bool = True
) -> nn.Embedding:
    """
    Creates an nn.Embedding from df[index_col] -> df[vector_col],
    filling missing with zero embeddings.

    Args:
      df:           DataFrame containing your data.
      index_col:    Column with item indices (0..N-1).
      vector_col:   Column containing np.ndarray embeddings or nulls.
      freeze:       If True, weights are not trainable.

    Returns:
      A torch.nn.Embedding layer of size (N, D).
    """
    tensor = _df_column_to_tensor(df, index_col, vector_col)
    return nn.Embedding.from_pretrained(tensor, freeze=freeze)


def build_all_embeddings(
    df: pd.DataFrame,
    index_col: str,
    vector_cols: List[str],
    freeze: bool = True
) -> Dict[str, nn.Embedding]:
    """
    Given multiple vector columns, build a dict of Embedding layers.

    Args:
      df:           DataFrame containing your data.
      index_col:    Column name for item indices.
      vector_cols:  List of column names, each holding np.ndarray embeddings or nulls.
      freeze:       If True, weights are not trainable.

    Returns:
      A dict mapping each column name to its nn.Embedding layer.
    """
    embeddings = {}
    for col in vector_cols:
        embeddings[col] = build_embedding_layer(df, index_col, col, freeze=freeze)
    return embeddings



cols = [
    "text_embeddings",
    "dino_embedding",
    "fashion_clip_embedding",
    "category_vector",
]



emb_layers = build_all_embeddings(
    df = pd.read_parquet("/tmp/pycharm_project_190/preproc_data/metadata.parquet"),
    index_col="item_idx",
    vector_cols=cols,
    freeze=True
)


