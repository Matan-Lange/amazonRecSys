from typing import Dict, List
import random
import numpy as np

from torch.utils.data import Sampler


class UserUniformBatchSampler(Sampler[List[int]]):
    """
    Each mini‑batch contains:
        batch_users users  ×  k interactions per user  = batch_size rows.
    """

    def __init__(
            self,
            user2idx: Dict[int, List[int]],
            batch_users: int = 256,
            k: int = 1,
            shuffle: bool = True,
            drop_last: bool = False,
    ):
        self.user2idx = user2idx
        self.users = np.array(list(user2idx.keys()))
        self.batch_users = batch_users
        self.k = k
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        users = self.users.copy()
        if self.shuffle:
            np.random.shuffle(users)

        for start in range(0, len(users), self.batch_users):
            batch_u = users[start: start + self.batch_users]
            if len(batch_u) < self.batch_users and self.drop_last:
                break

            batch_indices = []
            for u in batch_u:
                rows = self.user2idx[u]
                # sample with replacement if user has < k rows
                picks = (
                    random.choices(rows, k=self.k)
                    if len(rows) < self.k
                    else random.sample(rows, k=self.k)
                )
                batch_indices.extend(picks)

            yield batch_indices

    def __len__(self) -> int:
        n = len(self.users) // self.batch_users
        if not self.drop_last and len(self.users) % self.batch_users:
            n += 1
        return n
