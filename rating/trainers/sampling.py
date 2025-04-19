from torch.utils.data import DataLoader
from rating.trainers.base import BaseTrainer
from rating.datasets.sampler import UserUniformBatchSampler


class SamplingTrainer(BaseTrainer):
    def __init__(self, model, train_dataset, val_dataset, config):
        super().__init__(model, train_dataset, val_dataset, config)
        self.batch_size = getattr(config, 'batch_size', 256)
        self.k = getattr(config, 'k', 2)
        self.drop_last = getattr(config, 'drop_last', False)

        sampler = UserUniformBatchSampler(
            user2idx=train_dataset.user_to_row_indices(),
            batch_users=self.batch_size,
            k=self.k,
            shuffle=True,
            drop_last=self.drop_last
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=getattr(config, 'num_workers', 5),
            pin_memory=True
        )
