#!/usr/bin/env python
# encoding: utf-8

from torchtext.data import BucketIterator


class MiniBatchWrapper(object):
    """
    wrap the simple torchtext iter
    """
    def __init__(self, dl, source_var, target_var):
        self.dl, self.source_var, self.target_var = dl, source_var, target_var

    def __iter__(self):
        for batch in self.dl:
            source_seq = getattr(batch, self.source_var) # we assume only one input in this wrapper
            target_seq = getattr(batch, self.target_var) # we assume only one input in this wrapper
            # if self.y_vars is  not None:
            # temp = [getattr(batch, feat).unsqueeze(1) for feat in self.y_vars]
            # y = torch.cat(temp, dim=1).float()
            # else:
                # y = torch.zeros((1))
            # yield (x, y)
            yield (source_seq, target_seq)

    def __len__(self):
        return len(self.dl)


class AnutshellIterator(object):
    def __init__(self, train, valid, device="cuda", batch_size=2):
        self.train_dataset = train.get("Dataset")
        self.valid_dataset = valid.get("Dataset")
        self.field_names = [field[0] for field in train.get("Field")]
        self.valid_dataset_dict = valid
        self.device = device
        self.batch_size = batch_size


    @classmethod
    def splits(cls, train, valid, batch_size=2, device="cuda"):
        train_dataset = train.get("Dataset")
        valid_dataset = valid.get("Dataset")
        field_names = [field[0] for field in train.get("Field")]
        device = device

        train_iter, valid_iter = BucketIterator.splits((train_dataset, valid_dataset),
                                                       batch_size=batch_size,
                                                       device=device,
                                                       sort_key=lambda x: len(vars(x)[field_names[0]]),
                                                       sort_within_batch=True)

        train_dataloader = MiniBatchWrapper(train_iter, field_names[0], field_names[1])
        valid_dataloader = MiniBatchWrapper(valid_iter, field_names[0], field_names[1])

        return train_dataloader, valid_dataloader


