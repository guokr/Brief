#!/usr/bin/env python
# encoding: utf-8

from torchtext.data import TabularDataset

class AnutshellDatasetInitError(Exception):
    pass


class AnutshellDataset(object):
    def __init__(self, train, valid, fields, **kwargs):
        self._parse_path(train, valid)
        self.fields = fields


    def _parse_path(self, train_filepath, valid_filepath):
        tr_ = train_filepath.split("/")
        val_ = valid_filepath.split("/")
        prefix_tr = "/".join(tr_[:-1])
        prefix_val = "/".join(val_[:-1])
        if prefix_tr != prefix_val:
            raise NutshellDatasetInitError("train file and valid file should be placed in the same dir")

        self.prefix_dir = prefix_tr
        self.train_file = tr_[-1]
        self.valid_file = val_[-1]


    def splits(self):
        train_data, valid_data = TabularDataset.splits(path=self.prefix_dir,
                                                       format="tsv",
                                                       train=self.train_file,
                                                       validation=self.valid_file,
                                                       skip_header=True,
                                                       fields=self.fields)

        return {"Dataset": train_data, "Field": self.fields}, {"Dataset": valid_data, "Field": self.fields}
