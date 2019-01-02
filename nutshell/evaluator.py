#!/usr/bin/env python
# encoding: utf-8

import torch
import numpy as np
import math


class Evaluator(object):
    def __init__(self, criterion, top_k=5):
        self.accumulate_num_samples = 0
        self.accumulate_loss = 0.0
        self.accumulate_perplexity = 0.0
        self.criterion = criterion


    def clear(self):
        self.accumulate_num_samples = 0
        self.accumulate_loss = 0.0
        self.accumulate_perplexity = 0.0


    def evaluate(self, preds, target, mode="train"):
        """
        do evaluation for each batch
        """
        # preds/target: batch_size x num_labels
        batch_size = preds.size(0)
        target_length = preds.size(1)
        # output shap: [batch_size, sequence length, output_dim/target vocab_dim]

        ## -> target_seq shape to [batch_size * (sequence length - 1)]
        ## -> ouput shape to [batch_size * (sequence length -1), output dim]
        ## should be ?

        self.accumulate_num_samples += batch_size

        # target = target[:, 1:]
        target = target.contiguous().view(-1)
        # preds = preds[:, 1:, :]
        preds = preds.contiguous().view(-1, preds.shape[2])

        batch_loss = self.criterion(preds, target)

        # return the batch ave recall, precision and fscore at once
        if mode == "train":
            batch_loss.backward()

        batch_loss_cpu = batch_loss.item()

        self.accumulate_loss += batch_loss_cpu * batch_size * (target_length)

        batch_ppl = math.e**batch_loss_cpu

        return batch_loss, batch_ppl
