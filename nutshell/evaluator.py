#!/usr/bin/env python
# encoding: utf-8

import torch
import numpy as np
import math
from rouge import Rouge

rouge = Rouge()

def compute_rouge(preds, targets):

    batch_size = preds.size(0)

    preds = preds[:, 1:, :]
    # output = output.contiguous().view(-1, output.shape[2])
    topv, topi = preds.topk(k=1)

    preds_cpu = topi.squeeze(-1).data.cpu().numpy()
    targets_cpu = targets[:, 1:].data.cpu().numpy()

    batch_rouge_f_1 = 0.0
    batch_rouge_f_2 = 0.0

    for sent in range(batch_size):
        preds_sent = " ".join([str(_) for _ in preds_cpu[sent] if _!=3 and _!=1])
        targets_sent = " ".join([str(_) for _ in targets_cpu[sent] if _!=3 and _!=1])
        try:
            rouge_score = rouge.get_scores(preds_sent, targets_sent)[0]
            batch_rouge_f_1 += rouge_score["rouge-1"]["f"]
            batch_rouge_f_2 += rouge_score["rouge-2"]["f"]
        except:
            pass
            # print("Unvalid prediction, skipped")

    return batch_rouge_f_1 / batch_size, batch_rouge_f_2 / batch_size



class Evaluator(object):
    def __init__(self, criterion):
        self.accumulate_num_samples = 0
        self.accumulate_loss = 0.0
        self.accumulate_perplexity = 0.0
        self.accumulate_rougef1 = 0.0
        self.accumulate_rougef2 = 0.0
        self.criterion = criterion
        self.accumulate_num_samples = 0


    def clear(self):
        self.accumulate_num_samples = 0
        self.accumulate_loss = 0.0
        self.accumulate_perplexity = 0.0


    def evaluate(self, preds, target, mode="train"):
        """
        do evaluation for each batch
        """
        batch_size = preds.size(0)
        target_length = preds.size(1)
        self.accumulate_num_samples += batch_size


        if mode == "eval":
            batch_rougef1, batch_rougef2 =  compute_rouge(preds, target)
            self.accumulate_rougef1 += batch_rougef1
            self.accumulate_rougef2 += batch_rougef2

        # print(self.accumulate_rougef1, self.accumulate_rougef2)

        # output shap: [batch_size, sequence length, output_dim/target vocab_dim]

        ## -> target_seq shape to [batch_size * (sequence length - 1)]
        ## -> ouput shape to [batch_size * (sequence length -1), output dim]
        ## should be ?
        self.accumulate_num_samples += batch_size

        ######### metrics computing
        # target_cpu = target.data.cpu().numpy()
        # preds_cpu = preds.data.cpu().numpy()

        ########### loss computing based on concating all sentences in one minibatccccch
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
        self.accumulate_perplexity += batch_ppl

        if mode=="eval":
            return list(map(lambda x: x/self.accumulate_num_samples,
                            [self.accumulate_loss, self.accumulate_perplexity, self.accumulate_rougef1, self.accumulate_rougef2]))
        return list(map(lambda x: x/self.accumulate_num_samples,
                        [self.accumulate_loss, self.accumulate_perplexity]))

