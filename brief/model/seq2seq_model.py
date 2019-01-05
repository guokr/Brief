#!/usr/bin/env python
# encoding: utf-8

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import dill as pickle

from .encoder import EncoderGRU
from .decoder import DecoderGRU


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def load(self, path):
        raise NotImplementedError

    def update_args(self):
        raise NotImplementedError

# encoder_instance = EncoderLSTM()
# decoder_instance = DecoderLSTM()

class Seq2SeqModel(nn.Module):
    def __init__(self, path=None, encoder=None, decoder=None, device="cuda"):
        super().__init__()
        if encoder!=None and decoder!=None:
            self._encoder_model = encoder
            self._decoder_model = decoder
            self._device = device
            self._teacher_forcing_ratio = 0.5
            self._vocab_size = decoder._vocab_size
            self._target_field = None
            self._source_filed = None
        elif path!=None:
            # restore from checkpoint dir
            self._device = device
            self.load(path, self._device)


    def get_args(self):
        return vars(self)


    def brief(self, text):
        """
        text: ["This should be a simple sntence"]
        """
        source_vocab = self._source_field.vocab.stoi
        source_batch_indexed = [[source_vocab[token] for token in sent.split()] for sent in text]
        source_batch_indexed = torch.LongTensor(source_batch_indexed).to(self._device)

        TEST_CASE_tgt = ["<sos>"]
        target_voab = self._target_field.vocab.stoi
        target_batch_indexed = [[target_voab[token] for token in sent.split()] for sent in TEST_CASE_tgt]
        target_batch_indexed = torch.LongTensor(target_batch_indexed).to(self._device)

        source_seq = source_batch_indexed
        target_seq = target_batch_indexed

        output = self.forward(source_seq, target_seq, 0, MAX_LENGTH=100)
        # target = target_seq[:, 1:]
        # target = target.contiguous().view(-1)
        output = output[:, 1:, :]
        # output = output.contiguous().view(-1, output.shape[2])
        topv, topi = output.topk(k=1)
        topi = topi.squeeze(-1)
        # print(topi)
        # print(topi.shape)
        for sentence in topi:
            words = [self._target_field.vocab.itos[word_idx] for word_idx in sentence]
            print("Predicted -> {}".format(" ".join(words)))



    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5, MAX_LENGTH=None):
        # input_seq shape: [batch_size, sequence length]
        # target_seq shape: [batch_size, sequence length]
        batch_size = target_seq.size(0)

        if MAX_LENGTH == None:
            MAX_LENGTH = target_seq.size(1)

        outputs = torch.zeros(batch_size, MAX_LENGTH, self._vocab_size).to(self._device)
        # print("--inside seq final all outputs shape", outputs.shape)


        encoder_outputs, hidden = self._encoder_model(input_seq)

        decoder_input = target_seq[:, 0]

        # print("--inside seq decoder input shape", decoder_input.shape)

        for t in range(1, MAX_LENGTH):
            output, hidden = self._decoder_model(decoder_input, hidden, encoder_outputs)
            outputs[:, t, :] = output
            # print(outputs)
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            decoder_input = (target_seq[:, t] if teacher_force else top1)

        return outputs
