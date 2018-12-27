#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLSTM(nn.Module):
    def __init__(self, dictionary, embedding_dim=128, hidden_size=128, num_layers=1, dropout=0.3,
                 bidirectional=False, padding=0, batch_firse=True):
        super().__init__()
        ## starts with _ means normal attr, otherwise nn layers
        self._embedding_dim = embedding_dim
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._bidirectional = bidirectional
        self._batch_first = True

        self.embedding = nn.Embedding(len(dictionary), self._embedding_dim)
        self.lstm = nn.LSTM(
            input_size=self._embedding_dim,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            dropout=self._dropout,
            bidirectional=self._bidirectional
        )

    def forward(self, sequence):
        batch_size = sequence.size(0)
        # sequence_length = sequence.size(1)
        state_size = batch_size, self._num_layers, self._hidden_size

        embedded = self.embedding(sequence)
        h0 = embedded.data.new(*state_size).zero_()
        c0 = embedded.data.new(*state_size).zero_()

        output, hidden, cell = self.lstm(embedded, h0, c0)


