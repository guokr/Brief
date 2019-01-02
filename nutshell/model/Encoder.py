#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=128, num_layers=1, dropout=0.3,
                 bidirectional=False, batch_first=True):
        super().__init__()
        ## starts with _ means normal attr, otherwise nn layers
        self._embedding_dim = embedding_dim
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._bidirectional = bidirectional
        self._batch_first = batch_first
        self._vocab_size = vocab_size

        ## ends with _layer means torch nn layers
        self.embedding_layer = nn.Embedding(self._vocab_size, self._embedding_dim)
        self.dropout_layer = nn.Dropout(self._dropout)
        self.lstm_layer = nn.LSTM(
            input_size=self._embedding_dim,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            # dropout=self._dropout,
            bidirectional=self._bidirectional,
            batch_first=self._batch_first
        )


    def forward(self, sequence):
        # print("---inside encoder---")
        # print("--inside encoder", sequence.shape)

        # batch_size = sequence.size(0)
        # sequence_length = sequence.size(1)
        # state_size = batch_size, self._num_layers, self._hidden_size
        # print("state shape", state_size)

        embedded = self.dropout_layer(self.embedding_layer(sequence))
        ## --> embedded shape: [batch_size, sequence length, embeding_dim]
        # print("encoder embedded shape", embedded.shape)

        #h0 = embedded.data.new(*state_size).zero_()
        #print("h0", h0)
        #print(h0.shape)
        #c0 = embedded.data.new(*state_size).zero_()

        outputs, (hidden, cell) = self.lstm_layer(embedded)
        # print("--inside encoder outputs shape", outputs.shape)
        # print("--inside encoder hiden shape", hidden.shape)
        # print("--inside encoder cell shape", cell.shape)
        ## outputs shape: [batch_size, sequence length, hidden_size x directions ]
        ## hidden shape: [n_layers x directions, batch_size, hidden_dim]
        ## cell shape: [n_layers x directions, batch_size, hidden_dim]

        # reorder?
        return hidden, cell


class EncoderGRU(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self.embedding = nn.Embedding(self._vocab_size, self._hidden_size)
        self.gru = nn.GRU(self._hidden_size, self._hidden_size, batch_first=True)

    def forward(self, sequence):
        # print("encoder input shape")
        # print(sequence.shape)
        batch_size = sequence.size(0)

        embedded = self.embedding(sequence)
        output, hidden = self.gru(embedded)
        # print("encoder inside gru output", output.shape)
        # print("encoder inside gru hidden", hidden.shape)
        return output, hidden
