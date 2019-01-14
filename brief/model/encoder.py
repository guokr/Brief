#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if "weigth" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_size=512, num_layers=2,
                 dropout_in = 0.3, dropout_out = 0.3, bidirectional=False):
        super().__init__()
        ## starts with _ means normal attr, otherwise nn layers
        self._embedding_dim = embedding_dim
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout_in = dropout_in
        self._dropout_out = dropout_out
        self._bidirectional = bidirectional
        self._vocab_size = vocab_size

        ## ends with _layer means torch nn layers
        self.embedding_layer = nn.Embedding(self._vocab_size, self._embedding_dim)
        self.dropout_layer = nn.Dropout(self._dropout_in)
        self.lstm_layer = LSTM(
            input_size=self._embedding_dim,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            dropout=self._dropout_out if self._num_layers > 1 else 0.,
            bidirectional=self._bidirectional,
        )


    def forward(self, sequence):
        # print("---inside encoder---")
        # print("--inside encoder", sequence.shape)

        batch_size, source_seq_length = sequence.size()
        # sequence_length = sequence.size(1)
        # state_size = batch_size, self._num_layers, self._hidden_size
        # print("state shape", state_size)

        embedded = self.dropout_layer(self.embedding_layer(sequence))

        embedded = embedded.transpose(0, 1)
        # > t x b x c
        state_size = self._num_layers, batch_size, self._hidden_size

        h0 = embedded.data.new(*state_size).zero_()
        c0 = embedded.data.new(*state_size).zero_()

        outputs, (hidden, cell) = self.lstm_layer(embedded, (h0, c0))
        # print("--inside encoder outputs shape", outputs.shape)
        # print("--inside encoder hiden shape", hidden.shape)
        # print("--inside encoder cell shape", cell.shape)
        ## outputs shape: [batch_size, sequence length, hidden_size x directions ]
        ## hidden shape: [n_layers x directions, batch_size, hidden_dim]
        ## cell shape: [n_layers x directions, batch_size, hidden_dim]

        return {"encoder_out": (outputs, hidden, cell)}


class EncoderGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, encoder_hidden_size=128, decoder_hidden_size=128,
                 dropout=0.3, batch_first=True, bidirectional=True):
        super().__init__()
        self._encoder_hidden_size = encoder_hidden_size
        self._decoder_hidden_size = decoder_hidden_size
        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._dropout = dropout

        self.embedding_layer = nn.Embedding(self._vocab_size, self._embedding_dim)
        self.gru_layer = nn.GRU(input_size=self._embedding_dim,
                                hidden_size=self._encoder_hidden_size,
                                batch_first=True,
                                bidirectional=True)
        self.fc_layer = nn.Linear(self._encoder_hidden_size * 2, self._decoder_hidden_size)
        self.dropout_layer = nn.Dropout(self._dropout)


    def get_args(self):
        return vars(self)


    def forward(self, sequence):
        # sequence shape: [batch_size, sequence length]
        embedded = self.dropout_layer(self.embedding_layer(sequence))
        self.gru_layer.flatten_parameters()

        # embedded shape: [batch_size, sequence length, embedding dim]
        outputs, hidden = self.gru_layer(embedded)

        # outputs shape: [batch_size, sequence length, num_directions x hidden_size]
        # hidden shape: [num_layers x num_directions, batch_size, hidden_size]

        #################
        # hidden stacked as [forward1, backward1, forward2, backward2] at last timestep

        hidden = torch.tanh(self.fc_layer(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        ## concat the encoder 's bidir 'hidden  and tanh map to decoder size

        # hidden shape : [batch size, decoder_hidden_dim]

        return outputs, hidden
