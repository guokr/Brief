#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=128, num_layers=1,
                 dropout=0.3, use_attention=False, encoder_embedding_dim=128,
                 batch_first=True):
        super().__init__()

        ## start with _ means normal attr, otherwise nn layers
        self._dropout = dropout
        self._hidden_size = hidden_size
        self._use_attention = use_attention
        self._voab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._batch_first = batch_first
        self._num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, self._embedding_dim)

        self.lstm = nn.LSTMCell(input_size=self._embedding_dim + self._hidden_size,
                                hidden_size=self._hidden_size)
#        self.lstm = nn.LSTM(
#            batch_first=self._batch_first,
#            input_size=self._embedding_dim,
#            hidden_size=self._hidden_size,
#            num_layers=self._num_layers
#        )



    def forward(self, sequence, encoder_outputs):
        print("---inside decoder---")
        print("target seq", sequence.shape)
        batch_size, sequence_length = sequence.size()

        embedded = self.embedding(sequence)
        print("embedded", embedded.shape)
        # original encoder output
        (encoder_output, encoder_output_hidden, encoder_output_cell) = encoder_outputs

        print(encoder_output_hidden.shape)

        # encoder's last time step  hidden and cell, all layers concat
        # encoder_output_hiddens = [encoder_output_hidden[i] for i in range(self._num_layers)]
        # encoder_output_cells = [encoder_output_cell[i] for i in range(self._num_layers)]

        decoder_hidden_loop = torch.zeros(batch_size, self._hidden_size, device="cuda")

        # start decoder's own sequence loop
        outs = []

        for word_idx in range(sequence_length):
            decoder_own_input = torch.cat((embedded[:, word_idx, :], decoder_hidden_loop), dim=1)
            # decoder own input : batch_size, embedding_size + input_feed custom/ 4, 128+100
            print("---inside inside decoder own input", decoder_own_input.shape)
            decoder_hidden, decoder_cell = self.lstm(decoder_own_input,
                                                                       (encoder_output_hidden[0], encoder_output_cell[0]))

            decoder_hidden_loop = decoder_hidden
            outs.append(decoder_hidden)

        decoder_own_outputs = torch.cat(outs, dim=1)
        # batch_size, seq length x hidden size
        print("---decoder own output", decoder_own_outputs.shape)
        decoder_own_outputs.view(batch_size, sequence_length, self._hidden_size)

        attention_weights = None

        return decoder_own_outputs, attention_weights



