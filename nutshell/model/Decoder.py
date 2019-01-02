#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=128, num_layers=1, dropout=0.3,
                 batch_first=True):
        super().__init__()
        ## starts with _ means normal attr
        self._embedding_dim = embedding_dim
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._batch_first = batch_first

        ## ends with _layer means torch nn layers

        self.embedding_layer = nn.Embedding(self._vocab_size, self._embedding_dim)
        self.lstm_layer = nn.LSTM(input_size=self._embedding_dim,
                                  hidden_size=self._hidden_size,
                                  num_layers=self._num_layers,
                                  batch_first=self._batch_first)
        self.dropput_layer = nn.Dropout(self._dropout)
        self.predict_layer = nn.Linear(self._hidden_size, self._vocab_size)

    def forward(self, input, hidden, cell):
        # print("--inside decoder step ------------")
        # print("--inside decoder input shape", input.shape)

        # input: [batch_size]
        # hidden: [n_layers x directions, batch_size, hidden size]
        # cell: [n_layers x directions, batch_size, hidden size]

        input = input.unsqueeze(1)
        # print("--inside decoder input unsq shape", input.shape)
        ## input shape: [batch_size, 1]
        embedded = self.dropput_layer(self.embedding_layer(input))
        ## embedded shape: [batch_size, 1, embedding_dim]

        outputs, (hidden, cell) = self.lstm_layer(embedded, (hidden, cell))
        # print("--inside decoder outputs shape", outputs.shape)
        # print("--inside decoder hidden shape", hidden.shape)
        # print("--inside decoder cell shape", cell.shape)

        ## outputs: [batch_size, 1, hidden_dim]
        ## hidden: [n_layers, batch_size, hidden_dim]
        ## cell: [n_layers, batch_size, hidden_dim]
        prediction = self.predict_layer(outputs.squeeze(1))

        ## prediction : [batch_size, output_dim]
        # print("--inside decoder prediction shape", prediction.shape)

        return prediction, hidden, cell



class LinearAttentionLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # starts with _ means normal attr
        self._input_size = input_size
        self._output_size = output_size

    def forward(input_features):
        raise NotImplementedError


class DecoderLSTM_B(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=128, num_layers=1,
                 dropout=0.3, use_attention=False, encoder_embedding_dim=128,
                 batch_first=True):
        super().__init__()

        ## start with _ means normal attr, otherwise nn layers
        self._dropout = dropout
        self._hidden_size = hidden_size
        self._use_attention = use_attention
        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._batch_first = batch_first
        self._num_layers = num_layers

        self.embedding = nn.Embedding(self._vocab_size, self._embedding_dim)

        self.lstm = nn.LSTMCell(input_size=self._embedding_dim + self._hidden_size,
                                hidden_size=self._hidden_size)
        self.predictor = nn.Linear(self._hidden_size, self._vocab_size)


    def forward(self, sequence, encoder_outputs):
        # print("---inside decoder---")
        # print("target seq", sequence.shape)
        batch_size, sequence_length = sequence.size()

        embedded = self.embedding(sequence)
        # print("embedded", embedded.shape)
        # original encoder output
        (encoder_output, encoder_output_hidden, encoder_output_cell) = encoder_outputs

        # print(encoder_output_hidden.shape)

        # encoder's last time step  hidden and cell, all layers concat
        # encoder_output_hiddens = [encoder_output_hidden[i] for i in range(self._num_layers)]
        # encoder_output_cells = [encoder_output_cell[i] for i in range(self._num_layers)]

        # start decoder's own sequence loop
        outs = []
        decoder_hidden_loop = torch.zeros(batch_size, self._hidden_size, device="cuda")

        for word_idx in range(sequence_length):
            decoder_own_input = torch.cat((embedded[:, word_idx, :], decoder_hidden_loop), dim=1)
            # decoder own input : batch_size, embedding_size + input_feed custom/ 4, 128+100
            # print("---inside inside decoder own input", decoder_own_input.shape)
            decoder_hidden, decoder_cell = self.lstm(decoder_own_input,
                                                                       (encoder_output_hidden[0], encoder_output_cell[0]))

            decoder_hidden_loop = decoder_hidden
            outs.append(decoder_hidden)

        decoder_own_outputs = torch.cat(outs, dim=1)
        # batch_size, seq length x hidden size
        # print("---decoder own output", decoder_own_outputs.shape)
        decoder_own_outputs = decoder_own_outputs.view(batch_size, sequence_length, self._hidden_size)
        # print("---decoder own output shape", decoder_own_outputs.shape)

        # attention_weights = None
        decoder_own_outputs = self.predictor(decoder_own_outputs)

        # print("decoder logits shape", decoder_own_outputs.shape)

        decoder_own_outputs = F.log_softmax(decoder_own_outputs, dim=-1)
        # print(type(decoder_own_outputs))

        return decoder_own_outputs


class DecoderGRU(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._max_length = 10

        self.embedding = nn.Embedding(self._vocab_size, self._hidden_size)
        self.attn = nn.Linear(self._hidden_size * 2, self._max_length)
        self.attn_combine = nn.Linear(self._hidden_size * 2, self._hidden_size)

        self.gru = nn.GRU(self._hidden_size, self._hidden_size, batch_first=True)
        self.out = nn.Linear(self._hidden_size, self._vocab_size)


    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
