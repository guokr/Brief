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


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden_size=128, decoder_hidden_size=128):
        super().__init__()
        self._encoder_hidden_size = encoder_hidden_size
        self._decoder_hidden_size = decoder_hidden_size
        self._weight_operator = nn.Parameter(torch.rand(self._decoder_hidden_size))

        self.attn_layer = nn.Linear((self._encoder_hidden_size * 2) + self._decoder_hidden_size,
                                     self._decoder_hidden_size)


    def forward(self, decoder_hidden, encoder_outputs):
        # decoder hidden shape: [batch_size, decoder_hidden_dim]
        # encoder outputs shape [batch_size, source sequence length , encoder hidden dim x encoder directions]
        batch_size = encoder_outputs.size(0)
        source_sequence_length = encoder_outputs.size(1)

        hidden = decoder_hidden.unsqueeze(1).repeat(1, source_sequence_length, 1)
        # hidden shape : [batch_size, source_seq length, decoder_hidden_dim]

        attn_affect = torch.tanh(self.attn_layer(torch.cat((hidden, encoder_outputs), dim=2)))
        attn_affect = attn_affect.permute(0, 2, 1)

        weight_op = self._weight_operator.repeat(batch_size, 1).unsqueeze(1)

        attention = torch.bmm(weight_op, attn_affect).squeeze(1)

        return F.softmax(attention, dim=1)


atten_instance = BahdanauAttention()

class DecoderGRU(nn.Module):
    def __init__(self, vocab_size, attention=atten_instance, embedding_dim=128, encoder_hidden_size=128, decoder_hidden_size=128,
                 dropout=0.3):
        super().__init__()

        self._embedding_dim = embedding_dim
        self._encoder_hidden_size = encoder_hidden_size
        self._decoder_hidden_size = decoder_hidden_size
        self._vocab_size = vocab_size
        self._dropout = dropout

        self.attention_layer = attention
        self.embedding_layer = nn.Embedding(self._vocab_size, self._embedding_dim)
        self.gru_layer = nn.GRU(self._encoder_hidden_size*2+self._embedding_dim,
                                self._decoder_hidden_size)
        self.dropout_layer = nn.Dropout(self._dropout)
        self.predict_layer = nn.Linear(self._encoder_hidden_size*2+self._decoder_hidden_size+self._embedding_dim,
                                       self._vocab_size)


    def forward(self, input, hidden, encoder_outputs):
        # input shape: [batch_size]
        # decoder hidden : [batch_size, decoder hidden size]
        # encoder outputs : [batch_size, source seq length, encoder_hidden dim x directions]
        input = input.unsqueeze(1)
        # input shape: [batch_size, 1]
        embedded = self.dropout_layer(self.embedding_layer(input))

        # embedded shape: [batch_size, 1, embedding_dim]

        a = self.attention_layer(hidden, encoder_outputs)

        # a shape: [batch_size, source seq lenght]

        a = a.unsqueeze(1)

        # a shape: [batch_size, 1, source seq lenght]
        weighted = torch.bmm(a, encoder_outputs)
        # weighted : [batch_size, 1, encoder hidden dim x directions]

        gru_input = torch.cat((embedded, weighted), dim=2)
        # gru input : [batch_size, 1, encoder hidden dim x directions + embedding_dim]
        output, hidden = self.gru_layer(gru_input, hidden.unsqueeze(1))

        # output [batch_size, target seq len, decoder_hidden dim x directions]
        # hidden [n layers x directions, batch_size, decoder_hidden dim]

        ## here seq length/ layer / dir all one
        # output: [batch_size, 1, decoder hidden dim]
        # hidden: [1, batch_size, decoder hidden dim]

        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = output.squeeze(1)

        output = self.predict_layer(torch.cat((output, weighted, embedded), dim=1))

        # output [batch_size, vocab_size]

        return output, hidden.squeeze(0)
