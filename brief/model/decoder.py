#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=128, num_layers=1,
                 output_embedding_dim=128,
                 dropout_in=0.3, dropout_out=0.3,
                 encoder_embedded_dim=128, encoder_output_units=128,
                 batch_first=True):
        super().__init__()
        ## starts with _ means normal attr
        self._embedding_dim = embedding_dim
        self._output_embedding_dim = output_embedding_dim
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._num_layers = num_layers
        self._dropout_in = dropout_in
        self._dropout_out = dropout_out
        self._batch_first = batch_first

        assert encoder_output_units == hidden_size, "encoder output units should equal to decoder hidden size"

        self._encoder_output_units = encoder_output_units

        ## ends with _layer means torch nn layers
        self.embedding_layer = nn.Embedding(self._vocab_size, self._embedding_dim)

        self.lstm_layer = nn.ModuleList([
            nn.LSTMCell(
                input_size=self._encoder_output_units + self._embedding_dim if layer==0 else hidden_size,
                hidden_size=hidden_size
            )
            for layer in range(self._num_layers)
        ])

        if hidden_size != output_embedding_dim:
            self.additional_linear = nn.Linear(self._hidden_size, output_embedding_dim)

        self.dropput_layer = nn.Dropout(self._dropout_in)
        self.final_linear = nn.Linear(self._output_embedding_dim, self._vocab_size)


    def forward(self, sequence, encoder_out):
        batch_size, sequence_length = sequence.size()

        encoder_outputs, encoder_hiddens, encoder_cells = encoder_out["encoder_out"]

        x = self.embedding_layer(sequence)
        x = F.dropout(x, p=self._dropout_in)

        x = x.transpose(0, 1)
        # batch_size, seq length, embed dim  => seq_length, batch_size, embed dim

        prev_hiddens = [encoder_hiddens[i] for i in range(self._num_layers)]
        prev_cells = [encoder_cells[i] for i in range(self._num_layers)]

        input_feed = x.data.new(batch_size, self._encoder_output_units).zero_()

        outs = []
        for j in range(sequence_length):
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            for i, lstm in enumerate(self.lstm_layer):
                hidden, cell = lstm(input, (prev_hiddens[i], prev_cells[i]))

                input = F.dropout(hidden, p=self._dropout_out)
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            out = hidden
            out = F.dropout(out, p=self._dropout_out)
            input_feed = out

            outs.append(out)

        x = torch.cat(outs, dim=0).view(sequence_length, batch_size, self._hidden_size)
        x = x.transpose(1, 0)

        x = self.final_linear(x)

        return x










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
                                self._decoder_hidden_size,
                                batch_first=True)
        self.dropout_layer = nn.Dropout(self._dropout)
        self.predict_layer = nn.Linear(self._encoder_hidden_size*2+self._decoder_hidden_size+self._embedding_dim,
                                       self._vocab_size)


    def get_args(self):
        return vars(self)


    def forward(self, input, hidden, encoder_outputs):
        # input shape: [batch_size]
        # decoder hidden : [batch_size, decoder hidden size]
        # encoder outputs : [batch_size, source seq length, encoder_hidden dim x encoder directions]
        input = input.unsqueeze(1)
        # input shape: [batch_size, 1]
        embedded = self.dropout_layer(self.embedding_layer(input))

        # embedded shape: [batch_size, 1, embedding_dim]

        a = self.attention_layer(hidden, encoder_outputs)

        # a shape: [batch_size, source seq lenght]

        a = a.unsqueeze(1)

        # a shape: [batch_size, 1, source seq lenght]
        weighted = torch.bmm(a, encoder_outputs)
        # weighted : [batch_size, 1, encoder hidden dim x encoder directions]

        gru_input = torch.cat((embedded, weighted), dim=2)
        # gru input : [batch_size, 1, encoder hidden dim x directions + embedding_dim]
        # here hidden should be [n_layer x direct, batch_size, decoder_hid dim]
        output, hidden = self.gru_layer(gru_input, hidden.unsqueeze(0))

        # output [batch_size, target seq len, decoder_hidden dim x directions]
        # hidden [n layers x directions, batch_size, decoder_hidden dim]

        ## here seq length/ layer / dir all one
        # output: [batch_size, 1, decoder hidden dim]
        # hidden: [1, batch_size, decoder hidden dim]

        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)

        output = self.predict_layer(torch.cat((output, weighted, embedded), dim=1))

        # output [batch_size, vocab_size]

        return output, hidden.squeeze(0)
