#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .Encoder import EncoderLSTM
from .Decoder import DecoderLSTM


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def load(self, path):
        raise NotImplementedError

    def update_args(self):
        raise NotImplementedError


# encoder_instance = EncoderLSTM()
# decoder_instance = DecoderLSTM()


class BriefModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self._encoder_model = encoder
        self._deocder_model = decoder
        self._device = "cuda"
        self._teacher_forcing_ratio = 0.5
        self._vocab_size = decoder._vocab_size


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
            output, hidden = self._deocder_model(decoder_input, hidden, encoder_outputs)
            outputs[:, t, :] = output
            # print(outputs)
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            decoder_input = (target_seq[:, t] if teacher_force else top1)

        return outputs
