#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class NutshellModel(BaseModel):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder_model = encoder
        self.deocder_model = decoder

    def forward(self, input_seq, target_seq):
        raise NotImplementedError
