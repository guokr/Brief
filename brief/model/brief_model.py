#!/usr/bin/env python
# encoding: utf-8

import os
import torch
import dill as pickle
from .seq2seq_model import Seq2SeqModel
from .encoder import EncoderGRU
from .decoder import DecoderGRU


class BriefModel(object):
    """
    Wrapper for Seq2Seq Model in inference
    """
    def __init__(self, path=None, device="cpu"):
        self._device = device
        self.load(path)


    def load(self, path):
        loaded_checkpoint = torch.load(os.path.join(path, "checkpoint_10.pt"),
                                       map_location=self._device)

        # loaded two field pickle files
        self._source_field = pickle.load(open(os.path.join(path, "SourceField.p"), "rb"))
        self._target_field = pickle.load(open(os.path.join(path, "TargetField.p"), "rb"))

        # rebuild the inside seq2seq model
        encoder_model = EncoderGRU(vocab_size=len(self._source_field.vocab))
        decoder_model = DecoderGRU(vocab_size=len(self._target_field.vocab))
        self._inside_model = Seq2SeqModel(encoder=encoder_model, decoder=decoder_model)
        self._inside_model.load_state_dict(loaded_checkpoint["model_state_dict"])
        self._inside_model.to(self._device)
        self._inside_model.eval()


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

        output = self._inside_model.forward(source_seq, target_seq, 0, MAX_LENGTH=300)
        output = output[:, 1:, :]
        topv, topi = output.topk(k=1)
        topi = topi.squeeze(-1)
        for sentence in topi:
            words = [self._target_field.vocab.itos[word_idx] for word_idx in sentence]
            print("Predicted -> {}".format(" ".join(words)))

