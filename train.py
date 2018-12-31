#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import torch
import dill as pickle
from torchtext.data import Field, TabularDataset, BucketIterator
from nutshell.model import EncoderLSTM, DecoderLSTM, EncoderGRU, DecoderGRU
from nutshell.utils import MiniBatchWrapper
from nutshell.dataset import NutshellSourceField, NutshellTargetField, NutshellDataset, NutshellIterator
from tqdm import tqdm



parser = argparse.ArgumentParser(description="Nutshell training")
# parser.add_argument("--model", type=str)
parser.add_argument("--train_filename", type=str)
parser.add_argument("--valid_filename", type=str)
parser.add_argument("--checkpoint_dir", type=str)
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--batch_size", default=4)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_args():
    print("=============== Command Line Tools Args ===============")
    for arg, value in vars(args).items():
        print("{:>20} <===> {:<20}".format(arg, value))
    print("=======================================================")


    status = True
#    if not os.path.exists(os.path.join(args.input_data_dir, args.train_filename)):
#        status = False
#        print("|ERROR| train file doesn't exist")
#
#    if not os.path.exists(os.path.join(args.input_data_dir, args.valid_filename)):
#        status = False
#        print("|ERROR| valid file doesn't exist")
#
#    if torch.cuda.is_available() == False:
#        status = False
#        print("|ERROR| Currently we dont support CPU training")
#
#    if torch.cuda.device_count() == 1 and args.multi_gpu == True:
#        status = False
#        print("|ERROR| We only detected {} GPU".format(torch.cuda.device_count()))
#
#    if os.path.isdir(args.checkpoint_dir) and len(os.listdir(args.checkpoint_dir)) != 0:
#        status = False
#        # exist but not empty
#        print("|ERROR| save dir must be empty")
#
#    if not os.path.isdir(args.checkpoint_dir):
#        print("|NOTE| Doesn't find the save dir, we will create a default one for you")
#        os.mkdir(args.checkpoint_dir)
#
#    if not os.path.isdir(args.output_data_dir):
#        print("|NOTE| Doesn't find the output data dir, we will create a default one for you")
#        os.mkdir(args.output_data_dir)

    return status


def preprocess():
    print("|LOGGING| Processing tokens and datasets...")

    SourceField = NutshellSourceField()
    TargetField = NutshellTargetField()

    from itertools import islice
    columns = []

    with open(args.train_filename) as f_input:
        for row in islice(f_input, 0, 1):
            for _ in row.strip().split("\t"):
                columns.append(_)

    tv_datafields = [(columns[0], SourceField), (columns[1], TargetField)]

    dataset = NutshellDataset(train=args.train_filename,
                              valid=args.valid_filename,
                              fields=tv_datafields)

    train_data, valid_data = dataset.splits()

    SourceField.build_vocab(train_data)
    TargetField.build_vocab(train_data)


    print(SourceField.vocab.itos[:10])
    print(SourceField.vocab.freqs.most_common(20))

    print(TargetField.vocab.itos[:10])
    print(TargetField.vocab.freqs.most_common(20))

    # pickle.dump(TEXT, open(os.path.join(args.output_data_dir, "TEXT.p"), "wb"))

    ############# pre-process done
    return train_data, valid_data, SourceField, TargetField


import torch.optim as optim
def train(train_data, valid_data, SourceField, TargetField):
    print("| Building batches...")
    # device = torch.device("cuda:{}".format(args.master_device))
    # build dataloader

    train_dataloader, valid_dataloader = NutshellIterator.splits(train=train_data, valid=valid_data)

    # train_dataloader = MiniBatchWrapper(train_iter, columns[0], columns[1])
    # valid_dataloader = MiniBatchWrapper(valid_iter, columns[0], columns[1])

    # print("| Building model...")
    # encoder_model = EncoderLSTM(vocab_size=len(SourceField.vocab))
    encoder_model = EncoderGRU(vocab_size=len(SourceField.vocab))

    encoder_model.to(device)
    decoder_model = DecoderGRU(vocab_size=len(TargetField.vocab))
    decoder_model.to(device)

    encoder_optimizer = optim.SGD(encoder_model.parameters(), lr=0.001)
    decoder_optimizer = optim.SGD(decoder_model.parameters(), lr=0.001)

    for epoch in range(1, args.epoch+1):
        train_step(encoder_model, decoder_model, train_dataloader, epoch, encoder_optimizer, decoder_optimizer)
        eval_test(SourceField, TargetField, encoder_model, decoder_model)


import torch.nn.functional as F
# criterion = F.nll_loss()
import torch.nn as nn
criterion = nn.NLLLoss()
def train_step(encoder_model, decoder_model, train_dataloader, epoch, encoder_optimizer, decoder_optimizer):
    tqdm_progress = tqdm(train_dataloader, desc="| Training epoch {}/{}".format(epoch, args.epoch))
    for source_seq, target_seq in tqdm_progress:
        print("outside ", source_seq.shape)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        print("source seq", source_seq.shape)
        print(source_seq)
        print("target seq", target_seq.shape)
        print("target seq", target_seq)

        target_seq_pack = target_seq.view(-1)

        # print("target seq pac", target_seq_pack.shape)
        # print("target seq pac", target_seq_pack)
        encoder_outputs = torch.zeros(10, 128, device=device)
        # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        encoder_outputs_, encoder_hidden = encoder_model(source_seq)
        for ei in range(source_seq.size(1)):
            encoder_outputs[ei]  = encoder_outputs_[0, 0]

        print("outside encoder outputs", encoder_outputs.shape)
        # print("outside encoder outputs", encoder_outputs)
        print("outside encoder hidden", encoder_hidden.shape)

        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([[2]]).to(device)
        print("decoder input", decoder_input)
        print("decoder input ", decoder_input.shape)
        loss = 0.0


        for ei in range(target_seq.size(1)):
            decoder_output, decoder_input, decoder_attention = decoder_model(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # print(decoder_output)
            # print(decoder_output.shape)
            decoder_input = target_seq[:, ei]
            # print(target_seq[:, ei].unsqueeze(0))
            # print(target_seq[:, ei].unsqueeze(0).shape)

            # loss = F.nll_loss(decoder_output, target_seq[:, ei+1].unsqueeze(0))

            loss += criterion(decoder_output, target_seq[:, ei])

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        tqdm_progress.set_postfix({"Loss":"{:.4f}".format(loss.item())})
        # print("loss value", loss.item())



TEST_CASE_src = ["stop! <eos>", "cheers! <eos>"]
TEST_CASE_tgt = ["<sos> Arrête-toi !", "<sos> Santé !"]


def eval_test(SourceField, TargetField, encoder_model, decoder_model):
    source_vocab = SourceField.vocab.stoi
    source_batch_indexed = [[source_vocab[token] for token in sent.split()] for sent in TEST_CASE_src]
    source_batch_indexed = torch.LongTensor(source_batch_indexed).to(device)

    encoder_outputs = encoder_model(source_batch_indexed)

    target_vocab = TargetField.vocab.stoi
    target_batch_indexed = [[target_vocab[token] for token in sent.split()] for sent in TEST_CASE_tgt]
    target_batch_indexed = torch.LongTensor(target_batch_indexed).to(device)

    final_outs = decoder_model(target_batch_indexed, encoder_outputs)
    out_v, out_i = torch.topk(final_outs, dim=2, k=1)
    out_i.squeeze(2)
    print(out_i.shape)
    print(out_i)
    for sample in out_i:
        for _ in sample:
            print(TargetField.vocab.itos[_])



if __name__ == "__main__":
    tr, vl, src, tgt = preprocess()
    train(tr, vl, src, tgt)
