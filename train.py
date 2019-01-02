#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import torch
import dill as pickle
import math
from torchtext.data import Field, TabularDataset, BucketIterator
from nutshell.model import EncoderLSTM, DecoderLSTM, EncoderGRU, DecoderGRU, NutshellModel
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


    print(SourceField.vocab.itos[:100])
    print(SourceField.vocab.freqs.most_common(20))

    print(TargetField.vocab.itos[:100])
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
    encoder_model = EncoderLSTM(vocab_size=len(SourceField.vocab))
    decoder_model = DecoderLSTM(vocab_size=len(TargetField.vocab))

    nutshell_model = NutshellModel(encoder_model, decoder_model)
    nutshell_model.to(device)
    optimizer = optim.Adam(nutshell_model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epoch+1):
        train_step(nutshell_model, train_dataloader, optimizer, criterion, epoch)
        eval_step(nutshell_model, valid_dataloader, optimizer, criterion, epoch)
        eval_test(SourceField, TargetField, nutshell_model)


import torch.nn.functional as F
import torch.nn as nn
def train_step(seq2seq_model, train_dataloader, optimizer, criterion, epoch):
    tqdm_progress = tqdm(train_dataloader, desc="| Training epoch {}/{}".format(epoch, args.epoch))
    seq2seq_model.train()
    for source_seq, target_seq in tqdm_progress:
        # print("--- in train source_seq shape", source_seq.shape)
        # print("--- in train target seq shape", target_seq.shape)
        optimizer.zero_grad()
        output = seq2seq_model(source_seq, target_seq)
        # print("--in train output shape", output.shape)
        # output shap: [batch_size, sequence length, output_dim/target vocab_dim]

        ## -> target_seq shape to [batch_size * (sequence length - 1)]
        ## -> ouput shape to [batch_size * (sequence length -1), output dim]
        target = target_seq[:, 1:]
        target = target.contiguous().view(-1)
        output = output[:, 1:, :]
        output = output.contiguous().view(-1, output.shape[2])
        # print(output.shape)
        # print(target.shape)
        # print(target.contiguous().view(-1))
        # print("--- in train loss target shape", target.shape)
        # print("--- in train loss output shape", target.shape)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        tqdm_progress.set_postfix({"Loss":"{:.4f}".format(loss.item()),
                                   "PPL":"{:.4f}".format(math.e**loss.item())})


def eval_step(seq2seq_model, valid_dataloader, optimizer, criterion, epoch):
    seq2seq_model.eval()
    tqdm_progress = tqdm(valid_dataloader, desc="| Training epoch {}/{}".format(epoch, args.epoch))
    with torch.no_grad():
        for source_seq, target_seq in tqdm_progress:
            output = seq2seq_model(source_seq, target_seq, 0)
            target = target_seq[:, 1:]
            target = target.contiguous().view(-1)
            output = output[:, 1:, :]
            output = output.contiguous().view(-1, output.shape[2])
            loss = criterion(output, target)
            tqdm_progress.set_postfix({"Loss":"{:.4f}".format(loss.item()),
                                       "PPL":"{:.4f}".format(math.e**loss.item())})


TEST_CASE_src = ["wait! <eos>", "cheers! <eos>"]
TEST_CASE_tgt = ["<sos> <sos> <sos>", "<sos> <sos> <sos>"]


def eval_test(SourceField, TargetField, seq2seq_model):
    source_vocab = SourceField.vocab.stoi
    source_batch_indexed = [[source_vocab[token] for token in sent.split()] for sent in TEST_CASE_src]
    source_batch_indexed = torch.LongTensor(source_batch_indexed).to(device)

    print(source_batch_indexed)

    target_voab = TargetField.vocab.stoi
    target_batch_indexed = [[target_voab[token] for token in sent.split()] for sent in TEST_CASE_tgt]
    target_batch_indexed = torch.LongTensor(target_batch_indexed).to(device)

    print(target_batch_indexed)

    source_seq = source_batch_indexed
    target_seq = target_batch_indexed

    output = seq2seq_model(source_seq, target_seq, 0)
    # target = target_seq[:, 1:]
    # target = target.contiguous().view(-1)
    output = output[:, 1:, :]
    # print(output)
    # output = output.contiguous().view(-1, output.shape[2])
    topv, topi = output.topk(k=1)
    # print(topi)
    print(topi.shape)

    output_cpu = topi
    for batch in output_cpu:
        for sequence in batch:
            for word in sequence:
                print("word is {}".format(TargetField.vocab.itos[word]))



if __name__ == "__main__":
    tr, vl, src, tgt = preprocess()
    train(tr, vl, src, tgt)
