#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import torch
import dill as pickle
from torchtext.data import Field, TabularDataset, BucketIterator
from nutshell.model import EncoderLSTM
from nutshell.utils import MiniBatchWrapper
from tqdm import tqdm



parser = argparse.ArgumentParser(description="Nutshell training")
# parser.add_argument("--model", type=str)
parser.add_argument("--input_data_dir", type=str)
parser.add_argument("--train_filename", type=str)
parser.add_argument("--valid_filename", type=str)
parser.add_argument("--checkpoint_dir", type=str)
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--batch_size", default=32)

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
    tokenize = lambda x: x.split()

    TEXT = Field(sequential=True, tokenize=tokenize, lower=True, batch_first=True)

    from itertools import islice
    columns = []

    with open(os.path.join(args.input_data_dir, args.train_filename)) as f_input:
        for row in islice(f_input, 0, 1):
            for _ in row.strip().split("\t"):
                columns.append(_)

    tv_datafields = []

    for c in columns:
        tv_datafields.append((c, TEXT))

    train_data, valid_data = TabularDataset.splits(path=args.input_data_dir,
                                                   format="tsv",
                                                   train=args.train_filename,
                                                   validation=args.valid_filename,
                                                   skip_header=True,
                                                   fields=tv_datafields)

    TEXT.build_vocab(train_data)

    print(TEXT.vocab.itos[:10])
    print(TEXT.vocab.freqs.most_common(20))

    # pickle.dump(TEXT, open(os.path.join(args.output_data_dir, "TEXT.p"), "wb"))

    ############# pre-process done
    return train_data, valid_data, TEXT, columns


def train(train_data, valid_data, TEXT, columns):
    print("| Building batches...")
    # device = torch.device("cuda:{}".format(args.master_device))
    # build dataloader
    train_iter, valid_iter = BucketIterator.splits((train_data, valid_data),
                                batch_size=args.batch_size,
                                device=device,
                                sort_key=lambda x: len(x.english),
                                sort_within_batch=True)

    train_dataloader = MiniBatchWrapper(train_iter, columns[0], columns[1])
    # valid_dataloader = MiniBatchWrapper(valid_iter, columns[0], columns[1])

    # print("| Building model...")
    for epoch in range(1, args.epoch+1):
        train_step(train_dataloader, epoch)

def train_step(train_dataloader, epoch):
    tqdm_progress = tqdm(train_dataloader, desc="| Training epoch {}/{}".format(epoch, args.epoch))
    for source_seq, target_seq in  tqdm_progress:
        print(source_seq, target_seq)

if __name__ == "__main__":
    tr, vl, TEXT, columns = preprocess()
    train(tr, vl, TEXT, columns)
