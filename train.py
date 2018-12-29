#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import torch
import dill as pickle
from torchtext.data import Field, TabularDataset, BucketIterator
from nutshell.model import EncoderLSTM, DecoderLSTM
from nutshell.utils import MiniBatchWrapper
from tqdm import tqdm



parser = argparse.ArgumentParser(description="Nutshell training")
# parser.add_argument("--model", type=str)
parser.add_argument("--input_data_dir", type=str)
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


import torch.optim as optim
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
    encoder_model = EncoderLSTM(vocab_size=len(TEXT.vocab))
    encoder_model.to(device)
    decoder_model = DecoderLSTM(vocab_size=len(TEXT.vocab))
    decoder_model.to(device)

    encoder_optimizer = optim.SGD(encoder_model.parameters(), lr=0.001)
    decoder_optimizer = optim.SGD(decoder_model.parameters(), lr=0.001)

    for epoch in range(1, args.epoch+1):
        train_step(encoder_model, decoder_model, train_dataloader, epoch, encoder_optimizer, decoder_optimizer)


import torch.nn.functional as F

# criterion = F.nll_loss()

def train_step(encoder_model, decoder_model, train_dataloader, epoch, encoder_optimizer, decoder_optimizer):
    tqdm_progress = tqdm(train_dataloader, desc="| Training epoch {}/{}".format(epoch, args.epoch))
    for source_seq, target_seq in tqdm_progress:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # print("source seq", source_seq.shape)
        # print(source_seq)
        # print("target seq", target_seq.shape)
        # print("target seq", target_seq)

        target_seq_pack = target_seq.view(-1)

        # print("target seq pac", target_seq_pack.shape)
        # print("target seq pac", target_seq_pack)
        encoder_outputs = encoder_model(source_seq)
        decoder_outputs = decoder_model(target_seq, encoder_outputs)
        # print("decoder final ouput", decoder_outputs.shape)
        # print("decoder final ouput", decoder_outputs)
        decoder_outputs_pack = decoder_outputs.view(-1, decoder_outputs.size(-1))
        # print("decoder final ouput pack", decoder_outputs_pack.shape)
        # print("decoder final ouput pack", decoder_outputs_pack)

        loss = F.nll_loss(decoder_outputs_pack, target_seq_pack)
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        tqdm_progress.set_postfix({"Loss":"{:.4f}".format(loss.item())})
        # print("loss value", loss.item())






if __name__ == "__main__":
    tr, vl, TEXT, columns = preprocess()
    train(tr, vl, TEXT, columns)
