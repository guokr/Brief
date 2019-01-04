#!/usr/bin/env python
# encoding: utf-8

import argparse
import torch
from torchtext.data import Field, TabularDataset, BucketIterator
from anutshell.model import EncoderLSTM, DecoderLSTM, AnutshellModel
from anutshell.evaluator import Evaluator
from anutshell.dataset import AnutshellSourceField, AnutshellTargetField, AnutshellDataset, AnutshellIterator
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Nutshell training")
# parser.add_argument("--model", type=str)
parser.add_argument("--train_filename", type=str)
parser.add_argument("--valid_filename", type=str)
parser.add_argument("--checkpoint_dir", type=str)
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=4)

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

    SourceField = AnutshellSourceField()
    TargetField = AnutshellTargetField()

    from itertools import islice
    columns = []

    with open(args.train_filename) as f_input:
        for row in islice(f_input, 0, 1):
            for _ in row.strip().split("\t"):
                columns.append(_)

    tv_datafields = [(columns[0], SourceField), (columns[1], TargetField)]

    dataset = AnutshellDataset(train=args.train_filename,
                              valid=args.valid_filename,
                              fields=tv_datafields)

    train_data, valid_data = dataset.splits()

    SourceField.build_vocab(train_data)
    TargetField.build_vocab(train_data)

    print("Source dataset ---")
    print(SourceField.vocab.itos[:100])
    print(SourceField.vocab.freqs.most_common(20))
    print("Target dataset ---")
    print(TargetField.vocab.itos[:100])
    print(TargetField.vocab.freqs.most_common(20))

    # pickle.dump(TEXT, open(os.path.join(args.output_data_dir, "TEXT.p"), "wb"))

    ############# pre-process done
    return train_data, valid_data, SourceField, TargetField


import torch.optim as optim
def train(train_data, valid_data, SourceField, TargetField):
    print("| Building batches...")

    train_dataloader, valid_dataloader = AnutshellIterator.splits(train=train_data,
                                                                 valid=valid_data,
                                                                 batch_size=args.batch_size)

    print("| Building model...")

    encoder_model = EncoderLSTM(vocab_size=len(SourceField.vocab))
    decoder_model = DecoderLSTM(vocab_size=len(TargetField.vocab))

    anutshell_model = AnutshellModel(encoder_model, decoder_model)
    anutshell_model.to(device)

    optimizer = optim.Adam(anutshell_model.parameters())
    criterion = nn.CrossEntropyLoss()
    valid_loss_history = {}

    print("| Training...")

    for epoch in range(1, args.epoch+1):
        train_step(anutshell_model, train_dataloader, optimizer, criterion, epoch)
        eval_step(anutshell_model, valid_dataloader, optimizer, criterion, valid_loss_history, epoch)
        test_step(SourceField, TargetField, anutshell_model)

import torch.nn.functional as F
import torch.nn as nn

def train_step(seq2seq_model, train_dataloader, optimizer, criterion, epoch):
    tqdm_progress = tqdm(train_dataloader, desc="| Training epoch {}/{}".format(epoch, args.epoch))
    seq2seq_model.train()
    evaluator = Evaluator(criterion)
    for source_seq, target_seq in tqdm_progress:
        optimizer.zero_grad()
        output = seq2seq_model(source_seq, target_seq)
        loss, ppl = evaluator.evaluate(output, target_seq)
        optimizer.step()
        tqdm_progress.set_postfix({"Loss": "{:.4f}".format(loss),
                                   "PPL": "{:.4f}".format(ppl)})


def eval_step(seq2seq_model, valid_dataloader, optimizer, criterion, eval_loss_history, epoch):
    seq2seq_model.eval()
    evaluator = Evaluator(criterion)
    tqdm_progress = tqdm(valid_dataloader, desc="| Validating epoch {}/{}".format(epoch, args.epoch))
    with torch.no_grad():
        for source_seq, target_seq in tqdm_progress:
            output = seq2seq_model(source_seq, target_seq, 0)
            loss, ppl, rougef1, rougef2 = evaluator.evaluate(output, target_seq, mode="eval")
            tqdm_progress.set_postfix({"Loss": "{:.4f}".format(loss),
                                       "PPL": "{:.4f}".format(ppl),
                                       "Rouge-1": "{:.4f}".format(rougef1),
                                       "Rouge-2": "{:.4f}".format(rougef2)})


TEST_CASE_tgt = ["<sos>"]
a = """
滁 州 市 气 象 台   2015   年   07   月   12   日   15   时   20   分 发 布 雷 电 黄 色 预 警 信 号 ： 目 前 我 市 西 部 有 较 强 对 流 云 团 向 东 南 方 向 移 动 ， 预 计   6   小 时 内 我 市 部 分 地 区 将 发 生 雷 电 活 动 ， 并 可 能 伴 有 短 时 强 降 水 、 大 风 、 局 部 冰 雹 等 强 对 流 天 气 ， 请 注 意 防 范 。 图 例 标 准 防 御 指 南   6   小 时 内 可 能 发 生 雷 电 活 动 ， 可 能 会 造 成 雷 电 灾 害 事 故 。   1   、 政 府 及 相 关 部 门 按 照 职 责 做 好 防 雷 工 作 ；   2   、 密 切 关 注 天 气 ， 尽 量 避 免 户 外 活 动 。
"""
TEST_CASE_src = [a]

def test_step(SourceField, TargetField, seq2seq_model):
    source_vocab = SourceField.vocab.stoi
    source_batch_indexed = [[source_vocab[token] for token in sent.split()] for sent in TEST_CASE_src]
    source_batch_indexed = torch.LongTensor(source_batch_indexed).to(device)

    target_voab = TargetField.vocab.stoi
    target_batch_indexed = [[target_voab[token] for token in sent.split()] for sent in TEST_CASE_tgt]
    target_batch_indexed = torch.LongTensor(target_batch_indexed).to(device)

    source_seq = source_batch_indexed
    target_seq = target_batch_indexed

    output = seq2seq_model(source_seq, target_seq, 0, MAX_LENGTH=100)
    # target = target_seq[:, 1:]
    # target = target.contiguous().view(-1)
    output = output[:, 1:, :]
    # output = output.contiguous().view(-1, output.shape[2])
    topv, topi = output.topk(k=1)
    topi = topi.squeeze(-1)
    # print(topi)
    # print(topi.shape)
    for sentence in topi:
        words = [TargetField.vocab.itos[word_idx] for word_idx in sentence]
        print("Predicted -> {}".format(" ".join(words)))


if __name__ == "__main__":
    tr, vl, src, tgt = preprocess()
    train(tr, vl, src, tgt)
