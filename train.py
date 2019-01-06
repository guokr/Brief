#!/usr/bin/env python
# encoding: utf-8

import argparse
import torch
import os
from brief.model import DecoderGRU, EncoderGRU, Seq2SeqModel
from brief.evaluator import Evaluator
from brief.dataset import BriefSourceField, BriefTargetField, BriefDataset, BriefIterator
from tqdm import tqdm
import dill as pickle


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

    SourceField = BriefSourceField()
    TargetField = BriefTargetField()

    from itertools import islice
    columns = []

    with open(args.train_filename) as f_input:
        for row in islice(f_input, 0, 1):
            for _ in row.strip().split("\t"):
                columns.append(_)

    tv_datafields = [(columns[0], SourceField), (columns[1], TargetField)]

    dataset = BriefDataset(train=args.train_filename,
                              valid=args.valid_filename,
                              fields=tv_datafields)

    train_data, valid_data = dataset.splits()

    SourceField.build_vocab(train_data, )
    TargetField.build_vocab(train_data, )

    print("Source dataset ---")
    print(SourceField.vocab.itos[:100])
    print(len(SourceField.vocab.itos))
    print(SourceField.vocab.freqs.most_common(20))
    print("Target dataset ---")
    print(TargetField.vocab.itos[:100])
    print(TargetField.vocab.freqs.most_common(20))

    pickle.dump(SourceField, open(os.path.join(args.checkpoint_dir, "SourceField.p"), "wb"))
    pickle.dump(TargetField, open(os.path.join(args.checkpoint_dir, "TargetField.p"), "wb"))

    ############# pre-process done
    return train_data, valid_data, SourceField, TargetField


import torch.optim as optim
def train(train_data, valid_data, SourceField, TargetField):
    print("| Building batches...")

    train_dataloader, valid_dataloader = BriefIterator.splits(train=train_data,
                                                                 valid=valid_data,
                                                                 batch_size=args.batch_size)

    print("| Building model...")

    encoder_model = EncoderGRU(vocab_size=len(SourceField.vocab))
    decoder_model = DecoderGRU(vocab_size=len(TargetField.vocab))

    seq2seq_model = Seq2SeqModel(encoder=encoder_model,
                                 decoder=decoder_model)
    seq2seq_model.to(device)

    optimizer = optim.Adam(seq2seq_model.parameters())
    criterion = nn.CrossEntropyLoss()
    valid_loss_history = {}

    print("| Training...")

    for epoch in range(1, args.epoch+1):
        train_step(seq2seq_model, train_dataloader, optimizer, criterion, epoch)
        eval_step(seq2seq_model, valid_dataloader, optimizer, criterion, valid_loss_history, epoch)
        test_step(SourceField, TargetField, seq2seq_model)

import torch.nn as nn


def train_step(seq2seq_model, train_dataloader, optimizer, criterion, epoch):
    seq2seq_model.train()
    evaluator = Evaluator(criterion)
    tqdm_progress = tqdm(train_dataloader, desc="| Training epoch {}/{}".format(epoch, args.epoch))
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
    torch.save({"model_args": seq2seq_model.get_args(),
                "model_state_dict": seq2seq_model.state_dict()},
               os.path.join(args.checkpoint_dir, "checkpoint_{}.pt".format(epoch)))


TEST_CASE_tgt = ["<sos>"]
a = """
滁 州 市 气 象 台   2015   年   07   月   12   日   15   时   20   分 发 布 雷 电 黄 色 预 警 信 号 ： 目 前 我 市 西 部 有 较 强 对 流 云 团 向 东 南 方 向 移 动 ， 预 计   6   小 时 内 我 市 部 分 地 区 将 发 生 雷 电 活 动 ， 并 可 能 伴 有 短 时 强 降 水 、 大 风 、 局 部 冰 雹 等 强 对 流 天 气 ， 请 注 意 防 范 。 图 例 标 准 防 御 指 南   6   小 时 内 可 能 发 生 雷 电 活 动 ， 可 能 会 造 成 雷 电 灾 害 事 故 。   1   、 政 府 及 相 关 部 门 按 照 职 责 做 好 防 雷 工 作 ；   2   、 密 切 关 注 天 气 ， 尽 量 避 免 户 外 活 动 。
"""
text = ("有 “ 抗 癌 药 代 购 第 一 人 ” 之 称 陆 勇 为 千 余 网 友 分 享 购 买 印 度 抗 癌 药 渠 道 ， 于 去 年 7 月 因 “ 妨 害 信 用 卡 管 理 ” 和 “ 销 售 假 药 罪 ” 被 检 察 机 关 起 诉 。 检 察 机 关 现 已 撤 回 诉 讼 ， 陆 勇 获 释 ")

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

    output = seq2seq_model(source_seq, target_seq, 0, MAX_LENGTH=30)
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
