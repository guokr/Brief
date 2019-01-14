#!/usr/bin/env python
# encoding: utf-8

import argparse
import torch
import os
from brief.model import DecoderGRU, EncoderGRU, Seq2SeqModel, NSeq2SeqModel, EncoderLSTM, DecoderLSTM
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

    SourceField.build_vocab(train_data, min_freq=2)
    TargetField.build_vocab(train_data, min_freq=2)

    print("Source dataset ---")
    # print(SourceField.vocab.itos[:])
    # print([(k,v) for k,v in enumerate(SourceField.vocab.itos[:])])
    print(len(SourceField.vocab.itos))
    print(SourceField.vocab.freqs.most_common(20))
    print("Target dataset ---")
    # print(TargetField.vocab.itos[:])
    # print([(k,v) for k,v in enumerate(TargetField.vocab.itos[:])])
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

    encoder_model = EncoderLSTM(vocab_size=len(SourceField.vocab))
    decoder_model = DecoderLSTM(vocab_size=len(TargetField.vocab))

    seq2seq_model = NSeq2SeqModel(encoder=encoder_model,
                                  decoder=decoder_model)
    seq2seq_model.to(device)

    optimizer = optim.Adam(seq2seq_model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=1)
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
        # print(source_seq)
        # print(target_seq)
        optimizer.zero_grad()
        output = seq2seq_model(source_seq, target_seq)
        # print(output.shape)
        # print(target_seq.shape)
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
aa = """为 期 三 个 月 的 全 国 公 路 执 法 专 项 整 改 工 作 刚 结 束 ，
整 治 重 点 包 括 对 非 法 超 限 运 输 车 辆 只 收 费 不 卸 载 、
伙 同 社 会 闲 散 人 员 擅 自 放 行 等 。 然 而 ， 在 重 要 省 道 滨 唐 公 路 津 冀 交 界 处 ，
执 法 治 超 沦 为 摆 设 ， 大 肆 收 费 后 擅 自 放 行 ， 超 载 问 题 严 重 失 控 。 <eos>"""
### 津 冀 交 界 公 路 治 超 载 乱 象 官 卡 执 法 沦 为 摆 设
a = """i am lazy. <eos>"""
b = """i am cold. <eos>"""
TEST_CASE_src = [aa]

def test_step(SourceField, TargetField, seq2seq_model):
    source_vocab = SourceField.vocab.stoi
    source_batch_indexed = [[source_vocab[token] for token in sent.split()] for sent in TEST_CASE_src]
    source_batch_indexed = torch.LongTensor(source_batch_indexed).to(device)
    # print(source_batch_indexed)

    target_voab = TargetField.vocab.stoi
    target_batch_indexed = [[target_voab[token] for token in sent.split()] for sent in TEST_CASE_tgt]
    target_batch_indexed = torch.LongTensor(target_batch_indexed).to(device)

    source_seq = source_batch_indexed
    target_seq = target_batch_indexed

    seq2seq_model.eval()
    with torch.no_grad():

        output = seq2seq_model.infer(source_seq, target_seq)
        # target = target_seq[:, 1:]
        # target = target.contiguous().view(-1)
        # output = output[:, 1:, :]
        # output = output.contiguous().view(-1, output.shape[2])
        # topv, topi = output.topk(k=1)
        # topi = topi.squeeze(-1)
        # print(topi)
        # print(topi.shape)

        # print("out ---")
        # print(output)

        words = [TargetField.vocab.itos[word_idx] for word_idx in output]
        print("Predicted -> {}".format(" ".join(words)))


if __name__ == "__main__":
    tr, vl, src, tgt = preprocess()
    train(tr, vl, src, tgt)
