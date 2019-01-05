#!/usr/bin/env python
# encoding: utf-8

from brief import BriefModel
import torch


brief_model = BriefModel(path="/data_hdd/brief_dev/checkpoint_exp", device="cuda")


TEST_CASE_tgt = ["<sos>"]
a = """
滁 州 市 气 象 台   2015   年   07   月   12   日   15   时   20   分 发 布 雷 电 黄 色 预 警 信 号 ： 目 前 我 市 西 部 有 较 强 对 流 云 团 向 东 南 方 向 移 动 ， 预 计   6   小 时 内 我 市 部 分 地 区 将 发 生 雷 电 活 动 ， 并 可 能 伴 有 短 时 强 降 水 、 大 风 、 局 部 冰 雹 等 强 对 流 天 气 ， 请 注 意 防 范 。 图 例 标 准 防 御 指 南   6   小 时 内 可 能 发 生 雷 电 活 动 ， 可 能 会 造 成 雷 电 灾 害 事 故 。   1   、 政 府 及 相 关 部 门 按 照 职 责 做 好 防 雷 工 作 ；   2   、 密 切 关 注 天 气 ， 尽 量 避 免 户 外 活 动 。
"""
TEST_CASE_src = [a]

def test_step(SourceField, TargetField, seq2seq_model):
    source_vocab = SourceField.vocab.stoi
    source_batch_indexed = [[source_vocab[token] for token in sent.split()] for sent in TEST_CASE_src]
    source_batch_indexed = torch.LongTensor(source_batch_indexed).to("cpu")

    target_voab = TargetField.vocab.stoi
    target_batch_indexed = [[target_voab[token] for token in sent.split()] for sent in TEST_CASE_tgt]
    target_batch_indexed = torch.LongTensor(target_batch_indexed).to("cpu")

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

brief_model.brief(TEST_CASE_src)
