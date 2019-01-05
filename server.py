#!/usr/bin/env python
# encoding: utf-8

from brief import BriefModel

brief_model = BriefModel(path="/data_hdd/brief_dev/checkpoint_exp", device="cuda")

text = ["""
滁 州 市 气 象 台   2015   年   07   月   12   日   15   时   20   分 发 布 雷 电 黄 色 预 警 信 号 ： 目 前 我 市 西 部 有 较 强 对 流 云 团 向 东 南 方 向 移 动 ， 预 计   6   小 时 内 我 市 部 分 地 区 将 发 生 雷 电 活 动 ， 并 可 能 伴 有 短 时 强 降 水 、 大 风 、 局 部 冰 雹 等 强 对 流 天 气 ， 请 注 意 防 范 。 图 例 标 准 防 御 指 南   6   小 时 内 可 能 发 生 雷 电 活 动 ， 可 能 会 造 成 雷 电 灾 害 事 故 。   1   、 政 府 及 相 关 部 门 按 照 职 责 做 好 防 雷 工 作 ；   2   、 密 切 关 注 天 气 ， 尽 量 避 免 户 外 活 动 。
"""]

brief_model.brief(text)
