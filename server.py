#!/usr/bin/env python
# encoding: utf-8

from brief import BriefModel

brief_model = BriefModel(path="/data_hdd/brief_dev/checkpoint_exp")

text = ("针 对 政 府 采 购 活 动 中 也 暴 露 出 质 量 不 高 、 效 率 低 下 等 问 题 ， \
        特 制 定 该 条 例 ， 完 善 政 府 采 购 制 度 ， 进 一 步 促 进 政 府 采 购 的 规 范 化 、 \
        法 制 化 ， 构 建 规 范 透 明 、 公 平 竞 争 、 监 督 到 位 、 严 格 问 责 的 政 府 采 购 工 作 机 制 。")


## “ 抗 癌 药 代 购 第 一 人 ” 已 获 释 检 方 主 动 撤 诉
brief_model.brief([text])
