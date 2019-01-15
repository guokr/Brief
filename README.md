<h1 align="center">Brief</h1>

Brief is a text summarizer based on sequence to sequence framework, implemented in Python and Facebook's <a href="https://pytorch.org/">PyTorch</a> project. In a nutshell, this is Brief.

<p align="center">
  <a href="https://pypi.org/project/brief/">
      <img src="https://img.shields.io/pypi/v/brief.svg?colorB=brightgreen"
           alt="Pypi package">
    </a>
  <a href="https://github.com/guokr/brief/releases">
      <img src="https://img.shields.io/github/release/guokr/brief.svg"
           alt="GitHub release">
  </a>
  <a href="https://github.com/guokr/brief/issues">
        <img src="https://img.shields.io/github/issues/guokr/brief.svg"
             alt="GitHub issues">
  </a>
  <a href="https://travis-ci.org/guokr/Brief/">
    <img src="https://travis-ci.org/guokr/Brief.svg"
         alt="Travis CI">
  </a>
</p>

<p align="center">
  <a href="#quick-demo">Demo</a> •
  <a href="#requirements">Requirements</a> •
  <a href="#install">Install</a> •
  <a href="#did-you-guys-have-some-pre-trained-models">Pre-trained models</a> •
  <a href="#how-to-train-on-your-own-dataset">Train</a> •
  <a href="#more-examples">Examples</a> •
  <a href="https://guokr.github.io/Caver/">Document</a>
</p>

<h2 align="center">Quick demo</h2>

```python
from brief import BriefModel
model = BriefModel("./checkpoint_transformer")

long_text = ("为 期 三 个 月 的 全 国 公 路 执 法 专 项 整 改 工 作 刚 结 束 ，
              整 治 重 点 包 括 对 非 法 超 限 运 输 车 辆 只 收 费 不 卸 载 、 
              伙 同 社 会 闲 散 人 员 擅 自 放 行 等 。 然 而 ， 在 重 要 省 道 滨 唐 公 路 津 冀 交 界 处 ，
              执 法 治 超 沦 为 摆 设 ， 大 肆 收 费 后 擅 自 放 行 ， 超 载 问 题 严 重 失 控 。")
             
model.summarize([long_text])
>>> 津 冀 交 界 公 路 治 超 载 乱 象 严 重 ， 官 卡 执 法 沦 为 摆 设 。

long_text = ("眼 下 ， 白 酒 业 “ 塑 化 剂 门 ” 继 续 发 酵 ， 
              业 内 业 外 各 有 说 法 。 酒 鬼 酒 公 司 股 票 继 续 停 牌 。 
              记 者 走 访 郑 州 市 场 发 现 ， 商 家 并 未 将 酒 鬼 酒 和 其 他 白 酒 下 架 ， 
              白 酒 销 量 暂 时 稳 定 ， 未 受 影 响 。 专 家 提 醒 ： 塑 化 剂 溶 于 酒 精 ， 
              应 避 免 用 塑 料 制 品 盛 装 白 酒 。")
              
model.summarize([long_text])
>>> 河 南 白 酒 未 受 “ 塑 化 剂 门 ” 冲 击 ， 喝 酒 别 用 塑 料 杯 。
```

<h2 align="center">Requirements</h2>

* PyTorch
* tqdm
* torchtext
* dill
* numpy
* Python3

<h2 align="center">Install</h2>

```bash
$ pip install brief --user
```

<h2 align="center">Did you guys have some pre-trained models</h2>
Yes, we will release two pre-trained models on LCSTS dataset on word-level and char-level separately.

<h2 align="center">How to train on your own dataset</h2>

```bash
python3 train.py --train_filename train_full.tsv 
                 --valid_filename valid_full.tsv
                 --checkpoint_dir /data_hdd/brief_dev/checkpoint_exp
                 --epoch 10
                 --batch_size 128
```
<h2 align="center">More examples</h2>
Updating...
