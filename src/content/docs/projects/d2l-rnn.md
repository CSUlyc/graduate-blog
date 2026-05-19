---
title: D2L / RNN 实验探索
description: 记录我学习 D2L 第 8 章序列模型与 RNN 的过程、实验和调参结果
---

这篇文章记录我在学习《动手学深度学习》第 8 章时，对序列模型和 RNN 的理解、实验过程以及遇到的问题。

## 研究背景

我主要关注的问题是：RNN 如何处理序列数据，以及不同超参数对训练效果有什么影响。

## 实验环境

- 平台：Kaggle / PAI-DSW
- 框架：PyTorch
- 数据集：时间机器数据集
- 任务：字符级语言模型预测

## 核心实验

### 实验一：不同隐藏单元数的影响

我设置了不同的 `num_hiddens`，观察 perplexity 的变化。

| num_hiddens | learning rate | final perplexity |
|---:|---:|---:|
| 512 | 1.0 | 5.05 |
| 512 | 0.5 | 6.85 |
| 512 | 0.1 | 10.48 |

从结果看，学习率为 `1.0` 时模型收敛效果最好。

## 代码片段

```python
num_hiddens = 512
lr = 1.0
num_epochs = 500

model = RNNModelScratch(
    len(vocab),
    num_hiddens,
    device,
    get_params,
    init_rnn_state,
    rnn
)