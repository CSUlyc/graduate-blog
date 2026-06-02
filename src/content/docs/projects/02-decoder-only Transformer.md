---
title: "手搓 Tiny Transformer：从字符预测到注意力可视化"
description: "记录我在阿里云 DSW 上从零实现一个 Decoder-only Tiny Transformer 字符级语言模型的过程，包括数据构造、Embedding、Transformer Block、训练流程、自回归生成和注意力热力图分析。"
---

本文记录我在阿里云 DSW 上手搓一个 Tiny Transformer 的过程。前面学习了 Bahdanau 注意力、多头注意力、自注意力、位置编码和 Transformer 结构之后，我想进一步通过代码把这些模块真正串起来。

这次实现的目标不是训练一个大型模型，而是先做一个可以完整跑通的 **decoder-only Tiny Transformer 字符级语言模型**。它的任务很简单：

> 给定前面的字符，预测下一个字符。

例如：

```text
输入：我正在学习 Transforme
预测：r
```

通过这个小项目，我主要练习了以下内容：

- 字符级词表与 token 编号
- token embedding 与 position embedding
- 缩放点积注意力
- causal mask
- 多头自注意力
- 残差连接
- LayerNorm
- FeedForward 网络
- Transformer Block
- 交叉熵 loss 与参数更新
- 自回归文本生成
- 注意力权重可视化

---

## 1. 项目目标

这次手搓的是一个简化版 GPT 风格模型，也就是 **decoder-only Transformer**。

它和完整 Transformer 编码器-解码器不同，没有 encoder，也没有 encoder-decoder attention。它只保留 decoder 侧最核心的 masked self-attention，用来做自回归语言建模。

整体结构可以先概括为：

```text
token ids
↓
Token Embedding
↓
Position Embedding
↓
tok_emb + pos_emb
↓
Transformer Block × 4
↓
Final LayerNorm
↓
lm_head Linear
↓
logits
↓
预测下一个字符
```

我的理解是：

> 这个项目相当于把 Transformer 解码器中最核心的部分单独拿出来，用字符预测任务验证它是否真的可以学习序列规律。

---

## 2. 整体数据流概览

为了更直观地说明这个 Tiny Transformer 的完整流程，我先把模型从原始文本到训练、生成的整体数据流画成一张图。图中左侧是模型前向计算主干，右侧分别是训练分支和生成分支；下方补充了 Transformer Block 内部结构、关键张量形状和符号说明。

![Tiny Transformer 整体数据流与流程图](/images/tiny-transformer-flow.png)

这张图的阅读顺序可以概括为：

```text
原始文本
↓
字符级词表
↓
encode 编码成 token ids
↓
get_batch 构造训练输入 x 和目标 y
↓
Token Embedding + Position Embedding
↓
Transformer Blocks × 4
↓
Final LayerNorm
↓
lm_head 线性输出层
↓
logits
↓
训练时计算 loss，生成时采样下一个 token
```

需要特别注意的是，训练流程和生成流程并不完全一样。训练时使用 `get_batch` 同时得到输入 `x` 和目标 `y`，然后根据 `logits` 与 `targets y` 计算交叉熵损失；生成时没有 `targets y`，模型只根据当前上下文得到最后一个位置的 `logits`，再经过 softmax 采样下一个 token，并把新 token 拼接回输入序列继续预测。

这次模型有两条流程：一条是**训练流程**，一条是**生成流程**。这两条流程都经过同一个模型主体，但是输入和输出用途不同。

### 2.1 训练流程

训练时有真实标签 `y`，所以模型可以计算 loss 并更新参数。

```text
原始 text
↓
字符级词表 stoi / itos
↓
encode
↓
data: (N,)
↓
get_batch
↓
x: (B, T), y: (B, T)
↓
Token Embedding(x)
↓
Position Embedding
↓
x = tok_emb + pos_emb: (B, T, 64)
↓
Transformer Blocks × 4
↓
Final LayerNorm
↓
lm_head
↓
logits: (B, T, 73)
↓
logits + targets y
↓
Cross Entropy Loss
↓
backward()
↓
optimizer.step()
```

### 2.2 生成流程

生成时没有标签 `y`，模型只能根据当前上下文预测下一个 token，然后把预测结果拼接回输入序列，继续预测。

```text
start_text
↓
encode
↓
idx: 当前上下文 token ids
↓
Token Embedding
↓
Position Embedding
↓
Transformer Blocks × 4
↓
Final LayerNorm
↓
lm_head
↓
logits
↓
取最后一个位置 logits
↓
softmax
↓
采样下一个 token
↓
拼接回 idx
↓
继续下一轮预测
```

需要注意的是：

> `get_batch` 只属于训练流程。生成阶段不会随机采样训练片段，也不需要标签 `y`，它只是不断把预测出的 token 拼接回当前上下文。

---

## 3. 实验环境

实验在阿里云 DSW 中完成，使用 PyTorch 实现。

检查环境代码如下：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    device = "cuda"
else:
    device = "cpu"

print("当前设备:", device)
```

本次实验使用 GPU 运行，训练速度较快。

---

## 4. 训练数据

为了先跑通模型，我没有下载大型语料，而是直接构造了一小段和注意力机制相关的中文文本：

```text
我正在学习深度学习。
我正在学习注意力机制。
我正在学习 Transformer。
Transformer 使用自注意力机制。
自注意力可以让序列中的每个位置关注其他位置。
多头注意力可以从多个子空间学习不同关系。
位置编码可以让模型知道每个 token 的顺序。
解码器需要使用因果掩码，防止模型看到未来 token。
```

原始文本长度为：

```text
162
```

为了方便训练演示，我把这段文本重复了多次，使模型可以采样到更多 batch。

需要注意的是：

> 因为训练数据很短，并且被重复了很多次，所以这个模型的目标不是泛化，而是验证 Transformer 的完整训练流程。

---

## 5. 字符级词表构建

这次模型使用字符级建模，也就是把每一个汉字、字母、标点都看作一个 token。

构建词表的代码如下：

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
```

其中：

| 名称 | 含义 |
|---|---|
| `stoi` | string to index，字符到编号 |
| `itos` | index to string，编号到字符 |
| `vocab_size` | 字符表大小 |

本次实验得到的词表大小为：

```text
vocab_size = 73
```

也就是说，模型最终输出层要对 73 个字符分别给出预测分数。

---

## 6. 编码和解码函数

为了让模型处理文本，需要先把字符转成编号。

```python
def encode(s):
    return [stoi[ch] for ch in s]

def decode(ids):
    return ''.join([itos[i] for i in ids])
```

例如：

```text
原始文本：我正在学习
编码结果：[若干 token 编号]
解码结果：我正在学习
```

我的理解是：

> `encode` 负责把文本变成模型可以处理的 token ids，`decode` 负责把模型生成的 token ids 还原成人类可读文本。

---

## 7. 训练样本构造：get_batch

语言模型的训练目标是：

> 用当前位置之前的 token，预测下一个 token。

例如原始序列是：

```text
我 正 在 学 习
```

那么训练样本可以构造成：

```text
输入 x：我 正 在 学
标签 y：正 在 学 习
```

也就是说，`x` 和 `y` 是错开一位的。

代码中使用 `block_size` 控制每条训练序列的长度：

```python
batch_size = 32
block_size = 32

def get_batch(split):
    data_source = train_data if split == "train" else val_data

    ix = torch.randint(0, len(data_source) - block_size - 1, (batch_size,))

    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])

    return x.to(device), y.to(device)
```

实际采样结果示例：

```text
输入文本:
注意力机制。
自注意力可以让序列

标签文本:
意力机制。
自注意力可以让序列中
```

这个结果说明训练目标是正确的：

> 每个位置都要预测下一个字符。

在本次实验中：

```text
x.shape = (32, 32)
y.shape = (32, 32)
```

含义是：

```text
32 条样本
每条样本长度 32
```

---

## 8. Token Embedding：字符编号变成 64 维向量

模型不能直接处理 token id，因为 token id 只是编号，本身没有语义大小关系。

例如：

```text
“我” → 42
“学” → 38
```

这里的 42 和 38 只是词表索引，不代表“我”比“学”更大或更重要。

所以需要 token embedding：

```python
self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
```

在本次实验中：

```text
vocab_size = 73
n_embd = 64
```

因此它创建了一张可训练表：

```text
token_embedding_table: (73, 64)
```

意思是：

```text
73 个字符
每个字符一个 64 维向量
```

在前向传播中：

```python
tok_emb = self.token_embedding_table(idx)
```

如果：

```text
idx.shape = (32, 32)
```

那么：

```text
tok_emb.shape = (32, 32, 64)
```

可以理解为：

> batch 中每个字符编号都被查表转换成了一个 64 维向量。

例如输入：

```text
自注意力可以让序列
```

其中“自”“注”“意”等字符会分别查表得到自己的 64 维 token embedding。

这些向量一开始是随机初始化的，训练过程中会通过反向传播不断更新。

---

## 9. Position Embedding：位置编号变成 64 维向量

Transformer 的 self-attention 本身不按顺序处理序列。如果没有位置编码，模型只知道有哪些字符，不容易知道它们的先后顺序。

所以需要 position embedding：

```python
self.position_embedding_table = nn.Embedding(block_size, n_embd)
```

在本次实验中：

```text
block_size = 32
n_embd = 64
```

所以它创建了一张位置表：

```text
position_embedding_table: (32, 64)
```

含义是：

```text
最多支持 32 个位置
每个位置对应一个 64 维位置向量
```

在前向传播中：

```python
pos = torch.arange(T, device=idx.device)
pos_emb = self.position_embedding_table(pos)
```

如果当前序列长度为 9：

```text
自注意力可以让序列
```

那么：

```text
pos = [0, 1, 2, 3, 4, 5, 6, 7, 8]
pos_emb.shape = (9, 64)
```

如果训练时 `T = 32`，那么：

```text
pos_emb.shape = (32, 64)
```

这里用的是**可学习位置编码**，它和教材里的正弦余弦位置编码不同。

| 类型 | 怎么产生 | 是否训练 | 本项目是否使用 |
|---|---|---|---|
| 可学习位置编码 | `nn.Embedding(block_size, n_embd)` 随机初始化 | 会训练 | 是 |
| 正弦余弦位置编码 | 用 sin/cos 公式计算 | 通常固定 | 否 |

我的理解是：

> token embedding 回答“这个字符是什么”，position embedding 回答“这个字符在哪里”。

---

## 10. `x = tok_emb + pos_emb`：字符信息和位置信息相加

模型中有这句：

```python
x = tok_emb + pos_emb
```

其中：

```text
tok_emb: (B, T, 64)
pos_emb: (T, 64)
```

PyTorch 会自动广播，把同一套位置向量加到 batch 中的每一条样本上。

所以结果是：

```text
x.shape = (B, T, 64)
```

在本次训练中：

```text
x.shape = (32, 32, 64)
```

对于某个位置来说：

```text
x_i = token_embedding(token_i) + position_embedding(i)
```

例如：

```text
x_0 = token_emb("自") + pos_emb(0)
x_1 = token_emb("注") + pos_emb(1)
x_2 = token_emb("意") + pos_emb(2)
```

这样每个位置的输入向量就同时包含：

```text
这个 token 是什么
这个 token 在第几个位置
```

如果只看一个 3 维的简化例子：

```text
token_emb("自") = [0.20, 0.50, -0.10]
pos_emb(0)      = [0.01, 0.03,  0.02]

x_0 = [0.21, 0.53, -0.08]
```

真实模型中不是 3 维，而是 64 维。

---

## 11. 单头自注意力 Head

在进入完整 Transformer Block 前，我先实现了单个 masked self-attention head。

核心流程是：

```text
输入 x
↓
生成 Q、K、V
↓
计算 QK^T / sqrt(d)
↓
使用 causal mask 屏蔽未来位置
↓
softmax 得到注意力权重
↓
注意力权重乘 V
↓
输出
```

核心代码如下：

```python
class Head(nn.Module):
    """单个 masked self-attention head"""

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
        self.last_attn = None

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        self.last_attn = wei.detach()

        out = wei @ v
        return out
```

在 `Head.forward(self, x)` 里，输入的 `x` 不是原始文本，也不是 token id，而是已经加上位置编码后的向量表示：

```text
x.shape = (B, T, 64)
```

在本次实验中：

```text
x.shape = (32, 32, 64)
```

---

## 12. Q、K、V 是怎么来的

在单头注意力中有三层线性变换：

```python
self.key = nn.Linear(n_embd, head_size, bias=False)
self.query = nn.Linear(n_embd, head_size, bias=False)
self.value = nn.Linear(n_embd, head_size, bias=False)
```

本次设置：

```text
n_embd = 64
head_size = 16
```

所以每个线性层都是：

```text
Linear(64 → 16)
```

前向传播时：

```python
k = self.key(x)
q = self.query(x)
v = self.value(x)
```

形状变化为：

```text
x: (B, T, 64)
q: (B, T, 16)
k: (B, T, 16)
v: (B, T, 16)
```

可以这样理解：

| 向量 | 作用 |
|---|---|
| Query | 当前字符想找什么信息 |
| Key | 当前字符能被别人如何匹配 |
| Value | 当前字符真正提供的内容 |

例如位置 6 是“让”：

```text
q_让：表示“让”这个位置想找什么历史信息
```

位置 2 是“意”：

```text
k_意：表示“意”这个位置有什么特征可以被匹配
v_意：表示“意”这个位置真正提供的信息
```

---

## 13. 注意力得分和 causal mask

注意力得分通过下面这句计算：

```python
wei = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
```

其中：

```text
q: (B, T, 16)
k.transpose(-2, -1): (B, 16, T)
```

所以：

```text
wei: (B, T, T)
```

如果当前序列长度是 9，那么每个样本的注意力矩阵是：

```text
wei: (9, 9)
```

它表示每个位置对每个位置的注意力得分。

例如：

```text
wei[6, 2]
```

表示：

> Query 位置 6 “让” 对 Key 位置 2 “意” 的匹配分数。

这里除以：

```text
sqrt(head_size) = sqrt(16) = 4
```

是为了让点积结果更稳定，避免 softmax 过于极端。

接下来使用 causal mask：

```python
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
```

`self.tril` 是下三角矩阵。如果序列长度为 9，它的含义是：

| Query 位置 | 可以关注的 Key 位置 |
|---:|---|
| 0 | 0 |
| 1 | 0, 1 |
| 2 | 0, 1, 2 |
| 3 | 0, 1, 2, 3 |
| 4 | 0, 1, 2, 3, 4 |
| 5 | 0, 1, 2, 3, 4, 5 |
| 6 | 0, 1, 2, 3, 4, 5, 6 |
| 7 | 0, 1, 2, 3, 4, 5, 6, 7 |
| 8 | 0, 1, 2, 3, 4, 5, 6, 7, 8 |

以位置 6 “让”为例：

```text
自 注 意 力 可 以 让 序 列
0  1  2  3  4  5  6  7  8
```

当模型处理“让”时，它可以关注：

```text
自、注、意、力、可、以、让
```

不能关注：

```text
序、列
```

这一步保证 decoder-only 模型不会偷看未来字符。

---

## 14. softmax 和 Value 加权求和

mask 之后使用 softmax：

```python
wei = F.softmax(wei, dim=-1)
```

softmax 会把注意力得分变成权重。

例如对于位置 6 “让”，可能得到：

| Key 位置 | 字符 | 注意力权重 |
|---:|---|---:|
| 0 | 自 | 0.17 |
| 1 | 注 | 0.08 |
| 2 | 意 | 0.38 |
| 3 | 力 | 0.04 |
| 4 | 可 | 0.06 |
| 5 | 以 | 0.11 |
| 6 | 让 | 0.24 |
| 7 | 序 | 0.00 |
| 8 | 列 | 0.00 |

未来位置“序”“列”的权重是 0，因为它们被 causal mask 屏蔽了。

最后：

```python
out = wei @ v
```

对于位置 6 “让”，可以理解为：

```text
out_让 =
0.17 * v_自
+ 0.08 * v_注
+ 0.38 * v_意
+ 0.04 * v_力
+ 0.06 * v_可
+ 0.11 * v_以
+ 0.24 * v_让
+ 0.00 * v_序
+ 0.00 * v_列
```

这就是注意力的核心：

> 用注意力权重，对历史位置的 Value 向量做加权求和。

因此，`out_让` 不再只是“让”自己的信息，而是融合了前面历史字符的信息。

---

## 15. 单头注意力测试结果

测试输入形状为：

```text
xb shape: torch.Size([4, 16])
x_emb shape: torch.Size([4, 16, 64])
```

经过单头注意力后：

```text
attention output shape: torch.Size([4, 16, 16])
attention weights shape: torch.Size([4, 16, 16])
```

含义如下：

| 张量 | 形状 | 含义 |
|---|---|---|
| `xb` | `(4, 16)` | 4 条样本，每条 16 个 token |
| `x_emb` | `(4, 16, 64)` | 每个 token 变成 64 维向量 |
| `out` | `(4, 16, 16)` | 单个 head 输出，每个位置 16 维 |
| `last_attn` | `(4, 16, 16)` | 每个位置对其他位置的注意力权重 |

注意力矩阵中右上角为 0，说明 causal mask 生效。

---

## 16. 多头注意力 Multi-Head Attention

单个注意力头只能从一个子空间学习关系，所以我又实现了多头注意力。

本次设置：

```text
n_embd = 64
num_heads = 4
head_size = 16
```

也就是：

```text
4 个 head × 每个 head 16 维 = 64 维
```

实现代码如下：

```python
class MultiHeadAttention(nn.Module):
    """多个 masked self-attention head 并行"""

    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()

        assert n_embd % num_heads == 0

        head_size = n_embd // num_heads

        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
```

每个 Head 输出：

```text
(B, T, 16)
```

4 个 Head 拼接后：

```text
(B, T, 64)
```

再经过 `self.proj` 输出投影层，仍然保持：

```text
(B, T, 64)
```

我的理解是：

> 多头注意力不是简单重复计算，而是让不同 Head 在不同子空间中学习不同的注意力模式。

---

## 17. 前馈网络 FeedForward

Transformer Block 中除了注意力层，还有前馈网络。

实现如下：

```python
class FeedForward(nn.Module):
    """Transformer 中的前馈网络"""

    def __init__(self, n_embd, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

这里使用了：

```text
Linear(64 → 256)
ReLU
Linear(256 → 64)
Dropout
```

输入输出形状都是：

```text
(B, T, 64)
```

我的理解是：

> 注意力层负责不同位置之间的信息交互，FeedForward 负责对每个位置的表示进行非线性变换。

也就是说：

```text
Self-Attention：决定“看谁”
FeedForward：加工“看完后得到的表示”
```

---

## 18. Transformer Block：Pre-LN 与残差连接

一个 Transformer Block 由以下部分组成：

```text
LayerNorm
↓
Masked Multi-Head Self-Attention
↓
Residual Add
↓
LayerNorm
↓
FeedForward
↓
Residual Add
```

本次实现采用 Pre-LN 写法：

```python
class Block(nn.Module):
    """一个 Transformer decoder block"""

    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()

        self.sa = MultiHeadAttention(n_embd, num_heads, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

这两句是 Block 的核心：

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

---

## 19. `x = x + self.sa(self.ln1(x))` 做了什么

这句可以拆成三步：

```text
self.ln1(x)
↓
self.sa(...)
↓
x + attention_output
```

第一步：

```python
self.ln1(x)
```

`ln1` 是：

```python
nn.LayerNorm(n_embd)
```

也就是：

```text
LayerNorm(64)
```

输入输出形状不变：

```text
(B, T, 64) → (B, T, 64)
```

它的作用是稳定每个 token 的 64 维向量分布，让训练更稳定。

第二步：

```python
self.sa(self.ln1(x))
```

`sa` 是多头 masked self-attention。它会让每个位置从自己和历史位置中读取信息。

输入输出形状依然是：

```text
(B, T, 64) → (B, T, 64)
```

第三步：

```python
x = x + self.sa(self.ln1(x))
```

这是残差连接。

对于位置 6 “让”来说，可以理解为：

```text
新的 x_让 = 原来的 x_让 + 注意力层从历史字符中提取到的信息
```

残差连接的意义是：

```text
保留原始表示
同时加入上下文信息
让深层网络更容易训练
```

如果没有残差连接，每层都会直接替换掉原始信息；有残差连接后，模型可以在原有信息基础上逐步补充新信息。

---

## 20. `x = x + self.ffwd(self.ln2(x))` 做了什么

第二句结构类似：

```python
x = x + self.ffwd(self.ln2(x))
```

它可以拆成：

```text
LayerNorm
↓
FeedForward
↓
Residual Add
```

其中 FeedForward 做的是：

```text
Linear(64 → 256)
↓
ReLU
↓
Linear(256 → 64)
```

它不会直接混合不同位置的信息，而是对每个位置自己的 64 维向量做非线性加工。

我的理解是：

> 一个 Block 就是一轮“历史信息交互 + 单点特征加工”。

具体来说：

```text
第一句：用 self-attention 让每个位置读取历史信息
第二句：用 FeedForward 加工每个位置融合后的表示
```

Block 的输入输出形状保持不变：

```text
输入: (B, T, 64)
输出: (B, T, 64)
```

这样才能连续堆叠 4 个 Block。

---

## 21. `x = self.blocks(x)` 做了什么

完整模型里有：

```python
self.blocks = nn.Sequential(*[
    Block(n_embd, num_heads, block_size, dropout)
    for _ in range(num_layers)
])
```

本次设置：

```text
num_layers = 4
```

所以：

```python
x = self.blocks(x)
```

等价于：

```python
x = block1(x)
x = block2(x)
x = block3(x)
x = block4(x)
```

每个 Block 的输入输出形状都是：

```text
(B, T, 64)
```

所以：

```text
进入 blocks 前: (32, 32, 64)
经过 Block 1:  (32, 32, 64)
经过 Block 2:  (32, 32, 64)
经过 Block 3:  (32, 32, 64)
经过 Block 4:  (32, 32, 64)
```

形状不变，但是内容不断变化。

刚进入 Blocks 前，`x` 主要包含：

```text
字符信息 + 位置信息
```

经过 4 层 Block 后，`x` 变成：

```text
融合历史上下文后的字符表示
```

例如位置 6 的“让”，一开始只知道：

```text
我是“让”
我在位置 6
```

经过多层 masked self-attention 后，它可以融合前面的：

```text
自、注、意、力、可、以、让
```

因此，最终的 `x_让` 会更适合预测下一个字符“序”。

---

## 22. 完整 Tiny Transformer 模型

完整模型包含：

```text
Token Embedding
Position Embedding
Transformer Block × 4
Final LayerNorm
Linear 输出层
```

模型代码如下：

```python
class TinyTransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, num_heads, num_layers, block_size, dropout):
        super().__init__()

        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[
            Block(n_embd, num_heads, block_size, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)

        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)

        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape

            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss
```

完整前向传播的数据形状为：

```text
idx:      (B, T)
tok_emb:  (B, T, 64)
pos_emb:  (T, 64)
x:        (B, T, 64)
blocks:   (B, T, 64)
ln_f:     (B, T, 64)
logits:   (B, T, 73)
```

---

## 23. Final LayerNorm 与 lm_head

经过 4 个 Transformer Block 后，模型会执行：

```python
x = self.ln_f(x)
logits = self.lm_head(x)
```

其中：

```python
self.ln_f = nn.LayerNorm(n_embd)
```

作用是在输出层之前再做一次归一化，让最后的隐藏表示更稳定。

`lm_head` 是：

```python
self.lm_head = nn.Linear(n_embd, vocab_size)
```

本次参数为：

```text
Linear(64 → 73)
```

输入：

```text
x: (B, T, 64)
```

输出：

```text
logits: (B, T, 73)
```

每个位置都会得到一个 73 维向量，表示模型对 73 个字符的预测分数。

例如某个位置的 logits 可以理解为：

```text
“序” 的分数：8.2
“可” 的分数：1.1
“注” 的分数：-0.5
“。” 的分数：0.3
...
```

训练时，模型会根据真实下一个字符计算交叉熵 loss。

---

## 24. loss 计算

模型输出的 logits 形状是：

```text
logits: (B, T, vocab_size)
```

在本次实验中：

```text
logits: (32, 32, 73)
```

标签形状是：

```text
targets: (32, 32)
```

但是 `F.cross_entropy` 需要输入形状为：

```text
(样本数, 类别数)
```

因此代码中做了展平：

```python
B, T, C = logits.shape

logits_flat = logits.view(B * T, C)
targets_flat = targets.view(B * T)

loss = F.cross_entropy(logits_flat, targets_flat)
```

展平后：

```text
logits_flat:  (1024, 73)
targets_flat: (1024,)
```

这里：

```text
1024 = 32 × 32
```

表示一个 batch 中共有 1024 个位置，每个位置都做一次“下一个字符分类”。

loss 的意义是：

> 衡量模型预测的下一个字符和真实下一个字符之间的差距。

---

## 25. 自回归生成函数

模型训练完成后，需要能够自己生成文本。

生成逻辑是：

```text
输入已有 token
↓
取最后 block_size 个 token
↓
模型预测下一个 token 概率
↓
按概率采样一个 token
↓
拼接到序列后面
↓
重复
```

代码如下：

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.block_size:]

        logits, loss = self(idx_cond)

        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

生成时最关键的是：

```python
logits = logits[:, -1, :]
```

因为生成阶段只关心当前上下文最后一个位置预测出的下一个 token。

例如输入：

```text
我正在学习
```

模型预测下一个 token 可能是：

```text
T
```

拼接后变成：

```text
我正在学习T
```

然后继续预测：

```text
我正在学习Tr
```

不断重复，最终生成完整文本。

---

## 26. 模型参数设置

本次实验使用的主要参数如下：

| 参数 | 数值 |
|---|---:|
| `vocab_size` | 73 |
| `block_size` | 32 |
| `batch_size` | 32 |
| `n_embd` | 64 |
| `num_heads` | 4 |
| `head_size` | 16 |
| `num_layers` | 4 |
| `dropout` | 0.1 |
| `learning rate` | 1e-3 |
| `max_iters` | 1000 |

模型打印结果如下：

```text
TinyTransformerLanguageModel(
  (token_embedding_table): Embedding(73, 64)
  (position_embedding_table): Embedding(32, 64)
  (blocks): Sequential(
    Transformer Block × 4
  )
  (ln_f): LayerNorm((64,))
  (lm_head): Linear(in_features=64, out_features=73)
)
```

模型总参数量为：

```text
210,761
```

---

## 27. 参数量分析

### 27.1 Token Embedding

```text
Embedding(73, 64)
```

参数量：

```text
73 × 64 = 4,672
```

### 27.2 Position Embedding

```text
Embedding(32, 64)
```

参数量：

```text
32 × 64 = 2,048
```

### 27.3 单个 Transformer Block

每个 Block 包含：

```text
MultiHeadAttention
FeedForward
LayerNorm × 2
```

多头注意力部分：

```text
4 个 Head
每个 Head 有 key、query、value 三个 Linear(64 → 16)
```

每个 Head 参数量：

```text
64 × 16 × 3 = 3,072
```

4 个 Head：

```text
3,072 × 4 = 12,288
```

输出投影层：

```text
Linear(64 → 64)
参数量 = 64 × 64 + 64 = 4,160
```

所以多头注意力部分参数量：

```text
12,288 + 4,160 = 16,448
```

FeedForward 部分：

```text
Linear(64 → 256): 64 × 256 + 256 = 16,640
Linear(256 → 64): 256 × 64 + 64 = 16,448
```

FeedForward 合计：

```text
33,088
```

两个 LayerNorm：

```text
2 × (64 + 64) = 256
```

单个 Block 总参数量：

```text
16,448 + 33,088 + 256 = 49,792
```

4 个 Block：

```text
49,792 × 4 = 199,168
```

### 27.4 输出层

```text
Linear(64 → 73)
```

参数量：

```text
64 × 73 + 73 = 4,745
```

### 27.5 总参数量

| 模块 | 参数量 |
|---|---:|
| Token Embedding | 4,672 |
| Position Embedding | 2,048 |
| Transformer Block × 4 | 199,168 |
| Final LayerNorm | 128 |
| 输出层 lm_head | 4,745 |
| **总计** | **210,761** |

---

## 28. 训练结果

训练前，模型第一次前向传播得到：

```text
logits shape: torch.Size([32, 32, 73])
loss: 4.4363
```

其中：

| 维度 | 含义 |
|---|---|
| 32 | batch size |
| 32 | 序列长度 |
| 73 | 词表大小 |

训练 1000 步后的 loss 变化如下：

| step | train loss | val loss |
|---:|---:|---:|
| 0 | 4.4451 | 4.4414 |
| 100 | 0.2276 | 0.2285 |
| 200 | 0.0643 | 0.0638 |
| 300 | 0.0480 | 0.0501 |
| 400 | 0.0464 | 0.0469 |
| 500 | 0.0428 | 0.0410 |
| 600 | 0.0403 | 0.0425 |
| 700 | 0.0400 | 0.0386 |
| 800 | 0.0356 | 0.0377 |
| 900 | 0.0365 | 0.0399 |
| 1000 | 0.0351 | 0.0357 |

可以看到，loss 从 4.4 左右迅速下降到 0.03 左右。

我的理解是：

> 因为训练文本很短，并且重复了很多次，所以模型很快记住了文本模式。这个结果说明模型的数据流和训练流程是正确的，但不代表它具备真正的大规模泛化能力。

---

## 29. 文本生成结果

以：

```text
我正在学习
```

作为开头，模型生成了下面的内容：

```text
我正在学习 Transformer。
Transformer 使用自注意力机制。
自注意力可以让序列中的每个位置关注其他位置。
多头注意力可以从多个子空间学习不同关系。
位置编码可以让模型知道每个 token
```

生成结果基本符合训练文本内容。

这说明模型已经学到了训练文本中的字符顺序和局部模式。

---

## 30. 注意力可视化

训练完成后，我进一步查看了模型内部的注意力权重。

输入序列为：

```text
自注意力可以让序列
```

位置编号如下：

| 位置 | 字符 |
|---:|---|
| 0 | 自 |
| 1 | 注 |
| 2 | 意 |
| 3 | 力 |
| 4 | 可 |
| 5 | 以 |
| 6 | 让 |
| 7 | 序 |
| 8 | 列 |

注意力热力图中：

> 横轴是 Key position，表示被关注的位置。  
> 纵轴是 Query position，表示当前正在更新的位置。  
> 颜色越亮，表示注意力权重越大。

---

## 31. Block 1 注意力图理解

第一张图是 Block 1 的 4 个注意力头。

Block 1 靠近输入层，此时输入主要还是：

```text
字符 embedding + 位置 embedding
```

所以 Block 1 的注意力模式更偏底层。

可以观察到：

- 有些 Head 更关注当前位置附近。
- 有些 Head 更关注句首位置。
- 有些 Head 沿着对角线分布，表示关注自己或相邻位置。
- 图的右上角基本为空，说明 causal mask 生效。

我的理解是：

> Block 1 主要在学习比较基础的局部依赖和位置关系。

---

## 32. Block 4 注意力图理解

第二张图是 Block 4 的 4 个注意力头。

Block 4 已经经过前面多层处理，输入表示已经融合了一定上下文信息。

相比 Block 1，Block 4 中有些 Head 的注意力更分散，也出现了一些跨位置关注。

可以观察到：

- 有些 Head 仍然关注句首。
- 有些 Head 关注中间关键位置。
- 有些 Head 不只关注前一个字符，而是跨多个历史位置建立联系。

我的理解是：

> 高层注意力不只是简单看相邻字符，而是在已经加工过的表示上选择更有用的历史信息。

---

## 33. 图中一个点的具体解释

以 Block 4 Head 4 中一个较亮的点为例。

假设这个点位于：

```text
Query pos = 6
Key pos = 2
```

根据位置表：

```text
Query pos 6 = 让
Key pos 2 = 意
```

这个点表示：

> 在第 4 层第 4 个注意力头中，模型在更新“让”这个位置的表示时，比较关注前面的“意”这个位置。

也就是说，当模型处理到：

```text
自注意力可以让
```

中的“让”时，它不是只看当前字符，也不是只看前一个字符“以”，而是把一部分注意力放到了更前面的“意”上。

不过由于模型很小、数据也很短，所以不能过度解释为真正的语义理解。更合理的说法是：

> 这个点说明模型已经学会在高层中跨位置关注历史字符。

---

## 34. 两张注意力图整体说明

这两张图说明了几个现象：

1. **causal mask 生效**  
   右上角未来位置基本不能被关注。

2. **不同 Head 学到不同模式**  
   4 个注意力头的亮点分布不同。

3. **不同层关注模式不同**  
   Block 1 更偏底层和局部，Block 4 的注意力更复杂。

4. **模型可以利用历史上下文预测下一个字符**  
   这正是 decoder-only Transformer 的核心能力。

一句话概括：

> 横轴是“看谁”，纵轴是“谁在看”，颜色越亮说明关注越强；低层注意力更偏基础位置关系，高层注意力会出现更复杂的历史信息选择。

---

## 35. 本次手搓项目串联的知识点

这个项目把前面学过的 Transformer 知识串了起来。

| 理论知识 | 项目中的实现 |
|---|---|
| token 编号 | `stoi` 和 `itos` |
| Token Embedding | `nn.Embedding(vocab_size, n_embd)` |
| Position Embedding | `nn.Embedding(block_size, n_embd)` |
| Query、Key、Value | `self.query`、`self.key`、`self.value` |
| 缩放点积注意力 | `q @ k.transpose(-2, -1) / sqrt(d)` |
| causal mask | 下三角矩阵 `tril` |
| softmax 权重 | `F.softmax(wei, dim=-1)` |
| 多头注意力 | 多个 `Head` 拼接 |
| 输出投影 | `self.proj` |
| 前馈网络 | `FeedForward` |
| 残差连接 | `x = x + sublayer(...)` |
| LayerNorm | `nn.LayerNorm(n_embd)` |
| 语言模型输出 | `lm_head` |
| 交叉熵损失 | `F.cross_entropy` |
| 自回归生成 | `generate` 函数 |

---

## 36. 阶段性理解

最开始我只是从教材中看到 Transformer 的结构图，知道它包括多头注意力、位置编码、前馈网络、残差连接和层归一化。

这次手搓之后，我对这些模块之间的数据流有了更具体的理解。

现在我可以把一个字符从输入到输出的过程串起来：

```text
字符
↓
字符编号
↓
token embedding
↓
position embedding
↓
两者相加
↓
masked multi-head self-attention
↓
残差连接
↓
FeedForward
↓
残差连接
↓
Linear 输出层
↓
预测下一个字符
```

我对 decoder-only Transformer 的理解是：

> 它通过 causal mask 保证每个位置只能看历史信息，再通过多层 masked self-attention 不断融合上下文，最后根据当前位置的隐藏表示预测下一个 token。

---

## 37. 项目不足

这个项目仍然是一个很小的实验。

主要不足包括：

- 训练数据太短，模型主要是在记忆文本。
- 使用的是字符级建模，不能体现真正词级语义。
- 没有加入验证更强泛化能力的数据集。
- 位置编码使用的是可学习 embedding，没有实现正弦余弦版本。
- 没有实现完整 encoder-decoder Transformer。
- 没有加入学习率调度、梯度裁剪等训练技巧。

不过作为第一版手搓项目，它已经完整覆盖了 decoder-only Transformer 的核心模块。

---

## 38. 后续改进方向

后续可以继续扩展：

1. 使用更大的中文文本数据训练。
2. 从字符级建模改成词级或 BPE token 建模。
3. 实现正弦余弦位置编码并与可学习位置编码对比。
4. 可视化不同层、不同 Head 的注意力变化。
5. 加入学习率 warmup 和梯度裁剪。
6. 实现 encoder-decoder Transformer，用于机器翻译任务。
7. 对比 RNN 语言模型和 Transformer 语言模型的训练效果。

---

## 39. 总结

这次项目从零实现了一个可以训练、可以生成文本、可以可视化注意力权重的 Tiny Transformer。

最终模型配置为：

```text
vocab_size = 73
block_size = 32
n_embd = 64
num_heads = 4
num_layers = 4
参数量 = 210,761
```

训练结果为：

```text
初始 loss: 4.4363
step 1000 train loss: 0.0351
step 1000 val loss: 0.0357
```

生成结果能够复现训练文本中的主要模式。

这次手搓项目让我把 Transformer 的核心机制从公式和结构图落实到了代码中。尤其是 token embedding、position embedding、causal mask、多头注意力、残差连接、LayerNorm 和自回归生成这些部分，只有真正写一遍之后，才能更清楚地理解它们在模型中各自起什么作用。

用一句话总结：

> Tiny Transformer 的核心就是：每个位置通过 masked self-attention 只能读取历史信息，再经过多层注意力和前馈网络加工，最后预测下一个 token。
