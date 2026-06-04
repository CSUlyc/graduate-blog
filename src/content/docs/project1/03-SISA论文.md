---

title: "精读论文：《Forget Attention: Importance-Aware Attention Is All You Need》"
description: "记录我精读论文《Forget Attention: Importance-Aware Attention Is All You Need》的过程，包括 arXiv 论文检索、Zotero 文献管理、双语阅读、高亮规范、Attention/SSM/SDPA 基础理解，以及 SISA 方法和实验结果分析。"
----------------------------------------------------------------------------------------------------------------------------------------------------------------------

本文记录我第一次系统精读一篇大模型结构类论文的过程。论文标题为：

> **Forget Attention: Importance-Aware Attention Is All You Need**

这篇论文主要研究 **Attention 与 State Space Model（SSM）如何更深层次融合**。作者提出了 **SISA（SSM-Informed Softmax Attention）**，核心思想是在 Attention 的 score 计算中加入 SSM 产生的顺序重要性信号，让注意力分数不仅反映内容相似度，也反映序列动态中的重要性。

这次阅读过程不只是读懂一篇论文，也是在练习完整的论文精读流程，包括：

* 在 arXiv 上筛选论文
* 判断论文类型和阅读优先级
* 使用 Zotero 管理论文
* 使用双语翻译辅助理解
* 制定高亮颜色规范
* 拆解 Attention、SSM、SDPA 等基础概念
* 阅读 Related Work、Method、Experimental Setup 和 Results
* 形成可复用的论文笔记框架

---

论文结构
![SISA论文流程图](/images/SISA论文.png)

## 1. 为什么选择这篇论文

我在 arXiv 的 `cs.CL`、`cs.LG`、`cs.AI` 等分类中浏览论文时，看到了这篇：

```text
Forget Attention: Importance-Aware Attention Is All You Need
```

论文的关键词包括：

```text
Attention
SSM
Mamba
Hybrid Architecture
Long Context
Transformer
SISA
```

从标题和摘要可以判断，它不是普通的 LLM 应用论文，也不是 Agent 或 RAG 论文，而是偏向 **模型结构创新** 的方法类论文。

它主要讨论的问题是：

```text
Transformer 具备全局检索能力，但不一定能显式判断哪些 token 更重要；
SSM 能建模顺序中的重要性，但不擅长回头检索任意历史位置。
```

作者希望把二者结合起来：

```text
Attention 的全局检索能力
+
SSM 的顺序重要性建模能力
```

因此，这篇论文很适合作为我理解 Transformer 改进、SSM/Mamba、长上下文建模的精读材料。

---

## 2. 论文保存与文献管理

为了后续长期管理论文，我使用 Zotero 保存文献。

操作流程是：

```text
1. 打开 arXiv 摘要页
2. 安装 Zotero 桌面端
3. 安装 Zotero Connector 浏览器插件
4. 在 arXiv 的 /abs 页面点击 Zotero Connector
5. 保存到 Zotero 分类 Transformer_SSM
6. 检查是否自动保存 Full Text PDF
7. 给论文添加标签
```

最终在 Zotero 中形成如下结构：

```text
Transformer_SSM
└── Forget Attention: Importance-Aware Attention Is All You Need
    └── Full Text PDF
```

我给这篇论文添加的标签是：

```text
Attention
SSM
Mamba
Hybrid Architecture
Long Context
Transformer
SISA
精读
模型结构
```

这些标签方便之后按方向检索论文。

---

## 3. 我的论文高亮规范

精读论文时，如果整页都标黄，后面复习反而没有重点。因此我给自己制定了一套固定高亮规则：

| 颜色 | 标注内容                     |
| -- | ------------------------ |
| 黄色 | 研究背景、研究问题、动机             |
| 蓝色 | 核心方法、创新点                 |
| 紫色 | 公式、变量、关键定义               |
| 绿色 | 实验结果、关键结论                |
| 红色 | 不懂、存疑、需要回头查的地方           |
| 橙色 | Related Work、已有方法缺陷、对比方法 |

这篇论文中，我重点标注了：

```text
SISA
SSM-informed Softmax Attention
attention score
importance term
SDPA
NIAH
LAMBADA
score-level fusion
```

我的理解是：

> 高亮不是为了把论文变彩色，而是为了给后续复盘服务。每种颜色都应该对应一种阅读目的。

---

## 4. 中英文对照阅读方式

我采用了两条阅读线：

```text
浏览器 arXiv HTML + 沉浸式翻译：用于双语理解正文
Zotero PDF：用于正式高亮、批注和长期保存
```

具体流程是：

```text
1. 在 arXiv 页面打开 HTML experimental
2. 使用沉浸式翻译进行双语对照
3. 先理解 Abstract、Introduction、Method 和 Conclusion
4. 回到 Zotero PDF 进行正式高亮
5. 遇到公式或难懂段落再单独拆解
```

我发现对于这类结构类论文，不能只依赖中文翻译。很多关键词必须保留英文，例如：

```text
attention score
state space model
SSM-derived importance term
softmax attention
SDPA
query/key vectors
long-context retrieval
score-level fusion
```

这些词如果完全翻译成中文，后面读公式时容易对不上。

---

## 5. 论文核心问题

这篇论文的核心问题可以概括为：

```text
如何让 Attention 同时具备内容检索能力和顺序重要性判断能力？
```

普通 Transformer 的 Attention 通过 Q 和 K 的点积判断内容相似度：

```text
score_ij = q_i^T k_j / sqrt(d_h)
```

它回答的问题是：

```text
第 i 个 token 和第 j 个 token 在内容上是否相关？
```

但作者认为，仅仅判断内容相似度还不够。序列建模中还需要知道：

```text
第 j 个 token 在顺序动态中对第 i 个 token 是否重要？
```

这正是 SSM 擅长提供的信号。

---

## 6. Attention 是如何判断内容相似度的

在 Transformer 中，每个 token 会被映射成三个向量：

| 向量    | 直观含义             |
| ----- | ---------------- |
| Query | 我现在想找什么信息        |
| Key   | 我有什么特征，别人可以怎么找到我 |
| Value | 如果别人关注我，我能提供什么内容 |

Attention 的核心计算是：

```text
score_ij = q_i^T k_j
```

也就是当前位置的 Query 和其他位置的 Key 做点积。

点积越大，说明两个向量方向越接近，模型就认为它们越相关。

例如：

```text
q_他 = [0.9, 0.1]
k_小明 = [0.8, 0.2]
k_苹果 = [0.1, 0.9]
```

那么：

```text
q_他 · k_小明 = 0.9×0.8 + 0.1×0.2 = 0.74
q_他 · k_苹果 = 0.9×0.1 + 0.1×0.9 = 0.18
```

因此模型会更倾向于让“他”关注“小明”。

这些数字并不是人工设定的，而是模型通过训练学出来的。更准确地说：

```text
模型学习的是 W_Q、W_K、W_V、embedding 等参数；
q、k、v 是在当前输入下临时计算出来的中间结果。
```

公式为：

```text
q_i = x_i W_Q
k_i = x_i W_K
v_i = x_i W_V
```

训练过程可以理解为：

```text
预测错误
↓
loss 变大
↓
反向传播
↓
更新 W_Q、W_K、embedding 等参数
↓
让相关 token 的 Q-K 点积变大
↓
让不相关 token 的 Q-K 点积变小
```

所以 Attention 的学习过程本质上是：

> 让该匹配的 Query 和 Key 更匹配，不该匹配的 Query 和 Key 不匹配。

---

## 7. SSM 是什么

SSM 全称是：

```text
State Space Model
```

中文常叫：

```text
状态空间模型
```

在深度学习序列建模中，我可以先把它理解成：

> 一种按顺序处理序列，并不断维护隐藏状态的模型。

它和 Attention 的思路不同。

Transformer 的 Attention 是：

```text
每个 token 可以直接和其他 token 做两两比较。
```

而 SSM 是：

```text
按顺序读入 token，一边读一边更新隐藏状态 h。
```

最简形式可以写成：

```text
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

其中：

| 符号    | 含义           |
| ----- | ------------ |
| `x_t` | 当前输入         |
| `h_t` | 当前隐藏状态，也就是记忆 |
| `A`   | 旧状态如何保留      |
| `B`   | 当前输入如何写入状态   |
| `C`   | 如何从状态中读取信息   |
| `y_t` | 当前输出         |

直观理解：

```text
新记忆 = 保留一部分旧记忆 + 写入一部分新信息
```

Mamba 是现代 SSM 的代表结构之一。它通过选择机制动态决定哪些信息应该记住、哪些应该忘掉，因此可以更高效地处理长序列。

---

## 8. Attention 和 SSM 的差异

这篇论文中一句话非常关键：

```text
Transformers see everywhere but cannot prioritize;
SSMs know what matters but cannot revisit.
```

我对这句话的理解是：

| 模型                      | 优势              | 局限                  |
| ----------------------- | --------------- | ------------------- |
| Transformer / Attention | 能全局检索任意历史 token | 不一定显式知道哪些 token 更重要 |
| SSM / Mamba             | 能建模顺序动态和重要性     | 不擅长回头精准检索某个历史位置     |

所以作者提出：

```text
能不能让 Attention 在判断内容相似度的同时，也参考 SSM 的顺序重要性信号？
```

这就是 SISA 的出发点。

---

## 9. 三种 Attention-SSM 融合方式

论文中的核心图对比了三种融合方式：

```text
(a) Block fusion
(b) Head fusion
(c) Score fusion: SISA
```

### 9.1 Block Fusion

Block-level 融合是层级交替，例如：

```text
Transformer
↓
Mamba
↓
Transformer
↓
Mamba
```

这种方式的问题是：

```text
Attention 层在计算 attention score 时，并不知道 SSM 认为哪些 token 重要。
```

它们只是层与层之间传递输出。

### 9.2 Head Fusion

Head-level 融合是在同一层里同时使用 Attention head 和 SSM head：

```text
Attn head
Attn head
Attn head
SSM head
↓
concat
↓
output
```

这种方式比 block-level 更细，但本质上仍然是：

```text
Attention 和 SSM 各算各的，最后合并输出。
```

### 9.3 Score Fusion

SISA 属于 score-level fusion。

它不是在输出后融合，而是在 Attention 分数计算时就融合 SSM 信号。

普通 Attention：

```text
score = content match
```

SISA：

```text
score = content match + SSM importance match
```

这就是论文的核心创新。

---

## 10. SISA 的核心公式

普通 Attention 的分数是：

```text
s_ij = q_i^T k_j / sqrt(d_h)
```

SISA 改成：

```text
s_ij^SISA = q_i^T k_j / sqrt(d_h) + λ · C̄_i^T B̄_j
```

其中：

| 部分                      | 作用                  |
| ----------------------- | ------------------- |
| `q_i^T k_j / sqrt(d_h)` | 普通 Attention 的内容相似度 |
| `λ · C̄_i^T B̄_j`       | SSM 提供的顺序重要性匹配      |
| `λ`                     | 控制 SSM 项影响大小        |
| `i ≥ j`                 | 因果约束，只能看当前位置和历史位置   |

这表示：

> 第 i 个 token 要不要关注第 j 个 token，不只看内容上像不像，还要看 SSM 认为 j 对 i 是否重要。

---

## 11. 如何理解 `C̄_i^T B̄_j`

一开始我对 `C_i^T B_j` 不太理解。后来我把 SSM 想成一本“动态笔记本”。

每个 token 做两件事：

```text
把自己的信息写进记忆
从当前记忆中读取信息
```

因此可以这样理解：

| 符号          | 直观含义                                    |
| ----------- | --------------------------------------- |
| `B_j`       | 第 j 个 token 写入 SSM 状态的方式                |
| `C_i`       | 第 i 个 token 从 SSM 状态读取信息的方式             |
| `C_i^T B_j` | 第 i 个 token 想读的信息和第 j 个 token 写入的信息是否匹配 |

如果 `C_i^T B_j` 越大，说明：

```text
从 SSM 状态空间角度看，第 j 个 token 对第 i 个 token 更重要。
```

因此：

```text
q_i 和 k_j：判断内容上像不像
C_i 和 B_j：判断记忆读写上配不配
```

这是理解 SISA 的关键。

---

## 12. SSM Channels：B、C、Decay 和 Phase

在 Method 部分，作者从输入 `x_t` 中生成 SSM 相关通道：

```text
B_t = W_B x_t
C_t = W_C x_t
```

其中：

| 符号    | 直观含义                     |
| ----- | ------------------------ |
| `B_t` | 第 t 个 token 写入 SSM 记忆的方式 |
| `C_t` | 第 t 个 token 读取 SSM 记忆的方式 |

除此之外，作者还引入了：

```text
Decay
Phase
```

### 12.1 Decay

Decay 控制信息能保留多久。

如果衰减慢，说明信息能在序列中保留更久；如果衰减快，说明信息很快被遗忘。

直观理解：

```text
α_t 接近 1：信息保留得久
α_t 接近 0：信息很快衰减
```

### 12.2 Phase

Phase 可以理解为旋转或相位，用来加入一种和输入内容相关的顺序结构信息。

它类似 RoPE 中的位置旋转思想，但这里更强调 data-dependent，也就是与当前输入内容有关。

---

## 13. Augmented Q/K：SISA 的工程实现技巧

如果直接实现：

```text
score = q_i^T k_j / sqrt(d_h) + λ C̄_i^T B̄_j
```

可能需要修改底层 Attention kernel，工程上比较麻烦。

作者的技巧是：

```text
把 SSM 信息拼接到 Q 和 K 后面。
```

定义增强后的 Query 和 Key：

```text
Q̂_i = [q_i ; s C̄_i]
K̂_j = [k_j ; s B̄_j]
```

拼接后的点积为：

```text
Q̂_i^T K̂_j
=
q_i^T k_j + s² C̄_i^T B̄_j
```

再除以 `sqrt(d_h)`：

```text
Q̂_i^T K̂_j / sqrt(d_h)
=
q_i^T k_j / sqrt(d_h)
+
s² / sqrt(d_h) · C̄_i^T B̄_j
```

只要令：

```text
s² / sqrt(d_h) = λ
```

就得到 SISA 的 score。

这样 SISA 就可以直接调用标准 SDPA，而不用自己写新的 Attention kernel。

---

## 14. SDPA 是什么

SDPA 全称是：

```text
Scaled Dot-Product Attention
```

中文是：

```text
缩放点积注意力
```

它就是 Transformer 里标准的 Attention 计算方式：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

计算流程为：

```text
1. Q 和 K 做点积，计算相似度
2. 除以 sqrt(d_k)，缩放分数
3. softmax，把分数变成注意力权重
4. 用权重加权 V，得到输出
```

SISA 的巧妙之处是：

```text
把 SSM 信息拼进 Q/K，让标准 SDPA 自动完成：
内容匹配 + 顺序重要性匹配
```

这也是论文强调：

```text
single SDPA call
stock SDPA
no custom kernel
```

的原因。

---

## 15. Related Work 的理解

论文 Related Work 主要说明 SISA 和已有工作的区别。

### 15.1 SSM-Attention Hybrids

已有混合模型包括：

```text
Jamba
Samba
Zamba
Griffin
Hymba
Falcon-H1
```

它们主要采用：

```text
block-level fusion
head-level fusion
```

但这些方法的问题是：

```text
Attention 和 SSM 使用不同参数集分别计算，只有各自产生输出后才合并。
```

因此：

```text
SSM 信号没有进入 Attention score。
```

SISA 的新意就在于：

```text
直接在 attention score 层面融合 SSM 信号。
```

### 15.2 Attention Score Biases

论文还对比了已有的 attention bias 方法：

```text
ALiBi
T5 relative bias
DAPE
FoX
```

这些方法大多编码的是位置先验，例如距离、相对位置等。

但作者认为它们的问题是：

```text
不能利用输入序列本身的动态变化信息。
```

而这正是 SSM 擅长产生的信号。

因此可以总结为：

```text
前人要么在层级/头级拼接 Attention 和 SSM，
要么只给 Attention 加位置偏置；
SISA 的新意是在 Attention score 里直接融合 SSM 产生的动态重要性信号。
```

---

## 16. Experimental Setup：实验设置

作者比较了四种架构：

```text
Transformer
SISA
Mamba-2
Mamba-3
```

并在三个规模上实验：

```text
50M
152M
369M
```

实验中尽量匹配参数量，保证比较公平。

### 16.1 模型配置

以 152M 规模为例：

```text
Transformer: d/h/L/d_ff = 768/12/12/3072
SISA:        d/h/L/d_ff/d_s = 768/12/12/2748/32
```

SISA 增加了 `d_s=32` 的 SSM channel，同时减少 FFN 维度，从而控制总参数量接近。

这对应论文中的设计权衡：

```text
FFN-vs-SSM allocation
```

也就是：

> 把一部分 FFN 参数预算分给 SSM channel。

### 16.2 训练配置

所有模型都训练在：

```text
5B tokens of SlimPajama-6B
```

上。

训练配置包括：

```text
AdamW
weight decay 0.1
gradient clipping 1.0
cosine schedule
500-step warmup
effective batch 524K tokens
bf16
NVIDIA H100 80GB
```

作者也说明：

```text
50M 和 152M 训练较充分；
369M 可能有些训练不足。
```

不过因为所有架构共享相同 token budget，因此相对比较仍然有参考价值。

---

## 17. Evaluation：评估任务

论文使用了五个 benchmark：

| Benchmark  | 任务      | 测试能力      |
| ---------- | ------- | --------- |
| LAMBADA    | 最后一个词预测 | 长距离理解     |
| NIAH       | 找隐藏信息   | 检索 / 记忆保持 |
| HellaSwag  | 句子补全    | 常识推理      |
| ARC-Easy   | 科学问答    | 事实知识      |
| Winogrande | 代词消歧    | 共指消解      |

其中最重要的是：

```text
LAMBADA
NIAH
```

因为它们最能体现论文关于长上下文理解和检索能力的主张。

NIAH 是 Needle-in-a-Haystack，可以理解为“大海捞针”测试。论文中会在随机位置插入一句：

```text
The secret number is 42.
```

然后测试模型能否在长上下文中找回这个隐藏信息。

---

## 18. Results：152M 主实验结果

在 152M 主实验中，SISA 表现最突出。

LAMBADA 结果：

| 模型          | LAMBADA |
| ----------- | ------: |
| Transformer |    13.9 |
| Mamba-2     |    12.7 |
| Mamba-3     |    15.5 |
| SISA d_s=16 |    17.3 |

这说明 SISA 在长距离语言建模上优于 Transformer 和 Mamba-3。

NIAH 结果：

| 模型          |  NIAH |
| ----------- | ----: |
| Transformer | 100.0 |
| Mamba-2     |  82.5 |
| Mamba-3     |  99.0 |
| SISA        | 100.0 |

这里的关键不是 SISA 比 Transformer 更高，而是：

```text
SISA 在加入 SSM 信号后，没有损失 Attention 的检索能力。
```

也就是说，SISA 保留了 Transformer 的检索优势，同时加入了 SSM 的顺序重要性信号。

---

## 19. NIAH 训练收敛结果

论文中 NIAH 的训练过程非常关键。

表格显示：

| 模型          |    1K |    2K |    3K |    5K |    7K | Final |
| ----------- | ----: | ----: | ----: | ----: | ----: | ----: |
| Transformer |  61.5 |  78.0 |  93.5 |  98.5 | 100.0 | 100.0 |
| SISA        | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Mamba-2     |   0.0 |   6.5 |  42.0 |  71.0 |  78.5 |  82.5 |
| Mamba-3     |   0.0 |  61.0 |  96.5 |  86.0 |  97.5 |  99.0 |

核心结论是：

```text
SISA 从训练早期 step 1K 就达到 NIAH 100%。
```

相比之下，Transformer 到 step 7K 才达到 100%。

我的理解是：

> 普通 Transformer 需要逐渐学会如何进行长上下文检索，而 SISA 由于引入了 SSM-derived score bias，在训练早期就给 Attention 提供了结构性位置和重要性提示。

---

## 20. Scaling：不同规模下的表现

论文还比较了 50M、152M、369M 三个规模。

### 20.1 50M

在 50M 下，SISA 也有提升：

```text
Transformer LAMBADA: 13.4
SISA LAMBADA: 14.4
```

同时 SISA 保持 NIAH 100%。

这说明 SISA 不只是 152M 上偶然有效，在小模型上也能带来收益。

### 20.2 152M

152M 是论文主结果，SISA 表现最强：

```text
SISA d_s=16 LAMBADA 17.3
NIAH 100
HellaSwag 26.9
```

### 20.3 369M

在 369M 规模上，Mamba-3 在 LAMBADA 上反超：

```text
Mamba-3 LAMBADA 17.4
SISA d_s=128 LAMBADA 14.8
```

这说明 SISA 并不是所有规模和所有指标都绝对最优。

更准确的结论是：

```text
SISA 和 Mamba-3 具有互补优势。
```

---

## 21. Throughput：训练速度

论文中的吞吐量结果如下：

| 模型          |  tok/s |  相对速度 |
| ----------- | -----: | ----: |
| Transformer | 27,714 | 1.00× |
| SISA        | 16,783 | 0.61× |
| Mamba-2     | 10,719 | 0.39× |
| Mamba-3     | 13,460 | 0.49× |

可以看到：

```text
SISA 比标准 Transformer 慢；
但比 Mamba-2 和 Mamba-3 快。
```

作者强调：

```text
SISA runs 1.25× faster than Mamba-3.
```

我的理解是：

> SISA 增强 Q/K 后增加了 Attention 计算开销，因此慢于标准 Transformer；但它仍然可以复用 PyTorch 的标准 SDPA，不需要自定义 kernel，所以工程兼容性比 Mamba 系列更好。

---

## 22. 这篇论文的核心贡献

我总结这篇论文的贡献主要有三点：

### 22.1 提出 score-level fusion

以往 Attention-SSM 混合模型多是 block-level 或 head-level 融合，SISA 则在 Attention score 层面融合。

```text
以前：Attention 和 SSM 各算各的，最后合并输出
SISA：SSM 信号直接进入 Attention score
```

### 22.2 引入 SSM-derived importance term

SISA 的 attention score 为：

```text
s_ij^SISA = q_i^T k_j / sqrt(d_h) + λ · C̄_i^T B̄_j
```

其中第二项表示：

```text
从 SSM 顺序动态角度看，第 j 个 token 对第 i 个 token 是否重要。
```

### 22.3 用增强 Q/K 实现 single SDPA call

作者通过：

```text
Q̂_i = [q_i ; s C̄_i]
K̂_j = [k_j ; s B̄_j]
```

把 SSM bias 转化为标准点积的一部分，因此可以直接调用 SDPA。

这使得 SISA 不需要显式构造 `L × L` bias matrix，也不需要修改 attention kernel。

---

## 23. 我的阶段性理解

读这篇论文之前，我对 Attention 的理解主要停留在：

```text
Q、K、V
softmax
加权求和
```

读完之后，我对 Attention score 有了更深的理解。

普通 Attention 的 score 主要表达：

```text
内容匹配程度
```

而 SISA 试图让 score 同时表达：

```text
内容匹配程度
+
顺序重要性匹配程度
```

因此，我可以用一句话概括 SISA：

> SISA 不是抛弃 Attention，而是让 Attention 变得 importance-aware。

更具体地说：

```text
Attention 提供全局检索能力；
SSM 提供顺序重要性信号；
SISA 把 SSM 信号加入 Attention score；
再通过增强 Q/K 的方式复用标准 SDPA。
```

---

## 24. 这篇论文的不足和需要继续思考的地方

虽然 SISA 的思路很有启发，但它也不是完全没有问题。

我目前看到的几个点：

1. **SISA 在 369M 上并没有全面超过 Mamba-3**
   说明它的优势可能和模型规模、训练 token、参数分配有关。

2. **SISA 比标准 Transformer 慢**
   因为增强 Q/K 增加了计算维度，吞吐量只有 Transformer 的 0.61×。

3. **实验规模仍然有限**
   论文主要在 50M、152M、369M 规模下验证，还没有证明在更大 LLM 上一定有效。

4. **SSM 信号的可解释性仍需要进一步理解**
   例如 `C̄_i^T B̄_j` 虽然可以解释为读写匹配，但实际训练后每个维度具体代表什么，仍然比较抽象。

5. **参数预算分配是关键问题**
   SISA 需要在 FFN 和 SSM channel 之间分配参数，这个 trade-off 可能会影响不同规模下的表现。

---

## 25. 后续学习计划

为了真正读懂这篇论文，我后面还需要补充以下内容：

1. 复习 Transformer 中的 Scaled Dot-Product Attention。
2. 系统学习 RoPE 的旋转位置编码思想。
3. 了解 SSM、S4、Mamba 的基本原理。
4. 对比 Jamba、Hymba 等 Attention-SSM 混合架构。
5. 进一步理解 FlashAttention、SDPA 和 kernel 兼容性。
6. 尝试画一张 SISA 的计算流程图。
7. 如果有开源代码，可以尝试运行最小版本实验。

---

## 26. 总结

这次精读过程让我形成了一个比较完整的论文阅读流程：

```text
arXiv 筛选论文
↓
判断论文类型和优先级
↓
Zotero 保存和打标签
↓
双语对照泛读
↓
PDF 高亮精读
↓
拆解基础概念
↓
阅读 Related Work
↓
理解 Method 公式
↓
分析 Experimental Setup
↓
总结 Results
↓
形成博客笔记
```

对这篇论文，我目前最核心的理解是：

```text
SISA = Attention 的内容匹配 + SSM 的顺序重要性匹配
```

普通 Attention 判断：

```text
当前 token 和历史 token 在内容上是否相关？
```

SISA 进一步判断：

```text
历史 token 写入 SSM 状态的信息，是否正好是当前 token 想读取的信息？
```

因此，SISA 的价值在于：

> 它不是简单把 Attention 和 SSM 拼在一起，而是在 Attention score 内部直接融合 SSM 的动态重要性信号。

最后用一句话总结：

> SISA 的核心不是“Forget Attention”，而是让 Attention 在全局检索的基础上，具备对序列重要性的感知能力。
