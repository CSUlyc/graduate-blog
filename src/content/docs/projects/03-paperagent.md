---

title: "PAPERAGENT 项目架构：从论文阅读 Agent 到个人科研工作台"
description: "本文记录 PAPERAGENT 的项目结构与系统架构，包括 PDF 解析、RAG 检索增强、Evidence Pack、LangGraph 工作流、博客生成、论文库沉淀和 Web 可视化界面。"
pubDate: 2026-06-07
tags: ["Agent", "RAG", "论文阅读", "博客生成", "LangGraph", "Next.js", "FastAPI"]
-------------------------------------------------------------------------

PAPERAGENT 是我为研究生阶段论文阅读、知识沉淀和博客写作开发的一个本地 Agent 项目。

它最初的目标很简单：

```text
输入一篇 PDF 论文
↓
生成论文精读笔记
↓
生成博客 Markdown
```

但随着功能逐步扩展，它已经不只是一个论文总结脚本，而是变成了一套完整的本地科研工作流：

```text
本地 PDF 论文阅读
↓
RAG 检索增强
↓
Evidence Pack 可追溯证据
↓
精读笔记 / 方法分析 / 实验分析 / 复现建议
↓
博客 Markdown
↓
二段式发布
↓
论文库沉淀
↓
组会 PPT 大纲
↓
Web 可视化管理
```

我对 PAPERAGENT 的定位是：

> 一个面向个人科研场景的论文阅读、博客生成和知识沉淀工作台。

---

## 一、为什么要设计成 Agent 工作流

读论文不是一个单步任务。

一篇论文从 PDF 到博客，通常要经历很多阶段：

```text
解析论文
↓
理解背景
↓
拆解方法
↓
分析实验
↓
判断复现价值
↓
写成博客
↓
检查格式
↓
发布到个人博客
```

如果直接让大模型一次性生成博客，很容易出现几个问题：

1. 论文内容缺少依据。
2. 方法和实验混在一起。
3. 生成结果不可追溯。
4. 博客风格不稳定。
5. 后续很难积累成论文库。
6. 出错时不知道是哪一步出了问题。

因此，PAPERAGENT 没有采用“一次性总结”的方式，而是把论文阅读拆成多个阶段，每个阶段由不同模块负责。

整体流程可以概括为：

```text
论文输入
↓
结构化解析
↓
RAG evidence
↓
多 Agent 分析
↓
博客生成
↓
审核与发布
```

---

## 二、PAPERAGENT 总体架构图

![paperagent 整体数据流与流程图](/images/paperagent.png)

---

## 三、用户交互层：CLI 与 Web 控制台

PAPERAGENT 目前同时支持两种使用方式：

```text
CLI 命令行
Web 可视化控制台
```

CLI 更适合开发、调试和批量处理。

常用命令包括：

```bash
python -m app.cli doctor
python -m app.cli inspect --pdf ./storage/papers/example.pdf
python -m app.cli read-paper --pdf ./storage/papers/example.pdf --mode deep
python -m app.cli batch-read --dir ./storage/papers --mode quick
python -m app.cli show-run --run-id your_run_id
python -m app.cli list-papers
```

Web 控制台则更适合日常使用。

它提供了：

```text
Dashboard
New Paper
Papers
Paper Detail
Runs
Run Detail
Blog Review
Settings
```

其中 Run Detail 页面是最重要的页面之一，因为它能展示：

```text
运行报告
Markdown 输出
Evidence Pack
RAG 效果分析
输出文件路径
```

这样我可以直接在浏览器里检查 Agent 是否真的找到了正确 evidence。

---

## 四、服务接口层：FastAPI 后端

为了让 Web 前端调用已有 Python 能力，我增加了 FastAPI 后端。

它的作用不是重写论文处理逻辑，而是把现有 CLI 和工具模块封装成 HTTP API。

主要接口包括：

| 接口                            | 作用            |
| ----------------------------- | ------------- |
| `/api/health`                 | 服务健康检查        |
| `/api/config`                 | 查看非敏感配置       |
| `/api/doctor`                 | 环境诊断          |
| `/api/papers`                 | 论文库列表         |
| `/api/papers/upload`          | 上传 PDF        |
| `/api/runs/start`             | 启动论文处理工作流     |
| `/api/runs/{id}`              | 查看 Run Report |
| `/api/runs/{id}/outputs`      | 查看生成内容        |
| `/api/runs/{id}/evidence`     | 查看 evidence   |
| `/api/runs/{id}/rag-analysis` | 查看 RAG 效果分析   |
| `/api/blogs/lint`             | 博客格式检查        |
| `/api/blogs/publish`          | 发布到本地发布目录     |
| `/api/blogs/deploy`           | 部署到真实博客目录     |

这种设计让前端只负责展示和交互，真正的论文处理仍然由 Python 后端完成。

---

## 五、论文输入与解析层

PAPERAGENT 的输入是本地 PDF。

PDF 解析主要做三件事：

```text
读取 PDF 文本
保留页码标记
切分为结构化 chunk
```

PDF Loader 会使用 PyMuPDF 提取文本，并保留页码：

```text
[Page 1]
第一页内容……

[Page 2]
第二页内容……
```

随后 chunker 会把论文切成多个 chunk。

每个 chunk 至少包含：

```json
{
  "chunk_id": "chunk_001",
  "section": "Method",
  "page": 4,
  "text": "...",
  "prev_chunk_id": "chunk_000",
  "next_chunk_id": "chunk_002",
  "char_count": 2800
}
```

我后来发现，chunk 质量对 RAG 非常重要。

如果 section 识别不准，后面的 query planner 再好，也可能检索到错误章节。

例如：

```text
实验分析命中了 Related Work
方法分析命中了 References
复现计划过度依赖 unknown section
```

这些问题本质上都和 chunker 的结构识别有关。

---

## 六、RAG 检索增强层

RAG 是 PAPERAGENT 的核心部分之一。

当前 RAG 不是简单的 `top_k` 检索，而是多了一层 query planning。

整体流程是：

```text
Agent 任务
↓
purpose 分类
↓
生成 query plan
↓
按章节优先检索
↓
向量检索
↓
词法回退
↓
相邻 chunk 扩展
↓
Evidence Pack
```

目前支持的 purpose 包括：

| purpose      | 作用     |
| ------------ | ------ |
| overview     | 论文整体精读 |
| method       | 方法分析   |
| experiment   | 实验分析   |
| reproduction | 复现计划   |
| limitation   | 局限性分析  |
| blog         | 博客生成   |
| slides       | 组会大纲   |
| related_work | 相关工作   |
| citation     | 引用分析   |
| references   | 参考文献分析 |

不同 purpose 会使用不同的检索策略。

例如：

```text
method 优先检索 Method / Model / Approach / Architecture
experiment 优先检索 Experiment / Evaluation / Results / Ablation
reproduction 优先检索 Implementation / Training / Appendix
overview 优先检索 Abstract / Introduction / Conclusion
```

这比直接问：

```text
这篇论文的方法是什么？
```

要稳定很多。

---

## 七、Evidence Pack：让 Agent 输出可追溯

PAPERAGENT 不是直接把全文丢给大模型，而是先为每个任务构造 Evidence Pack。

每条 evidence 会保存：

```json
{
  "chunk_id": "chunk_010",
  "section": "Results",
  "page": 7,
  "text": "...",
  "score": 0.658,
  "source": "chroma",
  "type": "experiment"
}
```

Evidence 的作用是：

```text
告诉 Agent 这次生成到底依据了哪些原文片段
```

不同 Agent 使用不同 evidence：

```text
Paper Reader Agent -> overview evidence
Method Analyzer Agent -> method evidence
Experiment Analyzer Agent -> experiment evidence
Reproduction Planner Agent -> reproduction evidence
```

这样生成的内容不是黑盒，而是可以回到原文 chunk 检查。

这对论文阅读非常重要。

因为论文博客最怕的问题不是写得不流畅，而是：

```text
写得很像真的，但论文其实没有这么说。
```

Evidence Pack 的目标就是降低这种风险。

---

## 八、Agent 分析生成层

在拿到 Evidence Pack 后，不同 Agent 开始生成自己的内容。

### 1. Paper Reader Agent

负责生成论文精读笔记，关注：

```text
研究问题
研究背景
核心思想
主要贡献
方法概览
实验结论
局限性
启发
```

### 2. Method Analyzer Agent

负责分析方法，关注：

```text
模型结构
关键模块
公式
输入输出
算法流程
和已有方法的差异
```

### 3. Experiment Analyzer Agent

负责分析实验，关注：

```text
数据集
baseline
评价指标
主实验结果
消融实验
效率分析
结果是否支撑创新点
```

### 4. Reproduction Planner Agent

负责生成复现建议，关注：

```text
是否有代码
数据集是否公开
算力要求
最小复现版本
复现步骤
适合博客还是组会汇报
```

### 5. Blog Writer Agent

负责把前面的中间产物整理成 Astro 支持的 Markdown 博客。

它会读取：

```text
prompts/blog_writer.md
prompts/blog_style.md
```

其中 `blog_style.md` 用来保持我自己的博客风格。

### 6. Reviewer Agent

负责检查生成结果，重点关注：

```text
是否有明显幻觉
是否缺少 evidence
Markdown 是否规范
Mermaid 是否可渲染
是否适合发布
```

---

## 九、Run Report：每次运行都有记录

每次执行 `read-paper`，系统都会生成一份 Run Report。

保存位置是：

```text
storage/runs/{run_id}/run_report.json
```

Run Report 记录：

```text
run_id
paper_id
input_pdf_path
started_at
ended_at
duration_seconds
model_name
base_url
rag_mode
outputs
evidence_files
errors
warnings
```

这相当于每次论文处理的“实验记录”。

如果某次输出效果不好，我可以回头查看：

```text
使用了什么模型
RAG 是否成功
检索到了多少 evidence
有没有 fallback
输出文件在哪里
有没有 warning
```

这对调试 Agent 非常有帮助。

---

## 十、博客生成与发布流程

PAPERAGENT 的博客生成不是一步直接发布，而是采用二段式发布。

第一步是生成博客草稿：

```text
storage/blogs/{paper_id}_blog.md
```

然后执行：

```bash
python -m app.cli publish --blog ./storage/blogs/example_blog.md
```

发布到项目内集中目录：

```text
published_blogs/
```

这一步相当于“内部发布”，方便人工检查。

确认没有问题后，再执行：

```bash
python -m app.cli deploy-to-blog --blog ./published_blogs/example_blog.md
```

部署到真实 Astro + Starlight 博客目录。

这样设计的原因是：

> 不让未经审核的 AI 生成内容直接进入正式博客。

在正式发布前，还会经过：

```text
lint-blog
validate-blog
```

检查内容包括：

```text
frontmatter
Markdown 代码块
Mermaid
公式
TODO
图片路径
超长段落
AI 模板化表达
```

---

## 十一、PaperCard 论文库

PAPERAGENT 还会把处理过的论文沉淀到论文库。

论文库保存为：

```text
storage/library/papers.jsonl
```

每篇论文对应一个 PaperCard，包含：

```text
paper_id
标题
作者
年份
venue
主题
方法名
任务
数据集
baseline
metric
状态
产物路径
```

常用命令：

```bash
python -m app.cli list-papers
python -m app.cli show-paper --paper-id your_paper_id
python -m app.cli search-papers --keyword agent
```

这一步让 PAPERAGENT 不只是一次性生成工具，而是逐渐变成个人论文知识库。

---

## 十二、组会 PPT 大纲

除了博客，PAPERAGENT 还支持生成组会 PPT 大纲。

命令如下：

```bash
python -m app.cli generate-slides-outline --paper-id your_paper_id
```

输出位置：

```text
storage/slides_outline/{paper_id}_slides_outline.md
```

目前它只生成 Markdown 大纲，不直接生成 `pptx`。

这样做更稳，因为组会 PPT 还需要人工根据老师和同学的关注点调整。

---

## 十三、Web 可视化界面

为了让 PAPERAGENT 更适合日常使用，我增加了 Web 控制台。

技术栈是：

```text
Next.js 15
React 19
TypeScript
Tailwind CSS
Recharts
react-markdown
Mermaid
KaTeX
FastAPI
```

启动后端：

```bash
uvicorn api.server:app --host 127.0.0.1 --port 8000
```

启动前端：

```bash
cd web
npm install
npm run dev
```

访问：

```text
http://localhost:3000
```

主要页面包括：

| 页面          | 作用               |
| ----------- | ---------------- |
| Dashboard   | 查看项目整体状态         |
| New Paper   | 上传或选择 PDF 并启动工作流 |
| Papers      | 查看论文库            |
| Runs        | 查看运行记录           |
| Run Detail  | 查看某次运行的所有输出      |
| Blog Review | 预览、检查和发布博客       |
| Settings    | 查看配置和环境状态        |

其中我最常用的是 Run Detail 页面。

它包含多个 Tab：

```text
Overview
Reading Note
Method
Experiment
Reproduction
Blog
Review
Evidence
Files
```

这让我可以非常直观地看到：

```text
Agent 生成了什么
参考了哪些 evidence
RAG 检索质量如何
博客是否可以发布
```

---

## 十四、RAG 效果分析页面

RAG 效果分析页面是我后面重点加入的功能。

它会统计：

```text
平均相似度
RAG 模式
chunk 数量
evidence 数量
每个模块的 evidence 数
向量命中数量
邻居扩展数量
词法降级数量
章节分布
类型分布
页码分布
```

它帮助我发现一个重要问题：

> 平均相似度高，不代表 evidence 一定来自正确章节。

例如：

```text
method 分析应该命中 Method / Model / Approach
experiment 分析应该命中 Experiment / Evaluation / Results
overview 应该命中 Abstract / Introduction / Conclusion
```

如果某个模块虽然相似度很高，但 evidence 来自 References 或 unknown section，就说明 RAG 仍然需要优化。

所以后续我准备继续增加：

```text
section_match_rate
main_body_ratio
appendix_ratio
references_ratio
unknown_ratio
neighbor_ratio
```

让 RAG 质量判断更细致。

---

## 十五、测试与质量控制

PAPERAGENT 当前已经加入了 pytest 测试。

测试覆盖包括：

```text
PDF 加载异常
chunk 结构与回退
metadata 提取回退
EvidencePack
RunReport
query planner
模式分支
博客 lint / validate
论文库
批量处理
slides outline
workflow 基本行为
FastAPI 后端接口
路径安全
```

运行测试：

```bash
python -m pytest
```

我现在的理解是：

> Agent 项目不是只要能生成内容就行，还必须能测试、能追踪、能回滚、能解释。

否则后期功能越多，越容易变成不可维护的黑盒。

---

## 十六、当前项目目录结构

当前 PAPERAGENT 的主要目录如下：

```text
app/               CLI、配置、LLM 客户端
agents/            精读、方法、实验、复现、博客、审核、PPT 大纲 Agent
docs/              架构、工作流、RAG、发布、Prompt、排障文档
prompts/           各 Agent 提示词，含 blog_style.md
published_blogs/   项目内集中发布目录
rag/               chunker、query planner、index builder、retriever、evidence builder
schemas/           PaperState、Evidence、RunReport、PaperCard、BatchReport
storage/           PDF、解析结果、笔记、博客、审核、向量索引、运行记录、论文库、PPT 大纲
tests/             单元测试
tools/             PDF、metadata、Markdown、发布、论文库、批量处理工具
workflows/         论文处理工作流
api/               FastAPI 后端服务
web/               Next.js 前端控制台
```

可以看到，它已经从一个简单脚本扩展成了一个比较完整的本地科研工具。

---

## 十七、我的阶段性理解

开发 PAPERAGENT 之后，我对 Agent 和 RAG 有了一个更清晰的认识。

Agent 的关键不是：

```text
让模型一次性输出一篇文章
```

而是：

```text
把复杂任务拆成多个可控节点
```

RAG 的关键也不是：

```text
检索到一些文本
```

而是：

```text
检索到适合当前任务的 evidence
```

所以 PAPERAGENT 真正要解决的问题是：

```text
如何让论文阅读过程结构化、可追溯、可审核、可沉淀
```

而不是简单地把 PDF 交给大模型总结。

---

## 十八、后续优化方向

接下来我准备继续优化几个方向。

### 1. Section Detection

重点解决：

```text
表格列名误判为 section
References 和 Appendix 识别
正文与附录区分
带编号标题识别
section_source 标注
```

### 2. RAG Quality Metrics

增加：

```text
section_match_rate
main_body_ratio
appendix_ratio
references_ratio
unknown_ratio
neighbor_ratio
vector_hit_ratio
```

### 3. Evidence Reviewer

让审核 Agent 检查：

```text
关键结论是否有 evidence
实验分析是否来自实验章节
方法分析是否来自方法章节
复现建议是否来自实现细节或训练配置
References 是否污染普通任务
```

### 4. 前端体验

继续完善：

```text
RAG 效果分析
Evidence 展开与筛选
博客预览
论文库搜索
批量任务管理
```

### 5. 组会材料生成

目前只生成 PPT 大纲，后续可以继续支持：

```text
组会讲稿
流程图
PPT 图片素材
自动生成 pptx
```

---

## 十九、总结

PAPERAGENT 当前已经形成了一个比较完整的闭环：

```text
论文 PDF
↓
RAG Evidence
↓
多 Agent 分析
↓
博客 Markdown
↓
二段式发布
↓
论文库沉淀
↓
Web 可视化管理
```

它的核心价值不是替我完全读论文，而是帮我完成：

```text
结构化整理
证据追溯
初稿生成
质量检查
长期沉淀
```

我后续会继续把它作为研究生阶段的长期项目维护。

最终我希望 PAPERAGENT 能成为一个真正服务个人科研工作的工具：

> 不是一个一次性的论文总结器，而是一个可追溯、可审核、可扩展的论文阅读与知识沉淀工作台。
