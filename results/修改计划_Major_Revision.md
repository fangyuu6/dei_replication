# 研究一 Major Revision 修改计划

> 基于两份编辑/审稿意见综合整理（Codex 版 + Nature Food 编辑版），按优先级排序。
> 制定日期：2026-02-24

---

## 一、两份 Review 核心共识

两份 review 均给出 **Major Revision**，并在以下 5 个核心问题上高度一致：

| # | 核心问题 | Codex 评语 | NF 编辑评语 | 严重度 |
|---|---------|-----------|------------|--------|
| 1 | DEI ≈ 1/E，H 贡献仅 0.8% | "DEI 是否只是环境代价排名的重命名" | "if DEI ≈ f(1/E), what value does H add?" | **致命** |
| 2 | H 构念效度不足（Yelp 代理偏差） | "Yelp 难完全剥离服务/价格/氛围" | "reviews conflate dish quality with restaurant quality" | **致命** |
| 3 | E 缺乏不确定性量化 | — | "No uncertainty propagation...LCA factors span an order of magnitude" | **重要** |
| 4 | 浪费空间分析不可操作 | — | "benchmark is almost always kimchi...not actionable" | **重要** |
| 5 | 缺乏明确假设/因果框架 | "因果表述降级" | "What hypothesis does DEI test?" | **重要** |

额外关注点（至少一份 review 提出）：
- 人工验证未完成（两份均提）
- 跨地域/跨平台稳健性（两份均提）
- 时间稳定性（NF 编辑提）
- BERT 微调样本量薄弱 + 单一 LLM 来源（NF 编辑提）
- E 等权假设（NF 编辑提）
- 营养维度缺失（两份均提，但定位为 Enhancement）
- 排名不确定性/概率排名（Codex 提）

---

## 二、多数据库交叉印证 H 分数（新增核心策略）

> **审稿人核心质疑**：H 仅来自 Yelp 单一数据源，可能是平台特异性现象而非真实味觉信号。
> **应对策略**：获取尽可能多的外部数据源，对 158 道菜的 H 进行跨平台/跨方法交叉验证。

### 2.1 可直接获取的数据源（立即下载）

| # | 数据源 | 规模 | 数据类型 | 获取方式 | 菜品覆盖预估 |
|---|--------|------|---------|---------|-------------|
| 1 | **Food.com (Kaggle, irkaal 版)** | 500K 食谱, 1.4M 评论 | 食谱名 + 1-5 星评分 + 评论文本 | [Kaggle 直接下载](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews) | ~130/158 |
| 2 | **Food.com (Kaggle, shuyangli94 版)** | 180K 食谱, 700K 评论 | 同上，含营养信息 | [Kaggle 直接下载](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) | ~120/158 |
| 3 | **Google Local Reviews (UCSD McAuley Lab)** | **6.66 亿条评论**, 490 万商家 | 评分 + 评论文本 + 坐标 | [UCSD 直接下载](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal_restaurants/) | ~140/158 (需NLP) |
| 4 | **Epicurious 食谱** | 20K 食谱 | 评分 0-5 + 营养信息 | [Kaggle 直接下载](https://www.kaggle.com/datasets/hugodarwood/epirecipes) | ~80/158 |
| 5 | **Amazon Grocery & Gourmet (2023)** | 1430 万评分 | 产品评分 + 评论文本 | [HuggingFace](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) | ~40/158 |
| 6 | **Amazon Fine Food Reviews** | 568K 评论 | 评分 1-5 + 评论文本 | [Kaggle/SNAP 直接下载](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) | ~30/158 |
| 7 | **TripAdvisor 6 城市** | 6 城市餐厅评论 | 评分 1-5 + 评论全文 | [Zenodo 直接下载](https://zenodo.org/records/6583422) | ~100/158 (需NLP) |
| 8 | **Reddit 食物板块 (Pushshift)** | 数百万帖子/评论 | 帖子文本 + upvote 分数 | [Academic Torrents](https://academictorrents.com/details/1614740ac8c94505e4ecb9d88be8bed7b6afddd4) | ~130/158 (需NLP) |
| 9 | **Meituan/大众点评 ASAP 数据集** | 46,730 条中文评论 | 18 维度情感标注 (含食物) | [GitHub 直接下载](https://github.com/Meituan-Dianping/asap) | ~80/158 (需NLP) |
| 10 | **Food-Pics Extended (学术)** | 896 种食物图片 | 适口性 VAS 评分, ~2000 评估者 | [免费下载](https://www.eat.sbg.ac.at/resources/food-pics) | ~60/158 |
| 11 | **CROCUFID (跨文化)** | 479 种食物图片 | 想吃度 VAS, 805 人 (英/美/日) | [OSF 免费下载](https://osf.io/5jtqx/) | ~50/158 |
| 12 | **TasteAtlas 全球菜品评分** | 18,912 种食物, 590K 评分 | 菜品名 + 0-5 评分 | 手动查询 158 道菜（无 API） | **~155/158** |
| 13 | **AllRecipes US (by State)** | 数千食谱 | 评分 + 评论 + 食材 | [Kaggle 直接下载](https://www.kaggle.com/datasets/dimitryzub/allrecipes-all-us-recipes-by-state) | ~70/158 |
| 14 | **Spoonacular API** | 365K 食谱 | aggregateLikes + 营养 | API 注册（免费额度有限） | ~100/158 |
| 15 | **Sensometric Society 数据集** | 多品类（面包/芝士/饮品） | 9 点享乐量表, 61-570 人 | [直接下载](https://sensometric.org/datasets) | ~10/158 |
| 16 | **Yummly (Harvard Dataverse)** | 27K 食谱 | 评分 + 食材 + 菜系 | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/W0OB30) | ~80/158 |

### 2.2 需申请/Outreach 的数据源

| # | 数据源 | 价值 | 联系方式 | Outreach 策略 |
|---|--------|------|---------|--------------|
| 1 | **UK Biobank 食物偏好问卷** | 150 种食物, **18 万人**, 9 点享乐量表 — 人群级金标准 | [UK Biobank 申请](https://biobank.ndph.ox.ac.uk/) | 正式研究员申请，需关联健康研究目的 |
| 2 | **Finlayson/Stubbs SatMap** | **436 种食物, 3,364 人**, 喜好度评分 — 最全面的食物级享乐数据集 | 通讯作者 Graham Finlayson, University of Leeds | 邮件请求补充材料中的 per-food 均值数据 |
| 3 | **US Armed Forces 食物偏好调查** | 378 种食物, 3,900 人, 9 点享乐量表 | [DTIC 报告](https://apps.dtic.mil/sti/tr/pdf/ADA110512.pdf) | DTIC 注册下载技术报告（含完整评分表） |
| 4 | **CSIRO Sensory-Diet Database** | 720+ 种食物的感官属性（专业面板评分） | 联系 CSIRO 作者 (Hendrie/Lease) | 邮件请求学术使用许可 |
| 5 | **TheFork (欧洲 11 国)** | 餐厅评论 + 评分 | [开发者门户](https://docs.thefork.io/) | 申请 B2B API 学术合作 |
| 6 | **Tabelog (日本)** | 日本最大餐评, 8500 万条评论 | Kakaku.com PR: pr@kakaku.com | 日语邮件，需日本合作机构 |
| 7 | **大众点评完整数据** | 中国最大餐评平台 | 美团研究院 | 通过中国合作机构申请 |

### 2.3 交叉验证分析框架

```
                        ┌─────────────────────────────┐
                        │    158 道菜 × N 个数据源     │
                        │    H 分数交叉验证矩阵        │
                        └──────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
    ┌─────────▼─────────┐ ┌───────▼────────┐ ┌────────▼───────┐
    │ 第一层：大规模NLP  │ │ 第二层：食谱评分 │ │ 第三层：学术基准│
    │                   │ │                │ │                │
    │ • Google Local    │ │ • Food.com     │ │ • Food-Pics    │
    │ • TripAdvisor     │ │ • Epicurious   │ │ • CROCUFID     │
    │ • Reddit          │ │ • AllRecipes   │ │ • UK Biobank   │
    │ • ASAP (中文)     │ │ • Spoonacular  │ │ • SatMap       │
    │                   │ │ • Yummly       │ │ • US Army      │
    │ 方法: 同 Yelp 的  │ │ 方法: 按菜名匹 │ │ 方法: 食物名模 │
    │ NLP 管线提取 H    │ │ 配，取平均评分  │ │ 糊匹配+相关分析│
    └───────────────────┘ └────────────────┘ └────────────────┘
                                   │
                        ┌──────────▼──────────────────┐
                        │       汇总报告               │
                        │ • 跨平台 H 相关矩阵          │
                        │ • 每道菜的"H 共识区间"       │
                        │ • 平台特异性菜品识别          │
                        │ • 多源 H 的 DEI 敏感性分析   │
                        └─────────────────────────────┘
```

### 2.4 Outreach 邮件模板

**Subject**: Data sharing request for cross-cultural food hedonic validation study (Nature Food submission)

```
Dear Prof. [Name],

We are preparing a manuscript for Nature Food that quantifies the
hedonic-environmental trade-off across 158 common dishes using NLP-derived
taste scores from 5.3 million Yelp reviews. A key reviewer concern is the
reliance on a single data source for hedonic validation.

Your [dataset name] ([N] foods, [N] raters) would provide an invaluable
independent benchmark. Specifically, we would use the per-food [metric]
scores to cross-validate our BERT-derived hedonic scores.

We would:
- Use the data solely for academic validation (non-commercial)
- Cite your work prominently and offer co-authorship if appropriate
- Share our cross-validation results with you prior to submission

Would it be possible to share the per-food mean scores for [specific request]?

Thank you for your time.

Best regards,
[Your name]
[Institution]
```

---

## 三、修改任务清单（按优先级 P0 → P3）

### P0：不做就退稿的项目

#### P0-1. 证明 DEI ≠ E 排名（回应核心问题 #1）

**目标**：证明 H 的加入为 DEI 提供了超越纯 1/E 排名的信息增量。

**具体任务**：
- [ ] **a. DEI vs 1/E 排名正式对比**
  - 计算 Kendall's tau-b 和 Rank-Biased Overlap (RBO) between log(DEI) ranking 和 log(1/E) ranking
  - 统计有多少道菜排名变化 ≥5 位
  - 列出"因 H 而排名显著改变"的典型案例
- [ ] **b. 同 E 区间内 H 的区分力**
  - 将菜品按 E 分为 3-4 个 bin，在每个 bin 内检验 H 是否仍能解释偏好差异
  - 报告 within-bin 的 H 方差和排名变化
- [ ] **c. 叙事重构**
  - 如果 H 信息增量有限，按 NF 编辑建议重新定框为："好消息——味觉差异小意味着消费者可以几乎无痛地选择低 E 食物"
  - 将 "H 方差小" 从"方法局限"转变为"核心实证发现+政策机会窗口"

**脚本**：新建 `code/07_dei_vs_e_ranking.py`

---

#### P0-2. H 的多源交叉验证 + 餐厅级控制（回应核心问题 #2）⭐ 最关键

**目标**：用多个独立数据源证明 H 是菜品味觉的可靠测量，而非 Yelp 平台/餐厅质量的人工制品。

**具体任务**：

##### a. 多数据库 H 交叉验证（新增核心分析）
- [ ] **下载 & 处理 Food.com** — 按菜名匹配 158 道菜，取平均评分作为 H_foodcom
- [ ] **下载 & 处理 Google Local (UCSD)** — 餐厅子集，应用现有 NLP 管线提取 H_google
- [ ] **手动收集 TasteAtlas** — 查询 158 道菜评分作为 H_tasteatlas
- [ ] **下载 & 处理 Epicurious** — 按菜名匹配，取评分作为 H_epicurious
- [ ] **下载 & 处理 TripAdvisor (Zenodo)** — 应用 NLP 管线提取 H_tripadvisor
- [ ] **下载 & 处理 Reddit 食物板块** — 提取菜品讨论的情感得分作为 H_reddit
- [ ] **下载 Food-Pics Extended** — 匹配食物图片适口性评分作为 H_foodpics
- [ ] **下载 CROCUFID** — 匹配跨文化想吃度评分作为 H_crocufid
- [ ] **计算跨平台 H 相关矩阵** — 报告所有两两 Spearman rho
- [ ] **构建"H 共识分数"** — 多源 H 的加权平均或因子分析提取公因子

##### b. 混合效应模型（分离餐厅 vs 菜品效应）
- [ ] 建立 H_ij = β₀ + β_dish_i + β_restaurant_j + ε_ij
- [ ] 报告 dish-level ICC
- [ ] 需要 `dish_mentions_scored.parquet` 中有 restaurant_id 信息

##### c. 地域分层
- [ ] 按 US 地区（coastal vs inland 或 top 5 州）分别计算 H
- [ ] 报告跨地域 H 的 Spearman 相关

##### d. 人工验证（200 样本）
- [ ] 样本已生成：`results/tables/human_validation_sample.csv`
- [ ] 设计评分指南（盲评，仅评味觉 1-10）
- [ ] 需要 2-3 名评估者，报告 Krippendorff's alpha

##### e. H 构念效度诊断表
- [ ] H 与星级、价格档、评论长度的偏相关
- [ ] 控制后 dish-level 排名变化

**脚本**：
- `code/07b_download_external_data.py` — 自动下载 & 预处理外部数据集
- `code/07c_cross_validate_h.py` — 跨平台 H 相关分析
- `code/07d_h_validity.py` — 混合效应 + 地域分层 + 构念效度

---

#### P0-3. E 不确定性量化与传播（回应核心问题 #3）

**目标**：给 E 和 DEI 加上置信区间，量化排名不确定性。

**具体任务**：
- [ ] **a. E 的置信区间**
  - 基于 (i) 食谱份量 ±30% 变异 和 (ii) LCA 因子文献不确定性范围
  - 报告每道菜 E 的 5th/95th percentile
- [ ] **b. 不确定性传播到 DEI**
  - 用 Monte Carlo（≥10,000 次）同时扰动 H 和 E（含 E 的 LCA 因子不确定性）
  - 报告每道菜进入 Top10/Bottom10 的概率
  - 报告 rank interval（中位排名 ± 90% CI）
  - 报告"可互换区间"（排名差异不显著的菜品组）
- [ ] **c. 排名层级稳定性**
  - 在 5th/95th percentile E 情景下，有多少菜品换层（Top/Middle/Bottom 三等分）
- [ ] **d. 食谱交叉验证**
  - 至少 20 道菜的食谱与 USDA FoodData Central 交叉比对

**脚本**：新建 `code/07e_uncertainty.py`

---

### P1：不做会被要求再修的项目

#### P1-1. 浪费空间重定义（回应核心问题 #4）

**具体任务**：
- [ ] **a. 品类内替代分析**
  - 定义功能品类（蛋白主菜、谷物菜、甜品、饮品、沙拉/冷菜、汤类）
  - 在每个品类内做 Pareto 分析 + 替代网络
- [ ] **b. 菜系内 Pareto 分析**
  - 对 13 个菜系分别识别 Pareto 前沿
  - 报告"菜系内最优替代"
- [ ] **c. 替代网络可视化**（Codex 建议主图）
  - 高 E → 低 E 的可替代边，标注 H 损失与 E 节省
  - 限定为"同品类/同菜系"的有意义替代

**脚本**：新建 `code/07f_within_category.py`

---

#### P1-2. 明确假设 + 非同义回归（回应核心问题 #5）

**具体任务**：
- [ ] **a. 陈述可检验假设**
  - H1: "菜品间味觉差异远小于环境代价差异"（核心发现）
  - H2: "植物基/生食菜品在味觉不损失的前提下环境效率更高"
  - H3: "菜品 DEI 主要由食材驱动，而非菜系传统"
- [ ] **b. 非 E 成分回归**
  - DEI rank ~ % plant-based + raw/cold 比例 + ingredient count + cuisine age + cultural diffusion
  - 这是真正有趣的问题：什么 non-E 特征预测高 DEI？
- [ ] **c. 行为/政策文献引用**
  - 引入 nudge 文献（Thaler & Sunstein）
  - 讨论 DEI 标签在什么条件下能改变消费行为
  - 引用现有可持续饮食标签研究（EWG Food Scores, Eco-Score, Planet-Score）

**脚本**：新建 `code/07g_hypothesis_regression.py`

---

#### P1-3. 排名概率图（Codex 建议主图 #1）

**具体任务**：
- [ ] 基于 P0-3 的 Monte Carlo 结果
- [ ] 绘制 rank distribution + Top/Bottom membership probability
- [ ] 报告中位 rank shift 和 90th percentile rank shift（而非仅"44.1% ±3 名"）

**脚本**：并入 `code/07e_uncertainty.py`

---

### P2：做了显著加分的项目

#### P2-1. 时间稳定性
- [ ] 统计评论年份分布
- [ ] 测试 H 是否随评论年份系统性变化（H ~ year 回归）

#### P2-2. 第二 LLM 交叉验证
- [ ] 用 GPT-4o 或 Claude 对相同 200 条评论评分
- [ ] 报告与 DeepSeek 评分的一致性（ICC 或 Krippendorff's alpha）

#### P2-3. 留一菜品交叉验证 (Leave-one-dish-out CV)
- [ ] 重新评估 BERT 微调的 dish-level r（避免菜品信息泄漏）

#### P2-4. 政策导向 E 权重
- [ ] 碳社会成本权重（carbon shadow price）
- [ ] 区域水资源稀缺性权重
- [ ] 报告不同权重下排名变化

#### P2-5. 样本代表性讨论
- [ ] 明确讨论 158 道菜的选择框架
- [ ] 承认缺失的品类：主食、早餐、植物肉替代品、超加工食品、非洲/中东/中亚/南美菜系
- [ ] 讨论扩展到 ≥300 道菜的可行性

#### P2-6. 统计呈现改进
- [ ] 全文报告 95% CI
- [ ] 分半信度加 bootstrap CI
- [ ] OLS 使用 HC3 robust standard errors
- [ ] Monte Carlo 报告中位 rank shift + P90 rank shift

---

### P3：锦上添花（可作为后续研究）

#### P3-1. 营养调整 DEI_N
- [ ] DEI_N = H / (E / N)，N 为营养密度（如 NRF 9.3）
- [ ] 讨论纯 DEI 不能替代膳食规划

#### P3-2. 消费者实验（pilot）
- [ ] Conjoint analysis 或 discrete choice experiment
- [ ] N≥200 的选择实验，测试 DEI 标签效果

#### P3-3. 动态 DEI（季节/地域调整 E）
- [ ] 子集菜品的季节性 E 变化

#### P3-4. 行星边界框架
- [ ] 将 E 嵌入 planetary boundaries（Rockström 2009, EAT-Lancet 2019）

#### P3-5. 供给侧分析
- [ ] 哪些替代对餐厅最有利可图？

---

## 四、写作层面修改

| # | 修改点 | 来源 |
|---|--------|------|
| 1 | **标题重构**：突出实证发现而非方法。建议方向："The Hedonic-Environmental Decoupling in Food: Most Dishes Deliver Similar Pleasure at Vastly Different Environmental Costs" | NF 编辑 |
| 2 | **摘要重构**：以反直觉发现（H 方差小、E 方差大）开头，而非方法管线 | NF + Codex |
| 3 | **主叙事**：从"我们发明了新指数"改为"我们量化了现实饮食的高浪费选择空间+可执行替代路径" | Codex |
| 4 | **Figure 1**：H vs E 散点图 + Pareto 前沿（告诉整个故事），NLP 管线移入 Methods/SI | NF 编辑 |
| 5 | **新增主图**：(a) 不确定性排名图 (b) 品类内替代网络图 (c) 跨平台 H 一致性图 (d) 跨地域效应对比图 | Codex + 新增 |
| 6 | **新增附表**：H 跨平台相关矩阵 + H 构念效度诊断（偏相关矩阵） | 新增 |
| 7 | **因果措辞降级**：全文改为关联与决策支持框架 | Codex |
| 8 | **与现有指数对比**：讨论 EWG Food Scores、Eco-Score、Planet-Score，DEI 的增量贡献 | NF 编辑 |
| 9 | **正文风格**：从技术报告改为 narrative arc（问题→洞察→启示） | NF 编辑 |

---

## 五、新增脚本规划

| 脚本 | 对应任务 | 预计复杂度 |
|------|---------|-----------|
| `code/07_dei_vs_e_ranking.py` | P0-1: DEI vs 1/E 对比 | 中 |
| `code/07b_download_external_data.py` | P0-2a: 下载 & 预处理外部数据集 | 高 |
| `code/07c_cross_validate_h.py` | P0-2a: 跨平台 H 相关分析 | 高 |
| `code/07d_h_validity.py` | P0-2b-e: 混合效应 + 地域分层 + 构念效度 | 高 |
| `code/07e_uncertainty.py` | P0-3 + P1-3: E 不确定性 + 排名概率 | 高 |
| `code/07f_within_category.py` | P1-1: 品类内替代 + 替代网络图 | 中 |
| `code/07g_hypothesis_regression.py` | P1-2: 非 E 回归 + 假设检验 | 中 |

---

## 六、执行路线图

```
阶段 0（今天开始，数据获取）⭐ 最优先
├── 立即下载：Food.com, Google Local, Epicurious, TripAdvisor, Reddit Pushshift
├── 立即下载：Food-Pics, CROCUFID, ASAP
├── 手动收集：TasteAtlas 158 道菜评分
├── 发送 Outreach 邮件：Finlayson/SatMap, CSIRO, US Army DTIC
└── 启动 UK Biobank 申请流程

阶段 1（数据到位后，纯计算）
├── P0-1: DEI vs 1/E 对比分析
├── P0-2a: 多源 H 交叉验证（Food.com + Epicurious + TasteAtlas 先行）
├── P1-3: 概率排名图（扩展现有 Monte Carlo）
├── P2-1: 时间稳定性
└── P2-6: 统计呈现改进（CI, HC3, bootstrap）

阶段 2（大数据处理，中等工作量）
├── P0-2a续: Google Local + TripAdvisor + Reddit 的 NLP 管线处理
├── P0-2b: 混合效应模型
├── P0-2c: 地域分层
├── P0-2e: 构念效度诊断
├── P0-3: E 不确定性量化
├── P1-1: 品类内替代分析
└── P1-2: 假设 + 非 E 回归

阶段 3（需外部资源/人力）
├── P0-2d: 人工验证 200 样本（需招评估者）
├── P2-2: 第二 LLM 交叉验证（需 API 调用）
├── P0-3d: USDA 食谱交叉验证（需手动比对）
├── 处理 Outreach 回复的数据（UK Biobank, SatMap 等）
└── P2-5: 样本扩展讨论

阶段 4（写作修改）
├── 标题 + 摘要重构
├── 叙事弧线重塑
├── 新图新表整合（含跨平台 H 验证图）
└── 因果措辞审查
```

---

## 七、风险评估

| 风险 | 影响 | 应对策略 |
|------|------|---------|
| 跨平台 H 相关低（不同数据源给出矛盾的 H 排名） | H 分数可信度崩塌 | 分析差异来源（平台偏差 vs 菜品差异），取多源共识 H |
| 混合效应模型显示 restaurant 效应 > dish 效应 | H 测的是餐厅而非菜品 | 提取 dish 残差作为"纯菜品 H"，重新计算 DEI |
| Google Local 数据量过大处理困难 | 时间和计算资源不足 | 按州采样 10% 处理 |
| 人工验证与 BERT H 相关低 | 动摇整个 H 管线 | (a) 分析不一致来源 (b) 用人评直接作为 H 的 gold standard |
| E 不确定性传播后排名大幅变化 | "排名无意义" | 转向聚类分析（高/中/低效率组）而非精确排名 |
| 第二 LLM 评分与 DeepSeek 差异大 | 质疑 LLM 标注一致性 | 取多 LLM 均值或训练集成模型 |
| DEI vs 1/E 几乎无差异 | "H 是冗余的" | 全面转向 "decoupling finding" 叙事 |
| Outreach 无回复 | 缺少学术基准数据 | 已有足够多的免费下载源覆盖 |

---

## 八、预期成果：多源 H 验证如何强化论文

如果跨平台 H 一致性高（Spearman rho > 0.7）：
- 证明 H 是菜品味觉的稳健测量，不是 Yelp 特异现象
- 论文增加一张 **"跨平台 H 一致性矩阵图"** 作为核心验证证据
- 可构建 **"多源共识 H"** 作为更可靠的享乐分数
- 直接回应两位审稿人关于 H 外部效度的核心质疑

如果跨平台 H 一致性低（rho < 0.5）：
- 这本身就是重要的实证发现："味觉评价具有平台/语境依赖性"
- 转向分析：是哪些菜品在不同平台上评价一致/分歧？
- 报告每个平台的 DEI，分析哪些结论跨平台稳健

无论结果如何，多源验证都大幅提升论文的方法论严谨度和可信度。

---

*本计划将根据阶段 0 数据获取进度和阶段 1 初步分析结果动态调整后续优先级。*
