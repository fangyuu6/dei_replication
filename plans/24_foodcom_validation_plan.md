# Food.com 跨情境验证 + 论文 Discussion 强化

## Context

当前论文最大的方法论弱点是：所有 H 验证（Yelp / Google Local / TripAdvisor）都来自**餐厅评论**平台，审稿人可以质疑 H 反映的是"餐厅体验"而非"食物本身"。Food.com 的评论来自**家庭厨师**评价自己烹饪的结果，天然消除了餐厅代理偏差。

但正如用户指出的：Food.com 验证修复了代理偏差问题，但"LLM 读文字 ≠ 人放进嘴里"的问题仍在。因此计划同时包含：
1. **Food.com 跨情境验证**（新脚本 `24_foodcom_validation.py`）
2. **论文 Discussion 重写**：统一段落诚实地界定文字测量的边界，并用 sensitivity 分析量化其影响上限

### 做完 Food.com 后的弱点层级

| 层级 | 弱点 | 状态 |
|------|------|------|
| ✅ 已修复 | 代理偏差（餐厅体验 vs 食物本身） | 若 ρ > 0.60，基本关闭 |
| ⚠️ 降级但未消失 | "LLM读文字 ≠ 人放进嘴里" | 循环论证风险——两个文本数据集互相验证。需定量界定影响边界 |
| ❌ 结构性不可修复 | BT 跨菜系可比性假设 | 只能透明披露 |
| ❌ 结构性不可修复 | 地理集中性（美国消费者视角） | Food.com 同样美国用户主导，不提供跨文化视角 |

**总体判断**：从"有趣但方法论存疑"→"有趣且方法论经过合理鲁棒性检验"

### 否决的备选方案：多源混池

曾考虑将 Yelp + Google Local + TripAdvisor + Food.com 合并为一个统一语料库来估计 H。否决原因：

1. **语境混淆**：Yelp 的 "The som tam here was explosive, best I've had in Philly" 与 Food.com 的 "Made this for dinner, substituted fish sauce with soy, kids loved it" 不是同一种语义信号。混合后 BERT 学到的是两种语境的平均值，而非更纯净的口味信号。
2. **丧失验证能力**：混池后无法再说"两个独立语料给出一致的排名"。独立来源的收敛才能回应"H 是否捕捉了真实口味"的质疑；而混池只是增大了 N、缩小了标准误，不能回答这个根本问题。
3. **语义空间不可比**：如果 BT 比较的 profile 是 2 条 Yelp + 1 条 Google + 1 条 Food.com，审稿人会问"你怎么知道这些评论在同一语义空间里可比"，无法回答。
4. **多一倍数据不等于更好**：统计功效的瓶颈不在 N（已有 77K+），而在 H 测量的 construct validity。

**结论**：保持 Yelp 做主源，Food.com 做完全独立的跨情境验证。

---

## 修改范围

### 文件 1: `code/24_foodcom_validation.py`（新建）

**模板**: 复用 `code/16_cross_platform_bert_scoring.py` 的 BERT 评分模式 + `code/07c_cross_validate_h.py` 的 Food.com 匹配逻辑 + `code/17_pairwise_ranking.py` 的 BT 管线。

#### Phase 0: 配置
- 路径：`ROOT/DATA/EXT/RESULTS/TABLES/FIGURES/MODEL_DIR`（同 16_）
- 加载 `data/combined_dish_DEI_v2.csv`（2,563 dishes）
- 构建 `DISH_ALIASES`（复用 07c 的 ~230 行别名字典 + `match_dish()` 函数）

#### Phase 1: 菜谱匹配
- 输入：`raw/external/recipes.parquet`（~522K 菜谱），已本地存在
- 两轮匹配：
  - Pass 1: `match_dish(recipe_name)` — 精确 + 前后缀剥离（homemade/easy/best/recipe 等）
  - Pass 2: 标准化后 token-set 匹配（菜名所有 token 出现在食谱名中）
- 输出：`data/foodcom_recipe_dish_map.csv`（RecipeId, recipe_name, dish_id, match_method）
- 预期匹配量：60-150 道菜（西方菜覆盖高，东南亚/中东覆盖低）

#### Phase 2: 评论提取与过滤
- 输入：`raw/external/reviews.parquet`（~1.4M 评论）
- Merge `recipe_dish_map`，过滤 `word_count >= 20`
- 每道菜最多 1,000 条（`MAX_PER_DISH = 1000`，同 16_ 模式）
- Food.com 评论整条就是针对单道菜，不需要 context window 提取
- 输出：`data/foodcom_mentions_bert.parquet`

#### Phase 3: BERT 评分 + 域偏移诊断
- 加载 `models/hedonic_bert_finetuned/`（BERT-base-uncased, regression）
- `score_batch(texts, batch_size=128, max_length=256)`，clip 到 [1.0, 10.0]
- **主分析**：直接用 Yelp 训练的 BERT，不做 domain adaptation
- **域偏移对照**：用 Food.com 的星级评分（Rating 1-5）对 BERT 原始输出做线性校准（`H_calibrated = a * H_bert_raw + b`，OLS fit on Food.com stars×2），然后比较校准前后的跨平台 ρ：
  - 若校准后 ρ 提升不大（Δρ < 0.05）→ 原始 BERT 已足够泛化，域偏移可忽略
  - 若校准后 ρ 明显提升（Δρ > 0.10）→ 域偏移是真实问题，需报告两个 ρ 值
  - 这个对照让我们在 ρ 中等（0.40-0.55）时能区分"真实情境差异"vs"BERT 域偏移"vs"两者兼有"
- 输出：更新 `data/foodcom_mentions_bert.parquet` 加 `H_bert` 和 `H_bert_calibrated` 列

#### Phase 4: 菜品级聚合
- `groupby("dish_id")["H_bert"].agg(["mean", "std", "count"])`
- 过滤 `count >= 5`
- 输出：`data/foodcom_dish_h_bert.csv`（dish_id, H_foodcom_bert, std, n, ci95）

#### Phase 5: BT 两两排名
- **Step 5a**: 对每道匹配菜选 5 条代表性评论（按 BERT 分数百分位 10/30/50/70/90 分层抽样）
- **Step 5b**: 如果匹配菜 ≤ 200，用 exhaustive C(n,2)；否则 anchor-bridging (20 anchors)
  - 预期 ~80 道菜 → C(80,2) = 3,160 对，可全排
- **Step 5c**: DeepSeek v3.2 via OpenRouter
  - 批量 10 对/调用，`MAX_WORKERS = 50`
  - Prompt 修改：`"restaurant review excerpts"` → `"home cooking review excerpts"`
  - Checkpoint 每 100 批次保存
- **Step 5d**: `choix.ilsr_pairwise(n, comparisons, alpha=0.01)`，归一化到 [1, 10]
- 输出：`data/foodcom_pairwise_wins.csv`，`data/foodcom_dish_h_pairwise.csv`

#### Phase 6: 统计比较（核心结果）
4 层对比（带 bootstrap 95% CI，2000 次）：

| 比较 | 类型 | 含义 |
|------|------|------|
| Food.com BERT H vs Yelp BERT H | NLP-NLP 跨情境 | 同一模型在不同文字语境下的收敛 |
| Food.com BT H vs Yelp BT H | Pipeline-Pipeline 跨情境 | 完整管线的独立复现 |
| Food.com star vs Yelp BERT H | 方法对比 | 星级评分（天花板效应）的参照 |
| Food.com BERT H vs Food.com star | 内部方法对比 | NLP 相对星级的增量信息 |

输出：`results/tables/foodcom_cross_context_validation.csv`

额外诊断：
- **排名移动者分析**（关键！）：Top 10 移动最大的菜——这些菜恰好证明 H 不是 100% 食物内在属性，有情境依赖成分。诚实报告，并论证"即使存在 X% 的情境依赖，E 仍主导 DEI 方差"
- H 分布对比（CV 对比）
- 域偏移诊断：校准前后 ρ 的变化量 Δρ

#### Phase 7: 可视化
- `results/figures/foodcom_cross_context_validation.png`：2 panel scatter (BT-BT, BERT-BERT)
- `results/figures/foodcom_h_distribution.png`：密度叠加图
- `results/figures/foodcom_rank_movers.png`：排名变动条形图

---

### 文件 2: `paper/main.tex` 修改

#### 修改 1: Section 3.4 "Cross-platform validation" → 新增段落（line ~293 后）

在现有 Google Local / TripAdvisor 结果之后，新增一段 Food.com 跨情境验证：

> To test whether hedonic rankings generalize beyond restaurant settings, we performed a cross-context validation using Food.com, where ~1.6M reviews are written by home cooks evaluating dishes they prepared themselves. Applying the same finetuned BERT model to N matched dishes yields ρ = X.XX (p < Y). The full BT pipeline independently replicated on Food.com reviews produces ρ = X.XX. These cross-context correlations—from entirely different consumption settings (restaurants vs. home kitchens)—provide evidence that H captures food-intrinsic hedonic quality rather than restaurant-experience confounders.

#### 修改 2: Discussion "Strengths and limitations" 重写（lines 372-384）

当前 limitations 是分散的、轻描淡写的。重写核心策略：**承认弱点、但论证弱点影响有界**

**段落 A**: 收敛的双重解释——主动提出，不让审稿人来提

> The cross-context convergence between restaurant reviews (Yelp) and home-cooking reviews (Food.com) admits two interpretations: (a) H captures a stable intrinsic hedonic property of food, independently measured across consumption settings; or (b) both platforms share a common American food discourse—users encode "deliciousness" using the same cultural vocabulary, producing correlated rankings without necessarily reflecting objective taste quality. We cannot fully distinguish these interpretations, because both user populations are predominantly American. The rank-mover analysis provides partial evidence: N dishes show substantial rank shifts between contexts (e.g., [examples]), indicating that H is not purely food-intrinsic but contains a context-dependent component of approximately X%. Crucially, this context-dependent variation does not alter the core finding: even after excluding the most context-sensitive dishes, E accounts for >Y% of Var(log DEI).

**段落 B**: 文字测量局限性的定量边界

> Our hedonic scores are text-derived proxies, not controlled sensory panel measurements. The cross-context convergence (ρ = X.XX) demonstrates stability across consumption settings, but does not rule out systematic biases shared by both text corpora. Sensitivity analysis bounds the practical impact: the H CV threshold required to overturn E-dominance (H contributing >50% of Var(log DEI)) is approximately 40%, whereas both independent text corpora yield H CV of 3–4% (BERT) or 10–34% (BT). Even under extreme survivorship assumptions (2,563 ghost dishes, Δ=2.0), E accounts for 92.6% of variance. Text-based measurement is a genuine limitation, but its quantitative impact is bounded well below the tipping threshold.

**段落 C**: 跨文化边界的诚实界定

> Both Yelp and Food.com user bases are predominantly American. The H estimates for Pareto-optimal dishes from Thai, Lebanese, and Korean cuisines reflect American consumers' perception of these foods, not hedonic valuation within the originating cultures. Our finding that "som tam is highly delicious" is more precisely stated as "som tam is perceived as highly delicious by American restaurant-goers and home cooks." This distinction does not undermine the policy relevance of our findings—dietary sustainability interventions in Western consumer markets target precisely this population—but limits the generalizability of dish-level H rankings to non-Western dietary contexts.

#### 修改 3: Extended Data Table 2（robustness summary）新增行

```latex
Food.com cross-context (BERT) & $\rho$ = X.XX & $p$ < Y, N dishes \\
Food.com cross-context (BT) & $\rho$ = X.XX & $p$ < Y, N dishes \\
```

#### 修改 4: Abstract（line ~49）新增

在 robustness 列表中添加 `cross-context home-cooking validation (ρ = X.XX)`

#### 修改 5: Methods section 新增子段

描述 Food.com 数据来源、匹配方法、评分管线。

---

## 执行步骤

1. **编写 `code/24_foodcom_validation.py`** — Phase 0-7 全部整合在一个脚本
2. **运行脚本** — 预计 ~20 min（BERT 评分 + API 调用）
3. **根据实际 ρ 值更新 `paper/main.tex`** — 5 处修改
4. **验证论文编译** — `pdflatex main.tex`

## 关键复用

| 功能 | 来源文件 | 复用内容 |
|------|----------|----------|
| 菜名匹配 | `07c_cross_validate_h.py:47-250` | `DISH_ALIASES` + `EXTRA_ALIASES` + `match_dish()` |
| BERT 加载/评分 | `16_cross_platform_bert_scoring.py:248-277` | `load_bert_model()` + `score_batch()` |
| 评论抽样 | `16_cross_platform_bert_scoring.py:382-392` | `sample_mentions(max_per_dish=1000)` |
| 聚合 | `16_cross_platform_bert_scoring.py:425-443` | `aggregate_dish_h()` |
| BT 管线 | `17_pairwise_ranking.py:52-234` | `select_representative_reviews()` + `make_batch_prompt()` + BT fit |
| API 调用 | `17_pairwise_ranking.py` | OpenRouter/DeepSeek, ThreadPoolExecutor |

## 风险与预案

| 风险 | 影响 | 预案 |
|------|------|------|
| 匹配率低（<40 道菜） | 统计功效不足 | 如实报告；40 道菜仍可给出 ρ 估计 |
| ρ < 0.40 | 弱收敛 | 检查域偏移对照（校准后 ρ 是否提升）；拆分西方菜 vs 非西方菜 |
| ρ 中等（0.40-0.55） | 解释模糊 | 域偏移对照是关键：若 Δρ_calibration > 0.10 → 域偏移是主因；若 Δρ < 0.05 → 真实情境差异 |
| ρ > 0.80 | 太强，可能被质疑为共享文化编码 | 用排名移动者分析展示非100%收敛的菜品，论证情境依赖成分 |
| API 调用失败 | BT 阶段中断 | Checkpoint 机制，可断点续跑 |

## 验证方式

1. 脚本运行完成后检查输出文件完整性
2. 核对 `results/tables/foodcom_cross_context_validation.csv` 中 ρ 值和 p 值
3. 目视检查 scatter plot 的分布合理性
4. 论文中所有 X.XX 占位符替换为实际数值
5. `pdflatex` 编译无错误
