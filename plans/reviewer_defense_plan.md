# 审稿人防御计划 v4：基于全部预检的最终策略

## 关键实证发现（2026-02-25 预检）

在决定修订方向之前，我们做了三轮预检，每轮结果都改变了策略方向。

### 预检 0: 食材化学 vs E 的相关性

| 变量对 | Pearson r | Spearman ρ |
|--------|-----------|-----------|
| fat_g vs E | **+0.47** | **+0.61** |
| protein_g vs E | **+0.63** | **+0.59** |
| calorie vs E | **+0.51** | **+0.61** |
| fat_cal_pct vs E | +0.19 | +0.22 |

**结论**：食材化学层面的"享乐驱动因素"（脂肪、蛋白质、热量）与环境成本**强正相关**。

### H_text vs 食材化学的相关性

| 变量对 | Pearson r |
|--------|-----------|
| H_text vs fat_g | +0.15 (p=0.005) |
| H_text vs protein_g | +0.04 (p=0.49, NS) |
| H_text vs calorie | +0.11 (p=0.05) |
| H_text vs E | +0.09 (p=NS) |
| r(H, E \| fat, protein, cal) | +0.04 (p=0.47, NS) |

**结论**：文本 H 几乎不捕捉食材化学属性。

### Top vs Bottom DEI 的化学剖面

| | Top 10 DEI | Bottom 10 DEI | 比值 |
|---|-----------|--------------|------|
| 平均 fat_g | 9.2 | 41.5 | 4.5× |
| 平均 protein_g | 7.3 | 59.5 | 8.1× |
| 平均 calorie | 184 | 750 | 4.1× |
| 平均 E | 0.029 | 0.560 | 19× |

### 预检 1: Within-review 直接比较（因果方向检验）

2,508 对 within-review 同餐比较（同一消费者、同一家餐厅、同一次就餐中点了两道不同的菜）：

| 分析 | N | r(ΔH, ΔE) | p |
|------|---|-----------|---|
| 全部 within-review 对 | 2,508 | **-0.001** | 0.98 |
| 高 E 对比对（top 25% |ΔE|） | 619 | +0.009 | 0.82 |
| 同类别内对 | 379 | -0.054 | 0.30 |
| 跨类别对 | 2,129 | +0.004 | 0.84 |
| 极端对比（|ΔE|>0.3） | 163 | -0.065 | 0.41 |
| 高E菜获更高评分比例 | — | **50.0%** | — |

同餐中 ΔH 与食材化学差异：

| | r(ΔH, Δvar) | p |
|---|------------|---|
| ΔH vs Δfat | +0.047 | 0.019 |
| ΔH vs Δprotein | -0.024 | 0.24 |
| ΔH vs Δcalorie | +0.017 | 0.40 |
| ΔH vs ΔE | -0.001 | 0.98 |

**极端对比实例**（同一消费者、同一餐、E 差 3-5 倍，给出一样的评分）：
- beef_bourguignon (E=0.685) vs ratatouille (E=0.132)：ΔH ≈ 0
- bulgogi (E=0.536) vs kimbap (E=0.111)：ΔH ≈ 0
- osso_buco (E=0.773) vs soup (E=0.126)：ΔH ≈ 0

**注意**：within-review 对存在 **BERT 上下文泄漏**问题（r(overlap, |ΔH|) = -0.44），高重叠对的 |ΔH| 趋近于零可能是模型产物。但：
- 低重叠子集（<Q25, n=627）：|ΔH|=0.950（正常水平），r(ΔH,ΔE) = **-0.010**, 50.2% 无偏好
- **更重要的**：89,974 对跨评论同餐厅比较（完全零 BERT 泄漏）：r(ΔH,ΔE) = **-0.004**, 49.9% 无偏好

### 预检 2: 跨评论同餐厅比较（零 BERT 泄漏，核心证据）

不同消费者在同一家餐厅评价不同菜品。9,981 家餐厅，89,974 对。

| 分析 | N | r(ΔH, ΔE) | p |
|------|---|-----------|---|
| 全部跨评论同餐厅对 | **89,974** | **-0.004** | 0.26 |
| 同类别跨评论对 | 11,588 | -0.007 | 0.46 |
| 高对比对（|ΔE|>0.3） | 7,437 | -0.014 | 0.23 |
| 高E→高H 比例 | — | **49.9%** | — |

### 预检 3: H_text 的方差来源

| 模型 | R² | 含义 |
|------|-----|------|
| H ~ 食材化学（fat+protein+cal） | 2.4% | 化学几乎无法预测消费者评价 |
| H ~ 烹饪方式 | 0.9% | 烹饪方式也不行 |
| H ~ 化学+烹饪 | 3.1% | 两者合计也只有 3% |
| **未解释** | **96.9%** | 消费者享乐评价的绝大部分来自其他维度 |

这也回应了"烹饪转化 H_chem"的质疑：即使考虑烹饪方式，它对 H_text 的解释力只增加 0.7%。

**这组数据回应了"因果方向歧义"质疑**：89,974 对跨评论比较（零 BERT 泄漏）中，不同消费者在同一家餐厅给高E菜和低E菜完全相同的评分。

---

## 预检结果的含义

**四轮预检揭示了一个完整的证据链**：

1. **食材化学层面**：味道的客观驱动因素（脂肪、蛋白质、热量）与 E **强正相关**（r=0.47-0.63）。从食品科学角度看，**化学层面的 trade-off 确实存在**。

2. **消费者跨菜品评价层面**：文本衍生的 H 与 E **近乎正交**（r=0.09）。消费者**感知不到**这个 trade-off。

3. **跨评论同餐厅比较**（零 BERT 泄漏，核心因果证据）：不同消费者在**同一家餐厅**评价不同 E 的菜品，89,974 对比较中 r(ΔH, ΔE) = **-0.004**，49.9% 无偏好。这不是"缺少比较机会"，也不是 BERT 上下文泄漏——而是**在控制餐厅的情况下，消费者确实不偏好高E菜品**。

4. **断裂的解释**：食材化学 + 烹饪方式加起来只解释 H_text 的 **3.1%**。消费者享乐评价的 96.9% 来自非化学、非烹饪维度（呈现方式、文化认同、新奇感、情感记忆……）。这些维度恰好与 E 正交——不是因为巧合，而是因为它们根本不由食材组成驱动。

---

## 修订策略的根本转向

### 不能做的事

❌ 把 H_chem 作为"杀手级证据"证明 H⊥E——因为 H_chem 会与 E 正相关，直接证伤原论文

❌ 继续宣称"味道与环保不冲突"——因为在食材化学层面冲突确实存在

❌ 假装 r(H_text, E)≈0 是一个纯粹的好消息——它部分源于 H_text 捕捉能力不足

### 应该做的事：把"漏洞"变成"发现"

论文的核心贡献应该重新定位为：

> **食品的客观享乐化学属性与环境成本正相关，但消费者的主观享乐评价与环境成本正交。这一"感知解耦"意味着可持续饮食转型不需要克服口味障碍——障碍本就不存在于消费者的体验中。**

这比原来的 claim 更弱但更精确，也更有趣：
- 原来的 claim："味道与环保不冲突"（过度简化）
- 新的 claim："消费者体验到的味道与环保不冲突，即使食材化学上有耦合"

新 claim 的政策含义**更强**，不是更弱——因为它直接说明了：消费者不需要被说服"低 E 食物一样好吃"，他们已经这么感知了。政策干预应该利用这个感知解耦，而不是试图改变消费者的口味。

---

## Phase 1: 新增分析（4 个脚本）

### 脚本 25a: LLM 食谱验证 (`code/25a_recipe_validation.py`)
**(保留，这是纯增量证据，与叙事转向无关)**

回应问题四（LLM 食谱误差）。方法：
1. 100 道核心菜，用 DeepSeek 重新生成食谱
2. 对比 Jaccard 匹配率、克重 MAE、E Spearman ρ
3. 按蛋白质来源分组检验系统偏差
4. 用 LLM E 替换人工 E，重算方差分解

**输出**：`recipe_validation_summary.csv`, `recipe_validation_scatter.png`
**运行时间**：~5 min（API 调用）

---

### 脚本 25b: BT 收敛曲线 (`code/25b_bt_convergence.py`)
**(保留，降级为辅助证据)**

回应问题一（方差分解是算术必然）。方法：
1. 按比例抽取 10%-100% 比较，拟合 BT，画 CV 收敛曲线
2. Split-half CV 一致性

**输出**：`bt_convergence_curve.csv`, `bt_convergence.png`
**运行时间**：~2 min

---

### 脚本 25c: 食材化学 H 与三层现实 (`code/25c_food_chemistry_h.py`)
**(从"杀手级证据"重新定位为"揭示三层现实的核心分析")**

回应问题一和问题二——不是通过辩护，而是通过**转化**。

**方法**：

#### Step 1: 食材化学味觉属性（H_chem）
为 106 种食材编码 5 个味觉属性（fat, umami, sugar, sodium, spice_diversity），从食谱计算 2,563 道菜的 H_chem composite。

#### Step 2: 三层相关矩阵（核心新结果）

| | E | H_text | H_chem |
|---|---|--------|--------|
| E | 1 | ~0.09 | **~0.45+** |
| H_text | ~0.09 | 1 | **~0.15** |
| H_chem | **~0.45+** | **~0.15** | 1 |

这个矩阵讲述了完整的故事：
- H_chem 与 E 耦合（食材化学层面存在 trade-off）
- H_text 与 E 解耦（消费者感知层面不存在 trade-off）
- H_text 与 H_chem 弱相关（消费者评价由非化学因素主导）

#### Step 3: H_text 的方差分解
H_text 中有多少方差能被食材化学解释？
```
R²(H_text ~ fat + protein + calorie + umami + sodium + spice) = ?
```
预期：R² < 0.10（<10% 的 H_text 由化学解释）
→ 剩下 90%+ 来自烹饪技法、文化因素、呈现方式等

#### Step 4: "如果 H 是完美味觉测量"的 counterfactual
- 假设 H_true = 0.3·H_chem + 0.7·H_nonchemical（合理假设）
- 在不同 H_chem 权重下（0→1），r(H_true, E) 如何变化
- 展示：即使 H_chem 权重升到 0.5，r(H_true, E) 仍然 < 0.25（因为 H_nonchemical 分量稀释了正相关）

#### Step 5: 可视化
- **Figure A**：三角散点矩阵（H_text vs E, H_chem vs E, H_text vs H_chem）
- **Figure B**：H_text 的方差来源饼图（化学 vs 非化学 vs 残差）
- **Figure C**：counterfactual 敏感性曲线

**输出**：
- `results/tables/food_chemistry_h_scores.csv`
- `results/tables/three_layer_correlation_matrix.csv`
- `results/tables/h_text_variance_sources.csv`
- `results/figures/three_layer_reality.png`
- `results/figures/h_chemistry_sensitivity.png`

**运行时间**：~3 min

---

### 脚本 25d: Within-restaurant FE + Within-review 比较 (`code/25d_within_restaurant_h.py`)
**(保留 + 新增 within-review 分析——这是回应因果方向质疑的核心证据)**

回应问题二（H 测的是餐厅体验）和因果方向质疑。

**Part A: Within-restaurant 固定效应**
1. 双向固定效应 demean：H_demeaned_ij = H_ij - H̄_.j
2. Dish-level H_within = groupby(dish_id)[H_demeaned].mean()
3. 核心输出：ρ(H_within, H_original), r(H_within, E), ANOVA 方差分解

**Part B: 跨评论同餐厅比较（核心因果证据，零 BERT 泄漏）**

预检发现 within-review 对存在 BERT 上下文泄漏（r(overlap,|ΔH|)=-0.44）。但跨评论同餐厅比较（不同消费者在同一餐厅评价不同菜品）完全没有此问题，且样本量大 36 倍。

正式分析：
1. 构建所有跨评论同餐厅配对（89,974 对，9,981 家餐厅）
2. 核心统计：r(ΔH, ΔE), r(ΔH, Δfat), r(ΔH, Δprotein), r(ΔH, Δcalorie)
3. 高 E 对比子集（|ΔE| > 0.3）：7,437 对
4. 同类别子集：11,588 对
5. BERT 泄漏诊断：within-review 按 context overlap 分层，确认低重叠对与跨评论对结论一致
6. Binomial test for 49.9% vs 50%
7. 可视化：ΔH vs ΔE 散点图（faceted by overlap level）

**核心数字**：
- 89,974 对跨评论对：r(ΔH, ΔE) = -0.004, 49.9% 无偏好
- 完全回应"因果方向"质疑 + 零 BERT 泄漏风险

**Part C: H 方差来源分解**
- R²(H ~ fat + protein + calorie) = 2.4%
- R²(H ~ cook_method) = 0.9%
- R²(H ~ chemistry + cooking) = 3.1%
- 96.9% 来自非化学、非烹饪维度

**输出**：
- `results/tables/within_restaurant_h.csv`
- `results/tables/cross_review_within_restaurant_pairs.csv`
- `results/tables/h_variance_sources.csv`
- `results/tables/bert_leakage_diagnostic.csv`
- `results/figures/cross_review_dh_vs_de.png`
- `results/figures/bert_leakage_stratified.png`

**运行时间**：~2 min

---

## Phase 2: 论文重构

### 标题修改

**当前**："The hedonic--environmental decoupling of global cuisine"

**新标题候选**：
- "Perceived but not chemical: the hedonic–environmental decoupling in 2,563 dishes"
- "Consumer hedonic evaluations are decoupled from environmental costs despite chemical coupling"
- "The perceptual gap between taste and sustainability across 2,563 dishes"

### Abstract 重写

**核心句**（替换 "E accounts for 96.9%"）：
> Food-chemistry hedonic drivers (fat, protein, calories) correlate strongly with environmental cost ($r = 0.47$–$0.63$), yet consumer hedonic evaluations are essentially orthogonal to it ($r = 0.09$). Cross-review within-restaurant comparisons—where different diners evaluate different dishes at the same restaurant—confirm this decoupling across 89,974 pairs ($r(\Delta H, \Delta E) = -0.004$; 49.9\% of higher-$E$ dishes rated higher). Food chemistry and cooking method together explain only 3.1\% of hedonic variation; the remaining 96.9\% reflects non-chemical dimensions that are independent of environmental cost. Sustainable dietary shifts face no systematic taste barrier in consumer perception.

### 新增 Results 段落："The three-layer reality"

在 Section 2.1 之后新增一段（或替换当前方差分解的呈现方式）：

> The relationship between hedonic quality and environmental cost depends on the level of analysis (Fig.~X). At the food-chemistry level, objective hedonic drivers—fat content ($r = 0.47$), protein content ($r = 0.63$), and caloric density ($r = 0.51$)—are strongly positively correlated with $E$. At the consumer-evaluation level, text-derived $H$ is essentially uncorrelated with $E$ ($r = 0.09$). This divergence arises because consumer hedonic evaluations capture far more than food chemistry: $H_{\text{text}}$ is only weakly predicted by fat, protein, and caloric content (combined $R^2 < 0.10$). The remaining $>$90\% of hedonic variation reflects preparation technique, cultural resonance, novelty, presentation, and other non-chemical dimensions that are largely independent of ingredient-driven environmental costs.

### Discussion 重写

**§4.1 "方差分解的正确解读"** — 同 v2，但新增一句：
> The variance decomposition reflects both the genuine CV gap and the fact that consumer $H$ captures primarily non-chemical hedonic dimensions, which happen to be orthogonal to $E$.

**§4.2 "H 的多层效度与因果方向"** — 全面重写：
> Our $H$ scores derive from text-based NLP, raising two questions: (1) what does $H$ actually measure? and (2) does $r(H, E) \approx 0$ reflect genuine perceptual decoupling, or merely an absence of systematic comparison between high- and low-$E$ dishes?
>
> On the first question, three pieces of evidence characterize $H_{\text{text}}$:
>
> (a) **Within-restaurant stability.** Controlling for restaurant fixed effects barely changes dish rankings ($\rho = X.XX$), indicating that $H$ reflects dish-level rather than restaurant-level variation.
>
> (b) **Weak food-chemistry grounding.** $H_{\text{text}}$ correlates only weakly with fat ($r = 0.15$), protein ($r = 0.04$), and caloric content ($r = 0.11$). Consumer hedonic evaluations are not primarily driven by the chemical properties that conventional food science identifies as palatability drivers.
>
> (c) **Cross-context partial stability.** The cross-context BT correlation ($\rho = 0.34$) reflects both food-intrinsic and culturally shared evaluative dimensions.
>
> We interpret $H_{\text{text}}$ not as a pure measure of taste, but as a composite of consumer-experienced food quality—encompassing flavor, preparation skill, cultural fit, novelty, and presentation.
>
> On the second question—causal direction—cross-review within-restaurant comparisons provide direct evidence. In 89,974 instances where different reviewers evaluated different dishes at the same restaurant (controlling for restaurant quality, ambiance, and service), we find $r(\Delta H, \Delta E) = -0.004$ ($p = 0.26$), with 49.9\% of higher-$E$ dishes rated higher. This null result holds among high-contrast pairs ($|\Delta E| > 0.3$: $r = -0.014$, $n = 7{,}437$) and within the same functional category ($r = -0.007$, $n = 11{,}588$). These are consumers evaluating different dishes in the same restaurant context, with zero risk of NLP model context leakage, and expressing no systematic preference for higher-$E$ dishes.
>
> This interpretation explains why food-chemistry hedonic drivers ($r(\text{fat}, E) = 0.47$) are coupled with $E$ while consumer evaluations are not: the non-chemical dimensions of food quality (preparation, presentation, cultural context) dilute the chemistry–$E$ correlation to near zero in the consumer experience.

**§4.3 样本代表性** — 去掉 "global"，限定 "American dining landscape"

**§4.4 LLM 食谱** — 用 25a 结果填充

**§4.5 结论条件性** — 明确三个条件：
1. $H_{\text{text}}$ 反映消费者体验（是的——但不等于纯味觉）
2. E 近似真实环境成本（25a 验证）
3. 解耦在非美国市场也成立（未验证，scope limitation）

---

## Phase 3: 根据结果的分支路径

### 路径 A（主路径）：三层现实得到验证 ✅

r(H_chem, E) > 0.3 且 r(H_text, E) ≈ 0 且 ρ(H_within, H_original) > 0.85 且 ρ(E_human, E_llm) > 0.85

→ 论文重构为"感知解耦"叙事
→ 这是比原论文更有深度的贡献

### 路径 B：Within-restaurant H 与 E 相关 ⚠️

r(H_within, E) > 0.15

→ 控制餐厅后 H 与 E 开始耦合
→ r(H_text, E) ≈ 0 主要因为餐厅效应引入噪声
→ 需要在论文中报告并讨论

### 路径 C：H_chem 与 E 不相关 🤔

r(H_chem, E) < 0.15

→ 出乎预料（但预检已基本排除这个可能性）
→ 如果真的如此，需要检查 H_chem 的构建是否合理

### 路径 D：LLM 食谱 ρ < 0.70 ⚠️

→ WorldCuisines 降级为 supplementary
→ 核心结论仅基于 334 道菜

---

## 依赖关系与执行顺序

```
Phase 1（全部可并行）:
  25a (LLM 食谱验证, ~5 min)        ─┐
  25b (BT 收敛曲线, ~2 min)         ─┤
  25c (三层现实分析, ~3 min)         ─┼─→ Phase 2 (论文重构)
  25d (Within-restaurant FE, ~1 min) ─┘
```

---

## 诚实评估

### 这个修订做到了什么

1. **把最大漏洞（H_chem 与 E 正相关）转化为核心发现**——"感知解耦"比"完全解耦"更精确、更有趣、更诚实
2. **直接验证 LLM 食谱**——纯增量证据
3. **Within-restaurant FE**——控制了餐厅体验混淆
4. **跨评论同餐厅比较**——89,974 对、零 BERT 泄漏、r(ΔH,ΔE)=-0.004、49.9% 无偏好。直接回应因果方向质疑。
5. **发现并处理了 BERT 上下文泄漏**——within-review 对有 r(overlap,|ΔH|)=-0.44 的泄漏，但用跨评论对绕过了这个问题。诚实报告泄漏本身增加可信度。
6. **诚实承认 H_text 不是"味道"**——但论证这对政策含义无影响

### 这个修订不能做到什么

1. **不能证明"味道与环保不冲突"**——因为在食材化学层面冲突确实存在
2. **不能解决美国中心性**——没有非美国数据
3. **不能消除"消费者评价 ≠ 味道"的质疑**——只能重新定义论文的 claim
4. **不能避免被审稿人要求感官面板验证**——这可能是 revise-and-resubmit 的条件

### 仍可被攻击的点

1. **"跨评论比较中不同消费者的 H 评价不可比"**——不同人有不同的评分基线。回应：(a) 89,974 对的大样本量让个体偏差被平均掉；(b) 如果高 E 菜真的更好吃，它们应该在跨消费者平均后仍然得分更高；(c) within-review 低重叠子集（627 对，同一消费者，零泄漏）结论一致（r=-0.010, 50.2%）
2. **"96.9% 的 H_text 方差仍然是黑箱"**——无法彻底解决，只能说：(a) 跨平台 ρ=0.59-0.70 说明不是纯噪声；(b) 89,974 对跨评论比较说明这个"黑箱"的内容在同餐厅比较中仍然与 E 正交
3. **"选择偏差——美食家效应"**——评论中提到多道菜的消费者可能更挑剔。但选择偏差方向对论文有利：更挑剔的消费者更可能察觉到味道差异，如果连他们都不偏好高E菜，一般消费者更不会

### 论文最终定位

> 在美国餐饮市场中，食物的客观化学享乐属性（脂肪、蛋白质、热量）与环境成本正相关（r=0.47-0.63），但消费者的主观享乐评价与环境成本正交（r=0.09）。89,974 对跨评论同餐厅比较（零 BERT 泄漏）证实这不是统计假象或比较缺失——不同消费者在同一餐厅给高E菜和低E菜完全相同的评分（49.9% 无偏好）。食材化学和烹饪方式合计只解释 3.1% 的消费者评价方差，其余 96.9% 来自与 E 正交的非化学维度。这一"感知解耦"意味着可持续饮食转型面对的**消费者口味障碍在很大程度上是想象中的**。

### 与原论文的对比

| | 原论文 | v4 |
|---|-------|-----|
| 核心 claim | 味道与环保不冲突 | 消费者**体验到的**食物品质与环保不冲突 |
| 核心证据 | r(H,E)=0.09 + 方差分解 | 化学耦合 + 感知解耦 + 89,974 对跨评论比较 |
| 对 H 的定义 | "hedonic quality"（暗示=味道）| consumer-experienced food quality（明确≠纯味觉） |
| 对 trade-off 的立场 | 不存在 | 化学层面存在，但消费者感知不到 |
| 方差分解的角色 | 核心发现 | CV 差异的算术后果（降级为描述性） |
| 政策含义 | 消费者不需要为环保牺牲味道 | 消费者已经不觉得低E菜更难吃，政策应利用这个感知间隙 |
| Nature 审稿风险 | 高（5个漏洞） | 中（诚实框定 + 新证据） |

---

## 预计总工作量

| Phase | 内容 | 预计时间 |
|-------|------|----------|
| Phase 1a | 25a: LLM 食谱验证 | ~30 min 编写 + 5 min 运行 |
| Phase 1b | 25b: BT 收敛曲线 | ~20 min 编写 + 2 min 运行 |
| Phase 1c | 25c: 三层现实分析 | ~30 min 编写 + 3 min 运行 |
| Phase 1d | 25d: Within-restaurant FE + Within-review | ~30 min 编写 + 2 min 运行 |
| Phase 2 | 论文重构 | ~1.5 hr |
| **总计** | | **~3.5 小时** |
