# DEI 论文修改计划：回应五项批评 + 新概念

## 背景
论文收到五项批评：(C1) H分数压缩不合理，(C2) Yelp代理偏差，(C3) 缺少营养维度，(C4) 菜品功能不对等，(C5) 需引入"精致化成本曲线"新概念。基于334道菜/29菜系的combined数据集进行全部修改。

---

## 新增脚本（6个）

### 11a: H分数压缩分析 (`code/11a_h_decompression.py`)
**解决 C1**：H range [6.05, 7.57]，CV=3.9%，top-bottom差仅0.04分不合理

**四项分析**：
1. **压缩量化**：对比Yelp星级的dish-level CV vs BERT H的CV，计算"压缩因子"
2. **Beta分布反压缩**：将H的经验CDF映射到beta分位数函数，扩展至[3,9]目标范围（保序）
3. **敏感性分析**（核心图）：模拟CV=5%/10%/15%/20%时H对Var(log DEI)的贡献百分比，找到"交叉点"（H贡献≥10%时需要多大CV）
4. **排名稳定性**：在各反压缩场景下DEI排名的Spearman相关

**输出**：`h_compression_analysis.csv`, `h_decompression_sensitivity.csv`, `h_decompression_sensitivity.png`, `h_rescaled_distribution.png`

---

### 11b: Yelp代理偏差控制 (`code/11b_yelp_proxy_controls.py`)
**解决 C2**：餐厅ICC=8.3% > 菜品ICC=2.4%，68.2% H方差由混杂因素解释

**方法**：两阶段残差化
1. Review级回归：`H_ij = α + β₁·biz_stars + β₂·price_range + β₃·log(review_count) + β₄·text_len + ε`
2. 提取残差作为"控制后H"：`H_controlled = H - (β₁·stars + β₂·price + β₃·log(count))`
3. 按dish聚合，重新计算 `log_DEI_controlled = log(H_controlled) - log(E)`
4. 对比原始DEI vs 控制后DEI：排名相关、tier变化、Pareto前沿变化

**数据源**：`dish_mentions_scored.parquet` + `expanded_dish_mentions.parquet` + `restaurants.parquet`
**输出**：`controlled_dei_rankings.csv`, `proxy_bias_summary.csv`, `controlled_vs_original_dei.png`

---

### 11c: 营养维度 (`code/11c_nutritional_dimension.py`)
**解决 C3**：kimchi vs brisket功能不对等，缺少蛋白质/热量/微量营养素维度

**步骤**：
1. **营养数据库**：为101种食材硬编码USDA FoodData Central数据（每kg: protein_g, fat_g, carb_g, fiber_g, iron_mg, zinc_mg, b12_ug, calcium_mg, vitamin_c_mg, calorie_kcal）
2. **菜品营养谱**：用DISH_RECIPES + EXPANDED_RECIPES计算334道菜的每份营养
3. **NDI指数**：采用Drewnowski (2009) NRF框架，7项鼓励营养素，每100kcal标准化
4. **DEI-N公式**：`log(DEI_N) = log(H) + α·log(NDI) - log(E)`，α默认0.5，敏感性[0,1]
5. **3D Pareto分析**：(H, E, NDI)三维空间的Pareto前沿
6. **餐点角色分类**：按热量分为 Side(<200kcal) / Light(200-400) / Full(400-700) / Heavy(>700)

**输出**：`ingredient_nutrients.csv`, `dish_nutritional_profiles.csv`, `dei_n_rankings.csv`, `3d_pareto_hne.png`, `dei_vs_dein_comparison.png`

---

### 11d: 类内分析 (`code/11d_within_category_analysis.py`)
**解决 C4**：跨类比较（沙拉 vs 主菜）夸大了替代可行性

**依赖**：11c的`dish_nutritional_profiles.csv`

**方法**：
1. **基于食谱的分类**：按主要蛋白质来源+热量角色分为10-12个功能类别（Red Meat / Poultry / Seafood / Plant Protein / Starch / Salad / Soup / Dessert / Beverage等），目标<5%落入"Other"
2. **类内DEI排名**：每个类别内的DEI排名和Pareto前沿
3. **类内方差分解**：类内H的CV是否高于全局？（预期更高，证明H在同类中更有区分力）
4. **营养约束替代**：替代品须满足 protein≥50%原品 AND 热量±50% AND E减少≥30% AND H损失<1分

**输出**：`comprehensive_category_assignment.csv`, `within_category_dei_rankings.csv`, `within_category_variance.csv`, `nutrition_constrained_substitutions_full.csv`, `within_category_dei_panels.png`

---

### 11e: 精致化成本曲线 (`code/11e_refinement_curve.py`)
**解决 C5**：新概念——同一菜品从基础到精致，E增长远快于H增长

**步骤**：
1. **定义17个菜品家族**：curry(10道), noodle(11道), rice(9道), chicken(9道), beef(10道), salad(8道), dessert(7道), seafood(8道) 等，每家族3-11个变体
2. **精致度量化**：`refinement = 0.3·log(n_ingredients) + 0.2·log(total_grams) + 0.2·animal_protein_fraction + 0.15·cooking_energy + 0.15·high_impact_share`
3. **核心模型**：`H_fi = H_base_f + α_f · log(E_fi / E_base_f) + ε`
   - α_f = "环境弹性"：每翻倍E获得的H增益（预期0.5-1.5分）
4. **缺失基础版本**：用食谱模拟（去除精致化食材、降级烹饪方式）生成base variants
5. **全局拟合**：混合效应模型 `H - H_base = α_global · log(E/E_base) + family_RE + ε`
6. **政策图**：展示每个家族中前50%精致化捕获了80%+的味觉提升

**输出**：`dish_families.csv`, `refinement_curves.csv`, `refinement_simulated_variants.csv`, `refinement_cost_curves.png`（核心新图）, `refinement_global_fit.png`, `hedonic_waste_by_family.png`

---

### 12: 整合脚本 (`code/12_integrated_revision.py`)
**依赖**：所有11a-11e输出

1. 汇总五项修改的影响矩阵
2. 生成修订后的Figure 1（控制后H、营养标注、类别编码）
3. 更新combined数据集新增列：`H_controlled`, `NDI`, `DEI_N`, `category_recipe`, `family`, `refinement_score`
4. 生成中文报告更新段落

**输出**：`revision_summary.csv`, `revision_impact_matrix.csv`, 更新 `combined_dish_DEI_revised.csv`

---

## 依赖关系与执行顺序

```
11a (H压缩)     ──┐
11b (代理偏差)   ──┤  可并行
11c (营养维度)   ──┤
11e (精致化曲线) ──┘
                    │
11d (类内分析)  ←── 11c  (需要营养数据)
                    │
12 (整合)       ←── 全部11x
```

## 关键依赖文件
- `code/config.py` — 需扩展：NUTRIENT_DATA, DISH_FAMILIES
- `code/04_env_cost_calculation.py` — DISH_RECIPES（158道）的E计算模板
- `code/09b_expanded_recipes.py` — EXPANDED_RECIPES（176道）
- `code/07d_h_validity.py` — ICC分解和残差化逻辑模板
- `data/combined_dish_DEI.csv` — 334道菜主数据
- `data/dish_mentions_scored.parquet` + `data/expanded_dish_mentions.parquet` — review级数据
- `data/restaurants.parquet` — 餐厅元数据

## 最大工作量：USDA营养数据
101种食材 × 10项营养素 = 1,010个数据点，需从USDA FoodData Central手动编码。采用与DISH_RECIPES相同的字典硬编码方式，注释中标注USDA FDC ID。

## 预期对核心结论的影响

| 批评 | 原始结论 | 修订后预期 | 影响程度 |
|------|---------|-----------|---------|
| C1 H压缩 | H贡献0.8% | 需CV>20%才贡献10%+，排名rho>0.95 | 低（结论不变，但需诚实披露） |
| C2 代理偏差 | 未控制 | 控制后DEI rho>0.85（E主导缓冲了H变化） | 中 |
| C3 营养 | 未纳入 | 牛肉菜升20-40位，3D Pareto更均衡 | **高** |
| C4 类内 | 跨类比较 | 类内结论相似，H区分力更强 | 中 |
| C5 精致化 | 无 | α≈0.5-1.5，翻倍E仅获0.5-1.5分H | **新贡献** |

## 验证方式
- 每个脚本末尾打印摘要统计
- 12_integrated 生成全面对比表
- 所有图表保存至 `results/figures/` 和 `results/tables/`
- 运行模式：`cd /c/project_EF && PYTHONIOENCODING=utf-8 $PYTHON code/11a_h_decompression.py`

**预计总运行时间**：~10-15分钟（无GPU/API调用，纯统计计算）
