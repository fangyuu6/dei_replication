# 计划：用 2,563 道菜完整数据集重新运行所有可扩展分析

## Context
论文已从 334 道菜扩展到 2,563 道菜（加入 WorldCuisines 2,229 道新菜），但多数核心分析（分类替代、膳食组合、NDI、OLS回归、refinement curves 等）仍只用旧 334 道菜数据。需要创建一个综合脚本重新运行这些分析，然后更新论文。

## 实施步骤

### Step 1: 创建 `code/23_full_dataset_analysis.py`

一个脚本完成所有工作，预计运行 <2 分钟，无需 API 调用。

#### 数据加载与统一
- 加载 `data/combined_dish_DEI_v2.csv` (2,563 dishes)
- 加载 `data/expanded_dish_env_costs_v2.csv` (2,230 new dishes, 有 `ingredients_json`, `primary_coarse`, `calories_approx`, `protein_g_approx`)
- 从 `04_env_cost_calculation.py` 和 `09b_expanded_recipes.py` 提取原始 334 道菜的食谱
- 从 `ingredients_json` 解析新菜食谱，构建统一 `all_recipes` 字典 (2,563 entries)

#### 数据丰富化（为 v2 增加 5 列）
1. **calorie_kcal, protein_g**: 从食谱食材计算（复用 `11c` 的 `NUTRIENT_DATA` 101种食材营养数据）
2. **NDI**: NRF-7 框架，7 种营养素/100kcal 密度指数
3. **meal_role**: 按热量分 4 档 (Side <200, Light 200-400, Full 400-700, Heavy ≥700)
4. **category**: 13 功能类别，先用食材权重分类（复用 `11d` 的 `classify_dish()`），再用 `primary_coarse` 做 fallback

保存丰富后的 `data/combined_dish_DEI_v2.csv`。

#### 分析 A: 类别内方差分解 (Section 2.3, Table 3)
- 13 类别 × 2,563 道菜的 within-category H/E 贡献
- 营养约束替代矩阵 (E↓≥30%, H↓<1, protein≥50%, cal±50%)
- 输出: `within_category_variance_v2.csv`, `nutrition_constrained_substitutions_v2.csv`
- 更新论文: "334 dishes"→"2,563 dishes", "1,246 viable swaps"→新数, "49.1%"→新值

#### 分析 B: 膳食级分析 (Section 2.7, Table 5)
- Within-role 方差分解 (4 roles)
- 膳食组合: mains × sides, 约束 500-1500kcal + ≥20g protein
- 热量等效替代配对
- 更新论文: "15,073 meal-level combinations"→新数, "6,611 substitution pairs"→新数

#### 分析 C: NDI / 营养维度 (Section 2.8)
- 全 2,563 道菜的 DEI-N 计算
- 3D Pareto (H, 1/E, NDI)
- 更新论文: "NDI ranges 1.0-22.3 across 334 dishes"→新范围, "18 Pareto-optimal"→新数

#### 分析 D: OLS 回归 (Ext Data Table 1)
- logDEI ~ log(E_carbon) + log(E_water) + log(E_energy), HC3, n=2,563
- 更新: R²=0.675 n=334 → 新值

#### 分析 E: Refinement Cost Curves (Section 2.9)
- 用关键词匹配扩展 dish families（新菜可归入 curry/noodle/rice 等家族）
- 重新拟合 per-family α 和 global α
- 更新: "15 families, 106 dishes, α=-1.42" → 新值

#### 分析 F: Monte Carlo 排名不确定性 (Ext Data Table 3)
- E ±CV 扰动, 1000 模拟, top/bottom 10 排名 CI
- 更新: 排名 CI 表

#### 分析 G: 幸存者偏差 (Section 3.8)
- Ghost-dish 模拟, K=[100..2563], Δ=[0..3.0], 200 reps
- 更新: heatmap + 临界点数值

### Step 2: 运行脚本，收集结果

### Step 3: 更新 `paper/main.tex`

根据脚本输出的数字，更新以下部分:
- Section 2.3: 分类替代分析 (n, swaps, H contrib)
- Table 3: 类别内方差分解表
- Section 2.7: 膳食级分析 (combos, subs)
- Table 5: Within-role 方差表
- Section 2.8: NDI/DEI-N (范围, Pareto数)
- Section 2.9: Refinement curves (families, α)
- Ext Data Table 1: OLS 回归
- Ext Data Table 3: MC 排名 CI
- Section 3.8: 幸存者偏差数值
- Discussion/Conclusion: 关键统计数据

## 不需要更新的部分（保留 334-core 分析）
- Section 3.1-3.5: BERT 验证、跨平台验证、时间稳定性（需要 Yelp 评论数据）
- Section 3.9: 地理稳定性
- Section 3.10: FNDDS 交叉验证
- Section 3.8 (satiety): 多维度饱腹感分析（需要评论文本关键词）

## 关键文件
| 文件 | 角色 |
|------|------|
| `code/23_full_dataset_analysis.py` | 新建 — 主分析脚本 |
| `data/combined_dish_DEI_v2.csv` | 输入+输出（增加 5 列） |
| `data/expanded_dish_env_costs_v2.csv` | 输入（新菜食谱数据） |
| `code/11c_nutritional_dimension.py` | 复用 NUTRIENT_DATA, DRV |
| `code/11d_within_category_analysis.py` | 复用 classify_dish() |
| `paper/main.tex` | 更新论文数字 |

## 验证
1. 脚本运行无错误，<2 分钟完成
2. `combined_dish_DEI_v2.csv` 增加了 category, meal_role, NDI, protein_g, calorie_kcal 列
3. 所有新生成的 v2 图表出现在 `results/figures/` 和 `results/tables/`
4. 论文中不再有引用旧数据的 334-dish 分析（保留为 core subset 对比的除外）
