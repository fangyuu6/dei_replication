# DEI 项目指南

## 项目概述
美味效率指数 (Deliciousness Efficiency Index) — 量化"每单位环境代价获得的味觉享受"。
目标期刊: Nature 级别。数据来源: Yelp 530万条评论 + Poore & Nemecek 2018 环境因子。

## 核心公式
- **主规格**: `log(DEI) = log(H) - log(E)`
- **稳健性**: `DEI_z = Z(H) - Z(E)`
- 原始 `DEI = H/E` 仅用于向后兼容，不应作为主分析指标

## 为什么用 log 而不是原始比值
H 的跨菜品变异极小 (CV=3.9%)，E 的变异极大 (CV=64.3%)。原始 H/E 中 99.2% 的方差来自 E，DEI 排名本质上就是 1/E 排名。log 变换将乘法关系转为加法，使 H 和 E 的贡献在同一尺度上可比较，并改善回归残差分布。

## 管线脚本 (按顺序运行)

| 脚本 | 功能 | 运行时间 |
|------|------|----------|
| `code/01_explore_yelp.py` | Yelp 数据 EDA | ~2 min |
| `code/01b_prepare_env_factors.py` | 101 种食材环境因子 | ~1 min |
| `code/02_extract_dishes.py` | 从评论提取 158 道菜提及 | ~5 min |
| `code/03_nlp_hedonic_scoring.py` | BERT 情感评分 + LLM 标注 | ~30 min (含 API) |
| `code/03b_finetune_bert.py` | 微调 BERT 回归模型 | ~10 min (GPU) |
| `code/03c_apply_finetuned.py` | 用微调模型重评全部 76,927 条 | ~15 min |
| `code/04_env_cost_calculation.py` | 158 道菜食谱级 E 计算 | ~1 min |
| `code/05_dei_computation.py` | DEI 计算 + 方差分解 + OLS + 图 | ~1 min |
| `code/06_validation_robustness.py` | 7 项验证测试 | ~3 min |
| `code/config.py` | 全局配置 (路径/权重/菜系等) | — |

## 关键数据文件

| 文件 | 内容 |
|------|------|
| `data/dish_DEI_scores.csv` | 158 道菜完整数据 (H, E, log_DEI, DEI_z, 菜系, 烹饪方式) |
| `data/dish_hedonic_scores.csv` | H 分数 (预训练 + 微调) |
| `data/dish_environmental_costs.csv` | E 三分量 (碳/水/能耗) |
| `data/dish_mentions_scored.parquet` | 76,927 条评论级评分 |
| `data/llm_annotations.parquet` | 1,896 条 DeepSeek v3.2 标注 |
| `models/hedonic_bert_finetuned/` | 微调 BERT 权重 |

## 当前关键结果 (2026-02-25)

- **334 道菜, 29 菜系**, pairwise Bradley-Terry H (CV=33.6%), E (CV=64.0%)
- **方差分解**: H 贡献 20.0%, E 贡献 80.0% (pairwise H); 旧 BERT H: 0.8%/99.2%
- **OLS R²** = 0.675 (pairwise H, 334 dishes, HC3)
- **H-E 相关**: r=0.081 (本质无关)
- **6 道 Pareto 最优**: som_tam, rojak, fattoush, ca_phe_sua_da, ceviche, dan_dan_noodles
- **Top**: som_tam (7.77), rojak (6.67), papaya_salad (6.25)
- **Bottom**: pot_roast (1.27), churrasco (1.28), fajita (1.46)

## 待办 / 可能的下一步

1. **H 信号增强**: pairwise ranking 模型，或扩大 LLM 标注量 (当前仅 1,096 有效)
2. **代理偏差控制**: 在 H 回归中加入餐厅星级/价格/评论长度作为控制变量
3. **人工验证**: 200 样本人工评分 (样本已生成: `results/tables/human_validation_sample.csv`)
4. **加权 DEI**: `log(DEI_w) = w·log(H) - (1-w)·log(E)` 的权重敏感性分析
5. **地理分层**: 按城市/州分析 DEI 差异

## 运行环境
```bash
# Python 路径 (Windows bash, 不能用 python 命令)
PYTHON="/c/Users/C/AppData/Local/Programs/Python/Python313/python.exe"

# 标准运行模式
cd /c/project_EF && PYTHONIOENCODING=utf-8 $PYTHON code/<script>.py

# LLM 标注需要 API key
export OPENROUTER_API_KEY="..."
```

## 报告
- 中文完整报告: `results/研究一_结果报告.md`
- 回归输出: `results/tables/dei_regression.txt`
- 方差分解: `results/tables/variance_decomposition.csv`
- 验证汇总: `results/tables/validation_summary.csv`
