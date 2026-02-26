# Plan: LLM 标注 2,000 条评论 + 微调 BERT

## Context
当前 H 分数使用预训练 BERT 情感模型（nlptown 5-star → 1-10 映射），这是一个通用情感模型，不是专门针对食物味觉享受的。通过 LLM 标注训练数据并微调专用模型，可以显著提升 H 分数的准确性和领域特异性。

## Step 1: 修改 `code/03_nlp_hedonic_scoring.py` — 添加 DeepSeek/OpenRouter 支持

- 修改 `annotate_with_openai()` 函数，支持自定义 `base_url` 参数
- 在 `main()` 中添加 `OPENROUTER_API_KEY` 环境变量检测
- 配置: `base_url="https://openrouter.ai/api/v1"`, `model="deepseek/deepseek-v3.2"`
- 分层采样 2,000 条评论（按菜品均匀分布，每道菜 ~13 条）
- 输出: `data/llm_annotations.parquet`

## Step 2: 创建 `code/03b_finetune_bert.py` — 微调脚本

- 加载 LLM 标注数据，过滤 score=0（信息不足）和 score=-1（错误）
- 训练/验证集 80/20 分割（按菜品分层）
- 微调 `bert-base-uncased` 做回归任务（输出 1-10 连续分）
- 训练参数: epochs=5, lr=2e-5, batch=16, warmup=10%
- 保存模型到 `models/hedonic_bert_finetuned/`
- 输出验证指标: MAE, RMSE, Pearson r, Spearman rho

## Step 3: 创建 `code/03c_apply_finetuned.py` — 用微调模型重新评分

- 加载微调后的 BERT 模型
- 对全部 76,927 条已采样评论重新评分
- 与预训练模型结果对比（Pearson r, rank correlation）
- 重新聚合 dish-level H 分数
- 输出: 更新 `data/dish_hedonic_scores.csv`

## Step 4: 重跑下游管线

- 重跑 `05_dei_computation.py` 和 `06_validation_robustness.py`
- 更新报告

## 关键文件
- 修改: `code/03_nlp_hedonic_scoring.py` (添加 OpenRouter 支持)
- 新建: `code/03b_finetune_bert.py`
- 新建: `code/03c_apply_finetuned.py`
- 输出: `data/llm_annotations.parquet`, `models/hedonic_bert_finetuned/`

## 验证
- LLM 标注成功率 > 90%（score 1-10 的有效比例）
- 微调后验证集 MAE < 1.0, Pearson r > 0.75
- 微调 H vs 预训练 H 的 Spearman rho（衡量排名变化幅度）
- DEI 排名变化分析
