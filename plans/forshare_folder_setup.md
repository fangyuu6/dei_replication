# Plan: 创建 forshare 文件夹（GitHub 分享用精简版项目）

## Context
需要创建一个 `forshare/` 文件夹，包含项目的精简版本，方便上传到 GitHub 分享。去掉所有敏感信息（API 密钥）、大文件（数据库 dump、大型 CSV/HTML 输出）和 IDE 配置，只保留核心代码和文档。

## 包含的内容

| 目录/文件 | 说明 |
|-----------|------|
| `config/` | settings.py + __init__.py（安全，只有 os.getenv） |
| `extraction/` | 所有 .py 文件（LLM 分析、评分、提示词等） |
| `ingestion/` | 所有 .py 文件（含 crawl/ 子目录） |
| `export/` | csv_export.py, html_report.py |
| `storage/` | policy_store.py, schema.sql, __init__.py |
| `scripts/` | 所有脚本（不含 .ps1 密码脚本） |
| `docs/` | csv_columns.md, data_sources.md, database_design.md |
| `input/` | URL 列表文件（小文件，不敏感） |
| 根目录 | README.md, PLAN.md, DEPLOY.md, requirements.txt, index.html |
| `.env.example` | 新建：带占位符的环境变量模板 |
| `.gitignore` | 适配 forshare 的版本 |

## 排除的内容

- `.env`（真实 API 密钥）
- `2026-02-public.pgdump`（9.5 GB）、`2026-02-schema.pgdump`
- `output/` 整个目录（最大 152 MB 单文件，总共 ~2 GB+）
- `website_deploy/`
- `__pycache__/`、`.claude/`、`.cursor/` 等 IDE/缓存目录
- `scripts/run_restore_with_password.ps1`（含数据库密码引用）

## 执行步骤

1. **创建目录结构** — `mkdir -p forshare/` 及所有子目录
2. **复制核心代码** — 用 `cp -r` 复制各个 Python 包目录（排除 `__pycache__`）
3. **复制文档和根文件** — README.md, PLAN.md, DEPLOY.md, requirements.txt, index.html, docs/
4. **复制 input/** — URL 列表文件
5. **创建 `.env.example`** — 列出所有需要的环境变量，值用占位符
6. **创建 forshare 专用 `.gitignore`** — 排除 output/、.env、__pycache__、IDE 配置等
7. **创建空 `output/` 目录** — 放一个 `.gitkeep` 文件保持目录结构

## 验证
- 检查 forshare/ 没有 `.env`、没有 `.pgdump`、没有大文件
- 检查所有 Python 源文件都在
- 总大小应在几 MB 以内
