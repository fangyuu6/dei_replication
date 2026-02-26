# 调整 MAX_WORKERS 测试并发性能

## Context
用户想测试把 MAX_WORKERS 从 500 调到 1000 后速度是否有提升，以确定当前并发是否已达到代理瓶颈。

## Steps
1. 等 2019 年抓取完成后，暂停所有爬虫程序
2. 修改 `scrape_worker.py` 第 74 行：`MAX_WORKERS = 500` → `MAX_WORKERS = 1000`

## File
- `c:\Users\C\Desktop\flwebscraping\scrape_worker.py:74` — `MAX_WORKERS = 500` 改为 `1000`

## Verification
- 重新启动爬虫后，观察日志中的抓取速率（条/秒）是否比之前 500 时更快
- 观察失败率是否上升
