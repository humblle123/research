# CLAUDE.md

竞品调研流水线 demo：输入产品名，并行检索三维度（官网/口碑/竞品）→ LLM 提炼 → 结构化报告。已部署 Vercel。

## 跑起来
- 本地：`pip install -r requirements.txt` + 配 `.env`（见 `.env.example`）+ `npx vercel dev`
- 部署：push 到 `main` 即自动部署（已连 GitHub `humblle123/research`）
- 线上：https://research-gilt-one.vercel.app/

## 技术栈
- 前端：根目录 `index.html`（原生 JS，无构建）
- 后端：`api/index.py`（Python serverless，仅标准库 HTTP + `ddgs`）
- LLM：多协议适配（responses / chat_completions / completions / anthropic），完全可配置，不绑 provider

## 约定
- LLM/搜索 key 一律走 `.env` 或 Vercel 环境变量，**绝不写进代码或提交**（`.env` 已 gitignore）
- LLM 调用用标准库 HTTP，不引入 openai SDK（控制冷启动）
- 前端协议/模型/搜索后端改动需同步改 `index.html` 下拉框 + `api/index.py` 白名单

## 状态与下一步
- 已 live：首页 + 4 协议切换 + 端到端调研均验证通过（2026-07-27）
- 已知限制：`completions`(legacy) 协议在当前 provider(sui-xiang) 返回 404，属平台限制非代码问题
- 待办：sui-xiang key 曾泄露，须用重置后的新 key 配 Vercel 环境变量
