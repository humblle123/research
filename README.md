# 竞品调研流水线（AI 应用自动化 Demo）

输入任意产品/公司名，**并行检索**官网信息、用户口碑、竞品对比三个维度，经过去重提炼后生成一份结构化调研报告。一个展示「多步骤 LLM 工作流编排」的可交互 demo。

## 架构

```
前端静态页 (public/index.html)
   │  fetch POST /api/research
   ▼
Python Serverless Function (api/research.py)
   ① 扩展搜索词   —— LLM 把随意输入扩成 3 组并行查询
   ② 并行检索     —— 官网信息 / 用户口碑 / 竞品对比（3 路并发）
   ③ 汇总提炼     —— 去重、事实与观点分离、每角度提炼要点
   ④ 结构化报告   —— 生成 JSON 报告（定位/功能/口碑/竞品/风险/结论）
```

- **零重型依赖**：LLM 走 OpenAI 兼容 `chat/completions`（标准库 HTTP），搜索用 `ddgs`，冷启动快。
- **完全可配置**：`base_url` / `api_key` / `model` 均可注入，不绑定任何 provider。
- **key 混合策略**：默认用服务端共享 key（带每 IP 每日限流防刷），访客也可在高级设置里填自己的 key。
- **降级与防护**：检索为空时如实标注"公开信息有限"不编造；base_url 强制 https 防 SSRF。

## 部署（Vercel）

```bash
npx vercel --prod
# 配置服务端共享 key（可选，不配则访客须填自己的 key）：
npx vercel env add LLM_BASE_URL
npx vercel env add LLM_API_KEY
npx vercel env add LLM_MODEL
npx vercel --prod   # 重新部署生效
```

环境变量：

| 变量 | 说明 |
|---|---|
| `LLM_BASE_URL` | OpenAI 兼容接口地址，如 `https://api.openai.com` |
| `LLM_API_KEY` | 服务端共享 key（可选） |
| `LLM_MODEL` | 默认模型名 |
| `TAVILY_API_KEY` | 可选，搜索后端用 Tavily 时更稳 |
| `RATE_LIMIT_PER_IP` | 共享 key 每 IP 每日上限，默认 20 |

## 本地运行

```bash
pip install -r requirements.txt
npx vercel dev   # 或任意能跑 serverless 的方式
```

> 本报告由 AI 基于公开网络信息生成，仅供参考。
