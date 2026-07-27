# 竞品调研流水线（AI 应用自动化 Demo）

输入任意产品/公司名，**并行检索**官网信息、用户口碑、竞品对比三个维度，经过去重提炼后生成一份结构化调研报告。一个展示「多步骤 LLM 工作流编排」的可交互 demo。

**线上地址**：https://research-gilt-one.vercel.app/ （打开即用，未填 key 时走服务端共享额度）

## 架构

```
根目录 index.html（静态前端，Vercel @vercel/static）
   │  fetch POST /api/index
   ▼
api/index.py（Python Serverless Function，Vercel @vercel/python）
   ① 扩展搜索词   —— LLM 把随意输入扩成 3 组并行查询
   ② 并行检索     —— 官网信息 / 用户口碑 / 竞品对比（3 路并发）
   ③ 汇总提炼     —— 去重、事实与观点分离、每角度提炼要点
   ④ 结构化报告   —— 生成 JSON 报告（定位/功能/口碑/竞品/风险/结论）
```

- **多协议适配**：同一套逻辑支持 4 种 LLM API 协议，前端下拉框一键切换，无需改代码——
  `responses`（/v1/responses，默认）、`chat_completions`（/v1/chat/completions）、
  `completions`（/v1/completions 老式）、`anthropic`（/v1/messages，Claude 原生）。
  各协议的请求结构、鉴权（Bearer vs x-api-key）、返回解析分别实现，后端白名单校验。
- **完全可配置**：`base_url` / `api_key` / `model` / 协议 / 搜索后端均可注入，不绑定任何 provider。
- **轻依赖**：LLM 调用用标准库 HTTP（不引 openai SDK），搜索用 `ddgs`，冷启动快。
- **key 混合策略**：默认用服务端共享 key（带每 IP 每日限流防刷），访客也可在高级设置里填自己的 key。
- **降级与防护**：检索为空时如实标注"公开信息有限"不编造；base_url 强制 https 防 SSRF。

## 部署（Vercel）

代码推送即自动部署（已连 GitHub）。首次手动部署：

```bash
npx vercel --prod
```

配置服务端共享 key（可选，不配则访客须填自己的 key）——在 Vercel 后台 **Settings → Environment Variables** 添加后 **Redeploy** 生效：

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
cp .env.example .env   # 或自行设置 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
npx vercel dev          # 本地起 serverless + 静态页
```

> 本报告由 AI 基于公开网络信息生成，仅供参考。
