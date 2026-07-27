"""Vercel Serverless Function: 竞品调研流水线。

POST /api/research
body: {
  product, base_url?, api_key?, model?, search_provider?, search_key?
}
返回: { report: {...}, logs: [...] }

设计要点：
- 搜索用纯 HTTP（DuckDuckGo HTML / Tavily），不依赖重型库，控制冷启动。
- 限流：用服务端共享 key 的请求按 IP 做内存计数（单实例内有效），防刷爆额度。
- key 策略：访客填了自己的 api_key 就用访客的；否则用服务端环境变量。
- base_url 强制 https，避免 SSRF 到内网。
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler

# ---------------- 配置 ----------------
DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com")
DEFAULT_API_KEY = os.environ.get("LLM_API_KEY", "")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gpt-5.5")
TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "")
MAX_SHARED_PER_IP = int(os.environ.get("RATE_LIMIT_PER_IP", "20"))

_ANGLES = ("official", "reviews", "competitors")
_ANGLE_LABEL = {"official": "基本信息", "reviews": "用户口碑", "competitors": "竞品对比"}

# 简易内存限流（同一实例内有效；serverless 多实例时为尽力而为）
_RATE: dict = {}


def _rate_ok(ip: str) -> bool:
    now = time.time()
    day = int(now // 86400)
    rec = _RATE.get(ip)
    if not rec or rec[0] != day:
        _RATE[ip] = [day, 1]
        return True
    if rec[1] >= MAX_SHARED_PER_IP:
        return False
    rec[1] += 1
    return True


# ---------------- HTTP 工具 ----------------
def _post(url: str, payload: dict, headers: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _get(url: str, headers: dict, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


# ---------------- LLM ----------------
def _extract_output_text(data: dict) -> str:
    """从 /v1/responses 的返回里抠出文本。优先 output_text，其次遍历 output[].content[].text。"""
    if data.get("output_text"):
        return data["output_text"].strip()
    for item in data.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text" and c.get("text"):
                    return c["text"].strip()
    raise ValueError("responses 返回中没有文本内容: status=" + str(data.get("status")))


def _chat_responses(base_url: str, api_key: str, model: str, system: str, user: str) -> str:
    """OpenAI Responses 协议（/v1/responses）。instructions 传 system，input 传 user。"""
    data = _post(
        base_url.rstrip("/") + "/v1/responses",
        {"model": model, "instructions": system, "input": user},
        {"Authorization": "Bearer " + api_key},
        timeout=90,
    )
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    return _extract_output_text(data)


def _chat_completions(base_url: str, api_key: str, model: str, system: str, user: str) -> str:
    """标准 Chat Completions 协议（/v1/chat/completions）。"""
    data = _post(
        base_url.rstrip("/") + "/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        {"Authorization": "Bearer " + api_key},
        timeout=90,
    )
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    return (data["choices"][0]["message"]["content"] or "").strip()


def _chat_legacy(base_url: str, api_key: str, model: str, system: str, user: str) -> str:
    """Legacy Completions 协议（/v1/completions），纯文本补全，无多轮角色。"""
    prompt = (system + "\n\n" + user).strip()
    data = _post(
        base_url.rstrip("/") + "/v1/completions",
        {"model": model, "prompt": prompt, "max_tokens": 2048},
        {"Authorization": "Bearer " + api_key},
        timeout=90,
    )
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    return (data["choices"][0].get("text") or "").strip()


def _chat_anthropic(base_url: str, api_key: str, model: str, system: str, user: str) -> str:
    """Anthropic Messages 协议（/v1/messages）。system 独立字段，鉴权用 x-api-key。"""
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/messages",
        data=json.dumps({
            "model": model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": 2048,
        }).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        data = json.load(r)
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    return "".join(parts).strip()


_PROTOCOLS = ("responses", "chat_completions", "completions", "anthropic")


def _chat(base_url: str, api_key: str, model: str, system: str, user: str, protocol: str = "responses") -> str:
    if protocol == "chat_completions":
        return _chat_completions(base_url, api_key, model, system, user)
    if protocol == "completions":
        return _chat_legacy(base_url, api_key, model, system, user)
    if protocol == "anthropic":
        return _chat_anthropic(base_url, api_key, model, system, user)
    return _chat_responses(base_url, api_key, model, system, user)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e <= s:
        raise ValueError("no json")
    return json.loads(text[s : e + 1])


def _chat_json(base_url, api_key, model, system, user, protocol="responses") -> dict:
    sys = system + "\n只返回一个 JSON 对象，不要任何额外文字、不要用 markdown 代码块。"
    try:
        return _extract_json(_chat(base_url, api_key, model, sys, user, protocol))
    except Exception:
        return _extract_json(
            _chat(base_url, api_key, model, sys + "\n上次不是合法 JSON，重新只输出 JSON。", user, protocol)
        )


# ---------------- 搜索 ----------------
_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def _ddg(query: str, max_results: int) -> list:
    """用 ddgs 库搜索（底层处理反爬）。失败抛异常由上层兜底。"""
    from ddgs import DDGS

    with DDGS() as d:
        hits = list(d.text(query, max_results=max_results))
    return [
        {"title": h.get("title", ""), "snippet": h.get("body", ""), "url": h.get("href", "")}
        for h in hits
    ]


def _tavily(query: str, api_key: str, max_results: int) -> list:
    data = _post(
        "https://api.tavily.com/search",
        {"api_key": api_key, "query": query, "max_results": max_results},
        {},
        timeout=20,
    )
    return [
        {"title": x.get("title", ""), "snippet": x.get("content", ""), "url": x.get("url", "")}
        for x in data.get("results", [])
    ]


def _search(query: str, provider: str, key: str, max_results: int = 5) -> list:
    try:
        if provider == "tavily" and key:
            return _tavily(query, key, max_results)
        return _ddg(query, max_results)
    except Exception:
        return []


# ---------------- 流水线 ----------------
def _fmt(results: list) -> str:
    if not results:
        return "（无检索结果）"
    return "\n".join(f"- {r['title']}: {r['snippet']}" for r in results)


def run_pipeline(product, base_url, api_key, model, provider, search_key, log, protocol="responses"):
    log("① 扩展搜索词…")
    q = _chat_json(
        base_url, api_key, model, "你是搜索词扩展助手。",
        f"用户想调研产品「{product}」（可能是俗称或拼写不准）。补全为官方名称并生成 3 组搜索查询，JSON："
        '{"official":"<官网/基本信息>","reviews":"<用户评价,带 评价/怎么样/吐槽>","competitors":"<竞品,带 对比/vs/平替>"}，每个查询≤20字。',
        protocol,
    )
    queries = {k: str(q.get(k, product)) for k in _ANGLES}
    for a in _ANGLES:
        log(f"  · {_ANGLE_LABEL[a]}：{queries[a]}")

    log("② 并行检索（3 路）…")
    def _one(angle):
        hits = _search(queries[angle], provider, search_key)
        log(f"  · {_ANGLE_LABEL[angle]}：{len(hits)} 条")
        return angle, hits
    with ThreadPoolExecutor(max_workers=3) as ex:
        raw = dict(ex.map(_one, _ANGLES))
    total = sum(len(v) for v in raw.values())
    if total == 0:
        log("  ⚠ 未检索到结果，将仅基于模型知识分析")

    log("③ 汇总去重 & 提炼…")
    joined = "\n\n".join(f"## {_ANGLE_LABEL[a]}\n{_fmt(raw[a])}" for a in _ANGLES)
    d = _chat_json(
        base_url, api_key, model, "你是信息提炼助手。",
        f"以下是关于「{product}」的三路检索结果。去重、区分事实与观点、每角度提炼≤5条要点。"
        'JSON，键 "official"/"reviews"/"competitors"，值为字符串数组，不足给空数组。\n\n' + joined,
        protocol,
    )
    distilled = {k: [str(x) for x in d.get(k, [])][:5] for k in _ANGLES}

    log("④ 生成结构化报告…")
    src = "\n\n".join(
        f"## {_ANGLE_LABEL[a]}\n" + ("\n".join("- " + p for p in distilled[a]) or "（无）")
        for a in _ANGLES
    )
    report = _chat_json(
        base_url, api_key, model, "你是产品调研分析师。",
        f"基于以下已提炼的关于「{product}」的信息生成调研报告，严格 JSON："
        '{"product_name":"","one_line":"","positioning":"","core_features":[],'
        '"user_voice":{"praises":[],"complaints":[]},'
        '"competitors":[{"name":"","difference":""}],"risk_notes":[],"conclusion":""}'
        "信息不足写\"公开信息有限\"，绝不编造。\n\n" + src,
        protocol,
    )
    report["_meta"] = {"search_provider": provider, "results_used": total, "queries": queries}
    log("✓ 完成")
    return report


# ---------------- Handler ----------------
class handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(200, {})

    def do_GET(self):
        self._send(200, {"ok": True, "hint": "这是 API 端点，请用 POST 调用；首页在 /"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._send(400, {"error": "请求体不是合法 JSON"})

        product = (body.get("product") or "").strip()
        if not product:
            return self._send(400, {"error": "缺少 product"})
        if len(product) > 100:
            return self._send(400, {"error": "product 过长"})

        # key 策略：访客优先，否则服务端共享 key（受限流保护）
        visitor_key = (body.get("api_key") or "").strip()
        using_shared = not visitor_key
        api_key = visitor_key or DEFAULT_API_KEY
        if not api_key:
            return self._send(400, {"error": "未配置 API Key，请在前端填写"})

        if using_shared:
            ip = self.headers.get("x-forwarded-for", "unknown").split(",")[0].strip()
            if not _rate_ok(ip):
                return self._send(429, {"error": "共享额度已用完（每 IP 每日限次），请在高级设置里填你自己的 API Key"})

        base_url = (body.get("base_url") or "").strip() or DEFAULT_BASE_URL
        if not base_url.startswith("https://"):
            return self._send(400, {"error": "base_url 必须是 https"})
        model = (body.get("model") or "").strip() or DEFAULT_MODEL
        provider = (body.get("search_provider") or "duckduckgo").strip()
        search_key = (body.get("search_key") or "").strip() or TAVILY_KEY
        protocol = (body.get("protocol") or "responses").strip()
        if protocol not in _PROTOCOLS:
            protocol = "responses"

        logs = []
        try:
            report = run_pipeline(product, base_url, api_key, model, provider, search_key, logs.append, protocol)
            return self._send(200, {"report": report, "logs": logs, "shared_key": using_shared})
        except Exception as e:
            return self._send(500, {"error": f"流水线执行失败：{e}", "logs": logs})

    def log_message(self, *args):  # 静音默认访问日志
        pass
