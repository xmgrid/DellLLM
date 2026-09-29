#!/usr/bin/env python3
"""两台 GB10 上的 DeepSeek-V4 集群控制台。

浏览器访问 :9090。文本对话先按 DeepSeek-V4 官方格式编码，再转给本机
vLLM 的 :8888。带图片的请求走 chat/completions，避免把图贴到助手标记后面。
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

from PIL import Image

from encoding_dsv4 import encode_messages, parse_message_from_completion_text

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
VLLM = "http://127.0.0.1:8888"
MODEL = "deepseek-v4-flash-0731"
WORKER = "linhaixiang@192.168.200.13"
HEAD = "192.168.0.78"
MAX_MODEL_LEN = 1048576


def run(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as exc:
        return 124, (exc.stdout or "") + (exc.stderr or "")


LANG_INSTRUCTION = {
    "zh": "请用中文回答。",
    "en": "Please answer in English.",
    "ja": "日本語で答えてください。",
    "auto": "请使用与用户最后一条消息相同的语言回答。",
}


def fit_max_tokens(prompt: str, requested: int, image_count: int = 0) -> int:
    """把生成长度收进「上下文上限 − 提示词」里。

    界面允许填 1048576，但那是提示词和回复加在一起的上限。若原样传给
    vLLM，渲染器会在分词前认为输入位置是 0 并直接拒绝。这里用字符数
    粗估提示词，每张图再留 512，保证两边检查都能过。
    """
    requested = max(16, int(requested or 512))
    prompt_tokens = len(prompt) + image_count * 512
    room = MAX_MODEL_LEN - prompt_tokens
    if room < 16:
        raise ValueError(
            f"提示词大约占了 {prompt_tokens} 个 token，上下文上限是 {MAX_MODEL_LEN}，剩下的位置不够生成。"
        )
    return min(requested, room)


def apply_language(messages: list, language: str) -> list:
    """只在发给模型的副本上追加语言要求，不写回对话历史。"""
    instruction = LANG_INSTRUCTION.get(language or "zh", LANG_INSTRUCTION["zh"])
    copied = [dict(message) for message in messages]
    for message in reversed(copied):
        if message.get("role") != "user":
            continue
        content = message.get("content") or ""
        if isinstance(content, str):
            message["content"] = (content + "\n" + instruction).strip()
        break
    return copied


def image_prompt_parts(prompt: str, images: list[str]) -> list[dict]:
    """把图片放进最后一轮用户内容，紧挨在问题文字前面。

    编码结果已经以助手标记结尾。如果把图片接在这个标记后面，模型会先开口
    回答，等于没看见图。
    """
    marker = "<｜User｜>"
    idx = prompt.rfind(marker)
    if idx < 0:
        head, tail = "", prompt
    else:
        cut = idx + len(marker)
        head, tail = prompt[:cut], prompt[cut:]
    parts = [{"type": "text", "text": head}]
    parts.extend({"type": "image_url", "image_url": {"url": url}} for url in images)
    parts.append({"type": "text", "text": tail})
    return parts


def normalize_images(urls: list[str]) -> list[str]:
    """打不开的上传直接拒绝。JPG/PNG 保持原始字节，不再重新压缩。"""
    normalized = []
    for url in urls:
        if not isinstance(url, str) or not url.startswith("data:") or "," not in url:
            normalized.append(url)
            continue
        _header, payload = url.split(",", 1)
        try:
            with Image.open(BytesIO(base64.b64decode(payload))) as image:
                image.load()
        except Exception as exc:
            raise ValueError("无法识别这张图片，请改用 JPG 或 PNG") from exc
        normalized.append(url)
    return normalized


def vllm_ok() -> bool:
    try:
        with urllib.request.urlopen(VLLM + "/v1/models", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def node_card(name: str, host: str, role: str, remote: str | None) -> dict:
    if remote:
        code, out = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", remote, "nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader"])
    else:
        code, out = run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader"])
    gpu = out.strip().splitlines()[0] if code == 0 and out.strip() else "不可用"
    return {"name": name, "host": host, "role": role, "gpu": gpu, "up": code == 0}


def state() -> dict:
    return {
        "ready": vllm_ok(),
        "model": MODEL,
        "api": f"http://{HEAD}:8888/v1",
        "console": f"http://{HEAD}:9090",
        "max_model_len": MAX_MODEL_LEN,
        # 官方卡片是 7。当前启动脚本为了速度使用 5，界面与脚本保持一致。
        "speculative": {"method": "dspark", "num_speculative_tokens": 5},
        "nodes": [
            node_card("promaxgb10-84ce", "192.168.0.78", "head", None),
            node_card("spark-8505", "192.168.0.79", "worker", WORKER),
        ],
    }


def systemctl(action: str, unit: str, remote: str | None = None) -> tuple[int, str]:
    inner = "sudo -n systemctl " + action + " " + unit
    if remote:
        return run(["ssh", "-o", "BatchMode=yes", remote, inner], timeout=30)
    return run(["sudo", "-n", "systemctl", action, unit], timeout=30)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(fmt % args)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: object) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            self._json(200, state())
            return
        if path == "/api/logs":
            which = "spark-head" if "which=worker" not in self.path else "spark-worker"
            if which == "spark-worker":
                code, out = run(["ssh", "-o", "BatchMode=yes", WORKER, "sudo -n journalctl -u spark-worker -n 200 --no-pager"], timeout=15)
            else:
                code, out = run(["sudo", "-n", "journalctl", "-u", "spark-head", "-n", "200", "--no-pager"], timeout=15)
            self._json(200, {"unit": which, "text": out, "ok": code == 0})
            return
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        file = (WEB / rel).resolve()
        if not str(file).startswith(str(WEB)) or not file.is_file():
            self._send(404, b"not found", "text/plain")
            return
        kind = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "text/javascript"}.get(file.suffix, "application/octet-stream")
        self._send(200, file.read_bytes(), kind)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        n = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        if path == "/api/serve/start":
            systemctl("start", "spark-worker", WORKER)
            code, out = systemctl("start", "spark-head")
            self._json(200, {"ok": code == 0, "output": out[-2000:]})
            return
        if path == "/api/serve/stop":
            systemctl("stop", "spark-head")
            code, out = systemctl("stop", "spark-worker", WORKER)
            self._json(200, {"ok": code == 0, "output": out[-2000:]})
            return
        if path == "/api/chat":
            self._chat(json.loads(raw.decode() or "{}"))
            return
        self._json(404, {"error": "not found"})

    def _chat(self, body: dict) -> None:
        messages = apply_language(body.get("messages") or [], body.get("language") or "zh")
        thinking = "thinking" if body.get("thinking") else "chat"
        effort = body.get("reasoning_effort") or "low"
        images = body.get("images") or []
        requested = int(body.get("max_tokens") or 512)
        try:
            images = normalize_images(images)
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        prompt = encode_messages(messages, thinking_mode=thinking, reasoning_effort=effort if thinking == "thinking" else None)
        try:
            max_tokens = fit_max_tokens(prompt, requested, len(images))
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        if images:
            # 看图时用温度 0，避免采样把画面描述带偏。纯文本仍用 0.6。
            payload = {
                "model": MODEL,
                "messages": [{
                    "role": "user",
                    "content": image_prompt_parts(prompt, images),
                }],
                "max_tokens": max_tokens,
                "stream": True,
                "temperature": 0,
            }
            url = VLLM + "/v1/chat/completions"
        else:
            payload = {
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stream": True,
                "temperature": 0.6,
            }
            url = VLLM + "/v1/completions"
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:2000]
            message = detail
            try:
                parsed = json.loads(detail)
                message = parsed.get("error", detail)
                if isinstance(message, str):
                    try:
                        inner = json.loads(message)
                        message = inner.get("error", {}).get("message", message)
                    except json.JSONDecodeError:
                        pass
                elif isinstance(message, dict):
                    message = message.get("message", detail)
            except json.JSONDecodeError:
                pass
            self._json(exc.code, {"error": message})
            return
        except Exception as exc:
            self._json(502, {"error": str(exc)})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        collected = []
        try:
            for line in resp:
                try:
                    self.wfile.write(line)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
                if line.startswith(b"data: ") and b"[DONE]" not in line:
                    try:
                        chunk = json.loads(line[6:])
                        if "choices" in chunk:
                            choice = chunk["choices"][0]
                            piece = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
                            collected.append(piece)
                    except Exception:
                        pass
        finally:
            resp.close()
        text = "".join(collected)
        try:
            parsed = parse_message_from_completion_text(text, thinking_mode=thinking)
        except Exception:
            parsed = {"role": "assistant", "content": text, "reasoning_content": "", "tool_calls": []}
        tail = json.dumps({"parsed": parsed}, ensure_ascii=False)
        try:
            self.wfile.write(f"event: parsed\ndata: {tail}\n\n".encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return


def main() -> None:
    os.chdir(ROOT)
    server = ThreadingHTTPServer(("0.0.0.0", 9090), Handler)
    print("sparkctl listening on 0.0.0.0:9090")
    server.serve_forever()


if __name__ == "__main__":
    main()
