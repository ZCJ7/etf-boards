import json
import os

import requests

KEY_FILE = "user_openai_key.txt"

# 国内站 compatible-mode；部分账号需用新版模型名
API_BASES = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
)
VISION_MODELS = ("qwen-vl-plus", "qwen2.5-vl-72b-instruct", "qwen2-vl-7b-instruct", "qwen-vl-max")


def get_api_key_path():
    return os.path.abspath(KEY_FILE)


def get_api_key():
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def set_api_key(key):
    with open(KEY_FILE, "w", encoding="utf-8") as f:
        f.write(key)


def _post_json(url: str, headers: dict, payload: dict, timeout: int = 90) -> requests.Response:
    return requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)


def ai_chat(
    prompt,
    api_key=None,
    system_prompt="你是专业的投资分析AI，请用简明中文回答。",
    model="qwen-plus",
    timeout: int = 120,
    max_tokens: int = 2200,
):
    api_key = api_key or get_api_key()
    if not api_key:
        return "未设置API Key"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    # 过长 prompt 易触发读超时：截断尾部次要内容
    text = str(prompt or "")
    if len(text) > 14000:
        text = text[:14000] + "\n\n…(内容过长已截断，请基于以上优先段落分析)"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "max_tokens": max_tokens,
    }
    url = f"{API_BASES[0]}/chat/completions"
    last_err = ""
    for attempt in range(2):
        try:
            # (connect, read) — 读超时单独加长
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=(15, timeout),
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            last_err = f"{response.status_code} {response.text[:300]}"
            if response.status_code in (429, 500, 502, 503, 504):
                continue
            return f"AI调用失败: {last_err}"
        except requests.exceptions.Timeout as e:
            last_err = f"超时({timeout}s): {e}"
            timeout = min(timeout + 60, 180)
        except Exception as e:
            last_err = str(e)
            break
    return f"AI调用失败: {last_err}。可点清缓存后减少资讯源再试，或检查网络/DashScope 服务。"


def ai_vision_chat(
    prompt: str,
    image_b64: str,
    mime_type: str = "image/jpeg",
    api_key=None,
    model: str | None = None,
) -> str:
    """多模态视觉识别（交易截图等），自动尝试多个模型。"""
    api_key = api_key or get_api_key()
    if not api_key:
        return "未设置API Key"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    models = (model,) if model else VISION_MODELS
    errors: list[str] = []

    for m in models:
        payload = {
            "model": m,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        url = f"{API_BASES[0]}/chat/completions"
        try:
            response = _post_json(url, headers, payload, timeout=120)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            errors.append(f"{m}: {response.status_code} {response.text[:180]}")
        except Exception as exc:
            errors.append(f"{m}: {exc}")

    return "AI调用失败: " + " | ".join(errors[:2])
