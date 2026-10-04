"""南方基金 ETFirst CLI 封装。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any


def _etfirst_bin() -> str:
    path = shutil.which("etfirst")
    if path:
        return path
    venv_bin = os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "etfirst.exe")
    if os.path.exists(venv_bin):
        return venv_bin
    raise FileNotFoundError("未找到 etfirst 命令，请先执行: pip install etfirst-0.2.3.tar.gz")


def run_etfirst(*args: str, timeout: int = 60) -> dict[str, Any]:
    cmd = [_etfirst_bin(), "--json", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8")
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(err or f"etfirst 执行失败: {' '.join(args)}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"etfirst 返回非 JSON: {proc.stdout[:200]}") from exc


def get_config() -> dict[str, Any]:
    cmd = [_etfirst_bin(), "config"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    return json.loads(proc.stdout)


def is_logged_in() -> bool:
    try:
        cfg = get_config()
        return bool(cfg.get("logged_in")) or bool(cfg.get("api_key_saved"))
    except Exception:
        return False


def login(api_key: str) -> str:
    cmd = [_etfirst_bin(), "auth", "login", "--api-key", api_key]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    return (proc.stdout or "登录成功").strip()


def etf_detail(product_code: str) -> dict[str, Any]:
    return run_etfirst("etf-detail", "all", "--product-code", product_code)


def index_detail(index_code: str) -> dict[str, Any]:
    return run_etfirst("index-detail", "all", "--index-code", index_code)


def list_etf(page_no: int = 1, page_size: int = 20, **kwargs: Any) -> dict[str, Any]:
    args = ["index-base", "list-etf", "--type", str(kwargs.get("type", 2)), "--page-no", str(page_no), "--page-size", str(page_size)]
    for key, value in kwargs.items():
        if key in {"type", "page_no", "page_size"}:
            continue
        args.extend([f"--{key.replace('_', '-')}", str(value)])
    return run_etfirst(*args)
