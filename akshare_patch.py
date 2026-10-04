"""修补 AkShare 访问东方财富接口时的请求头，解决 RemoteDisconnected。"""

from __future__ import annotations

import functools

import requests

_PATCHED = False

EASTMONEY_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

_EASTMONEY_HOSTS = (
    "eastmoney.com",
    "push2his.eastmoney.com",
    "push2delay.eastmoney.com",
    "88.push2.eastmoney.com",
    "82.push2.eastmoney.com",
)


def _needs_patch(url: str) -> bool:
    return any(host in str(url) for host in _EASTMONEY_HOSTS)


def install_akshare_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    _orig_request = requests.Session.request
    _orig_get = requests.get

    @functools.wraps(_orig_request)
    def patched_request(self, method, url, **kwargs):
        if _needs_patch(url):
            headers = dict(EASTMONEY_HEADERS)
            headers.update(kwargs.pop("headers", {}) or {})
            kwargs["headers"] = headers
            kwargs.setdefault("timeout", 20)
        return _orig_request(self, method, url, **kwargs)

    @functools.wraps(_orig_get)
    def patched_get(url, **kwargs):
        if _needs_patch(url):
            headers = dict(EASTMONEY_HEADERS)
            headers.update(kwargs.pop("headers", {}) or {})
            kwargs["headers"] = headers
            kwargs.setdefault("timeout", 20)
        return _orig_get(url, **kwargs)

    requests.Session.request = patched_request  # type: ignore[method-assign]
    requests.get = patched_get  # type: ignore[assignment]
    _PATCHED = True
