from akshare_patch import install_akshare_patch

install_akshare_patch()

import json
import os
import re
from typing import Any

from utils import clean_etf_symbol

CONFIG_DIR = "config"
HOLDINGS_FILE = os.path.join(CONFIG_DIR, "user_holdings.json")

# 常见指数代码（非场内 ETF，不可用于行情拉取）
INDEX_CODES = {
    "399006",
    "399001",
    "399005",
    "399300",
    "000001",
    "000016",
    "000300",
    "000905",
    "000852",
}


def _default_config() -> dict[str, Any]:
    return {
        "version": 1,
        "holdings": [],
        "watchlist": [],
        "notes": {},
    }


def load_holdings_config() -> dict[str, Any]:
    if not os.path.exists(HOLDINGS_FILE):
        return _default_config()
    try:
        with open(HOLDINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _default_config()
        data.setdefault("holdings", [])
        data.setdefault("watchlist", [])
        data.setdefault("notes", {})
        return data
    except (json.JSONDecodeError, OSError):
        return _default_config()


def save_holdings_config(config: dict[str, Any]) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(HOLDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def normalize_symbol(symbol: str) -> str:
    return clean_etf_symbol(str(symbol).strip())


def is_tradable_etf_symbol(symbol: str) -> tuple[bool, str]:
    code = normalize_symbol(symbol)
    if not re.fullmatch(r"\d{6}", code):
        return False, "请输入 6 位数字代码"
    if code in INDEX_CODES or code.startswith("399"):
        return False, f"{code} 是指数代码，不是场内 ETF"
    return True, ""


def _lookup_name_from_etf_list(symbol: str) -> str:
    try:
        from data import get_etf_list

        etf_list = get_etf_list()
        if etf_list.empty:
            return ""
        row = etf_list[etf_list["symbol"].astype(str) == symbol]
        if not row.empty:
            return str(row["name"].values[0])
    except Exception:
        pass
    return ""


def resolve_etf_name(symbol: str) -> str:
    symbol = normalize_symbol(symbol)
    name = _lookup_name_from_etf_list(symbol)
    if name:
        return name
    try:
        import akshare as ak

        df = ak.fund_etf_spot_em()
        if df is not None and not df.empty:
            code_col = "代码" if "代码" in df.columns else None
            name_col = "名称" if "名称" in df.columns else None
            if code_col and name_col:
                row = df[df[code_col].astype(str).str.replace(r"^(sh|sz)", "", regex=True) == symbol]
                if not row.empty:
                    return str(row[name_col].values[0])
    except Exception:
        pass
    return symbol


def _unique_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        symbol = normalize_symbol(item.get("symbol", ""))
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        result.append(
            {
                "symbol": symbol,
                "name": item.get("name") or resolve_etf_name(symbol),
                "enabled": item.get("enabled", True),
            }
        )
    return result


def get_enabled_holdings() -> list[dict[str, Any]]:
    config = load_holdings_config()
    holdings = _unique_items(config.get("holdings", []))
    return [h for h in holdings if h.get("enabled", True)]


def get_watchlist() -> list[dict[str, Any]]:
    config = load_holdings_config()
    return _unique_items(config.get("watchlist", []))


def get_holdings_pool(include_watchlist: bool = True) -> dict[str, str]:
    pool: dict[str, str] = {}
    for item in get_enabled_holdings():
        pool[item["symbol"]] = item.get("name") or item["symbol"]
    if include_watchlist:
        for item in get_watchlist():
            pool[item["symbol"]] = item.get("name") or item["symbol"]
    return pool


def get_all_symbols(include_watchlist: bool = True) -> list[str]:
    return list(get_holdings_pool(include_watchlist).keys())


def add_holding(symbol: str, name: str | None = None, to_watchlist: bool = False) -> tuple[bool, str]:
    ok, msg = is_tradable_etf_symbol(symbol)
    if not ok:
        return False, msg

    symbol = normalize_symbol(symbol)
    config = load_holdings_config()
    key = "watchlist" if to_watchlist else "holdings"
    items = _unique_items(config.get(key, []))

    for item in items:
        if item["symbol"] == symbol:
            label = "观察池" if to_watchlist else "持仓"
            return False, f"{symbol} 已在{label}中"

    other_key = "holdings" if to_watchlist else "watchlist"
    other_items = _unique_items(config.get(other_key, []))
    if any(item["symbol"] == symbol for item in other_items):
        return False, f"{symbol} 已在{'持仓' if to_watchlist else '观察池'}中，请先移除后再添加"

    resolved_name = name or resolve_etf_name(symbol)
    items.append({"symbol": symbol, "name": resolved_name, "enabled": True})
    config[key] = items
    save_holdings_config(config)
    _sync_favorite_etfs()
    label = "观察池" if to_watchlist else "持仓"
    return True, f"已添加 {symbol} ({resolved_name}) 到{label}"


def remove_holding(symbol: str, from_watchlist: bool = False) -> tuple[bool, str]:
    symbol = normalize_symbol(symbol)
    config = load_holdings_config()
    key = "watchlist" if from_watchlist else "holdings"
    items = _unique_items(config.get(key, []))
    new_items = [item for item in items if item["symbol"] != symbol]
    if len(new_items) == len(items):
        label = "观察池" if from_watchlist else "持仓"
        return False, f"{symbol} 不在{label}中"
    config[key] = new_items
    save_holdings_config(config)
    _sync_favorite_etfs()
    label = "观察池" if from_watchlist else "持仓"
    return True, f"已从{label}移除 {symbol}"


def set_holding_enabled(symbol: str, enabled: bool) -> tuple[bool, str]:
    symbol = normalize_symbol(symbol)
    config = load_holdings_config()
    items = _unique_items(config.get("holdings", []))
    found = False
    for item in items:
        if item["symbol"] == symbol:
            item["enabled"] = enabled
            found = True
            break
    if not found:
        return False, f"{symbol} 不在持仓列表中"
    config["holdings"] = items
    save_holdings_config(config)
    _sync_favorite_etfs()
    state = "启用" if enabled else "暂停"
    return True, f"已{state} {symbol}"


def remove_from_pool(symbol: str) -> tuple[bool, str]:
    """从持仓或观察池中移除（自动判断所在列表）。"""
    symbol = normalize_symbol(symbol)
    config = load_holdings_config()
    in_holdings = any(normalize_symbol(h.get("symbol", "")) == symbol for h in config.get("holdings", []))
    in_watchlist = any(normalize_symbol(w.get("symbol", "")) == symbol for w in config.get("watchlist", []))
    if in_holdings:
        return remove_holding(symbol, from_watchlist=False)
    if in_watchlist:
        return remove_holding(symbol, from_watchlist=True)
    return False, f"{symbol} 不在持仓或观察池中"


def _sync_favorite_etfs() -> None:
    """同步到 favorite_etfs.json，供组合回测等页面优先显示。"""
    favorite_file = os.path.join(CONFIG_DIR, "favorite_etfs.json")
    symbols = get_all_symbols(include_watchlist=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(favorite_file, "w", encoding="utf-8") as f:
        json.dump(symbols, f, ensure_ascii=False, indent=2)


def refresh_all_names() -> int:
    config = load_holdings_config()
    updated = 0
    for key in ("holdings", "watchlist"):
        items = _unique_items(config.get(key, []))
        for item in items:
            new_name = resolve_etf_name(item["symbol"])
            if new_name and new_name != item.get("name"):
                item["name"] = new_name
                updated += 1
        config[key] = items
    save_holdings_config(config)
    return updated
