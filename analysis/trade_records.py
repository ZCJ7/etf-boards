"""交易记录持久化与查询。"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from utils import clean_etf_symbol

CONFIG_DIR = "config"
TRADES_FILE = os.path.join(CONFIG_DIR, "user_trades.json")


@dataclass
class TradeRecord:
    symbol: str
    action: str  # 买入 / 卖出
    date: str  # YYYY-MM-DD
    price: float | None = None
    quantity: float | None = None
    note: str = ""
    match_hint: str = ""
    source: str = "screenshot"
    id: str = ""

    def __post_init__(self) -> None:
        self.symbol = clean_etf_symbol(self.symbol)
        self.action = "买入" if self.action in {"买入", "买", "B", "buy", "BUY"} else (
            "卖出" if self.action in {"卖出", "卖", "S", "sell", "SELL"} else self.action
        )
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


def _default_data() -> dict[str, Any]:
    return {"version": 1, "trades": []}


def load_trades() -> list[TradeRecord]:
    if not os.path.exists(TRADES_FILE):
        return []
    try:
        with open(TRADES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data.get("trades", [])
        return [TradeRecord(**{**r, "symbol": clean_etf_symbol(r.get("symbol", "")), "match_hint": r.get("match_hint", "")}) for r in rows if r.get("symbol")]
    except (json.JSONDecodeError, OSError, TypeError):
        return []


def save_trades(trades: list[TradeRecord]) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    payload = {"version": 1, "trades": [asdict(t) for t in trades]}
    with open(TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def merge_trades(existing: list[TradeRecord], new_items: list[TradeRecord]) -> list[TradeRecord]:
    seen = {(t.symbol, t.action, t.date, t.price) for t in existing}
    merged = list(existing)
    for t in new_items:
        key = (t.symbol, t.action, t.date, t.price)
        if key not in seen:
            merged.append(t)
            seen.add(key)
    merged.sort(key=lambda x: x.date)
    return merged


def get_trades_for_symbol(trades: list[TradeRecord], symbol: str) -> list[TradeRecord]:
    code = clean_etf_symbol(symbol)
    return [t for t in trades if t.symbol == code]


def align_trade_date(trade_date: str, trading_index: pd.DatetimeIndex) -> pd.Timestamp | None:
    if trading_index.empty:
        return None
    try:
        target = pd.Timestamp(trade_date)
    except Exception:
        return None
    if target in trading_index:
        return target
    prior = trading_index[trading_index <= target]
    if not prior.empty:
        return prior[-1]
    return trading_index[0]


def trades_to_dataframe(trades: list[TradeRecord]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["代码", "方向", "日期", "价格", "数量", "备注", "匹配", "来源", "id"])
    return pd.DataFrame(
        [
            {
                "代码": t.symbol,
                "方向": t.action,
                "日期": t.date,
                "价格": t.price,
                "数量": t.quantity,
                "备注": t.note,
                "匹配": t.match_hint,
                "来源": t.source,
                "id": t.id,
            }
            for t in trades
        ]
    )


def dataframe_to_trades(df: pd.DataFrame) -> list[TradeRecord]:
    if df is None or df.empty:
        return []
    out: list[TradeRecord] = []
    for _, row in df.iterrows():
        sym = str(row.get("代码", "")).strip()
        if not sym:
            continue
        out.append(
            TradeRecord(
                symbol=sym,
                action=str(row.get("方向", "买入")),
                date=str(row.get("日期", ""))[:10],
                price=float(row["价格"]) if pd.notna(row.get("价格")) and row.get("价格") != "" else None,
                quantity=float(row["数量"]) if pd.notna(row.get("数量")) and row.get("数量") != "" else None,
                note=str(row.get("备注", "") or ""),
                match_hint=str(row.get("匹配", "") or ""),
                source=str(row.get("来源", "manual") or "manual"),
                id=str(row.get("id") or "") or str(uuid.uuid4())[:8],
            )
        )
    return out
