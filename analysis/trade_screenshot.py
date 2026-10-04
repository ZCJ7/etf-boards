"""交易截图识别：本地 OCR + 券商流水格式解析 + 可选视觉模型。"""

from __future__ import annotations

import base64
import json
import re
from datetime import datetime
from typing import Any

from analysis.etf_name_resolver import resolve_code_by_name, resolve_code_for_trade
from analysis.trade_ocr import ocr_image_bytes
from analysis.trade_records import TradeRecord
from ai_utils import ai_vision_chat, get_api_key
from utils import clean_etf_symbol

# 创业板ETF(159915)买入成交 / 中证500ETF(510500)卖出成交
_RE_TRADE_TITLE = re.compile(
    r"(?P<name>[^\d\(（]+?)[\(（](?P<code>\d{6})[\)）]\s*(?P<action>买入|卖出)(?:成交)?",
    re.I,
)
_RE_TRADE_ACTION_FIRST = re.compile(
    r"^(?P<action>买入|卖出)\s*(?P<name>.+?)(?:成交)?$",
    re.I,
)
# 买入 基金名 / 基金名 买入
_RE_TRADE_INLINE = re.compile(
    r"(?P<name>[^\d\(（]+?)\s*(?P<action>买入|卖出)(?:成交)?|"
    r"(?P<action2>买入|卖出)\s*(?P<name2>.+?)(?:成交)?",
    re.I,
)
_RE_CODE_IN_TEXT = re.compile(r"[\(（](\d{6})[\)）]")
_ETF_CODE = re.compile(r"\b(1[56]\d{4}|5[18]\d{4}|58\d{4})\b")
_RE_MONTH_HEADER = re.compile(r"(20\d{2})\s*年\s*(\d{1,2})\s*月")
_RE_DATETIME = re.compile(
    r"(?:(20\d{2})[年/.-])?(\d{1,2})[月/.-](\d{1,2})(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?|"
    r"(20\d{2})-(\d{2})-(\d{2})(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?|"
    r"(20\d{2})-(\d{1,2})-(\d{1,2})(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?"
)
_RE_AMOUNT = re.compile(r"([+-]?[\d,]+\.\d{2})")
_BUY_WORDS = re.compile(r"买入|买进|申购|买")
_SELL_WORDS = re.compile(r"卖出|卖出成交|赎回|卖")


def _guess_mime(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _normalize_date(y: int, m: int, d: int) -> str:
    return f"{y:04d}-{m:02d}-{d:02d}"


def _parse_datetime_from_line(line: str, default_year: int) -> str | None:
    for m in _RE_DATETIME.finditer(line):
        if m.group(1) and m.group(2) and m.group(3):
            return _normalize_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if m.group(4) and m.group(5) and m.group(6):
            return _normalize_date(int(m.group(4)), int(m.group(5)), int(m.group(6)))
        if m.group(7) and m.group(8) and m.group(9):
            return _normalize_date(int(m.group(7)), int(m.group(8)), int(m.group(9)))
        if m.group(2) and m.group(3):
            return _normalize_date(default_year, int(m.group(2)), int(m.group(3)))
    return None


def _detect_action(text: str) -> str | None:
    if _BUY_WORDS.search(text):
        return "买入"
    if _SELL_WORDS.search(text):
        return "卖出"
    return None


def _amount_to_action(amount: str | None) -> str | None:
    if not amount:
        return None
    val = amount.replace(",", "")
    if val.startswith("-"):
        return "买入"
    if val.startswith("+"):
        return "卖出"
    return None


def parse_broker_history_text(text: str) -> list[TradeRecord]:
    """
    解析券商/App 成交流水 OCR 文本。
    适配：名称(代码)买入成交 + 下一行 MM-DD HH:MM:SS + 右侧金额。
    """
    if not text or not text.strip():
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    now = datetime.now()
    current_year = now.year
    trades: list[TradeRecord] = []
    pending: dict[str, Any] | None = None

    def _flush_pending(fallback_line: str = "") -> None:
        nonlocal pending
        if not pending:
            return
        resolved = resolve_code_for_trade(
            pending.get("name", ""),
            explicit_code=pending.get("code"),
        )
        code = resolved.symbol
        if not code:
            pending = None
            return
        action = pending.get("action") or _detect_action(fallback_line)
        if not action:
            action = _amount_to_action(pending.get("amount"))
        if not action:
            pending = None
            return
        date = pending.get("date") or ""
        if not date:
            pending = None
            return
        note = str(pending.get("name") or "")
        trades.append(
            TradeRecord(
                symbol=clean_etf_symbol(code),
                action=action,
                date=date,
                price=None,
                quantity=None,
                note=note,
                match_hint=resolved.reason,
                source="broker_ocr",
            )
        )
        pending = None

    for i, line in enumerate(lines):
        mh = _RE_MONTH_HEADER.search(line)
        if mh:
            current_year = int(mh.group(1))
            _flush_pending()
            continue

        mt = _RE_TRADE_TITLE.search(line)
        if mt:
            _flush_pending()
            pending = {
                "name": mt.group("name").strip(),
                "code": clean_etf_symbol(mt.group("code")),
                "action": _detect_action(mt.group("action")),
                "date": _parse_datetime_from_line(line, current_year),
            }
            amt = _RE_AMOUNT.search(line)
            if amt:
                pending["amount"] = amt.group(1)
            continue

        # 买入 基金名 / 基金名买入（无括号代码，靠名称匹配）
        af = _RE_TRADE_ACTION_FIRST.match(line)
        if af and not _RE_CODE_IN_TEXT.search(line):
            _flush_pending()
            pending = {
                "name": af.group("name").strip(),
                "code": None,
                "action": _detect_action(af.group("action")),
                "date": _parse_datetime_from_line(line, current_year),
            }
            continue

        mi = _RE_TRADE_INLINE.search(line)
        if mi and not _RE_TRADE_TITLE.search(line) and not _RE_CODE_IN_TEXT.search(line):
            name = (mi.group("name") or mi.group("name2") or "").strip()
            action_raw = mi.group("action") or mi.group("action2")
            if name and action_raw and len(name) >= 2:
                _flush_pending()
                pending = {
                    "name": name,
                    "code": None,
                    "action": _detect_action(action_raw),
                    "date": _parse_datetime_from_line(line, current_year),
                }
                continue

        if pending:
            if not pending.get("date"):
                dt = _parse_datetime_from_line(line, current_year)
                if dt:
                    pending["date"] = dt
            if not pending.get("amount"):
                amt = _RE_AMOUNT.search(line)
                if amt:
                    pending["amount"] = amt.group(1)
            if pending.get("date"):
                _flush_pending()
                continue

        # 兜底：行内只有代码
        codes = _ETF_CODE.findall(line)
        if codes:
            ctx = "\n".join(lines[max(0, i - 1) : min(len(lines), i + 3)])
            action = _detect_action(ctx)
            if action:
                dt = _parse_datetime_from_line(ctx, current_year) or ""
                for code in codes:
                    trades.append(
                        TradeRecord(
                            symbol=clean_etf_symbol(code),
                            action=action,
                            date=dt,
                            source="broker_ocr",
                        )
                    )

    _flush_pending()
    return _dedupe_trades(trades)


def parse_trades_from_text(text: str) -> list[TradeRecord]:
    broker = parse_broker_history_text(text)
    if broker:
        return broker
    return _legacy_parse_trades_from_text(text)


def _legacy_parse_trades_from_text(text: str) -> list[TradeRecord]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    trades: list[TradeRecord] = []
    global_dates: list[str] = []
    for line in lines:
        dt = _parse_datetime_from_line(line, datetime.now().year)
        if dt:
            global_dates.append(dt)

    for i, line in enumerate(lines):
        codes = _ETF_CODE.findall(line)
        if not codes:
            continue
        ctx = "\n".join(lines[max(0, i - 2) : min(len(lines), i + 3)])
        action = _detect_action(ctx) or _detect_action(line)
        if not action:
            continue
        date = _parse_datetime_from_line(ctx, datetime.now().year) or (global_dates[0] if global_dates else "")
        for code in codes:
            trades.append(TradeRecord(symbol=code, action=action, date=date, source="ocr_text"))
    return _dedupe_trades(trades)


def dedupe_trades(trades: list[TradeRecord]) -> list[TradeRecord]:
    return _dedupe_trades(trades)


def _dedupe_trades(trades: list[TradeRecord]) -> list[TradeRecord]:
    seen: set[tuple] = set()
    out: list[TradeRecord] = []
    for t in trades:
        key = (t.symbol, t.action, t.date, t.price)
        if key in seen or not t.date:
            continue
        seen.add(key)
        out.append(t)
    return out


def _coerce_trade_items(raw: Any) -> list[dict]:
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        for key in ("trades", "records", "data", "items"):
            if isinstance(raw.get(key), list):
                return [x for x in raw[key] if isinstance(x, dict)]
    return []


def _items_to_trades(items: list[dict]) -> list[TradeRecord]:
    trades: list[TradeRecord] = []
    default_year = datetime.now().year
    for item in items:
        name = str(item.get("name") or item.get("名称") or item.get("note") or "")
        raw_sym = clean_etf_symbol(str(item.get("symbol") or item.get("代码") or item.get("code") or ""))
        explicit = raw_sym if raw_sym and len(raw_sym) == 6 else None
        resolved = resolve_code_for_trade(name, explicit_code=explicit)
        sym = clean_etf_symbol(resolved.symbol or "")
        if not sym or len(sym) != 6:
            continue

        action_raw = str(item.get("action") or item.get("方向") or item.get("side") or "")
        action = _detect_action(action_raw)
        if action_raw.lower() in {"sell", "s"}:
            action = "卖出"
        if not action:
            continue

        date_raw = str(item.get("date") or item.get("日期") or item.get("trade_date") or "")
        date = _parse_datetime_from_line(date_raw, default_year) or (date_raw[:10] if len(date_raw) >= 10 else "")
        if not date:
            continue

        price = item.get("price") or item.get("价格") or item.get("成交价")
        qty = item.get("quantity") or item.get("数量") or item.get("份额")
        trades.append(
            TradeRecord(
                symbol=sym,
                action=action,
                date=date,
                price=float(price) if price not in (None, "") else None,
                quantity=float(qty) if qty not in (None, "") else None,
                note=name,
                match_hint=resolved.reason,
                source="screenshot",
            )
        )
    return _dedupe_trades(trades)


def parse_screenshot(image_bytes: bytes) -> tuple[list[TradeRecord], str, str]:
    """
    识别交易截图：优先本地 OCR + 券商流水解析，可选 AI 视觉补强。
    返回 (交易列表, 说明, OCR原文片段)
    """
    if not image_bytes:
        return [], "未收到图片", ""

    msgs: list[str] = []
    trades: list[TradeRecord] = []
    ocr_snippet = ""

    ocr_text, ocr_msg = ocr_image_bytes(image_bytes)
    if ocr_text:
        ocr_snippet = ocr_text[:3000]
        trades = parse_broker_history_text(ocr_text)
        msgs.append(f"{ocr_msg}，解析 {len(trades)} 条")
        if trades:
            return trades, "；".join(msgs), ocr_snippet

    if get_api_key():
        mime = _guess_mime(image_bytes)
        b64 = base64.b64encode(image_bytes).decode("ascii")
        prompt = (
            "这是证券/基金 App 的成交/交易记录长截图。请逐条提取成交记录，"
            "只返回 JSON 数组。每条字段：symbol(6位ETF代码，从括号中提取)、"
            "name(基金名)、action(买入或卖出)、date(YYYY-MM-DD，年份参考截图月份标题)、"
            "price(null)、quantity(null)、note(空字符串)。"
            "格式示例：创业板ETF(159915)买入成交 对应 date 取同行或下一行日期。"
        )
        resp = ai_vision_chat(prompt, image_b64=b64, mime_type=mime)
        if not resp.startswith("AI调用失败") and not resp.startswith("未设置"):
            text = resp.strip()
            if "```" in text:
                text = re.sub(r"```json|```", "", text).strip()
            try:
                raw = json.loads(text)
                ai_trades = _items_to_trades(_coerce_trade_items(raw))
                if ai_trades:
                    msgs.append(f"AI 视觉识别 {len(ai_trades)} 条")
                    return ai_trades, "；".join(msgs), ocr_snippet
            except json.JSONDecodeError:
                ai_trades = parse_broker_history_text(resp)
                if ai_trades:
                    msgs.append(f"AI 文本规则解析 {len(ai_trades)} 条")
                    return ai_trades, "；".join(msgs), ocr_snippet
            msgs.append(f"AI 未解析成功：{resp[:120]}")
        else:
            msgs.append(resp)
    else:
        msgs.append("未配置 AI Key，已仅使用本地 OCR")

    if ocr_text:
        preview = ocr_text[:500].replace("\n", " | ")
        return [], f"未能从截图解析交易。{'；'.join(msgs)}。OCR片段：{preview}", ocr_snippet
    return [], f"未能识别截图。{'；'.join(msgs)}", ocr_snippet
