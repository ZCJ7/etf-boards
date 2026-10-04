"""榜单快照：GitHub Actions / 本地脚本写入，Streamlit 启动时回放。"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

BOARD_DIR = Path(__file__).resolve().parent.parent / "data" / "boards"
META_NAME = "meta.json"
BOARD_FILES = {
    "low_position": "low_position.json",
    "reversal": "reversal.json",
    "momentum": "momentum.json",
}

_CST = timezone(timedelta(hours=8))


def now_cst_iso() -> str:
    return datetime.now(_CST).strftime("%Y-%m-%d %H:%M:%S")


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    out = df.copy()
    return json.loads(out.to_json(orient="records", force_ascii=False))


def _records_to_df(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def save_boards_snapshot(
    *,
    asof: str,
    generated_at: str | None = None,
    low_position: pd.DataFrame | None = None,
    reversal: pd.DataFrame | None = None,
    momentum: pd.DataFrame | None = None,
    momentum_key: tuple[str, int] | None = None,
    source: str | None = None,
    errors: dict[str, str] | None = None,
) -> Path:
    BOARD_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = generated_at or now_cst_iso()
    payload = {
        "low_position": low_position,
        "reversal": reversal,
        "momentum": momentum,
    }
    for name, df in payload.items():
        path = BOARD_DIR / BOARD_FILES[name]
        path.write_text(
            json.dumps(_df_to_records(df) if df is not None else [], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    meta = {
        "asof": asof,
        "generated_at": generated_at,
        "timezone": "Asia/Shanghai",
        "source": source or "auto",
        "momentum_denominator": momentum_key[0] if momentum_key else "510300",
        "momentum_period": momentum_key[1] if momentum_key else 20,
        "rows": {
            "low_position": 0 if low_position is None else int(len(low_position)),
            "reversal": 0 if reversal is None else int(len(reversal)),
            "momentum": 0 if momentum is None else int(len(momentum)),
        },
        "errors": errors or {},
    }
    (BOARD_DIR / META_NAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return BOARD_DIR


def load_boards_snapshot() -> dict[str, Any] | None:
    meta_path = BOARD_DIR / META_NAME
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not meta.get("asof") and not meta.get("generated_at"):
        return None

    boards: dict[str, pd.DataFrame] = {}
    for name, fname in BOARD_FILES.items():
        path = BOARD_DIR / fname
        if not path.exists():
            boards[name] = pd.DataFrame()
            continue
        try:
            boards[name] = _records_to_df(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            boards[name] = pd.DataFrame()

    if all(df.empty for df in boards.values()):
        return None
    return {"meta": meta, **boards}


def snapshot_caption(meta: dict[str, Any] | None) -> str:
    if not meta:
        return ""
    asof = meta.get("asof") or "?"
    generated = meta.get("generated_at") or "?"
    return f"仓库快照 · 更新于 {generated} · 行情截止 {asof}"
