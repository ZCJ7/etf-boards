"""只重算 Top5 动量并写回手机页（复用现有榜单/levels，不重拉全市场）。"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis.board_store import load_boards_snapshot, now_cst_iso  # noqa: E402
from etf_data_fetcher import live_bar_note  # noqa: E402
from scripts.momentum_page import save_momentum_artifacts  # noqa: E402
from scripts.refresh_boards import _style_section_html, write_pages_html  # noqa: E402


def main() -> int:
    snap = load_boards_snapshot()
    if not snap:
        print("no boards snapshot", flush=True)
        return 1
    boards = {
        "low_position": snap["low_position"],
        "reversal": snap["reversal"],
        "momentum": snap["momentum"],
    }
    meta = snap["meta"]
    levels_path = os.path.join(ROOT, "docs", "levels.json")
    levels_json = ""
    support_html = resist_html = levels_script = ""
    if os.path.exists(levels_path):
        levels_json = open(levels_path, encoding="utf-8").read()
        from scripts.level_page import level_sections

        items = json.loads(levels_json)
        support_html, resist_html, levels_script = level_sections(items)
    print("==> Top5 动量", flush=True)
    artifacts = save_momentum_artifacts()
    print("==> 比价", flush=True)
    style_html, style_err = _style_section_html()
    if style_err:
        print("style fail", style_err, flush=True)
    write_pages_html(
        meta.get("asof") or "",
        now_cst_iso(),
        boards,
        meta.get("errors") or {},
        meta.get("source") or "sina",
        live_bar_note(),
        style_html,
        support_html,
        resist_html,
        levels_script,
        levels_json,
        artifacts["html"],
    )
    print("wrote docs/index.html", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
