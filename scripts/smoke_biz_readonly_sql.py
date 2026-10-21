"""烟测业务只读库连通。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from agents.tools.readonly_sql import (  # noqa: E402
    biz_readonly_enabled,
    get_biz_readonly_engine,
    run_select,
)


def main() -> None:
    get_settings.cache_clear()
    get_biz_readonly_engine.cache_clear()
    cfg = get_settings()
    print("enabled", biz_readonly_enabled())
    print("url_set", bool(cfg.biz_readonly_database_url))
    print("url_host_ok", "rr-bp1z4pw0710558o722o" in (cfg.biz_readonly_database_url or ""))

    r1 = run_select("SELECT 1 AS ok")
    print("select1", r1.get("ok"), r1.get("rows"))

    r2 = run_select(
        "SELECT TABLE_NAME FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA = DATABASE() ORDER BY TABLE_NAME LIMIT 15"
    )
    names = [x.get("TABLE_NAME") for x in (r2.get("rows") or [])]
    print("tables", r2.get("row_count"), names)


if __name__ == "__main__":
    main()
