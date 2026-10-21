# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from agents.diagnose.biz_verify import detect_biz_intent
from agents.tools.readonly_sql import get_biz_readonly_engine
from agents.tools.warranty_query import extract_license_plate, query_warranty_by_plate

get_settings.cache_clear()
get_biz_readonly_engine.cache_clear()

Q = "晋D019X0，请帮忙看下该车牌号有没有质保卡，如有，胎号是什么~"


def main() -> None:
    print("intent", detect_biz_intent(Q))
    print("plate", extract_license_plate(Q))
    plate = extract_license_plate(Q) or "晋D019X0"
    try:
        r = query_warranty_by_plate(plate)
        print("rows", r.get("row_count"))
        for row in (r.get("rows") or [])[:5]:
            print(
                row.get("license_plate"),
                row.get("WARRANTY_CARD"),
                row.get("TYRE_CODE"),
                row.get("WARRANTY_STATUS"),
            )
    except Exception as exc:
        print("query_failed", exc)


if __name__ == "__main__":
    main()
