"""连锁自动入库：检索 + 回答冒烟（文本模拟 OCR，可不贴真图）。"""

from __future__ import annotations

import os
import sys

os.environ.setdefault(
    "LANGGRAPH_DATABASE_URL",
    "postgresql://postgres:123456@127.0.0.1:5432/langgraph_mem",
)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agents.diagnose.chat_runtime import run_chat_turn  # noqa: E402
from agents.diagnose.graph import reset_chat_graph  # noqa: E402
from agents.diagnose.nodes.retrieve_code import run_retrieve  # noqa: E402
from app.services.checkpoint_service import ensure_checkpoint_ready, get_checkpointer  # noqa: E402

OCR_FAKE = """
胎号查询 9001462521
物料编码 100A7357
操作时间 操作主体 操作类型
2026-04-29 09:14:05 S0476001FWA 宁城佳通轮胎经销处 K2入库
2026-04-29 09:18:38 S0476001FWA 宁城佳通轮胎经销处 K2零出
2026-04-29 09:18:38 S04760003LS 宁城县一凡汽车养护中心 K2入库
"""

QUESTION = (
    "这两家店是连锁店，这个胎号A门店已经入跟出了，为什么还会自动生成B门店的入库记录？"
)


def main() -> None:
    print("setup", ensure_checkpoint_ready())
    reset_chat_graph()

    retr = run_retrieve(user_text=QUESTION, image_description=OCR_FAKE)
    hits = retr.get("code_hits") or []
    print("hit_count", len(hits))
    print("queries", retr.get("retrieve_queries"))
    for h in hits[:5]:
        print(
            "-",
            h.get("path"),
            h.get("symbol"),
            h.get("start_line"),
            h.get("end_line"),
            "score=",
            h.get("score"),
        )

    cid = 900002
    try:
        get_checkpointer().delete_thread(str(cid))
    except Exception as exc:  # noqa: BLE001
        print("delete_thread", type(exc).__name__, exc)

    # 不走真 OCR：直接把 OCR 文本塞进 user（模拟 build_human_content）
    from agents.diagnose.nodes.chat_answer import build_human_content
    from langchain_core.messages import HumanMessage
    from agents.diagnose.graph import get_chat_graph
    from app.services.checkpoint_service import thread_config

    human = build_human_content(QUESTION, image_description=OCR_FAKE)
    result = get_chat_graph().invoke(
        {
            "messages": [HumanMessage(content=human)],
            "user_text": QUESTION,
            "image_description": OCR_FAKE,
            "errors": [],
        },
        config=thread_config(cid),
    )
    reply = result.get("reply") or ""
    print("pipeline", result.get("pipeline"))
    print("reply_head", reply[:400])
    print("graph_hits", len(result.get("code_hits") or []))

    ok_retrieve = len(hits) > 0
    ok_cite = ":" in reply and any(c.isdigit() for c in reply)
    ok_biz = any(x in reply for x in ("连锁", "自动", "入库", "归属", "出库", "门店"))
    print("CHECK retrieve", ok_retrieve, "cite", ok_cite, "biz", ok_biz)
    ok = ok_retrieve and ok_biz
    print("ACCEPT_OK" if ok else "ACCEPT_FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
