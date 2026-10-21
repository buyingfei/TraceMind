"""两轮对话记忆冒烟：同 thread 第二轮应能引用第一轮事实。"""

from __future__ import annotations

import os
import sys

os.environ.setdefault(
    "LANGGRAPH_DATABASE_URL",
    "postgresql://postgres:123456@127.0.0.1:5432/langgraph_mem",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agents.diagnose.chat_runtime import _snapshot_messages, run_chat_turn  # noqa: E402
from agents.diagnose.graph import reset_chat_graph  # noqa: E402
from app.services.checkpoint_service import ensure_checkpoint_ready, get_checkpointer  # noqa: E402


def main() -> None:
    print("setup", ensure_checkpoint_ready())
    reset_chat_graph()
    cid = 900001
    try:
        get_checkpointer().delete_thread(str(cid))
    except Exception as exc:  # noqa: BLE001
        print("delete_thread skip:", type(exc).__name__, exc)

    r1 = run_chat_turn(
        cid,
        user_text="Remember: project code is WT-ALPHA, store code is HZ-001.",
    )
    print("R1 pipeline=", r1.get("pipeline"))
    print("R1 reply=", (r1.get("reply") or "")[:200])

    r2 = run_chat_turn(
        cid,
        user_text="What are the project code and store code I just told you? Answer with the two values only.",
    )
    print("R2 pipeline=", r2.get("pipeline"))
    print("R2 reply=", r2.get("reply"))

    msgs = _snapshot_messages(cid)
    print("checkpoint_msgs=", len(msgs))
    for m in msgs:
        print("-", getattr(m, "type", None), str(getattr(m, "content", ""))[:100])

    text = (r2.get("reply") or "").upper()
    ok = "WT-ALPHA" in text and "HZ-001" in text
    print("MEMORY_OK" if ok else "MEMORY_FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
