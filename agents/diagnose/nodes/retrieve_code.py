"""代码向量检索节点（仅 OCR + 用户原话）。"""

from __future__ import annotations

import logging
from typing import Any

from agents.diagnose.state import ChatAgentState
from agents.tools import code_search

logger = logging.getLogger(__name__)


def build_retrieve_queries(
    *,
    user_text: str,
    image_description: str = "",
) -> list[str]:
    """多路语义检索：合并问句、纯用户问、纯 OCR 摘要（若有）。"""
    user = (user_text or "").strip()
    ocr = (image_description or "").strip()
    ocr_snip = ocr[:600] if ocr else ""

    queries: list[str] = []
    if user and ocr_snip:
        queries.append(f"{user}\n{ocr_snip}")
        queries.append(user)
        queries.append(ocr_snip)
    elif user:
        queries.append(user)
    elif ocr_snip:
        queries.append(ocr_snip)
    else:
        queries.append("业务排查")

    seen: set[str] = set()
    uniq: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq


def _summarize_hits(hits: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    """给 SSE / prompt 用的精简 hit。"""
    out: list[dict[str, Any]] = []
    for h in (hits or [])[:limit]:
        text = str(h.get("text") or "")
        out.append(
            {
                "id": h.get("id"),
                "score": h.get("score"),
                "path": h.get("path") or "",
                "symbol": h.get("symbol") or "",
                "node_type": h.get("node_type") or "",
                "start_line": h.get("start_line"),
                "end_line": h.get("end_line"),
                "text": text[:1200],
            }
        )
    return out


def format_code_evidence(hits: list[dict[str, Any]], *, max_chars: int = 6000) -> str:
    """把命中拼成给 LLM 的证据块。"""
    if not hits:
        return "（未命中代码，请提示用户先完成 wintogether-bff 索引，或换表述重试）"
    parts: list[str] = []
    used = 0
    for i, h in enumerate(hits, start=1):
        path = h.get("path") or "?"
        symbol = h.get("symbol") or ""
        start = h.get("start_line")
        end = h.get("end_line")
        score = h.get("score")
        loc = f"{path}:{start}-{end}" if start is not None else path
        header = f"[{i}] {loc}"
        if symbol:
            header += f"  symbol={symbol}"
        if score is not None:
            header += f"  score={score}"
        body = str(h.get("text") or "").strip()
        block = f"{header}\n{body}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts).strip()


def run_retrieve(
    *,
    user_text: str,
    image_description: str = "",
) -> dict[str, Any]:
    """仅用用户原话 + OCR 多路检索。"""
    queries = build_retrieve_queries(
        user_text=user_text,
        image_description=image_description,
    )
    result = code_search.retrieve_for_diagnose(queries)
    hits = _summarize_hits(result.get("hits") or [])
    return {
        "code_hits": hits,
        "retrieve_queries": result.get("queries") or queries,
        "index": result.get("index") or {},
        "retrieve_error": result.get("error") or "",
    }


def retrieve_node(state: ChatAgentState) -> dict[str, Any]:
    """Graph 节点：从本轮 user_text / image_description 检索。"""
    messages = list(state.get("messages") or [])
    user_text = str(state.get("user_text") or "").strip()
    if not user_text:
        for msg in reversed(messages):
            role = getattr(msg, "type", None)
            if role in ("human", "user"):
                user_text = str(getattr(msg, "content", "") or "")
                break
    image_description = str(state.get("image_description") or "")
    try:
        out = run_retrieve(user_text=user_text, image_description=image_description)
    except Exception as exc:  # noqa: BLE001
        logger.exception("retrieve_node 失败")
        return {
            "code_hits": [],
            "retrieve_queries": [],
            "errors": list(state.get("errors") or []) + [f"retrieve: {exc}"],
        }
    errors = list(state.get("errors") or [])
    if out.get("retrieve_error"):
        errors.append(f"retrieve: {out['retrieve_error']}")
    return {
        "code_hits": out["code_hits"],
        "retrieve_queries": out["retrieve_queries"],
        "errors": errors,
    }
