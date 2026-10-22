"""代码向量检索节点（用户原话 + OCR + 关键路径补检索）。"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.diagnose.biz_verify import CHAIN_PATH_QUERIES, looks_like_chain_scan_scene
from app.agents.diagnose.state import ChatAgentState
from app.agents.tools import code_search
from app.core.logging import clip_text

logger = logging.getLogger(__name__)


def build_retrieve_queries(
    *,
    user_text: str,
    image_description: str = "",
) -> list[str]:
    """多路语义检索：问句/OCR + 连锁场景关键路径词。"""
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

    if looks_like_chain_scan_scene(user, ocr):
        queries.extend(CHAIN_PATH_QUERIES)

    seen: set[str] = set()
    uniq: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq


def _summarize_hits(hits: list[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
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


def format_code_evidence(hits: list[dict[str, Any]], *, max_chars: int = 8000) -> str:
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
    """向量检索入口：只找代码证据，不调用对话模型。

    返回的 code_hits 会由 chat_runtime / chat_answer 拼进本轮 user 消息
    （见 build_evidence_user_suffix），再交给 gpt-sol / Deep Agent 回答。
    链路：run_retrieve → code_hits → build_openai_messages_for_turn → 调模型。
    """
    # 多路 query：用户问句 / OCR /（连锁场景）关键路径关键词
    queries = build_retrieve_queries(
        user_text=user_text,
        image_description=image_description,
    )
    # 连锁/K2 自动入出库：多取几条，并优先保留关键路径命中
    path_boost = looks_like_chain_scan_scene(user_text, image_description)
    logger.info(
        "retrieve start queries=%s path_boost=%s user=%r",
        len(queries),
        path_boost,
        clip_text(user_text, limit=80),
    )
    # Milvus 多路 embedding 检索并合并（非 LLM）
    result = code_search.retrieve_for_diagnose(
        queries,
        final_top_k=10 if path_boost else 8,
        prefer_critical=path_boost,
    )
    # 截断正文，供后续注入 prompt / SSE
    hits = _summarize_hits(result.get("hits") or [], limit=10 if path_boost else 8)
    top = hits[0] if hits else {}
    logger.info(
        "retrieve done hits=%s err=%r top=%s:%s",
        len(hits),
        (result.get("error") or "")[:120],
        top.get("symbol") or "",
        top.get("path") or "",
    )
    return {
        "code_hits": hits,  # 下游必须带上，否则模型看不到检索结果
        "retrieve_queries": result.get("queries") or queries,
        "index": result.get("index") or {},
        "retrieve_error": result.get("error") or "",
        "path_boost": path_boost,
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
