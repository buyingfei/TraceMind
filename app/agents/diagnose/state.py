"""排查 Graph 状态定义。"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class HelloState(TypedDict):
    """骨架阶段 Hello Graph 状态（无模型）。"""

    name: str
    message: str


class ChatAgentState(TypedDict, total=False):
    """带记忆的对话 Agent 状态（短期记忆主体是 messages）。"""

    messages: Annotated[list, add_messages]
    user_text: str
    reply: str
    pipeline: str
    model_key: str
    image_description: str
    errors: list[str]
    code_hits: list[dict[str, Any]]
    retrieve_queries: list[str]


class DiagnoseState(TypedDict, total=False):
    """正式排查流水线状态（后续扩展）。"""

    ticket_text: str
    entities: dict
    hypotheses: list
    code_hits: list
    db_evidence: list
    final_answer_biz: str
    final_answer_tech: str
    confidence: float
