"""排查 Graph 状态定义。"""

from typing import TypedDict


class HelloState(TypedDict):
    """骨架阶段 Hello Graph 状态（无模型）。"""

    name: str
    message: str


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
