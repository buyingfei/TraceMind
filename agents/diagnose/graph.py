"""排查 Graph 组装。

- Hello Graph：骨架验证
- Chat Graph：retrieve → answer，挂 Postgres checkpointer
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.diagnose.nodes.chat_answer import answer_node
from agents.diagnose.nodes.hello import greet_node
from agents.diagnose.nodes.retrieve_code import retrieve_node
from agents.diagnose.state import ChatAgentState, HelloState
from app.services.checkpoint_service import get_checkpointer

_hello_graph = None
_chat_graph = None


def build_hello_graph():
    """构建并编译 Hello StateGraph。"""
    builder = StateGraph(HelloState)
    builder.add_node("greet", greet_node)
    builder.add_edge(START, "greet")
    builder.add_edge("greet", END)
    return builder.compile()


def get_hello_graph():
    global _hello_graph
    if _hello_graph is None:
        _hello_graph = build_hello_graph()
    return _hello_graph


def run_hello_graph(name: str = "World") -> HelloState:
    """同步执行 Hello Graph。"""
    graph = get_hello_graph()
    return graph.invoke({"name": name, "message": ""})


def build_chat_graph(*, checkpointer=None):
    """对话 Graph：START → retrieve → answer → END。"""
    builder = StateGraph(ChatAgentState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", END)
    cp = checkpointer if checkpointer is not None else get_checkpointer()
    return builder.compile(checkpointer=cp)


def get_chat_graph():
    global _chat_graph
    if _chat_graph is None:
        _chat_graph = build_chat_graph()
    return _chat_graph


def reset_chat_graph() -> None:
    """测试或热重载时清空单例。"""
    global _chat_graph
    _chat_graph = None
