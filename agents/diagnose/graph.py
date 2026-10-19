"""排查 Graph 组装。

骨架阶段仅提供 Hello Graph，验证 LangGraph 可运行且不依赖模型。
后续在此扩展：parse → hypothesize → retrieve_code → query_db → write_answer。
"""

from langgraph.graph import END, START, StateGraph

from agents.diagnose.nodes.hello import greet_node
from agents.diagnose.state import HelloState

_hello_graph = None


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
