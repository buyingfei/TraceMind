"""Hello 节点：不调用任何大模型。"""

from app.agents.diagnose.state import HelloState


def greet_node(state: HelloState) -> HelloState:
    name = (state.get("name") or "World").strip() or "World"
    return {
        "name": name,
        "message": f"Hello, {name}! diagnose-agent skeleton is ready.",
    }
