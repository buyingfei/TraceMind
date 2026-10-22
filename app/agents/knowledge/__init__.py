"""已迁移至 app.agents.skills（Agent Skills 规范）。保留再导出以免旧 import 断裂。"""

from __future__ import annotations

from app.agents.skills import knowledge_suffix_for_intent, load_knowledge

__all__ = ["knowledge_suffix_for_intent", "load_knowledge"]
