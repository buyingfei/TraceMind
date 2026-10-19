"""业务库只读 SQL（骨架占位）。

注意：不要与 app/db（本系统库）混用。
"""

from app.core.exceptions import NotImplementedCapabilityError


def run_select(sql: str, params: dict | None = None) -> list[dict]:
    """后续接入只读账号 + SELECT 白名单。"""
    raise NotImplementedCapabilityError("tools.readonly_sql.run_select")
