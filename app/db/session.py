"""数据库会话（骨架占位）。"""

from app.core.exceptions import NotImplementedCapabilityError


def get_session():
    """后续接入 SQLAlchemy AsyncSession。"""
    raise NotImplementedCapabilityError("db.session.get_session")
