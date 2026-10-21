"""兼容层：请改用 app.db.repositories。"""

from app.db.repositories.ticket import TicketRepository

__all__ = ["TicketRepository"]
