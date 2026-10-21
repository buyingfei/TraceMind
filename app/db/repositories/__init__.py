"""本系统库 Repository（CRUD）。"""

from app.db.repositories.chat import ChatRepository
from app.db.repositories.ticket import TicketRepository

__all__ = ["ChatRepository", "TicketRepository"]
