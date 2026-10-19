"""Repository 基类占位。"""

from app.core.exceptions import NotImplementedCapabilityError


class TicketRepository:
    """工单仓储（骨架占位）。"""

    def create(self, *args, **kwargs):
        raise NotImplementedCapabilityError("TicketRepository.create")
