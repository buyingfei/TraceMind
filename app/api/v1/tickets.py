"""工单接口占位：后续接建工单 / 查进度 / 确认结论。"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class TicketCreateRequest(BaseModel):
    title: str = Field(..., description="问题标题")
    description: str = Field(..., description="业务描述")


class TicketCreateResponse(BaseModel):
    ticket_id: str
    status: str
    message: str


@router.post("", response_model=TicketCreateResponse)
def create_ticket(body: TicketCreateRequest) -> TicketCreateResponse:
    """骨架占位：尚未接入真实工单存储与排查流水线。"""
    return TicketCreateResponse(
        ticket_id="ticket_placeholder_001",
        status="accepted",
        message=(
            f"已收到工单「{body.title}」。"
            "后续将在此触发 diagnose Graph；当前为骨架占位。"
        ),
    )


@router.get("/{ticket_id}")
def get_ticket(ticket_id: str) -> dict[str, str]:
    return {
        "ticket_id": ticket_id,
        "status": "placeholder",
        "message": "查询进度能力尚未实现",
    }
