"""Hello World 接口：演示 FastAPI + LangGraph（无模型）。"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from agents.diagnose.graph import run_hello_graph

router = APIRouter()


class HelloRequest(BaseModel):
    name: str = Field(default="World", description="打招呼对象")


class HelloResponse(BaseModel):
    message: str
    graph: str = "diagnose.hello"


@router.get("/hello", response_model=HelloResponse)
def hello_get(name: str = "World") -> HelloResponse:
    result = run_hello_graph(name=name)
    return HelloResponse(message=result["message"])


@router.post("/hello", response_model=HelloResponse)
def hello_post(body: HelloRequest) -> HelloResponse:
    result = run_hello_graph(name=body.name)
    return HelloResponse(message=result["message"])
