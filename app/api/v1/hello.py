"""Hello World 接口：演示 FastAPI + LangGraph（无模型）。"""

from fastapi import APIRouter

from app.agents.diagnose.graph import run_hello_graph
from app.schemas.hello import HelloRequest, HelloResponse

router = APIRouter()


@router.get("/hello", response_model=HelloResponse)
def hello_get(name: str = "World") -> HelloResponse:
    result = run_hello_graph(name=name)
    return HelloResponse(message=result["message"])


@router.post("/hello", response_model=HelloResponse)
def hello_post(body: HelloRequest) -> HelloResponse:
    result = run_hello_graph(name=body.name)
    return HelloResponse(message=result["message"])
