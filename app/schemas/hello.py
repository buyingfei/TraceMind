"""Hello 接口 DTO。"""

from pydantic import BaseModel, Field


class HelloRequest(BaseModel):
    name: str = Field(default="World", description="打招呼对象")


class HelloResponse(BaseModel):
    message: str
    graph: str = "diagnose.hello"
