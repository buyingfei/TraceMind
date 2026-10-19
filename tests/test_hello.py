"""Hello World 骨架测试。"""

from fastapi.testclient import TestClient

from agents.diagnose.graph import run_hello_graph
from app.main import app

client = TestClient(app)


def test_run_hello_graph() -> None:
    result = run_hello_graph(name="Tester")
    assert result["name"] == "Tester"
    assert "diagnose-agent skeleton is ready" in result["message"]


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_hello_get() -> None:
    resp = client.get("/api/v1/hello", params={"name": "Biz"})
    assert resp.status_code == 200
    body = resp.json()
    assert "Hello, Biz!" in body["message"]


def test_hello_post() -> None:
    resp = client.post("/api/v1/hello", json={"name": "Dev"})
    assert resp.status_code == 200
    assert "Hello, Dev!" in resp.json()["message"]
