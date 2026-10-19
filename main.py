"""本地启动入口。

用法:
    conda activate langgraph
    pip install -r requirements.txt
    python main.py
"""

import uvicorn

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "local",
    )


if __name__ == "__main__":
    main()
