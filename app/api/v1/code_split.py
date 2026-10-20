"""代码切块预览接口（Tree-sitter）。"""

from fastapi import APIRouter, HTTPException, Query

from app.services.treesitter_java_splitter import split_java_file, split_preview_from_config

router = APIRouter(prefix="/code-split", tags=["code-split"])


@router.get("/preview")
def preview_from_config() -> dict:
    """按 config/code_index.yaml 中的 preview.file 切割，返回完整结果供查看。"""
    try:
        return split_preview_from_config()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Tree-sitter 切割失败: {exc}") from exc


@router.get("/file")
def preview_file(
    path: str = Query(
        ...,
        description="Java 文件绝对路径，例如 D:/Codes/wintogether-bff/.../Xxx.java",
    ),
    include_preamble: bool = True,
    max_chars: int = 4000,
) -> dict:
    """对任意 Java 文件做 Tree-sitter 切割预览。"""
    try:
        chunks = split_java_file(
            path,
            include_preamble=include_preamble,
            max_chars=max_chars,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Tree-sitter 切割失败: {exc}") from exc

    return {
        "file": path,
        "chunk_count": len(chunks),
        "chunks": [c.to_dict() for c in chunks],
    }
