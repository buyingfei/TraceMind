"""PaddleOCR 引擎单例。"""

from __future__ import annotations

from functools import lru_cache


@lru_cache
def get_paddle_ocr():
    """懒加载 OCR 引擎（进程内单例）。"""
    import os

    # Windows + paddle 3.x 上 oneDNN/PIR 易报错，默认关闭
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "未安装 PaddleOCR，请执行: pip install paddlepaddle paddleocr opencv-python-headless pillow"
        ) from exc

    # 3.x: use_textline_orientation；关闭文档预处理与 mkldnn
    # 2.x: use_angle_cls
    for kwargs in (
        {
            "lang": "ch",
            "use_textline_orientation": True,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "enable_mkldnn": False,
        },
        {
            "lang": "ch",
            "use_textline_orientation": True,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
        },
        {"lang": "ch", "use_textline_orientation": True},
        {"lang": "ch", "use_angle_cls": True},
        {"lang": "ch"},
        {},
    ):
        try:
            return PaddleOCR(**kwargs)
        except (TypeError, ValueError):
            continue
    raise RuntimeError("无法初始化 PaddleOCR")
