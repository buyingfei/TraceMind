"""PaddleOCR：图片文字识别。"""

from __future__ import annotations

import base64
import logging
import re
import time
from typing import Any

import numpy as np

from app.common.text import clip_text
from app.infra.ocr import get_paddle_ocr

logger = logging.getLogger(__name__)

_DATA_URL_RE = re.compile(r"^data:(image/[\w.+-]+);base64,(.+)$", re.DOTALL | re.IGNORECASE)


def _decode_image_to_ndarray(image: str) -> np.ndarray:
    """data URL / 纯 base64 → BGR ndarray（OpenCV 风格）。"""
    text = (image or "").strip()
    if not text:
        raise ValueError("图片内容为空")
    match = _DATA_URL_RE.match(text)
    raw_b64 = match.group(2).replace("\n", "") if match else text.replace("\n", "")
    data = base64.b64decode(raw_b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    try:
        import cv2

        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        from io import BytesIO

        from PIL import Image

        pil = Image.open(BytesIO(data)).convert("RGB")
        rgb = np.array(pil)
        # RGB -> BGR
        img = rgb[:, :, ::-1].copy()
    if img is None or getattr(img, "size", 0) == 0:
        raise ValueError("无法解码图片")
    return img


def _run_engine_once(engine: Any, img: np.ndarray) -> None:
    """跑一次推理以预热底层算子（忽略识别结果）。"""
    if hasattr(engine, "predict"):
        try:
            engine.predict(img)
            return
        except TypeError:
            pass
    if hasattr(engine, "ocr"):
        try:
            engine.ocr(img, cls=True)
        except TypeError:
            engine.ocr(img)


def warmup_ocr(*, run_infer: bool = True) -> dict[str, Any]:
    """启动时预热：加载模型，可选对小图跑一次推理。"""
    t0 = time.perf_counter()
    engine = get_paddle_ocr()
    load_ms = int((time.perf_counter() - t0) * 1000)

    infer_ms = 0
    if run_infer:
        t1 = time.perf_counter()
        # 带少量文字的小图，触发 det+rec 路径
        img = np.full((64, 256, 3), 255, dtype=np.uint8)
        img[20:44, 20:236] = 0
        try:
            _run_engine_once(engine, img)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR 预热推理失败（模型已加载）: %s", exc)
        infer_ms = int((time.perf_counter() - t1) * 1000)

    total_ms = int((time.perf_counter() - t0) * 1000)
    info = {
        "status": "ok",
        "load_ms": load_ms,
        "infer_ms": infer_ms,
        "total_ms": total_ms,
    }
    logger.info("OCR warmup done: %s", info)
    return info


def _as_mapping(page: Any) -> dict[str, Any]:
    """把 3.x Result 对象或 dict 统一成可 get 的 mapping。"""
    if isinstance(page, dict):
        return page
    if hasattr(page, "json") and callable(page.json):
        try:
            data = page.json
            if callable(data):
                data = data()
            if isinstance(data, dict):
                # 有的版本包一层 {"res": {...}}
                return data.get("res", data) if isinstance(data.get("res"), dict) else data
        except Exception:  # noqa: BLE001
            pass
    if hasattr(page, "keys") and hasattr(page, "get"):
        try:
            return {k: page.get(k) for k in page.keys()}  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            pass
    # 常见属性名直读
    out: dict[str, Any] = {}
    for key in ("rec_texts", "rec_scores", "texts", "scores"):
        if hasattr(page, key):
            out[key] = getattr(page, key)
    return out


def _lines_from_ocr_result(result: Any) -> list[tuple[str, float]]:
    """从不同版本 PaddleOCR 返回结构中提取 (text, score)。"""
    lines: list[tuple[str, float]] = []
    if result is None:
        return lines

    # 2.x: result = [ [ [box, (text, score)], ... ] ]
    if isinstance(result, list) and result and isinstance(result[0], list):
        page = result[0]
        for item in page or []:
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    info = item[1]
                    if isinstance(info, (list, tuple)) and len(info) >= 2:
                        text, score = str(info[0]), float(info[1])
                        if text.strip():
                            lines.append((text.strip(), score))
            except Exception:  # noqa: BLE001
                continue
        if lines:
            return lines

    # 3.x predict: list[dict|Result] with rec_texts / rec_scores
    if isinstance(result, list):
        for page in result:
            mapping = _as_mapping(page)
            texts = mapping.get("rec_texts") or mapping.get("texts") or []
            scores = mapping.get("rec_scores") or mapping.get("scores") or []
            for i, text in enumerate(texts):
                score = float(scores[i]) if i < len(scores) else 1.0
                if str(text).strip():
                    lines.append((str(text).strip(), score))
    return lines


def ocr_image(
    image: str,
    *,
    min_score: float = 0.5,
) -> dict[str, Any]:
    """对单张图做 OCR，返回拼接文本与明细。"""
    t0 = time.perf_counter()
    img = _decode_image_to_ndarray(image)
    h, w = int(img.shape[0]), int(img.shape[1])
    engine = get_paddle_ocr()

    result = None
    # 优先 3.x predict；再回退 2.x ocr
    if hasattr(engine, "predict"):
        try:
            result = engine.predict(img)
        except TypeError:
            result = None
    if result is None and hasattr(engine, "ocr"):
        try:
            result = engine.ocr(img, cls=True)
        except TypeError:
            result = engine.ocr(img)

    pairs = _lines_from_ocr_result(result)
    kept = [(t, s) for t, s in pairs if s >= min_score]
    if not kept and pairs:
        kept = pairs  # 全被阈值时退回全部

    text = "\n".join(t for t, _ in kept).strip()
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    if not text:
        logger.info("OCR image empty size=%sx%s ms=%s", w, h, elapsed_ms)
        raise RuntimeError("OCR 未识别到有效文字")

    logger.info(
        "OCR image ok size=%sx%s ms=%s preview=%r",
        w,
        h,
        elapsed_ms,
        clip_text(text, limit=100),
    )
    return {
        "key": "paddleocr",
        "provider": "paddleocr",
        "model": "paddleocr-ch",
        "reply": text,
        "lines": [{"text": t, "score": s} for t, s in kept],
    }


def ocr_images(
    images: list[str],
    *,
    min_score: float = 0.5,
) -> dict[str, Any]:
    """多图 OCR，按图片序号拼接。"""
    if not images:
        raise ValueError("images 不能为空")
    t0 = time.perf_counter()
    logger.info("OCR batch start images=%s", len(images))
    blocks: list[str] = []
    all_lines: list[dict[str, Any]] = []
    for idx, image in enumerate(images, start=1):
        one = ocr_image(image, min_score=min_score)
        blocks.append(f"【图片{idx}】\n{one['reply']}")
        for line in one.get("lines") or []:
            all_lines.append({"image_index": idx, **line})
    text = "\n\n".join(blocks).strip()
    logger.info(
        "OCR batch done images=%s ms=%s preview=%r",
        len(images),
        int((time.perf_counter() - t0) * 1000),
        clip_text(text, limit=100),
    )
    return {
        "key": "paddleocr",
        "provider": "paddleocr",
        "model": "paddleocr-ch",
        "reply": text,
        "lines": all_lines,
    }
