"""本地 OCR：长截图分段识别。"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

_OCR_ENGINE = None


def _get_ocr():
    global _OCR_ENGINE
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE
    try:
        from rapidocr_onnxruntime import RapidOCR

        _OCR_ENGINE = RapidOCR()
        return _OCR_ENGINE
    except Exception:
        return None


def _prepare_image(img: "Image.Image") -> tuple["Image.Image", bool]:
    """窄长截图放大，提升 OCR 识别率。返回 (处理后图片, 是否极窄长图)。"""
    from PIL import Image, ImageEnhance, ImageOps

    width, height = img.size
    is_narrow_strip = width <= 120 and height >= 800
    target_width = 1080 if is_narrow_strip else (720 if width < 120 else 480 if width < 360 else width)
    if width < target_width:
        scale = target_width / max(width, 1)
        img = img.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
    if height > 6000:
        scale = 6000 / height
        img = img.resize((int(img.size[0] * scale), 6000), Image.Resampling.LANCZOS)
    img = ImageOps.grayscale(img).convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    return img, is_narrow_strip or (img.size[1] / max(img.size[0], 1) > 10)


def _ocr_pil_image(img: "Image.Image") -> str:
    engine = _get_ocr()
    if engine is None:
        return ""
    import numpy as np

    result, _ = engine(np.array(img))
    if not result:
        return ""
    lines = [str(item[1]).strip() for item in result if item and len(item) > 1]
    return "\n".join(lines)


def ocr_image_bytes(image_bytes: bytes, slice_height: int = 2200) -> tuple[str, str]:
    """
    对交易长截图做 OCR。过高图片按高度切片，提升小字识别率。
    返回 (识别文本, 说明)
    """
    try:
        from PIL import Image
    except ImportError:
        return "", "未安装 Pillow，无法本地 OCR"

    engine = _get_ocr()
    if engine is None:
        return "", "未安装 rapidocr-onnxruntime，无法本地 OCR"

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        return "", f"图片读取失败: {exc}"

    orig_w, orig_h = img.size
    if orig_w < 200:
        narrow_warn = (
            f"图片过窄（{orig_w}×{orig_h}），OCR 很难识别。"
            "请用手机重新截**完整屏幕宽度**的成交记录，不要只截文字条。"
        )
    else:
        narrow_warn = ""

    img, is_narrow_strip = _prepare_image(img)

    if img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size
    if is_narrow_strip:
        row_h = 120
        chunks: list[str] = []
        for top in range(0, height, row_h):
            crop = img.crop((0, top, width, min(top + row_h, height)))
            part = _ocr_pil_image(crop)
            if part:
                chunks.append(part)
        text = "\n".join(chunks)
        n_rows = (height + row_h - 1) // row_h
        hint = "（极窄长图已放大并按行识别）" if text else narrow_warn or "（极窄长图，建议重新截全屏宽度）"
        return text, f"本地 OCR 完成{hint}（{width}x{height}，{n_rows}行）"

    if height <= slice_height:
        text = _ocr_pil_image(img)
        msg = f"本地 OCR 完成（{width}x{height}）"
        if narrow_warn and not text:
            msg = f"{narrow_warn} {msg}"
        return text, msg

    chunks: list[str] = []
    for top in range(0, height, slice_height):
        bottom = min(top + slice_height, height)
        crop = img.crop((0, top, width, bottom))
        part = _ocr_pil_image(crop)
        if part:
            chunks.append(part)

    merged = "\n".join(chunks)
    n_slices = (height + slice_height - 1) // slice_height
    return merged, f"本地 OCR 完成（{width}x{height}，分{n_slices}段）"
