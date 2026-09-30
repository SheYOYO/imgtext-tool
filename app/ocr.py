"""离线 OCR 文字识别（基于 RapidOCR）。

仅作为「自动识别选区文字」的可选增强，严格遵循两条底线：

1. 离线可用：RapidOCR 的模型随包内置（ch_PP-OCRv4_det/rec/cls 三个 onnx），
   无需联网即可识别中英文。识别结果与 `original_bgr` 同坐标系（均为 BGR）。
2. 优雅降级：若 rapidocr-onnxruntime 未安装 / 模型缺失 / 推理异常，本模块
   一律返回空字符串、绝不让主流程崩溃；主流程会自动回退到「用户手动输入」。

引擎采用懒加载 + 单例缓存：首次调用 `ocr_region` 时才初始化（加载模型约 1~3s），
之后复用同一实例，避免每次框选都重建引擎导致卡顿。
"""
import threading
from typing import List

import numpy as np

# 引擎单例与错误标记（避免反复重试初始化）
_ENGINE = None          # None=未初始化；False=不可用；否则为 RapidOCR 实例
_ENGINE_LOCK = threading.Lock()
_IMPORT_OK = None       # None=未探测；True/False=import 是否成功
_DEFAULT_MIN_SCORE = 0.5


def is_ocr_supported() -> bool:
    """RapidOCR 是否已安装（import 是否可用）。不触发模型加载，开销极小。

    用于 GUI 在启动/框选时判断「能否自动识别文字」。
    """
    global _IMPORT_OK
    if _IMPORT_OK is None:
        try:
            import importlib
            importlib.import_module("rapidocr_onnxruntime")
            _IMPORT_OK = True
        except Exception:
            _IMPORT_OK = False
    return _IMPORT_OK


def _get_engine():
    """懒加载并缓存 RapidOCR 引擎单例。失败则标记不可用（返回 False）。"""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ENGINE = RapidOCR()
        except Exception:
            _ENGINE = False
        return _ENGINE


def ocr_region(bgr_region: np.ndarray, min_score: float = _DEFAULT_MIN_SCORE) -> str:
    """识别 BGR 区域中的文字，返回拼接后的字符串。

    设计：
    - 输入为 BGR numpy（与 OpenCV 读取一致）。
    - 仅保留置信度 >= min_score 的检测结果，过滤噪点/误检。
    - 多段检测结果直接拼接（替换场景多为单行文字）。
    - 任何异常（引擎不可用 / 推理失败 / 空输入）一律返回 ""，绝不抛出。

    返回空字符串时，调用方应回退到「用户手动输入」。
    """
    if bgr_region is None or bgr_region.size == 0:
        return ""
    engine = _get_engine()
    if engine is False or engine is None:
        return ""
    try:
        result, _ = engine(bgr_region)
    except Exception:
        return ""
    if not result:
        return ""

    texts: List[str] = []
    for item in result:
        # RapidOCR 结果项格式：[box(4点), text(str), score(float)]
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        try:
            score = float(item[2])
        except (TypeError, ValueError):
            continue
        if score >= min_score:
            texts.append(str(item[1]))
    return "".join(texts).strip()
