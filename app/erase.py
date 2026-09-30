"""旧文字擦除引擎。

使用 OpenCV inpainting（图像修复）抹除旧文字并还原背景，
尽量保留背景纹理、渐变、噪点、压缩痕迹。

策略：
1. 先做文字前景掩码（自动二值化得到文字笔画区域）。
2. 掩码做形态学膨胀，覆盖文字边缘的抗锯齿过渡。
3. 用 cv2.inpaint 的 TELEA / NS 算法修复。
4. 提供「擦除强度」控制掩码膨胀程度。
"""
import cv2
import numpy as np


def build_text_mask(bgr_region: np.ndarray) -> np.ndarray:
    """从区域中生成文字前景掩码（255=文字，0=背景）。"""
    gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 文字为少数像素；若白色占多数说明文字是深色，取反
    if (bw > 127).mean() > 0.5:
        bw = 255 - bw
    return bw


def erase_text(image_bgr: np.ndarray,
               box: tuple,            # (x0, y0, x1, y1)
               strength: float = 0.5,  # 0~1 擦除强度（掩码膨胀半径）
               ) -> np.ndarray:
    """在全图上擦除 box 区域内的旧文字，返回新图。

    image_bgr: 全图（BGR）
    box: 框选区域 (x0, y0, x1, y1)
    strength: 擦除强度，控制掩码膨胀半径与修复半径。
    """
    result = image_bgr.copy()
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, x1 = sorted([x0, x1])
    y0, y1 = sorted([y0, y1])
    h, w = image_bgr.shape[:2]
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return result

    region = image_bgr[y0:y1, x0:x1]
    mask = build_text_mask(region)

    # 膨胀半径随擦除强度增加，覆盖文字边缘抗锯齿过渡
    k = max(1, int(2 + strength * 6))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.dilate(mask, kernel, iterations=1)
    # 轻微闭运算，填补文字内部孔洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 修复半径
    radius = max(2, int(3 + strength * 5))
    repaired = cv2.inpaint(region, mask, radius, cv2.INPAINT_TELEA)
    result[y0:y1, x0:x1] = repaired
    return result
