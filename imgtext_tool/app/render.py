"""新文字渲染引擎。

将用户输入的新文字按提取/微调后的参数渲染，尽量贴近原图质感：
颜色、字号、字间距、行间距、倾斜角度、粗细、清晰度/模糊、边缘抗锯齿、
以及轻微噪点/压缩感/柔边。

使用 Pillow 渲染文字，再用 numpy/OpenCV 做后处理（模糊、锐化、噪点、柔边）。
"""
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def _load_font(path: str | None, em_size: int) -> ImageFont.FreeTypeFont:
    """按 em 字号加载字体，失败回退到默认字体。

    【本轮重构要点】全流程统一到 em 字号：feature_extract 输出的就是 em，
    渲染侧直接用它构造字体，测量/匹配/渲染三处同尺寸，口径天然自洽。
    旧实现在渲染侧再用「墨迹高」反推 em，与字体匹配时的测量尺寸不一致，
    字重、宽窄都在错位的口径下比较——这是长期对不准的根源之一。
    """
    try:
        if path:
            return ImageFont.truetype(path, max(6, int(em_size)))
    except Exception:
        pass
    return ImageFont.load_default()


def _resolve_font(path: str | None, size: int, bold: bool) -> ImageFont.FreeTypeFont:
    """加载字体，支持回退到默认字体。"""
    return _load_font(path, size)


def _font_ink_at_size(path: str, sample_text: str, size: int) -> float:
    """返回字体在指定 em 下，sample_text（首行）的墨迹竖直高度（像素）。

    用于诊断与对比（如验证 em 推定精度），不再参与渲染字号反推。
    """
    try:
        font = ImageFont.truetype(path, size)
        b = font.getbbox(sample_text)
        if b is None:
            return 0.0
        return float(b[3] - b[1])
    except Exception:
        return 0.0


def _resolve_font_for_ink(path: str | None,
                          target_ink_h: int,
                          text: str = "国") -> ImageFont.FreeTypeFont:
    """【已废弃，仅保留兼容】旧口径：由「目标墨迹高」反推 em 字号。

    重构后 em 由 feature_extract 直接给出，本函数退化为按 em 加载。
    保留它是为了不让既有调用点崩掉。
    """
    return _load_font(path, target_ink_h)


def _measure_with_spacing(draw: ImageDraw.ImageDraw,
                          text: str,
                          font: ImageFont.FreeTypeFont,
                          letter_spacing: int,
                          line_spacing: int,
                          ) -> Tuple[int, int]:
    """测量带字间距/行间距的文字块尺寸。"""
    lines = text.split("\n")
    max_w = 0
    total_h = 0
    ascent, descent = font.getmetrics()
    line_h = ascent + descent + line_spacing
    for line in lines:
        w = 0
        for ch in line:
            w += draw.textlength(ch, font=font) + letter_spacing
        if line:
            w -= letter_spacing  # 去掉最后一个字符多余间距
        max_w = max(max_w, int(w))
        total_h += line_h
    if lines:
        total_h -= line_spacing
    return max_w, total_h


def render_text(
    text: str,
    params: Dict,
    font_path: str | None = None,
) -> np.ndarray:
    """渲染新文字，返回 RGBA numpy 数组（透明背景，文字有抗锯齿）。

    params 需要包含：
    - color_rgb: (r,g,b)
    - font_size: int
    - letter_spacing: int
    - line_spacing: int  (相对字号的比例或像素，这里用像素)
    - skew_angle: float (度)
    - sharpness: float 0~1 (越大越锐)
    - blur: float 0~1 (额外模糊程度)
    - bold: bool (是否加粗)
    - noise: float 0~1 (边缘噪点强度)
    - opacity: float 0~1 (文字透明度)
    """
    return _render_core(text, params, font_path)[0]


def render_text_with_metrics(text: str,
                             params: Dict,
                             font_path: str | None = None,
                             ) -> Tuple[np.ndarray, Dict]:
    """渲染新文字，同时返回排版度量（含基线位置）。

    返回 (RGBA, metrics)。metrics 关键字段：
      baseline_y      第一行基线在画布中的 y —— **基线对齐的锚点**
      line_baselines  每行基线的 y 列表
      em_size         实际使用的 em 字号
      ascent/descent  字体度量

    调用方应把 baseline_y 对齐到原图文字的基线，而不是对齐墨迹包围盒中心。
    """
    return _render_core(text, params, font_path)


def _render_core(text: str,
                 params: Dict,
                 font_path: str | None = None,
                 ) -> Tuple[np.ndarray, Dict]:
    """渲染核心实现：按 em 加载字体、按基线绘制。

    【本轮重构要点】绘制锚点由默认的 "la"（左上角）改为 **"ls"（左-基线）**，
    并记录基线在画布中的精确 y。这样调用方可以做真正的基线对齐，
    而不再用「墨迹包围盒中心」这种随字符剧烈变化的伪锚点
    （diag 实测：em=64 时墨迹中心到基线的距离跨度达 23px）。
    """
    color = tuple(int(c) for c in params.get("color_rgb", (0, 0, 0)))
    em_size = max(6, int(params.get("font_size", 16)))   # 统一口径：em 字号
    letter_spacing = int(params.get("letter_spacing", 0))
    line_spacing = int(params.get("line_spacing", 0))
    skew_angle = float(params.get("skew_angle", 0.0))
    sharpness = float(params.get("sharpness", 0.7))
    blur = float(params.get("blur", 0.0))
    bold = bool(params.get("bold", False))
    noise = float(params.get("noise", 0.0))
    opacity = float(params.get("opacity", 1.0))
    # 原图实测笔画厚度（像素，全宽）。由 feature_extract 测量后传入。
    # 注意：本工具「不做渲染期描边校正」——字重完全由「字体选择」决定
    # （font_match 按此值挑选天然字重最接近的字体）。渲染阶段因此既不
    # 强行加粗、也不强行加细。该参数仅作信息保留，不参与描边计算。
    target_stroke_px = float(params.get("target_stroke_px", 0.0))

    # em 字号直接构造字体（不再由墨迹高反推）
    font = _load_font(font_path, em_size)
    try:
        ascent, descent = font.getmetrics()
    except Exception:
        ascent, descent = int(em_size * 1.0), int(em_size * 0.25)

    lines = text.split("\n")
    line_h = ascent + descent + line_spacing

    # 逐行测量宽度（逐字符累加，与绘制时的推进保持一致）
    tmp = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    line_widths = []
    for line in lines:
        w = 0.0
        for ch in line:
            w += d.textlength(ch, font=font) + letter_spacing
        if line:
            w -= letter_spacing
        line_widths.append(max(0, int(round(w))))
    tw = max(line_widths) if line_widths else 0

    # 字重控制：严格交由「字体选择」完成。渲染阶段 **不施加任何描边**，
    # 既不强行加粗、也不强行加细——新文字的笔画厚度 = 所选字体的天然厚度。
    # 「加粗」复选框仅作为用户主动叠加的手动覆盖项（非自动行为）。
    stroke_w = max(1, em_size // 24) if bold else 0
    stroke_w = max(0, int(stroke_w))

    pad = max(4, int(em_size * 0.5))
    W = tw + pad * 2
    # 画布高度按「基线 + 上下伸展」精确计算，避免字形被裁切
    H = pad + ascent + (len(lines) - 1) * line_h + descent + pad
    img = Image.new("RGBA", (max(1, W), max(1, H)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    fill = color + (int(255 * opacity),)
    baseline0 = pad + ascent          # 第一行基线
    baselines = []
    for i, line in enumerate(lines):
        by = baseline0 + i * line_h
        baselines.append(by)
        x = pad
        for ch in line:
            # anchor="ls"：以基线为锚，保证不同字符共享同一条基线
            draw.text((x, by), ch, font=font, anchor="ls", fill=fill,
                      stroke_width=stroke_w, stroke_fill=fill)
            x += draw.textlength(ch, font=font) + letter_spacing

    # 转 numpy RGBA
    arr = np.array(img)

    # 倾斜处理：画布尺寸与坐标都会变，基线需用同一变换矩阵同步换算
    if abs(skew_angle) > 0.05:
        h0, w0 = arr.shape[:2]
        arr, M = _apply_skew_with_matrix(arr, skew_angle)
        pt = np.array([w0 / 2.0, float(baseline0), 1.0])
        new_pt = M @ pt
        shift_x = new_pt[0] - w0 / 2.0
        shift_y = new_pt[1] - float(baseline0)
        baseline0 = int(round(baseline0 + shift_y))
        baselines = [int(round(b + shift_y)) for b in baselines]

    # 自动模糊质感融合：把特征提取得到的模糊度（blur 参数，0~3 的 sigma）
    # 施加到新文字上，使新字失焦程度贴近原图（关键修复：不再被 sharpness 条件门控，
    # 否则自动模式下 blur_level 永远到不了渲染层）。
    if blur > 0.05:
        arr = _gaussian_blur_alpha(arr, blur)

    # 清晰度（手动覆盖项）：sharpness 偏低时额外柔化，作为用户微调手段
    if sharpness < 0.45:
        sigma = (0.5 - sharpness) * 2.0
        if sigma > 0.1:
            arr = _gaussian_blur_alpha(arr, sigma)

    # 边缘柔化：给 alpha 通道做轻微模糊，避免硬边
    if params.get("edge_soften", 0.0) > 0:
        s = params.get("edge_soften", 0.0) * 2.0
        arr = _gaussian_blur_alpha(arr, s)

    # 边缘噪点/压缩颗粒
    if noise > 0.0:
        arr = _add_noise(arr, noise)

    metrics = {
        "baseline_y": baseline0,
        "line_baselines": baselines,
        "em_size": em_size,
        "ascent": ascent,
        "descent": descent,
        "line_height": line_h,
        "line_widths": line_widths,
    }
    return arr, metrics


def _apply_skew_with_matrix(arr: np.ndarray,
                            angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """对 RGBA 图像做倾斜（绕中心旋转），同时返回变换矩阵 M。

    返回矩阵是为了让调用方把「基线」等排版坐标同步变换到倾斜后的画布，
    否则倾斜后基线位置会失准。
    """
    h, w = arr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += (nw / 2) - w / 2
    M[1, 2] += (nh / 2) - h / 2
    out = cv2.warpAffine(arr, M, (nw, nh), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return out, M


def _apply_skew(arr: np.ndarray, angle: float) -> np.ndarray:
    """对 RGBA 图像做倾斜（绕中心旋转）。"""
    return _apply_skew_with_matrix(arr, angle)[0]


def _gaussian_blur_alpha(arr: np.ndarray, sigma: float) -> np.ndarray:
    """对 RGB 和 Alpha 都做高斯模糊。"""
    k = int(round(sigma * 4)) | 1
    if k < 1:
        return arr
    return cv2.GaussianBlur(arr, (k, k), sigma)


def _add_noise(arr: np.ndarray, strength: float) -> np.ndarray:
    """在文字 alpha 边缘加轻微噪点，模拟压缩颗粒感。"""
    h, w = arr.shape[:2]
    alpha = arr[:, :, 3].astype(np.float32) / 255.0
    # 只在半透明边缘（0 < alpha < 1）加噪，避免污染纯色内部
    edge = np.zeros_like(alpha)
    mask = (alpha > 0.05) & (alpha < 0.95)
    noise_arr = np.random.normal(0, strength * 30.0, (h, w)).astype(np.float32)
    edge[mask] = noise_arr[mask]
    # 也轻微扰动 RGB
    rgb = arr[:, :, :3].astype(np.float32)
    rgb[:, :, 0] += edge
    rgb[:, :, 1] += edge
    rgb[:, :, 2] += edge
    arr[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return arr


def composite_text(base_bgr: np.ndarray,
                   text_rgba: np.ndarray,
                   position: Tuple[int, int],  # (x, y) 左上角，相对 base
                   ) -> np.ndarray:
    """将渲染好的文字 RGBA 合成到背景 BGR 图上，返回新 BGR 图。"""
    result = base_bgr.copy()
    x, y = position
    th, tw = text_rgba.shape[:2]
    H, W = base_bgr.shape[:2]

    # 裁剪越界部分
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + tw); y1 = min(H, y + th)
    if x1 <= x0 or y1 <= y0:
        return result

    tx0 = x0 - x; ty0 = y0 - y
    tx1 = tx0 + (x1 - x0); ty1 = ty0 + (y1 - y0)

    sub = text_rgba[ty0:ty1, tx0:tx1]
    alpha = sub[:, :, 3].astype(np.float32) / 255.0
    # 渲染层来自 PIL（**RGB** 顺序），底图是 OpenCV 的 **BGR** —— 必须翻转通道，
    # 否则红字会被渲染成蓝字、蓝字变成红字（黑白灰三通道相同才看不出差别）。
    rgb = sub[:, :, 2::-1].astype(np.float32)

    roi = result[y0:y1, x0:x1].astype(np.float32)
    blended = rgb * alpha[:, :, None] + roi * (1 - alpha[:, :, None])
    result[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def text_ink_bbox(text_rgba: np.ndarray, threshold: int = 40):
    """计算渲染文字的实际墨迹包围盒 (x0, y0, x1, y1)，用于与原文字基线对齐。

    只统计 alpha 超过阈值的像素，忽略抗锯齿的极淡边缘。
    无墨迹时返回 None。
    """
    if text_rgba is None or text_rgba.size == 0:
        return None
    alpha = text_rgba[:, :, 3]
    ys, xs = np.where(alpha > threshold)
    if ys.size == 0 or xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def measure_blank_gap(text_rgba: np.ndarray, threshold: int = 40) -> int | None:
    """测量 RGBA 文字图中相邻字符墨迹之间的「空白列」宽度中位数（忽略首尾留白）。

    用于字间距闭环校准：让渲染结果与「原图字间距」逐像素一致。
    只统计字符**之间**的内部空白段（两端留白不计），取中位数抗异常；
    单字（无可测间隔）或字符重叠（无整列空白）返回 None。
    """
    if text_rgba is None or text_rgba.size == 0:
        return None
    alpha = text_rgba[:, :, 3]
    col = (alpha > threshold).any(axis=0)
    gaps, start, in_c = [], None, False
    for i, v in enumerate(col):
        if v:
            # 遇到前景列：若刚离开一段空白，记录该段完整宽度（首列→此列）
            if start is not None:
                gaps.append(i - start)
                start = None
            in_c = True
        else:
            # 仅在「由前景进入空白」的第一列记录起点；连续空白列不再覆盖，
            # 否则会只量到 1px 过渡而不是整段空白宽度。
            if in_c:
                start = i
            in_c = False
    gaps = [g for g in gaps if 1 <= g <= 400]
    if not gaps:
        return None
    return int(round(float(np.median(gaps))))
