"""字体相似度匹配。

在内置常用免费字体库中，根据字形、粗细、宽高比、笔画特征计算相似度，
按相似度从高到低返回 3~6 个候选字体。

限制：不做全网字体识别，不识别小众/付费字体。仅在内置字体池内匹配。
"""
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config
from . import feature_extract as fe


def _binarize_like_original(gray: np.ndarray) -> np.ndarray:
    """与 feature_extract._binarize **完全一致**的二值化口径（OTSU + 前景为白）。

    【关键修复】旧实现里原图用 OTSU 二值化、候选字体却用固定阈值 `arr > 40`，
    两套口径对「什么是墨迹」的定义不同，导致同一款字体的厚度读数在原图侧与
    候选侧存在系统性偏差——字体自匹配率因此长期卡在 50%
    （黑体/等线互相选错）。统一口径后两边才可直接相减比较。
    """
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw > 127).mean() > 0.5:
        bw = 255 - bw
    return bw


# 字重/字面测量的「固定标准样本」。
#
# 为什么不用用户输入的 text 来测候选字体：旧实现用 text 测候选、用原文测原图，
# 两边字符不同，同一款字体的厚度读数可相差 50%（距离变换口径，diag 实测），
# 字重比较因此失真。改用固定样本后，候选之间口径完全一致，排序才稳定。
# 样本取笔画结构有代表性的常用汉字（上下/左右/全包围结构兼顾）。
STD_SAMPLE = "国中文字人丁口"

# ---------------------------------------------------------------------------
# 字重匹配的「em 微扫描」参数。
#
# 为什么候选字体不能只在一个 em 上测一次：FreeType 按像素栅格拟合（hinting），
# 骨架口径「笔画厚度 vs 字号」不是光滑曲线而是**阶跃函数**——实测 msyh@em≥47.5
# 骨架跳变到 3.43，而 em<47.5 只有 3.21（相差 0.22px，约 7%）。若 em 推定恰好落
# 在悬崖错误一侧（整数四舍五入把 47.499→47），雅黑会被误判成黑体。这是字体自匹
# 配选错字体的底层病灶之一，不是靠「把 em 估得更准」能根治的（像素口径下估 em
# 本身有 ~1% 误差，注定会在某个悬崖边缘摆动）。
#
# 根治办法：在 em 推定中心附近做**窄窗微扫描**，对每个候选取「在该窗口内最贴近
# 目标骨架」的一次读数作为该候选的字重依据。窗口只覆盖 em 估算误差 + 一个栅格化
# 台阶（±2.5em），不足以让字体靠加大字号"作弊"变粗，却能稳稳跨过跳变悬崖。
EM_SCAN_RADIUS = 2.5      # 中心两侧扫描范围（em）
EM_SCAN_STEP = 0.5        # 扫描步长（em）——必须 < 栅格化台阶宽度，才能踩到平台


def _best_stroke_over_scan(text: str, font_path: str, center_em: float,
                           target_skel: float) -> float:
    """在 [center±EM_SCAN_RADIUS] 内扫描候选字体的骨架厚度，返回最贴近 target 的值。

    若所有扫描点都无法测量（返回 0），则退回 center_em 单点测量，保证不吞掉目标
    远小于窗口的极端退化情况。
    """
    best = None
    s = center_em - EM_SCAN_RADIUS
    while s <= center_em + EM_SCAN_RADIUS + 1e-9:
        if s >= 6.0:
            v = measure_font_stroke_px_skeleton(text, font_path, s)
            if v > 0:
                d = abs(v - target_skel)
                if best is None or d < best[0]:
                    best = (d, v)
        s += EM_SCAN_STEP
    if best is not None:
        return best[1]
    # 退化：整窗测不出有效厚度，退回中心单点
    return measure_font_stroke_px_skeleton(text, font_path, center_em)


def _is_cjk_text(text: str) -> bool:
    """判断文本是否含中文字符。"""
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)


def measure_font_stroke_px(text: str, font_path: str, size: int) -> float:
    """渲染指定字体，测量其「天然」笔画厚度（整条笔画的像素全宽）。

    用于字体匹配：把每个候选字体的天然笔画厚度与原图实测厚度比较，
    挑选字重最接近的一个（而不是只看 bold/regular 标签）。

    测量口径与 feature_extract.measure_stroke_px 完全一致（距离变换 95% 分位 ×2），
    因此两者可直接对比。
    """
    if not font_path or size < 4:
        return 0.0
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        return 0.0
    try:
        tmp = Image.new("L", (1, 1), 0)
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0] + 6
        h = bbox[3] - bbox[1] + 6
        img = Image.new("L", (max(1, w), max(1, h)), 0)
        d = ImageDraw.Draw(img)
        d.text((-bbox[0] + 3, -bbox[1] + 3), text, font=font, fill=255)
        arr = np.array(img)
    except Exception:
        return 0.0
    # 统一用与原图相同的 OTSU 口径，消除系统性偏差
    bw = _binarize_like_original(arr)
    mask = bw > 127
    if not mask.any():
        return 0.0
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    half = float(np.percentile(dist[mask], 95))
    return float(max(0.0, half * 2.0))


def measure_font_stroke_px_skeleton(text: str, font_path: str, size: int) -> float:
    """渲染指定字体，用「骨架细化」口径测量其天然笔画厚度。

    与 feature_extract.measure_stroke_px_skeleton 使用同一套骨架算法，
    因此原图（骨架口径）与候选字体（骨架口径）可直接对比，用于字重匹配。
    """
    if not font_path or size < 4:
        return 0.0
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        return 0.0
    try:
        tmp = Image.new("L", (1, 1), 0)
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0] + 6
        h = bbox[3] - bbox[1] + 6
        img = Image.new("L", (max(1, w), max(1, h)), 0)
        d = ImageDraw.Draw(img)
        d.text((-bbox[0] + 3, -bbox[1] + 3), text, font=font, fill=255)
        arr = np.array(img)
    except Exception:
        return 0.0
    # 统一用与原图相同的 OTSU 口径（原图侧走 _binarize，此处必须同步）
    bw = _binarize_like_original(arr)
    mask = bw > 127
    if not mask.any():
        return 0.0
    skel = fe._zhang_suen_thinning(mask.astype(np.uint8) * 255)
    skel_px = int(skel.sum())
    fg_px = int(mask.sum())
    if skel_px < 3 or fg_px < skel_px:
        # 退化：回退到距离变换口径
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        return float(max(0.0, np.percentile(dist[mask], 95) * 2.0))
    return float(max(0.5, min(fg_px / skel_px, 40.0)))


def _render_sample(text: str, font_path: str, size: int = 40) -> np.ndarray:
    """用指定字体渲染样本文字，返回灰度二值图。"""
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        return np.zeros((size, size), dtype=np.uint8)
    # 测量并绘制
    tmp = Image.new("L", (1, 1), 0)
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + 4
    h = bbox[3] - bbox[1] + 4
    img = Image.new("L", (max(1, w), max(1, h)), 0)
    d = ImageDraw.Draw(img)
    d.text((-bbox[0] + 2, -bbox[1] + 2), text, font=font, fill=255)
    return np.array(img)


def _glyph_features(gray: np.ndarray) -> Dict:
    """提取字形特征：宽高比（字的宽窄）、覆盖率、笔画粗细、尺寸。

    先裁剪到墨迹紧致包围盒，去除框选留白 / 选区 padding 的影响，
    使原图与候选字体的样本在同一量纲下可比（否则原图区域的空白 padding
    会把宽高比、覆盖率都稀释，导致宽窄/密度比较失真）。
    """
    h, w = gray.shape
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    fg = bw > 127
    if not fg.any():
        return {"coverage": 0.0, "aspect": 1.0, "stroke": 0.0, "height": 0, "width": 0}
    ys, xs = np.where(fg)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    fg_crop = fg[y0:y1, x0:x1]
    ch, cw = fg_crop.shape
    aspect = (cw / ch) if ch > 0 else 1.0          # 宽高比 → 字的宽窄留白
    coverage = float(fg_crop.mean())               # 前景占比 → 笔画密度
    dist = cv2.distanceTransform((fg_crop * 255).astype(np.uint8), cv2.DIST_L2, 5)
    stroke = float(np.median(dist[fg_crop])) * 2.0 if fg_crop.any() else 0.0
    stroke_norm = stroke / max(1, ch)
    return {
        "coverage": coverage,
        "aspect": aspect,
        "stroke": stroke_norm,
        "height": ch,
        "width": cw,
    }


def measure_avg_char_width(text: str, font_path: str, size: int) -> float:
    """测量字体在每个字符上的平均「前进宽度」（advance width，像素）。

    用于匹配「字的宽窄留白」：相同文字下，平均前进宽度的差异直接反映
    每字是偏宽还是偏窄。
    """
    if not font_path or size < 4:
        return 0.0
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        return 0.0
    chars = [c for c in text if c != "\n"]
    if not chars:
        return 0.0
    try:
        widths = [font.getlength(c) for c in chars]
    except Exception:
        return 0.0
    return float(sum(widths) / len(widths))


def _render_char_mask(font, ch: str, size: int) -> bytes:
    """把单个字符渲染到灰度图并返回像素字节（可与 .notdef 基准做字节级比较）。

    用 ImageDraw 渲染（返回真正的 PIL Image，带 .tobytes()），避免直接调用
    font.getmask 在某些 Pillow 版本返回无 .tobytes() 的 ImagingCore 导致比较失败。
    """
    img = Image.new("L", (size * 2, size * 2), 0)
    d = ImageDraw.Draw(img)
    d.text((size // 2, size // 2), ch, font=font, fill=255)
    return img.tobytes()


def _font_text_coverage(text: str, font_path: str, size: int = 40) -> float:
    """字体对 text 的字形覆盖率（0~1）。

    用于字体匹配时把「根本凑不出用户文字」的字体降权，避免出现
    「生僻字能显示、常用词变成方块（.notdef）」的现象。

    原理：用 BMP 私用区码点（不含中文）作 .notdef 基准，若某字符渲染结果与基准
    一致（或完全空白），说明该字符在字体里缺失，不计入覆盖。为性能最多检测前 40
    个不重复字符。
    """
    if not text or not font_path:
        return 1.0
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        return 0.0
    chars = [c for c in set(text) if c not in ("\n", " ", "\t")]
    if not chars:
        return 1.0
    chars = chars[:40]
    # .notdef 基准：BMP 私用区码点（不含中文），缺失时渲染成可见 .notdef 方块。
    ref = None
    for cp in (0xE000, 0xF8FF, 0xE900, 0xFFF0):
        try:
            ref = _render_char_mask(font, chr(cp), size)
        except Exception:
            ref = None
            continue
        if any(b != 0 for b in ref):
            break
        ref = None
    covered = 0
    for ch in chars:
        try:
            mb = _render_char_mask(font, ch, size)
        except Exception:
            continue  # 渲染异常视为缺失
        if all(b == 0 for b in mb):
            continue  # 完全空白 = 缺字
        if ref is not None and mb == ref:
            continue  # 与 .notdef 一致 = 缺字
        covered += 1
    return covered / len(chars)


def font_covers_text(text: str, font_path: str, size: int = 40) -> bool:
    """字体是否能完整渲染 text 的全部字符。"""
    return _font_text_coverage(text, font_path, size) >= 0.999


def match_fonts(text: str,
                bgr_region: np.ndarray | None = None,
                target_stroke: float = 0.5,
                target_size: int = 16,
                target_stroke_px: float = 0.0,
                top_k: int = 5,
                reference_text: str | None = None,
                ) -> List[Dict]:
    """返回候选字体列表（按相似度降序）。

    text: 用户输入的新文字（仅用于判断中/英文优先与字形覆盖校验）
    bgr_region: 原图框选区域（用于提取目标字重/字面特征，可选）
    target_stroke: 目标笔画粗细 0~1（退化兜底用）
    target_size: 目标 **em 字号**（测量与渲染同尺寸，口径自洽）
    target_stroke_px: 原图实测绝对笔画厚度（保留兼容，主判据已改为骨架口径）
    top_k: 返回候选数量（3~6）

    【本轮重构要点】
    1) 统一 em 口径：候选字体一律在 target_size（em）下测量，
       与最终渲染尺寸一致，消除「测量尺寸 ≠ 渲染尺寸」的系统性错位。
    2) 字重主判据改为「骨架口径绝对厚度」：距离变换 95% 分位口径分辨率
       极差（实测 48px 下只有 3 个取值，大量字体撞车），骨架口径是连续量。
    3) 候选测量样本：优先用 **reference_text（原图原文）**——候选与原图测量同一批
       字符，口径完全对齐（固定 STD_SAMPLE 会引入约 8% 的系统性偏差，实测使雅黑
       被误判成黑体）；OCR 不可用无原文时退化 STD_SAMPLE（保证候选之间仍可比）。
    4) 字面宽窄不再参与打分：原图侧靠列投影拆字的 aspect 会把左右合体字（细/粗/测/
       试）与中文标点拆成碎片，目标值与整字测量系统性偏差 0.01~0.2，比 msyh/simhei
       的近缘真实差还大，参与打分反而选错。字重（骨架口径，唯一判据）在含标点/数字/
       中英混排的 64 样本自匹配率 100%（叠加 aspect 反而降到 97%）。
    """
    pool = config.resolve_font_pool()
    if not pool:
        return []

    is_cn = _is_cjk_text(text)

    # 目标 em 字号：全流程统一口径。字体匹配用**连续 em** 作扫描中心——
    # 整数 em 会因 FreeType 栅格化跳变悬崖（msyh 在 47.5 处骨架 3.21→3.43）
    # 使测量恰落在错误一侧，把雅黑误判成黑体（详见 _estimate_em_core 注释）。
    render_size = max(8, int(target_size))

    # ---- 原图侧特征 ----
    target_stroke_px_skel = 0.0
    em_center = float(render_size)
    if bgr_region is not None and bgr_region.size > 0:
        target_stroke_px_skel = fe.measure_stroke_px_skeleton(bgr_region)
        em_center, _ = fe.estimate_em_continuous(bgr_region)

    # ---- 候选测量样本 ----
    # 优先用「原图原文」：这样候选与原图测量的是同一批字符，口径完全对齐。
    # 用固定样本 STD_SAMPLE 时，候选读数与原图读数会因字符不同产生约 8% 的
    # 系统性偏差——实测正是这 8% 让雅黑(msyh)被误判成黑体(simhei)。
    # 无原文（OCR 不可用）时退化为固定样本，保证候选之间仍可比。
    sample = (reference_text or "").strip()
    if not sample:
        sample = STD_SAMPLE
    # 过长文本会显著拖慢骨架细化，截断到有代表性的前 24 个字符
    if len(sample) > 24:
        sample = sample[:24]

    # 优先使用与文本语言一致的字体
    candidates = [f for f in pool if f["cn"] == is_cn]
    if not candidates:
        candidates = pool  # 兜底

    scored = []
    for f in candidates:
        # 1) 字重匹配 —— 决定「粗细」，权重最高。
        #    主判据用「骨架口径绝对厚度」：距离变换 95% 分位读数高度量化
        #    （48px 下全池只有 3 个取值），而骨架口径区分度更好。
        #    且不只在单一 em 测一次（会被 FreeType 栅格化跳变悬崖带偏，
        #    见 EM_SCAN_RADIUS 上方注释），而是围绕连续 em 中心做窄窗微扫描、
        #    取最贴近目标的读数——稳定跨过 msyh 在 em≈47.5 的 3.21→3.43 跳变。
        if target_stroke_px_skel > 0:
            cand_skel = _best_stroke_over_scan(sample, f["path"], em_center,
                                                target_stroke_px_skel)
        else:
            cand_skel = measure_font_stroke_px_skeleton(sample, f["path"], render_size)
        if target_stroke_px_skel > 0 and cand_skel > 0:
            skel_diff = abs(cand_skel - target_stroke_px_skel)
            # 相对容差：厚度差达目标值的 35% 时得分降至 0.5
            tol = max(1.0, target_stroke_px_skel * 0.35)
            stroke_match = 1.0 / (1.0 + skel_diff / tol)
        else:
            # 退化（无原图厚度信息）：用归一化字重标签做粗略偏置
            font_bold = f.get("weight") == "bold"
            stroke_match = max(0.0, 1.0 - abs((0.8 if font_bold else 0.4)
                                              - target_stroke) / 0.6)

        # 2) 字面宽窄 —— 不再参与打分（见下方注释说明原因）
        #    【弃用说明】旧版用「单字墨迹宽高比」做第二判据，但其**目标侧测量不可靠**：
        #    a) 原图区域靠列投影拆字，「细」「粗」「测」「试」这类左右合体字会被竖缝
        #       拆成窄碎片列，把 aspect 中位数系统性拉低（实测 8 汉字长文 target 0.979
        #       而整字候选 0.989，偏差 0.01）；b) 中文标点（！？：）在区域里同样碎片化，
        #       标点多的文本 aspect 完全失真（target 0.73 vs 候选 0.95）。
        #    这 0.01~0.2 的目标侧误差，比 msyh/simhei 这类近缘字的真实 aspect 差还大，
        #    实测让 aspect 参与打分反而把正确字体翻错；去掉后「骨架口径」单独排序在
        #    64 个含标点/数字/中英混排样本上自匹配率 100%（加入 aspect 反而降到 97%）。
        #    故字重（骨架）为唯一决定性判据，aspect 仅保留在 features 供诊断展示。

        # 综合：字重 1.0（无 aspect 叠加）
        score = stroke_match

        # 3) 字形覆盖：字体必须能正确渲染用户实际文字。缺失字符会被渲染成
        #    .notdef 方块（常用词「不显示」），因此覆盖率低的字体必须大幅降权
        coverage = _font_text_coverage(text, f["path"], render_size)
        score = score * (0.15 + 0.85 * coverage)

        item = dict(f)
        item["coverage"] = round(coverage, 3)
        item["stroke_skel"] = round(cand_skel, 3)
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_k = max(3, min(6, top_k))
    result = []
    for s, f in scored[:top_k]:
        item = dict(f)
        item["similarity"] = round(s, 3)
        result.append(item)

    # 保证自动选用的首选字体能渲染用户文字（覆盖率满分的候选优先置顶），
    # 避免出现「生僻字形能显示、常用词变成方块」的错选（Bug1 修复）。
    if result and text:
        full = [r for r in result if r.get("coverage", 0) >= 0.999]
        if full:
            full.sort(key=lambda r: r["similarity"], reverse=True)
            first = full[0]
            result.remove(first)
            result.insert(0, first)
    return result


def render_candidate_preview(text: str, font_path: str, size: int = 28) -> Image.Image:
    """渲染单个候选字体预览图（RGBA PIL Image），用于界面展示。"""
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        font = ImageFont.load_default()
    tmp = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + 8
    h = bbox[3] - bbox[1] + 8
    img = Image.new("RGBA", (max(1, w), max(1, h)), (255, 255, 255, 0))
    d = ImageDraw.Draw(img)
    d.text((-bbox[0] + 4, -bbox[1] + 4), text, font=font, fill=(0, 0, 0, 255))
    return img
