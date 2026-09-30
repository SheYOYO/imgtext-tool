"""文字区域视觉特征自动提取。

从用户框选的区域中提取：
- 文字主色（颜色）
- 字号（区域高度 + 行高 + 文字占比估算）
- 字间距 / 字符间距
- 行间距 / 基线位置
- 清晰度（边缘锐度）
- 字体粗细（笔画宽度）
- 倾斜角度

全部基于 OpenCV + numpy，本地离线，无云端。
"""
import math
from typing import Dict, Tuple, List

import cv2
import numpy as np


# ---------------------------------------------------------------- 字体度量常数
# 下列常数由 diag_metrics.py 对字体池内全部中文字体实测得到（em=100 基准），
# 不是经验猜测。用途：在原图「只有像素、没有字体文件」的情况下，
# 由墨迹 / 行高反推真实的 em 字号与基线位置。
#
# 四款中文字体实测区间：
#   k_ink   0.890 ~ 0.940（字面墨迹高 / em）
#   k_below 0.080 ~ 0.120（墨迹底边在基线下方 / em）
#   k_adv   1.000 ~ 1.000（汉字 advance / em，四款完全一致）
#   k_line  0.920 ~ 0.990（单行占据高 / em）
K_INK_HEIGHT = 0.900   # 汉字字面墨迹高 / em，用于「墨迹高 → em」
K_INK_BELOW = 0.098    # (墨迹底边 - baseline) / em，用于「墨迹底边 → baseline」
K_ADVANCE = 1.000      # 汉字 advance / em
K_LINE = 0.930         # 单行占据高 / em
K_INK_WIDTH = 0.827    # 汉字字面墨迹宽 / em，用于「墨迹宽 → em」（扁平字符兜底通道）

# ---------------------------------------------------------------------------
# 标定系数：上面是「物理实测值」，下面是「工程标定值」。
#
# 二者差一个 OTSU 二值化膨胀因子（约 1.07）：原图经 OTSU 二值化后墨迹会向
# 外膨胀约 3%~8%，若直接用物理系数反推 em，估算值会系统性偏高约 7%
# （diag_em.py 实测常规汉字 1.06~1.09x）。因此标定值 = 物理值 × 1.07，
# 让「估算 em / 真实 em」收敛到 1.00。改动请用 diag_em.py 重新标定。
# ---------------------------------------------------------------------------
CALIB = 1.07
K_INK_HEIGHT_CAL = K_INK_HEIGHT * CALIB   # 0.963
K_LINE_CAL = K_LINE * CALIB               # 0.995
K_INK_WIDTH_CAL = K_INK_WIDTH * CALIB     # 0.885


# ---------------------------------------------------------------- 四向参照阈值
# 「纯噪点/纯背景被误判成文字」的判定阈值（治本修复，见 extract_neighbor_reference）：
#   前景占比下限：低于它说明区域里几乎没墨迹，谈不上文字（排除边缘一两个杂点）。
#   前景占比上限：高于它说明噪点/杂点铺满画面（实测高斯噪点 σ=2 的纯背景窗口
#     前景占比高达 ~0.49，反超真文字），不可能是排版中的文字。
#   最小参照字号：过小的「字号」通常是孤立杂点的伪墨迹尺度，不是真字。
MIN_INK_RATIO = 0.02
MAX_INK_RATIO = 0.40
MIN_NEIGHBOR_EM = 6
# 参照字号相对「窗口高度」的上界：真文字的墨迹不会超过其所在窗口的高太多，
# 若推定的字号远超窗口高度，说明区域里是杂点/噪点被误读成巨大"字"。
MAX_FS_RATIO_OF_WIN_H = 2.2



# ---------------------------------------------------------------- 颜色
def extract_text_color(bgr_region: np.ndarray,
                       background_hint: Tuple[int, int, int] | None = None
                       ) -> Tuple[int, int, int]:
    """提取文字主色（RGB）。

    策略：
    1. 若给出背景色提示，则取与背景差异最大的聚类颜色作为文字色。
    2. 否则对区域做 k-means 聚类，取「饱和度/对比度较高且像素占比不极端」的簇，
       通常文字是深色（黑字白底）或浅色（白字深底）。
    """
    if bgr_region.size == 0:
        return (0, 0, 0)

    # 转 RGB 便于直觉处理
    rgb = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2RGB)
    data = rgb.reshape(-1, 3).astype(np.float32)
    n_pixels = data.shape[0]

    # 像素太少时 k-means 会退化（样本数 < 簇数），直接取区域均值兜底
    if n_pixels < 8:
        mean = data.mean(axis=0)
        return tuple(int(max(0, min(255, round(float(v))))) for v in mean)

    k = min(4, max(2, n_pixels // 50))  # 聚类数，随像素量自适应
    k = max(2, min(k, n_pixels))        # 簇数不得超过样本数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)

    # 每个簇的占比
    counts = np.bincount(labels.flatten(), minlength=k).astype(np.float32)
    ratios = counts / n_pixels

    # 亮度
    lum = centers.mean(axis=1)
    overall_lum = lum @ ratios  # 区域整体亮度

    # 背景一般占多数；文字占少数。
    # 文字颜色通常与整体亮度反差大。取「占比不是最大、且与整体亮度反差最大」的簇。
    best = None
    best_score = -1
    for i in range(k):
        if ratios[i] > 0.75:   # 纯背景色，跳过
            continue
        contrast = abs(float(lum[i]) - float(overall_lum))
        # 轻微偏好更饱和的颜色（文字常有颜色）
        sat = float(np.std(centers[i].astype(np.float32)))
        score = contrast + 0.3 * sat
        if score > best_score:
            best_score = score
            best = tuple(int(c) for c in centers[i])

    if best is None:
        # 退化：取与整体亮度反差最大的簇
        idx = int(np.argmax(np.abs(lum - overall_lum)))
        best = tuple(int(c) for c in centers[idx])

    return best  # RGB


# ---------------------------------------------------------------- 一维投影分段
def _segments(mask) -> List[Tuple[int, int]]:
    """把一维布尔掩码拆成连续 True 段的 [(start, end), ...]（闭区间）。"""
    segs, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            segs.append((s, i - 1))
            s = None
    if s is not None:
        segs.append((s, len(mask) - 1))
    return segs


# ---------------------------------------------------------------- 字号（em）
def estimate_font_size(bgr_region: np.ndarray) -> int:
    """估算 em 字号（像素）——整条渲染链路的唯一字号口径。

    【本次重构的核心改动】返回值由「墨迹像素高度」改为「em 字号」。
    原因：旧实现里「墨迹高」和「em」两套口径混用，渲染侧再用墨迹高反推 em，
    导致字体匹配时的测量字号与最终渲染字号不一致（字重/宽窄都在错位口径下比较）。
    统一为 em 后，测量、匹配、渲染三者同尺寸，口径天然自洽。

    三通道推定（互相校验，抗退化）：
      A 行高通道：单行占据高 / K_LINE_CAL
      B 字高通道：字符墨迹高中位数 / K_INK_HEIGHT_CAL
      C 字宽通道：字符墨迹宽中位数 / K_INK_WIDTH_CAL  ← 扁平字符（「一」「丶」）救命通道
    取 A 与 max(B, C) 的中位数作为最终 em。
    """
    em, _ = estimate_em_size(bgr_region)
    return em


def _merge_text_row_segments(bw: np.ndarray,
                             lines: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """把「整行全空」切出的行段做小间隙合并，避免把**单个密集字**误切成多行。

    病灶：汉字上下堆叠结构的字（赢=亡口月贝凡、臝、鬱…）内部存在**整行全空**
    的留白，行投影会把一个字切成一叠“小行”，每段高只有 ~1/4~1/2 em——实测
    「赢羸蠃臝」@48 的 em 被低估到 22~37px（-22%~-54%），正是用户报「结构紧凑
    的字自动识别偏小」的底层原因。

    合并规则（保守，不吞真行距）：
    - 相邻段间隙 ≤ max(2, 0.30×相邻段较矮者)（字内部留白通常 ≤0.3em，
      而正常行距的墨迹间隙 ≥0.4em，不会误并两行）；
    - 且两段墨迹的 x 方向交集 ≥ 60%（真多行段落左右对齐、交集 100% 也满足，
      但该情况由上面的行距阈值拦截）。
    """
    if len(lines) < 2:
        return list(lines)
    out = [list(lines[0])]
    for (s, e) in lines[1:]:
        ps, pe = out[-1]
        gap = s - pe - 1
        hA, hB = pe - ps + 1, e - s + 1
        if 0 < gap <= max(2, 0.30 * min(hA, hB)):
            xs = np.where((bw[ps:pe + 1] > 127).any(axis=0))[0]
            xe = np.where((bw[s:e + 1] > 127).any(axis=0))[0]
            if xs.size and xe.size:
                ov = min(xs[-1], xe[-1]) - max(xs[0], xe[0])
                mn = min(xs[-1] - xs[0] + 1, xe[-1] - xe[0] + 1)
                if mn > 0 and ov / mn >= 0.6:
                    out[-1][1] = e
                    continue
        out.append([s, e])
    return [tuple(x) for x in out]


def _estimate_em_core(bgr_region: np.ndarray) -> Tuple[float, bool]:
    """em 推定的共享核心：返回「连续浮点 em」与「可靠性」标记。

    与 estimate_em_size（返回四舍五入整数）区别在于**不舍去小数**。为什么必须
    保留连续值：msyh 在 em=47.5 处骨架口径会因 FreeType 像素栅格化发生跳变
    （3.21→3.43），em=47.499 与 47.5 仅差 0.001px 却差 0.22px 笔画读数。
    字体匹配若用四舍五入后的整数测候选，可能正好落在跳变悬崖的错误一侧，
    把雅黑误判成黑体。因此匹配侧必须用连续 em（见 font_match 的 em 扫描）。
    """
    if bgr_region is None or bgr_region.size == 0:
        return 16.0, False
    gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 4 or w < 4:
        return max(6.0, float(h * 0.9)), False
    bw = _binarize(gray)

    # 【关键修复】先做小间隙行段合并——「赢臝鬱」这类上下堆叠字会被行投影
    # 切成一叠小段，导致 em 被低估 22%~54%（用户报紧凑字识别偏小）。
    lines = _merge_text_row_segments(bw, _segments((bw > 127).any(axis=1)))
    if not lines:
        return max(6.0, float(h * 0.55)), False

    line_heights = [e - s + 1 for (s, e) in lines]
    char_hs: List[int] = []
    char_ws: List[int] = []
    for (s, e) in lines:
        band = bw[s:e + 1, :]
        # 列段 = 单个字符（字符间通常有一道空白列）
        segs = list(_segments((band > 127).any(axis=0)))
        if not segs:
            continue
        items = []
        for (cs, ce) in segs:
            sub = band[:, cs:ce + 1]
            rr = np.where((sub > 127).any(axis=1))[0]
            cc = np.where((sub > 127).any(axis=0))[0]
            if rr.size and cc.size:
                items.append((int(rr[-1] - rr[0] + 1), int(cc[-1] - cc[0] + 1)))
        if not items:
            continue
        # 【矮字形过滤（治本）】数字 / 半宽拉丁 / 中文标点/符号的墨迹高度只有汉字
        # 的 0.3~0.7 倍，会把 char_h/char_w 的中位数系统性拉低——实测混排文本
        # 「2026年第3期」@48 的 em 被低估到 39px（-19%），连带字体匹配的扫描
        # 中心错位、把黑体误判成雅黑。这里只保留高度 ≥ 0.78×行内最大高的字形
        # （汉字与满高全角标点都在 0.85em 以上；数字 ~0.7em、小写拉丁 ~0.5em 被剔）。
        # **不能按宽度过滤**：「国」「测」这类合体字会被列投影拆成「高而窄」的碎片
        # 列（竖缝），按宽度剔除会让字宽通道(C)标定口径失配而高估 em
        # （实测 msyh 国测字号 47→49）。按高度剔除矮字形后纯汉字行统计与旧口径
        # 一致、混排行只统计汉字满高字形，两全。
        mh = max(h for h, _ in items)
        for h, w in items:
            if h >= 0.78 * mh:
                char_hs.append(h)
                char_ws.append(w)

    if not char_hs:
        # 完全无法拆分字符：退回到行高通道，且标记不可靠
        return max(6.0, float(np.median(line_heights) / K_LINE_CAL)), False

    line_h = float(np.median(line_heights))
    char_h = float(np.median(char_hs))
    char_w = float(np.median(char_ws)) if char_ws else char_h

    em_from_line = line_h / K_LINE_CAL
    em_from_char = max(char_h / K_INK_HEIGHT_CAL, char_w / K_INK_WIDTH_CAL)
    em = float(np.median([em_from_line, em_from_char]))

    # ---- 可靠性判定 ----
    # 1) 字符墨迹高显著小于其宽度 → 整行是扁平字符（「一」「丶」「-」），
    #    此时高度通道与行高通道会同时崩溃，字宽通道也只是部分救回。
    flat = char_w > 0 and (char_h / max(1.0, char_w)) < 0.45
    # 2) 字符高相对行高过小 → 行内混有极端字符
    thin = char_h < 0.35 * max(1.0, line_h)
    # 3) 推定结果远小于区域高度 → 区域含大量留白或仅标点
    tiny = em < 0.25 * h
    # 4) 一致性检查：常规汉字的墨迹高应 ≈ K_INK_HEIGHT × em。
    #    若实测字符高远低于该预期，说明区域内根本没有「满高」字符
    #    （典型：纯标点「。，、」），此时 em 只是按小墨迹反推出来的假值。
    expected_h = K_INK_HEIGHT * em
    short = char_h < 0.55 * max(1.0, expected_h)
    reliable = not (flat or thin or tiny or short)

    return em, reliable


def estimate_em_size(bgr_region: np.ndarray) -> Tuple[int, bool]:
    """估算 em 字号（四舍五入到整数），并给出「可靠性」标记。

    返回 (em, reliable)。
    reliable=False 表示该区域自身信息不足以可信推定（整行是「一」「丶」这类
    扁平字符、纯标点、或字符严重粘连），调用方应当用「四向邻近文字」的 em 兜底，
    而不是拿一个错误值硬算——这是「自动采集周边文字作为参考」的真正落点。

    旧实现正是死在这里：遇到「一」时单字墨迹高只有 5px，直接被当成字号，
    渲染出的文字小到几乎不可见。

    【注意】这里返回的是**整数** em（供滑杆/渲染/显示等整数消费方用）。若要做
    字体匹配这类对栅格化跳变敏感的测量，请用 estimate_em_continuous 拿连续值，
    避免 47.499 被四舍五入成 47 而落在 msyh 骨架跳变悬崖（47.5）的错误一侧。
    """
    em, reliable = _estimate_em_core(bgr_region)
    return max(6, int(round(em))), reliable


def estimate_em_continuous(bgr_region: np.ndarray) -> Tuple[float, bool]:
    """估算连续（浮点）em 字号，并给出「可靠性」标记。

    与 estimate_em_size 唯一区别是**保留小数**，供对字号误差极敏感的字体匹配用。
    详见 _estimate_em_core 顶部注释（FreeType 栅格化跳变悬崖）。
    """
    em, reliable = _estimate_em_core(bgr_region)
    return max(6.0, float(em)), reliable


def estimate_char_aspect(bgr_region: np.ndarray) -> float:
    """原图文字的「单字墨迹宽高比」中位数（与字数无关的字体特征）。

    【为什么必须换成这个指标】旧实现用整块文字的 aspect（总墨迹宽 / 总墨迹高）
    参与字体匹配，该值与**字数强耦合**：原图 4 字 aspect≈3.96，候选 6 字
    aspect≈6.00，Δaspect 高达 2.04，宽窄得分直接从 1.000 崩到 0.000
    （diag_font_match 实测）。也就是说，用户只要改字数，字形匹配就整体失效。

    改用「单字」级别后，无论选区里是 1 个字还是 20 个字，该值都稳定反映
    字体的字面宽窄，可以在原文与新文字之间安全比较。

    返回 0.0 表示无法可靠测量（字符粘连/区域退化），调用方应跳过该指标。
    """
    if bgr_region is None or bgr_region.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < 4 or gray.shape[1] < 4:
        return 0.0
    bw = _binarize(gray)
    lines = _segments((bw > 127).any(axis=1))
    if not lines:
        return 0.0

    ratios = []
    for (s, e) in lines:
        band = bw[s:e + 1, :]
        for (cs, ce) in _segments((band > 127).any(axis=0)):
            sub = band[:, cs:ce + 1]
            rr = np.where((sub > 127).any(axis=1))[0]
            cc = np.where((sub > 127).any(axis=0))[0]
            if rr.size and cc.size:
                h_ch = int(rr[-1] - rr[0] + 1)
                w_ch = int(cc[-1] - cc[0] + 1)
                # 过滤扁平字符（「一」「丶」）：它们的宽高比不代表字体的字面宽窄
                if h_ch >= 3 and 0.2 <= (w_ch / h_ch) <= 5.0:
                    ratios.append(w_ch / h_ch)
    if not ratios:
        return 0.0
    return float(np.median(ratios))


def estimate_ink_height(bgr_region: np.ndarray) -> int:
    """原图文字的字面墨迹像素高度（旧 estimate_font_size 的语义），保留供诊断/兼容。"""
    if bgr_region is None or bgr_region.size == 0:
        return 16
    gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < 4 or gray.shape[1] < 4:
        return max(6, int(gray.shape[0] * 0.9))
    bw = _binarize(gray)
    lines = _segments((bw > 127).any(axis=1))
    if not lines:
        return max(6, int(gray.shape[0] * 0.55))
    char_heights = []
    for (s, e) in lines:
        band = bw[s:e + 1, :]
        for (cs, ce) in _segments((band > 127).any(axis=0)):
            sub = band[:, cs:ce + 1]
            rr = np.where((sub > 127).any(axis=1))[0]
            if rr.size:
                char_heights.append(int(rr[-1] - rr[0] + 1))
    if not char_heights:
        char_heights = [e - s + 1 for (s, e) in lines]
    return max(6, int(round(float(np.median(char_heights)))))


# ---------------------------------------------------------------- 字间距
def estimate_letter_spacing(bgr_region: np.ndarray) -> int:
    """估算字符间距（像素）。按列统计前景像素，测量相邻字符之间空白列的宽度中位数。"""
    gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw > 127).mean() > 0.5:
        bw = 255 - bw

    h, w = bw.shape
    col_sums = (bw > 127).sum(axis=0)
    fg_cols = col_sums > max(1, h * 0.02)

    # 找空白段（字符间隙），忽略两端的空白。
    # 关键：进入空白的第一列记录起点，连续空白列不再覆盖 start，
    # 否则 start 会被每个空白列刷新、量到的只是 1px 过渡而非整段空白宽度。
    gaps = []
    start = None
    in_content = False
    for i, v in enumerate(fg_cols):
        if v:
            # 遇到前景列：若刚离开一段空白，记录该段完整宽度
            if start is not None:
                gaps.append(i - start)
                start = None
            in_content = True
        else:
            if in_content:
                start = i
            in_content = False

    if not gaps:
        return 0
    # 用中位数，抗异常
    return int(np.median(gaps))


def estimate_avg_char_width(bgr_region: np.ndarray) -> float:
    """估算原图每个字符的平均墨迹宽度（像素），用于匹配「字的宽窄留白」。

    方法：找到文字墨迹的水平紧致跨度，再用列投影把文字拆成若干「字符列段」
    （两个字符之间通常有一道空白列），用 总墨迹宽 / 字符段数 得到每字平均宽。

    当字符被连笔/粘在一起、无法可靠拆分（只得到 1 段）时返回 0.0，
    由调用方回退到宽高比（aspect）比较，避免给出错误宽度。
    """
    if bgr_region is None or bgr_region.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
    bw = _binarize(gray)
    ys, xs = np.where(bw > 127)
    if ys.size == 0:
        return 0.0
    x0, x1 = int(xs.min()), int(xs.max())
    ink_w = x1 - x0 + 1

    # 列投影：统计每列的前景像素数，超过阈值的列视为「文字列」
    col_fg = (bw > 127).sum(axis=0)
    thr = max(2, bw.shape[0] * 0.02)
    is_fg = col_fg > thr

    # 统计列连通段数量（每个字符通常是一段）
    segs = 0
    in_seg = False
    for v in is_fg:
        if v and not in_seg:
            segs += 1
            in_seg = True
        elif not v:
            in_seg = False

    # 单段（连笔/无间隙）时无法可靠拆分，返回 0 触发回退
    if segs <= 1:
        return 0.0
    return float(ink_w) / segs


# ---------------------------------------------------------------- 清晰度
def estimate_sharpness(bgr_region: np.ndarray) -> float:
    """估算边缘锐度，返回 [0,1] 区间值，越大越锐利。

    使用灰度图拉普拉斯算子方差（Variance of Laplacian）。
    """
    gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())
    # 经验映射：var 约 50~2000 常见，用 sigmoid 归一化
    sharpness = 1.0 / (1.0 + math.exp(-(var - 400.0) / 300.0))
    return max(0.0, min(1.0, sharpness))


def estimate_blur_level(bgr_region: np.ndarray) -> float:
    """估算原文字的模糊程度，返回 [0,1]，越大越糊。

    用于全自动模式：决定新文字需要额外施加多少高斯模糊，
    使新文字的失焦程度贴近原图（而不是一刀切地锐利）。

    方法：拉普拉斯方差越低说明越糊；同时用梯度幅值的均值做二次校验，
    避免纯色区域（无文字）被误判为"模糊"。
    """
    gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())

    # 方差 → 模糊度：var 很大(>800) 视为清晰(0)，var 很小(<30) 视为很糊(1)
    blur = 1.0 - 1.0 / (1.0 + math.exp(-(var - 250.0) / 220.0))

    # 纯色区域（无文字）方差天然为 0，会被误判成"极糊"。
    # 用前景像素占比做抑制：几乎没有前景时，模糊度按"中等"处理。
    try:
        bw = _binarize(gray)
        fg_ratio = float((bw > 127).mean())
    except Exception:
        fg_ratio = 0.1
    if fg_ratio < 0.01:
        blur = 0.3
    return float(max(0.0, min(1.0, blur)))


def _binarize(gray: np.ndarray) -> np.ndarray:
    """统一的二值化：Otsu + 保证前景为白。"""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw > 127).mean() > 0.5:
        bw = 255 - bw
    return bw


def _measure_ink_ratio(bgr_region: np.ndarray) -> float:
    """计算区域二值化后的前景（墨迹）像素占比。

    用于区分「真文字」与「噪点/纯背景」：真文字区域墨迹通常占 5%~55%
    （汉字笔画实心但不铺满），而连片噪点可高达 0.5+（且几何上杂乱无章，
    无法被 em 三通道可信推定，故 em_reliable 已为 False）。见
    extract_neighbor_reference 的参照过滤逻辑。
    """
    try:
        if bgr_region is None or bgr_region.size == 0:
            return 0.0
        gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
        if gray.size == 0:
            return 0.0
        bw = _binarize(gray)
        total = float(bw.size)
        if total <= 0:
            return 0.0
        return float((bw > 127).sum()) / total
    except Exception:
        return 0.0


def _single_component_ratio(bgr_region: np.ndarray) -> float:
    """返回「最大连通域面积 / 总墨迹面积」。

    判别噪点连片与真文字的关键形态量：
      - 真文字：笔画断开成多个连通域，最大连通域只占墨迹一小部分（实测 ~0.2）。
      - 纯噪点 / 杂点铺满：OTSU 二值化后聚成 1 片大连通域，占墨迹 90%+（实测 ~0.95）。

    返回 1.0 表示「墨迹几乎就是一个连通大块」（几乎必然是噪点/色块而非文字）。
    """
    try:
        if bgr_region is None or bgr_region.size == 0:
            return 1.0
        gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
        if gray.size == 0:
            return 1.0
        bw = _binarize(gray)
        fg = (bw > 127).astype(np.uint8)
        if fg.sum() < 8:          # 墨迹太少：孤立杂点，判为非文字
            return 1.0
        n, _, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
        if n <= 1:
            return 1.0
        sizes = stats[1:, cv2.CC_STAT_AREA]
        return float(sizes.max()) / float(sizes.sum())
    except Exception:
        return 1.0


def estimate_antialias(bgr_region: np.ndarray) -> float:
    """估算边缘抗锯齿（过渡带柔和度），返回 [0,1]，越大过渡越柔和。

    原理：统计灰度介于「前景主灰度」与「背景主灰度」之间的中间调像素
    占全部边缘像素的比例。抗锯齿把边缘铺开成多级灰阶，
    中间调占比越高，说明边缘过渡越宽（即抗锯齿越强 / 越柔和）。

    返回 0 表示硬边（无抗锯齿，如 1bit 二值图），1 表示非常柔和。
    """
    gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bw = _binarize(gray.astype(np.uint8))

    fg_px = gray[bw > 127]
    bg_px = gray[bw <= 127]
    if fg_px.size < 10 or bg_px.size < 10:
        return 0.5

    fg_val = float(np.median(fg_px))
    bg_val = float(np.median(bg_px))
    span = abs(fg_val - bg_val)
    if span < 12:
        # 前景背景对比极低，无法可靠判断过渡带，给中性偏柔和值
        return 0.5

    lo, hi = min(fg_val, bg_val), max(fg_val, bg_val)
    # 只统计真正处于过渡带内（20%~80%）的像素
    band = (gray > lo + span * 0.2) & (gray < lo + span * 0.8) & (span > 0)

    # 边缘像素 = 前景经膨胀后减去腐蚀（即前景轮廓的环状区域）
    fg_mask = (bw > 127).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(fg_mask, k, iterations=1)
    ero = cv2.erode(fg_mask, k, iterations=1)
    ring = (dil - ero) > 0
    ring = np.logical_or(ring, band)  # 过渡带本身也算边缘

    ring_n = int(np.count_nonzero(ring))
    if ring_n == 0:
        return 0.5
    ratio = float(np.count_nonzero(band)) / ring_n
    # ratio 经验范围约 0.15(硬边) ~ 0.75(柔边)，线性映射到 0~1
    aa = (ratio - 0.15) / 0.60
    return float(max(0.0, min(1.0, aa)))


def estimate_noise_level(bgr_region: np.ndarray) -> float:
    """估算原图的噪点/压缩颗粒强度，返回 [0,1]，越大颗粒感越强。

    用于全自动模式：给新文字叠加等量的轻微噪点，
    让它带上和原图一致的截图压缩 / 传感器颗粒质感。

    方法：用 3x3 中值滤波的高频残差，取残差绝对值的**中位数**（MAD 思想）。
    文字边缘只占少数像素，中位数天然对边缘不敏感，
    因此它主要反映平坦区域（背景）的真实噪点水平。
    """
    gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if gray.size < 9:
        return 0.0
    med = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    resid = np.abs(gray - med)
    mad = float(np.median(resid))
    # 经验：干净图 mad ≈ 0~0.5；轻微压缩 ≈ 1~2；强压缩/颗粒 ≈ 3~6
    level = (mad - 0.4) / 4.0
    return float(max(0.0, min(1.0, level)))


def estimate_line_spacing(bgr_region: np.ndarray) -> int:
    """估算多行文字的行间距（相邻两行基线间距，像素）。

    单行或无法判断时返回 0（调用方按字号 * 1.2 兜底）。
    """
    gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
    bw = _binarize(gray)
    row_sums = (bw > 127).sum(axis=1)
    thr = max(2, bw.shape[1] * 0.02)
    fg_rows = row_sums > thr

    # 找出每一行的连续区间
    lines = []
    start = None
    for i, v in enumerate(fg_rows):
        if v and start is None:
            start = i
        elif not v and start is not None:
            lines.append((start, i))
            start = None
    if start is not None:
        lines.append((start, len(fg_rows)))

    if len(lines) < 2:
        return 0

    # 行间距 = 相邻行起始位置之差的中位数
    gaps = [lines[i + 1][0] - lines[i][0] for i in range(len(lines) - 1)]
    gaps = [g for g in gaps if g > 2]
    if not gaps:
        return 0
    return int(np.median(gaps))


# ---------------------------------------------------------------- 粗细
def measure_stroke_px(bgr_region: np.ndarray) -> float:
    """估算原图文字的「绝对」笔画厚度（像素，整条笔画全宽）。

    方法：对文字前景做距离变换，取前景像素距离变换的高分位（95%）作为
    笔画最厚处的「半宽」，×2 即得整条笔画宽度。

    抗锯齿/模糊边缘会让距离≈1 的浅色像素稀释高分位，因此用 95% 而非最大值，
    得到的厚度更接近真实「中心实心笔画」宽度，不会被描边糊边虚高。

    返回 0.0 表示无法测量（空区域 / 无前景 / 退化为极小值）。
    """
    if bgr_region is None or bgr_region.size == 0:
        return 0.0
    if bgr_region.shape[0] < 3 or bgr_region.shape[1] < 3:
        return 0.0
    gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw > 127).mean() > 0.5:
        bw = 255 - bw
    # 几乎没有前景（纯背景框）无法测量
    if (bw > 127).mean() < 0.005:
        return 0.0

    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    fg = dist[bw > 127]
    if fg.size == 0:
        return 0.0
    # 半笔画宽度（距离变换在笔画中心达到最大）
    half_width = float(np.percentile(fg, 95))
    stroke_px = half_width * 2.0
    return float(max(0.0, stroke_px))


# ---------------------------------------------------------------- 骨架细化
def _zhang_suen_thinning(bw: np.ndarray) -> np.ndarray:
    """Zhang-Suen 二值骨架细化，返回与 bw 同尺寸、前景为 1 的骨架图。

    用于从文字笔画骨架反推笔画厚度：骨架长度 ≈ 笔画中心线总长，
    前景像素总面积 ÷ 骨架中心线长度 ≈ 平均笔画宽度（即真实厚度）。
    向量化实现，对常规文字区域（数千像素）可秒级完成。
    """
    img = (bw > 127).astype(np.uint8)
    if img.sum() == 0:
        return img
    img = np.pad(img, 1)
    changed = True
    guard = 0
    while changed and guard < 200:
        guard += 1
        changed = False
        for step in (0, 1):
            p1 = img[1:-1, 1:-1]
            p2 = img[:-2, 1:-1]
            p3 = img[:-2, 2:]
            p4 = img[1:-1, 2:]
            p5 = img[2:, 2:]
            p6 = img[2:, 1:-1]
            p7 = img[2:, :-2]
            p8 = img[1:-1, :-2]
            p9 = img[:-2, :-2]
            # 四邻接前景数（cond1 用）
            S4 = p2 + p4 + p6 + p8
            # 0->1 跳变次数（环形），等价于「连通数 S(p1) == 1」
            vals = np.stack([p2, p3, p4, p5, p6, p7, p8, p9, p2], axis=0)
            transitions = ((vals[:-1] == 0) & (vals[1:] == 1)).sum(axis=0)
            if step == 0:
                cond1 = (S4 >= 2) & (S4 <= 6)
                cond3 = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond1 = (S4 >= 2) & (S4 <= 6)
                cond3 = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            cond2 = (transitions == 1)
            m = (p1 == 1) & cond1 & cond2 & cond3
            if m.any():
                img[1:-1, 1:-1][m] = 0
                changed = True
    return img[1:-1, 1:-1]


def measure_stroke_px_skeleton(bgr_region: np.ndarray) -> float:
    """使用「骨架细化」算法计算原图文字的笔画厚度（像素）。

    用户硬性要求：用骨架细化算法计算笔画厚度，自动匹配对应字重，
    保证新文字粗细和原图一致。

    步骤：
    1) 二值化文字区域得到前景；
    2) Zhang-Suen 骨架细化得到单像素宽的笔画中心线；
    3) 笔画厚度 = 前景像素总面积 ÷ 骨架中心线长度（平均笔画宽度）。

    该口径与 measure_stroke_px（距离变换）互相独立、互相校验，
    字体匹配阶段两者加权融合，挑选天然字重最接近的字体。

    骨架过短（<3 像素，如单点或噪点）时回退到距离变换口径，避免异常值。
    """
    if bgr_region is None or bgr_region.size == 0:
        return 0.0
    if bgr_region.shape[0] < 3 or bgr_region.shape[1] < 3:
        return 0.0
    gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
    bw = _binarize(gray)
    if (bw > 127).mean() < 0.005:
        return 0.0
    skel = _zhang_suen_thinning(bw)
    skel_px = int(skel.sum())
    fg_px = int((bw > 127).sum())
    if skel_px < 3 or fg_px < skel_px:
        # 退化：骨架不可靠，回退到距离变换口径
        return measure_stroke_px(bgr_region)
    thickness = fg_px / skel_px
    return float(max(0.5, min(thickness, 40.0)))


def measure_stroke_normalized(bgr_region: np.ndarray, em_size: int = 0) -> float:
    """归一化笔画厚度 = 骨架厚度 / em。跨字号、跨字重可直接比较的不变量。

    为什么弃用旧的绝对像素厚度（stroke_px）作字重匹配依据：
      1) 绝对厚度与字号强耦合：字号一变数值就变，不是字重的不变量；
      2) 距离变换 95% 分位口径分辨率极差 —— diag_font_match 实测：字体池在
         48px 下的读数只落在 3 个值上（4.00 / 4.39 / 5.60），大量字体「撞车」，
         根本分不出字重。这是「笔画粗细对不上」的底层原因之一；
      3) 骨架口径（前景面积 ÷ 骨架长度）是连续量，分辨率显著更高
         （同样条件下可区分 4/4 款，且跨度更大）。

    归一化后：同一款字体在 24px 与 96px 下测得的值应当一致，
    因此「原图厚度」与「候选字体厚度」可放心直接相减比较。
    """
    if bgr_region is None or bgr_region.size == 0:
        return 0.0
    if em_size <= 0:
        em_size, _ = estimate_em_size(bgr_region)
    if em_size <= 0:
        return 0.0
    t = measure_stroke_px_skeleton(bgr_region)
    if t <= 0:
        return 0.0
    return float(t / em_size)


# ---------------------------------------------------------------- 基线
def estimate_baseline_y(bgr_region: np.ndarray, em_size: int = 0) -> int | None:
    """推定选区内文字的基线 y（区域相对坐标，像素）。返回 None 表示无法推定。

    基线是排版对齐的唯一正确锚点。旧实现用「墨迹包围盒中心」对齐，而墨迹中心
    到基线的距离随字符剧烈变化 —— diag_render_core 实测：em=64 时该偏移跨度达
    **23px**（「一」24px vs「。」1px）。所以只要替换文字与原文字符构成不同，
    中心对齐就必然导致文字忽高忽低。这是长期「治标不治本」的核心缺陷。

    难点：墨迹底边分布里同时混着两类干扰字符，且方向相反——
      · 下伸字符（g/y/p/q/，；）底边偏**大**
      · 扁平字符（「一」「丶」）底边偏**小**
    只靠分位数无法同时避开两者（实测：中位数被下伸带偏 9px，
    下四分位又被「一」带偏 14px）。

    算法（先筛族、再取分位）：
      1) 按行拆分，每行内按列投影拆出单个字符，记录每个字符的「高度」与「底边」；
      2) **筛出满高字符族**：保留高度 ≥ 0.85 × 最大高度的字符。
         汉字与英文大写属于满高族，扁平字「一」首先被剔除；
      3) 在满高族的底边中取 **25 分位**（下四分位）：此时族内剩下的干扰
         主要来自下伸字符，下四分位可稳定落在正常底边上
         （实测「中国gpy」由中位数误差 9px 降至 2px）；
      4) baseline = 下四分位墨迹底边 − K_INK_BELOW × em
         （汉字字面下沿通常略低于基线，diag_metrics 实测该偏移 = 0.098em）

    注意：本函数依赖 em 的正确性。「一」这类扁平字符会让 em 推定崩溃，
    连带基线出错——该场景由调用方用「四向邻近文字」的 em/基线兜底解决，
    而不是在此处硬凑。

    多行时返回**第一行**的基线（后续行由行距递推即可）。
    """
    if bgr_region is None or bgr_region.size == 0:
        return None
    gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < 4 or gray.shape[1] < 4:
        return None
    bw = _binarize(gray)
    lines = _segments((bw > 127).any(axis=1))
    if not lines:
        return None
    if em_size <= 0:
        em_size, _ = estimate_em_size(bgr_region)

    per_line_bottom = []
    for (s, e) in lines:
        band = bw[s:e + 1, :]
        # (字符高度, 字符墨迹底边)
        pairs = []
        for (cs, ce) in _segments((band > 127).any(axis=0)):
            sub = band[:, cs:ce + 1]
            rr = np.where((sub > 127).any(axis=1))[0]
            if rr.size:
                h_ch = int(rr[-1] - rr[0] + 1)
                pairs.append((h_ch, float(s + int(rr[-1]))))
        if not pairs:
            continue
        # ① 筛满高字符族：剔除「一」「丶」这类扁平字符
        max_h = max(h for h, _ in pairs)
        tall = [b for (h, b) in pairs if h >= 0.85 * max_h]
        if not tall:
            tall = [b for _, b in pairs]
        # ② 下四分位：剔除 g/y/p/q 等下伸字符
        per_line_bottom.append(float(np.percentile(tall, 25)))
    if not per_line_bottom:
        return None

    # 以第一行（最靠上的一行）为锚
    base_bottom = per_line_bottom[0]
    baseline = base_bottom - K_INK_BELOW * em_size
    return int(round(baseline))


def estimate_stroke_weight(bgr_region: np.ndarray) -> float:
    """估算笔画粗细（相对值 0~1）。

    用 measure_stroke_px 得到的绝对厚度，再按「文字实际行高」归一化，
    因为框选区域上下留白不应让归一化值偏低。
    常规字体笔画宽度约为字高的 9%~12%：
      norm ≈ 0.08 视为细体，0.11 视为常规，>0.16 视为粗体。
    """
    stroke_px = measure_stroke_px(bgr_region)
    if stroke_px <= 0:
        return 0.5

    gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw > 127).mean() > 0.5:
        bw = 255 - bw

    # 用文字实际行高（前景笔画竖直跨度）而非区域高度做归一化
    row_sums = (bw > 127).sum(axis=1)
    fg_rows = np.where(row_sums > max(2, bw.shape[1] * 0.02))[0]
    text_h = (fg_rows.max() - fg_rows.min() + 1) if fg_rows.size else bw.shape[0]
    text_h = max(1, text_h)

    norm = stroke_px / text_h
    # 线性映射：norm 0.06→0.25, 0.11→0.55, 0.18→0.95
    weight = (norm - 0.06) / 0.12 * 0.7 + 0.25
    return float(max(0.05, min(0.95, weight)))


# ---------------------------------------------------------------- 倾斜角度
def _pca_skew_angle(bw: np.ndarray) -> float | None:
    """用前景像素协方差的主轴方向估计倾斜（度），单行/少字符文字效果好。"""
    ys, xs = np.where(bw > 127)
    if xs.size < 12:
        return None
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    pts = pts - pts.mean(axis=0)
    cov = pts.T @ pts / (pts.shape[0] - 1)
    try:
        w, v = np.linalg.eigh(cov)
    except Exception:
        return None
    principal = v[:, int(np.argmax(w))]
    ang = math.degrees(math.atan2(principal[1], principal[0]))
    if ang > 90:
        ang -= 180
    if ang < -90:
        ang += 180
    return float(ang)


def estimate_skew_angle(bgr_region: np.ndarray) -> float:
    """估算文字倾斜角度（度）。

    综合两种方法：
    - minAreaRect：对多字符整体框效果好；
    - PCA 主轴：对单/少字符、minAreaRect 退化为 0 的情况更敏感。
    取两者中更可信（minAreaRect 退化≈0 而 PCA 检出明显倾斜）的结果，
    最终限制在 ±15°，超出视为不可靠返回 0。
    """
    gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw > 127).mean() > 0.5:
        bw = 255 - bw

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mrect = 0.0
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) >= 10:
            rect = cv2.minAreaRect(biggest)
            a = rect[2]
            if a < -45:
                a = 90 + a
            mrect = float(a)

    pca = _pca_skew_angle(bw)
    # 优先用 minAreaRect 的可靠结果；若其退化（≈0）而 PCA 检出明显倾斜，则用 PCA
    if abs(mrect) < 1.0 and pca is not None and abs(pca) > 0.5:
        result = pca
    else:
        result = mrect
    if abs(result) > 15:
        return 0.0
    return float(result)


# ---------------------------------------------------------------- 行数
def estimate_line_count(bgr_region: np.ndarray) -> int:
    """估算行数。"""
    gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw > 127).mean() > 0.5:
        bw = 255 - bw
    h, w = bw.shape
    row_sums = (bw > 127).sum(axis=1)
    fg_rows = row_sums > max(2, w * 0.02)
    count = 0
    in_line = False
    for v in fg_rows:
        if v and not in_line:
            count += 1
            in_line = True
        elif not v:
            in_line = False
    return max(1, count)


# ---------------------------------------------------------------- 文字包围盒
def foreground_bbox(bgr_region: np.ndarray):
    """返回文字前景的包围盒 (x0, y0, x1, y1)，相对区域坐标；无前景返回 None。

    用于新文字的基线对齐：把新文字的笔画竖直范围对齐到原文字的竖直范围。
    """
    gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw > 127).mean() > 0.5:
        bw = 255 - bw
    ys, xs = np.where(bw > 127)
    if ys.size == 0 or xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


# ---------------------------------------------------------------- 上下行参照样本
def extract_neighbor_reference(bgr_img: np.ndarray,
                              box: Tuple[int, int, int, int],
                              font_size_hint: int = 16) -> Dict | None:
    """提取选区「上下左右紧邻文字」的排版 / 质感参照特征（四向上下文）。

    用途（用户硬性流程第二步）：替换文字时，用选区四周紧邻的文字作为参照样本，
    对齐『排版间距、字号、笔画粗细』以及颜色/倾斜/模糊/噪点质感。

    为什么用四周做参照，而不只用选区自身：
    选区文字在擦除/替换后会消失，而同一段落的上下行、同一行左右相邻文字是
    稳定参照，其字号、字间距、笔画粗细、颜色、倾斜往往与选区文字同属一套排版风格，
    用它来校准能保证替换后的新文字与上下文对齐，不突兀。

    提取策略：
    1) 以选区实测字号估计单行行高 line_h（hint 的 1.6 倍）与单字宽 char_w（hint 的 1.2 倍）。
    2) 四向窗口：
       - 上：[y0-line_h, y0-2] × [x0-pad, x1+pad]
       - 下：[y1+2, y1+line_h] × [x0-pad, x1+pad]
       - 左：[y0-pad, y1+pad] × [x0-char_w, x0-2]
       - 右：[y0-pad, y1+pad] × [x1+2, x1+char_w]
    3) 四个窗口各自 extract_features；只保留「确实含文字」（text_bbox 非 None）的参照。
    4) 多参照取中值融合：字号/字间距/笔画粗细/颜色/倾斜/模糊/噪声。
    5) 同时回传 neighbor_boxes：各命中窗口中文字的**图像坐标矩形**，供第四步
       「渲染防重叠」使用（新文字不得与这些原始邻近文字重叠）。
    6) 四向均无有效文字时返回 None（调用方应回退到选区自身特征）。

    全程异常兜底：任何一步出错返回 None，绝不让识别/校准流程崩掉。
    """
    if bgr_img is None or bgr_img.size == 0:
        return None
    try:
        x0, y0, x1, y1 = [int(v) for v in box]
        h, w = bgr_img.shape[:2]
        line_h = max(4, int(round(font_size_hint * 1.6)))
        char_w = max(4, int(round(font_size_hint * 1.2)))
        pad = max(2, int(line_h * 0.3))
        x0e = max(0, x0 - pad)
        x1e = min(w, x1 + pad)
        y0e = max(0, y0 - pad)
        y1e = min(h, y1 + pad)

        # (窗口图像坐标, 该窗口在整图中的偏移)
        windows = []

        # 上
        if y0 - 2 > 0:
            uy0 = max(0, y0 - line_h); uy1 = max(0, y0 - 2)
            if uy1 > uy0:
                windows.append(((x0e, uy0, x1e, uy1), (x0e, uy0)))
        # 下
        if y1 + 2 < h:
            dy0 = min(h, y1 + 2); dy1 = min(h, y1 + line_h)
            if dy1 > dy0:
                windows.append(((x0e, dy0, x1e, dy1), (x0e, dy0)))
        # 左
        if x0 - 2 > 0:
            lx0 = max(0, x0 - char_w); lx1 = max(0, x0 - 2)
            if lx1 > lx0:
                windows.append(((lx0, y0e, lx1, y1e), (lx0, y0e)))
        # 右
        if x1 + 2 < w:
            rx0 = min(w, x1 + 2); rx1 = min(w, x1 + char_w)
            if rx1 > rx0:
                windows.append(((rx0, y0e, rx1, y1e), (rx0, y0e)))

        refs = []
        neighbor_boxes = []
        for (wx0, wy0, wx1, wy1), (ox, oy) in windows:
            region = bgr_img[wy0:wy1, wx0:wx1]
            if region is None or region.size == 0:
                continue
            try:
                f = extract_features(region)
            except Exception:
                continue
            tb = f.get("text_bbox")
            if tb is None:
                continue
            # ---- 关键修复（治本）：过滤「噪点/纯背景被误判成文字」的假参照 ----
            # 旧实现只判 text_bbox 非 None 就采信，导致噪点连片的窗口（纯背景 + 高斯
            # 噪点）被当成文字参照，字号被荒谬拉大（实测纯噪点窗口 font_size 高达
            # 154，四向融合后主区 43px 被平均到 84px——这正是「生成字号忽大忽小」的
            # 真正底层病灶之一）。
            #
            # 判据设计（用实测分离度而非经验堆叠）：
            #   噪点区 OTSU 二值化后墨迹铺满近半窗口（高斯 σ=2 实测 ink_ratio≈0.49），
            #   而真文字笔画稀疏（参照窗口内单字实测 ink_ratio≤0.15，整段也 <0.4）。
            #   → ink_ratio 是区分「噪点铺满」与「真文字」的最可靠判据。
            #   辅以 fs 合理性剔除孤立杂点（墨迹太少）与伪大号噪点（fs 远超窗口高）。
            ink_ratio = f.get("ink_ratio")
            if ink_ratio is None:
                ink_ratio = _measure_ink_ratio(region)
            if not (MIN_INK_RATIO <= ink_ratio <= MAX_INK_RATIO):
                continue
            fs = f.get("font_size", 0) or 0
            if fs < MIN_NEIGHBOR_EM:
                continue
            win_h = max(1, wy1 - wy0)
            if fs > MAX_FS_RATIO_OF_WIN_H * win_h:
                continue
            # 通过判据才算「有效文字参照」：转成整图坐标，同时记录其 ink 占比
            f = dict(f)
            f["_ink_ratio"] = ink_ratio
            neighbor_boxes.append((ox + tb[0], oy + tb[1], ox + tb[2], oy + tb[3]))
            refs.append(f)

        if not refs:
            return None

        keys = ["font_size", "letter_spacing", "stroke_px", "stroke_px_skeleton",
                "stroke_weight", "skew_angle", "blur_level", "noise_level", "sharpness"]
        merged = {}
        for k in keys:
            vals = [r[k] for r in refs if k in r]
            merged[k] = float(np.median(vals)) if vals else 0.0
        # 颜色：取第一个有效参照（同段落通常一致）
        merged["color_rgb"] = refs[0].get("color_rgb", (0, 0, 0))
        merged["_ref_count"] = len(refs)
        merged["neighbor_boxes"] = neighbor_boxes
        merged["has_up"] = any(b[3] < y0 for b in neighbor_boxes)
        merged["has_down"] = any(b[1] > y1 for b in neighbor_boxes)
        merged["has_left"] = any(b[2] < x0 for b in neighbor_boxes)
        merged["has_right"] = any(b[0] > x1 for b in neighbor_boxes)
        return merged
    except Exception:
        return None


# ---------------------------------------------------------------- 汇总
def extract_features(bgr_region: np.ndarray) -> Dict:
    """汇总提取所有特征。bgr_region 为框选区域（BGR）。"""
    empty = {
        "color_rgb": (0, 0, 0),
        "font_size": 16,
        "letter_spacing": 0,
        "line_spacing": 0,
        "sharpness": 0.5,
        "blur_level": 0.3,
        "antialias": 0.5,
        "noise_level": 0.0,
        "stroke_weight": 0.5,
        "stroke_px": 0.0,
        "stroke_px_skeleton": 0.0,
        "skew_angle": 0.0,
        "line_count": 1,
        "text_bbox": None,
        "em_reliable": False,     # 字号是否可信赖（扁平字符/纯标点时为 False）
        "baseline_y": None,       # 基线 y（区域相对坐标）
        "stroke_normalized": 0.0,  # 归一化笔画厚度（骨架厚度/em，跨字号不变量）
        "char_aspect": 0.0,       # 单字墨迹宽高比（与字数无关，供字体匹配）
    }
    if bgr_region is None or bgr_region.size == 0:
        return empty

    # 极小区域（如误操作框出 1x1）无法做可靠统计，直接返回安全默认值，
    # 避免 k-means / 距离变换在退化输入上抛异常。
    if bgr_region.shape[0] < 3 or bgr_region.shape[1] < 3:
        return empty

    try:
        # em 字号与其可靠性一并推定：可靠性决定是否要用邻近文字兜底
        em_size, em_reliable = estimate_em_size(bgr_region)
        features = {
            "color_rgb": extract_text_color(bgr_region),         # (r,g,b)
            # font_size 统一为 em 字号（不再用墨迹高），测量/匹配/渲染三处同口径
            "font_size": em_size,
            "letter_spacing": estimate_letter_spacing(bgr_region),  # int 像素
            "line_spacing": estimate_line_spacing(bgr_region),    # int 像素，单行时为 0
            "sharpness": estimate_sharpness(bgr_region),          # 0~1
            "blur_level": estimate_blur_level(bgr_region),        # 0~1
            "antialias": estimate_antialias(bgr_region),          # 0~1
            "noise_level": estimate_noise_level(bgr_region),      # 0~1
            "stroke_weight": estimate_stroke_weight(bgr_region),  # 0~1
            "stroke_px": measure_stroke_px(bgr_region),           # 绝对厚度(px) 距离变换口径
            "stroke_px_skeleton": measure_stroke_px_skeleton(bgr_region),  # 绝对厚度(px) 骨架细化口径
            "stroke_normalized": measure_stroke_normalized(bgr_region, em_size),  # 厚度/em
            "skew_angle": estimate_skew_angle(bgr_region),        # 度
            "line_count": estimate_line_count(bgr_region),        # int
            "text_bbox": foreground_bbox(bgr_region),             # (x0,y0,x1,y1) 或 None
            "em_reliable": em_reliable,
            # 基线必须在 em 已知后推定（baseline = 下四分位墨迹底边 − K_INK_BELOW × em）
            "baseline_y": estimate_baseline_y(bgr_region, em_size),
            "char_aspect": estimate_char_aspect(bgr_region),
        }
    except Exception:
        # 任何一路特征提取失败都不能让整个框选流程崩掉，
        # 退回安全默认值交由用户手动微调。
        return empty
    return features
