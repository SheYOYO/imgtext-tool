"""em 字号推定算法验证（先在诊断脚本里验证算法，通过后再落地到 app）。

难点：整行只有扁平字符（「一」「丶」「-」）时，
      行高通道与字高通道会同时崩溃（都只有 5px）。
方案：引入「字宽通道」——汉字字面墨迹宽 ≈ K_INK_WIDTH × em，
      取 em = max(由高推, 由宽推)，扁平字符靠宽度通道救回。

本脚本：
  1) 实测 K_INK_WIDTH
  2) 用候选算法在【含扁平字符、粘连、中英混排】的困难样本上验证精度
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import config                 # noqa: E402
from app import feature_extract as fe  # noqa: E402

LINE = "=" * 78
K_INK_HEIGHT = 0.900
K_LINE = 0.930


def _binarize(gray):
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw > 127).mean() > 0.5:
        bw = 255 - bw
    return bw


def _segments(mask):
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


# ---------------------------------------------------------- 1 实测 K_INK_WIDTH
def measure_k_ink_width():
    em = 100
    sample = "国中文字人丁口大王小囗目日田"
    print(LINE)
    print("1. 实测 K_INK_WIDTH（汉字字面墨迹宽 / em）")
    print(LINE)
    vals = []
    for f in [x for x in config.resolve_font_pool() if x.get("cn")]:
        try:
            font = ImageFont.truetype(f["path"], em)
        except Exception:
            continue
        ws = []
        for ch in sample:
            img = Image.new("RGBA", (em * 3, em * 3), (0, 0, 0, 0))
            ImageDraw.Draw(img).text((em, em), ch, font=font, fill=(0, 0, 0, 255))
            a = np.array(img)[:, :, 3]
            xs = np.where((a > 40).any(axis=0))[0]
            if xs.size:
                ws.append(xs.max() - xs.min() + 1)
        k = np.median(ws) / em
        vals.append(k)
        print(f"  {os.path.basename(f['path']):<22} 字面墨迹宽/em = {k:.3f}")
    kk = float(np.median(vals))
    print(f"\n  >> K_INK_WIDTH = {kk:.3f}（建议取值）")
    return kk


# ---------------------------------------------------------- 2 候选算法
def estimate_em_candidate(bgr_region, k_width):
    """候选 em 推定算法（待验证）。返回 (em, reliable)。"""
    if bgr_region is None or bgr_region.size == 0:
        return 16, False
    gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 4 or w < 4:
        return max(6, int(h * 0.9)), False
    bw = _binarize(gray)

    rows = _segments((bw > 127).any(axis=1))
    if not rows:
        return max(6, int(h * 0.55)), False

    line_heights = [e - s + 1 for (s, e) in rows]
    char_hs, char_ws = [], []
    for (s, e) in rows:
        band = bw[s:e + 1, :]
        for (cs, ce) in _segments((band > 127).any(axis=0)):
            sub = band[:, cs:ce + 1]
            rr = np.where((sub > 127).any(axis=1))[0]
            cc = np.where((sub > 127).any(axis=0))[0]
            if rr.size:
                char_hs.append(int(rr[-1] - rr[0] + 1))
            if cc.size:
                char_ws.append(int(cc[-1] - cc[0] + 1))
    if not char_hs:
        return max(6, int(np.median(line_heights) / K_LINE)), False

    line_h = float(np.median(line_heights))
    char_h = float(np.median(char_hs))
    char_w = float(np.median(char_ws)) if char_ws else char_h

    em_from_line = line_h / K_LINE
    em_from_h = char_h / K_INK_HEIGHT
    em_from_w = char_w / k_width
    # 扁平字符（「一」「丶」）高度通道崩溃，由宽度通道救回
    em_from_char = max(em_from_h, em_from_w)
    em = float(np.median([em_from_line, em_from_char]))
    return max(6, int(round(em))), True


# ---------------------------------------------------------- 3 困难样本验证
def verify(k_width):
    print()
    print(LINE)
    print("2. em 推定算法验证（目标：估算 em / 真实 em ≈ 1.00）")
    print(LINE)
    fonts = [x for x in config.resolve_font_pool() if x.get("cn")]
    font_path = fonts[0]["path"]

    cases = [
        ("中国文字", 64, "常规汉字"),
        ("一", 64, "★单字扁平（旧算法崩溃点）"),
        ("一一一一", 64, "★全扁平字符"),
        ("中国文字替换测试内容", 64, "长串汉字"),
        ("ABCabc123", 64, "纯英文数字"),
        ("中国ABC文字abc", 64, "中英混排"),
        ("中 国 文 字", 64, "带空格"),
        ("中国文字\n第二行文字", 64, "多行"),
        ("中国文字", 24, "小字号"),
        ("中国文字", 96, "大字号"),
        ("。，、", 64, "★纯标点"),
        ("中国文字一二三", 64, "混合含扁平字"),
    ]
    print(f"  {'样本':<24}{'真实em':>8}{'估算em':>8}{'偏差':>9}  说明")
    print("  " + "-" * 70)
    bad = 0
    total = 0
    for text, em_true, note in cases:
        font = ImageFont.truetype(font_path, em_true)
        # 渲染到白底
        tmp = Image.new("RGB", (1, 1), (255, 255, 255))
        d0 = ImageDraw.Draw(tmp)
        bb = d0.multiline_textbbox((0, 0), text, font=font)
        img = Image.new("RGB", (bb[2] - bb[0] + 16, bb[3] - bb[1] + 16), (255, 255, 255))
        ImageDraw.Draw(img).multiline_text((8 - bb[0], 8 - bb[1]), text,
                                           font=font, fill=(0, 0, 0))
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        est, ok = estimate_em_candidate(bgr, k_width)
        err = est / em_true
        total += 1
        flag = ""
        if abs(err - 1.0) > 0.15:
            flag = "  ❌"
            bad += 1
        print(f"  {text!r:<24}{em_true:>8}{est:>8}{err:>8.2f}x {flag} {note}")

    print("  " + "-" * 70)
    rate = (total - bad) / total
    print(f"\n  >> 合格率：{total - bad}/{total} = {rate * 100:.0f}%"
          f"（偏差在 ±15% 内为合格）")
    print(f"  >> 判定：{'❌ 算法不合格' if rate < 0.8 else '✅ 算法可用'}")
    return rate


def main():
    k_width = measure_k_ink_width()
    rate = verify(k_width)
    print()
    print(LINE)
    print(f"结论：K_INK_WIDTH={k_width:.3f}，em 推定合格率 {rate * 100:.0f}%")
    print(LINE)


if __name__ == "__main__":
    main()
