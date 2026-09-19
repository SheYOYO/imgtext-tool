"""字体度量系数实测：为底层重构提供真实参数（不靠经验猜测）。

需要测定的系数（用于 em 字号推定 + 基线推定）：
  k_ink    = 汉字字面墨迹高 / em          （由墨迹高反推 em）
  k_below  = (墨迹底边 - baseline) / em   （由墨迹底边反推 baseline）
  k_adv    = 汉字平均 advance / em        （字宽基准）
  k_line   = 单行占据高度 / em            （由行高反推 em）
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import config  # noqa: E402

LINE = "=" * 78
EM = 100  # 用 100 做基准，系数直接是百分比


def _ink_rows_of_char(font, ch, em):
    """渲染单个字符，返回 (ink_top, ink_bottom) 相对 baseline 的偏移（正=基线下方）。"""
    img = Image.new("RGBA", (em * 3, em * 3), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y_top = em
    d.text((em, y_top), ch, font=font, fill=(0, 0, 0, 255))
    a = np.array(img)[:, :, 3]
    ys = np.where((a > 40).any(axis=1))[0]
    if ys.size == 0:
        return None
    ascent, _ = font.getmetrics()
    baseline = y_top + ascent
    return int(ys.min()) - baseline, int(ys.max()) - baseline


def main():
    pool = config.resolve_font_pool()
    cn = [f for f in pool if f.get("cn")] or pool

    # 典型汉字样本：覆盖上下结构、左右结构、全包围
    sample = "国中文字人丁口大王小"
    print(f"基准 em = {EM}，样本汉字：{sample}")
    print(LINE)
    print(f"  {'字体':<22}{'k_ink':>9}{'k_below':>9}{'k_adv':>9}{'k_line':>9}"
          f"{'ascent/em':>11}{'descent/em':>12}")
    print(LINE)

    k_inks, k_belows, k_advs, k_lines = [], [], [], []
    for f in cn:
        try:
            font = ImageFont.truetype(f["path"], EM)
        except Exception:
            continue
        ascent, descent = font.getmetrics()

        # k_ink：单字墨迹高 / em
        hs = []
        for ch in sample:
            r = _ink_rows_of_char(font, ch, EM)
            if r:
                hs.append(r[1] - r[0] + 1)
        k_ink = (np.median(hs) / EM) if hs else 0.0

        # k_below：(墨迹底边 - baseline)/em，取中位数（排除下伸）
        belows = []
        for ch in sample:
            r = _ink_rows_of_char(font, ch, EM)
            if r:
                belows.append(r[1])
        k_below = (np.median(belows) / EM) if belows else 0.0

        # k_adv：平均 advance / em
        advs = [font.getlength(ch) for ch in sample]
        k_adv = (float(np.mean(advs)) / EM) if advs else 0.0

        # k_line：单行渲染占据高度 / em（用整串的墨迹高近似行占据高）
        img = Image.new("RGBA", (EM * 20, EM * 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((EM, EM), sample, font=font, fill=(0, 0, 0, 255), anchor="ls")
        a = np.array(img)[:, :, 3]
        ys = np.where((a > 40).any(axis=1))[0]
        k_line = ((ys.max() - ys.min() + 1) / EM) if ys.size else 0.0

        k_inks.append(k_ink)
        k_belows.append(k_below)
        k_advs.append(k_adv)
        k_lines.append(k_line)

        print(f"  {os.path.basename(f['path']):<22}{k_ink:>9.3f}{k_below:>9.3f}"
              f"{k_adv:>9.3f}{k_line:>9.3f}{ascent / EM:>11.3f}{descent / EM:>12.3f}")

    print(LINE)
    print(f"  {'中位数':<22}{np.median(k_inks):>9.3f}{np.median(k_belows):>9.3f}"
          f"{np.median(k_advs):>9.3f}{np.median(k_lines):>9.3f}")
    print(f"  {'最小值':<22}{min(k_inks):>9.3f}{min(k_belows):>9.3f}"
          f"{min(k_advs):>9.3f}{min(k_lines):>9.3f}")
    print(f"  {'最大值':<22}{max(k_inks):>9.3f}{max(k_belows):>9.3f}"
          f"{max(k_advs):>9.3f}{max(k_lines):>9.3f}")
    print(LINE)
    print("\n重构应采用的常数（取中位数）：")
    print(f"  K_INK_HEIGHT = {np.median(k_inks):.3f}   # 字面墨迹高 / em")
    print(f"  K_INK_BELOW  = {np.median(k_belows):.3f}   # (墨迹底边 - baseline) / em")
    print(f"  K_ADVANCE    = {np.median(k_advs):.3f}   # 汉字 advance / em")
    print(f"  K_LINE       = {np.median(k_lines):.3f}   # 单行占据高 / em")


if __name__ == "__main__":
    main()
