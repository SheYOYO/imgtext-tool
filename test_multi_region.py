"""多选区独立编辑数据流自测。

验证「多区域编辑互不覆盖」的数据流：以 working_bgr 为累积底图，
先应用框1、再应用框2（基于含框1结果的底图），框1 的成果必须保留。

说明：GUI 层的「切框前固化草稿 / on_leave_box」负责把未应用的预览固化为
可累积的 working_bgr，本测试验证其底层数据流（erase+render+composite 的
累积语义）正确——只要数据流正确，配合 on_leave_box 即可保证「上一处修改
不自动消失、多个区域独立保存」。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from app import config
from app import render as render_mod
from app import erase as erase_mod


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_output")
os.makedirs(OUT, exist_ok=True)


def _cn_fonts():
    pool = config.resolve_font_pool()
    return [f for f in pool if f.get("cn")]


def _center(box, rgba):
    x0, y0, x1, y1 = box
    th, tw = rgba.shape[:2]
    return (x0 + max(0, ((x1 - x0) - tw) // 2),
            y0 + max(0, ((y1 - y0) - th) // 2))


def _ink_presence(bgr, box, color=(20, 20, 20), tol=60):
    """统计 box 区域内接近指定文字色的像素占比（用于判断该区域是否仍有文字）。"""
    x0, y0, x1, y1 = box
    sub = bgr[y0:y1, x0:x1]
    if sub.size == 0:
        return 0.0
    diff = np.abs(sub.astype(int) - np.array(color)).sum(axis=2)
    return float((diff < tol).mean())


def test_multi_region_independent():
    fonts = _cn_fonts()
    if not fonts:
        print("  [SKIP] 无可用中文字体，跳过")
        return
    fp = fonts[0]["path"]

    H, W = 300, 600
    base = np.full((H, W, 3), 245, np.uint8)

    box1 = (40, 60, 220, 160)
    box2 = (360, 60, 540, 160)

    def _params():
        return {
            "color_rgb": (20, 20, 20), "font_size": 48,
            "letter_spacing": 0, "line_spacing": 0, "skew_angle": 0.0,
            "sharpness": 0.7, "blur": 0.0, "edge_soften": 0.0, "noise": 0.0,
            "bold": False, "target_stroke_px": 4.0, "opacity": 1.0,
        }

    # 框1 应用：擦除 + 渲染「苹果」
    erased1 = erase_mod.erase_text(base, box1, strength=0.5)
    rgba1 = render_mod.render_text("苹果", _params(), font_path=fp)
    img1 = render_mod.composite_text(erased1, rgba1, _center(box1, rgba1))

    # 框2 应用：基于「含框1结果」的 img1
    erased2 = erase_mod.erase_text(img1, box2, strength=0.5)
    rgba2 = render_mod.render_text("香蕉", _params(), font_path=fp)
    img2 = render_mod.composite_text(erased2, rgba2, _center(box2, rgba2))

    p1 = _ink_presence(img2, box1)
    p2 = _ink_presence(img2, box2)
    print(f"  框1 文字占比={p1:.3f}｜框2 文字占比={p2:.3f}")
    assert p1 > 0.02, "框1 的成果在应用框2后丢失（被覆盖）"
    assert p2 > 0.02, "框2 应用失败"
    print("  [PASS] 多选区独立编辑：框1 成果在应用框2后保留、互不覆盖")


if __name__ == "__main__":
    test_multi_region_independent()
    print("\n多选区独立编辑数据流自测通过 ✓")
