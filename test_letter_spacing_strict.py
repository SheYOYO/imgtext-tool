"""字间距严格对齐原图 —— 专项自测（用户硬性要求：替代文字字与字距离与原图一致）。

覆盖：
  - 不同原图字间距（0/6/14）下，替换文字可见间隙严格等于原图实测间隙（量化极限 ≤1px）
  - 不同中文字体（字池内 cn 字体）下均成立（闭环校准与具体字体无关）
  - 手动字间距覆盖优先级高于原图测量值
  - 单字（无内部间隙）/ 多行（含 \\n）跳过校准，letter_spacing 不被改写、不报错
"""
import os
import sys
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

from app import gui, config, render as render_mod, feature_extract as fe


def _cn_fonts():
    pool = config.resolve_font_pool()
    return [f["path"] for f in pool if f.get("cn")]


def _build_original(letter_spacing, fp):
    """生成「已知字间距」的原图（黑字合成到灰底），返回 BGR。"""
    orig = render_mod.render_text("原始文字六字", {
        "color_rgb": (20, 20, 20), "font_size": 40, "letter_spacing": letter_spacing,
        "line_spacing": 0, "skew_angle": 0.0, "sharpness": 0.9, "blur": 0.0,
        "bold": False, "noise": 0.0, "opacity": 1.0}, font_path=fp)
    oh, ow = orig.shape[:2]
    gray = np.full((oh, ow, 3), 245, np.uint8).astype(np.float32)
    a = orig[:, :, 3:4].astype(np.float32) / 255.0
    comp = (orig[:, :, :3].astype(np.float32) * a + gray * (1 - a)).astype(np.uint8)
    return cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)


_pass = 0
_fail = 0
def check(name, cond, info=""):
    global _pass, _fail
    if cond:
        _pass += 1
        print(f"  [PASS] {name}")
    else:
        _fail += 1
        print(f"  [FAIL] {name}  {info}")


def _measure_original_gap(bgr):
    """原图实测字符间隙（与 app 识别口径一致：estimate_letter_spacing 二值列投影）。"""
    return fe.estimate_letter_spacing(bgr)


# ---------------------------------------------------------------- 1. 多原图间距 × 多字体 严格对齐
def test_strict_across_fonts_and_spacings():
    print("\n== 1. 不同原图字间距 × 不同字体：替换文字间隙严格等于原图 ==")
    fonts = _cn_fonts()
    check("字池中存在中文字体", len(fonts) >= 1, f"fonts={fonts}")
    for fp in fonts:
        for orig_ls in [0, 6, 14]:
            bgr = _build_original(orig_ls, fp)
            orig_gap = _measure_original_gap(bgr)

            root = tk.Tk(); root.withdraw()
            app = gui.App(root)
            app.original_bgr = bgr
            app.features = {"letter_spacing": orig_gap, "line_count": 1}
            app.selected_font_path = fp
            base = {"color_rgb": (20, 20, 20), "font_size": 40, "letter_spacing": 0,
                    "line_spacing": 0, "skew_angle": 0.0, "sharpness": 0.9, "blur": 0.0,
                    "bold": False, "noise": 0.0, "opacity": 1.0}
            # 任意替换文字
            for t in ["替换文字", "新文字三", "中文测试替换字", "国国国国国国国国"]:
                cal = app._calibrate_letter_spacing(t, dict(base), fp)
                cur, _m = render_mod.render_text_with_metrics(t, cal, font_path=fp)
                gap = render_mod.measure_blank_gap(cur)
                if gap is not None:
                    check(f"[{os.path.basename(fp)}] 原图间距={orig_ls} 替换[{t}] 间隙差≤1px",
                          abs(gap - orig_gap) <= 1,
                          f"gap={gap} vs orig_gap={orig_gap}")
            root.destroy()


# ---------------------------------------------------------------- 2. 手动字间距覆盖优先级
def test_manual_override_priority():
    print("\n== 2. 手动字间距覆盖优先级高于原图测量值 ==")
    fp = _cn_fonts()[0]
    bgr = _build_original(6, fp)
    orig_gap = _measure_original_gap(bgr)
    root = tk.Tk(); root.withdraw()
    app = gui.App(root)
    app.original_bgr = bgr
    app.features = {"letter_spacing": orig_gap, "line_count": 1}
    app.selected_font_path = fp
    app.canvas._active_index = 0
    app._user_tweaks = {0: {"letter_spacing": 3}}   # 用户手动指定 3，应高于原图 8
    base = {"color_rgb": (20, 20, 20), "font_size": 40, "letter_spacing": 0,
            "line_spacing": 0, "skew_angle": 0.0, "sharpness": 0.9, "blur": 0.0,
            "bold": False, "noise": 0.0, "opacity": 1.0}
    cal = app._calibrate_letter_spacing("示例文字", dict(base), fp)
    cur, _m = render_mod.render_text_with_metrics("示例文字", cal, font_path=fp)
    gap = render_mod.measure_blank_gap(cur)
    check("手动字间距被采用（渲染间隙≈3，非原图 8）",
          gap is not None and abs(gap - 3) <= 1, f"gap={gap} vs manual=3 (orig={orig_gap})")
    root.destroy()


# ---------------------------------------------------------------- 3. 单字 / 多行 跳过校准
def test_single_char_and_multiline_skip():
    print("\n== 3. 单字 / 多行跳过校准：letter_spacing 不被改写、不报错 ==")
    fp = _cn_fonts()[0]
    bgr = _build_original(6, fp)
    orig_gap = _measure_original_gap(bgr)
    root = tk.Tk(); root.withdraw()
    app = gui.App(root)
    app.original_bgr = bgr
    app.features = {"letter_spacing": orig_gap, "line_count": 1}
    app.selected_font_path = fp
    base = {"color_rgb": (20, 20, 20), "font_size": 40, "letter_spacing": 0,
            "line_spacing": 0, "skew_angle": 0.0, "sharpness": 0.9, "blur": 0.0,
            "bold": False, "noise": 0.0, "opacity": 1.0}
    # 单字
    cal1 = app._calibrate_letter_spacing("示", dict(base), fp)
    check("单字：letter_spacing 锁定原图值（不崩溃、不收放）",
          cal1.get("letter_spacing") == orig_gap, f"ls={cal1.get('letter_spacing')} vs {orig_gap}")
    # 多行
    cal2 = app._calibrate_letter_spacing("第一行\n第二行", dict(base), fp)
    check("多行：letter_spacing 锁定原图值（行距由 line_spacing 管，不崩溃）",
          cal2.get("letter_spacing") == orig_gap, f"ls={cal2.get('letter_spacing')} vs {orig_gap}")
    check("多行：字体大小不变", cal2.get("font_size") == base.get("font_size"))
    root.destroy()


if __name__ == "__main__":
    test_strict_across_fonts_and_spacings()
    test_manual_override_priority()
    test_single_char_and_multiline_skip()
    print(f"\n字间距严格对齐 自测：通过 {_pass} 项，失败 {_fail} 项")
    sys.exit(1 if _fail else 0)
