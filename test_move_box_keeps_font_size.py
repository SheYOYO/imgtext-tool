"""移动选框后，用户手动调整过的文字参数（字号等）必须保持不变。

用户反馈：移动选框时，调过的文字大小会变（被打回 OCR 识别基线字号）。

真实链路（画布事件）：
  新建框 → OCR 填原文、字号滑杆=特征基线 A
  → 用户输入替换文字、把字号手动调到 B（滑块 = B）
  → 长摁拖动选框（_on_left_down → _on_left_drag → _on_left_up）
  → App.on_box_changed（此前会 _auto_apply_features() 把滑块打回 A）

修复：移动/缩放选框只动几何，绝不把滑块参数打回 OCR 基线 / 冻结值。
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tkinter as tk
from tkinter import messagebox
from app import gui, ocr as ocr_mod

messagebox.showinfo = lambda *a, **k: None
messagebox.showerror = lambda *a, **k: None
# 模拟用户真实环境：OCR 可用
ocr_mod.is_ocr_supported = lambda: True
ocr_mod.ocr_region = lambda bgr, min_score=0.5: "识别字"

from test_replace_erase_preview_guides import make_image_colored, _setup

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))


def _event(x, y):
    return types.SimpleNamespace(x=x, y=y)


def _ink_bbox(bgr):
    """任意颜色的「非浅色墨迹」包围盒：返回 (x0,y0,x1,y1) 或 None。

    替换文字沿用 OCR 提取的原字颜色（本例红字 → 替换字也是红色），
    因此不能假定蓝色；浅灰底(240) 上的任何深色/彩色墨迹即渲染出的文字。
    """
    if bgr is None:
        return None
    s = bgr.astype(int).sum(axis=2)
    ys, xs = np.where(s < 600)
    if ys.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def simulate_new_box(canvas, x0, y0, x1, y1):
    """用真实画布事件新建选框。"""
    canvas._on_left_down(_event(x0, y0))
    canvas._on_left_drag(_event(x1, y1))
    canvas._on_left_up(_event(x1, y1))


def simulate_move_box(canvas, cx, cy, dx, dy):
    """用真实画布事件把选框（从框内按下）向右下拖动 (dx, dy)。"""
    canvas._on_left_down(_event(cx, cy))
    canvas._on_left_drag(_event(cx + dx // 2, cy + dy // 2))
    canvas._on_left_drag(_event(cx + dx, cy + dy))
    canvas._on_left_up(_event(cx + dx, cy + dy))


def test_move_box_keeps_manual_font_size():
    root, app, full = _setup(text="原始文字六字", size=40, rgb=(255, 0, 0))
    c = app.canvas
    c.fit_scale = 1.0
    c.zoom = 1.0
    c.pan_x = 0.0
    c.pan_y = 0.0
    c.image_cv = app.original_bgr

    # 1) 真实事件新建选框（覆盖原文字）→ OCR 填原文、滑杆 = 特征基线字号
    x0, y0, x1, y1 = full[0] - 10, full[1] - 10, full[2] + 10, full[3] + 10
    simulate_new_box(c, x0, y0, x1, y1)
    _flush(app)
    check("真实事件新建选框成功", len(c._boxes) == 1)
    fs_a = int(float(app.vars["font_size"].get()))
    check("OCR 基线字号已进滑块", fs_a > 0, f"fs_a={fs_a}")

    # 2) 用户输入替换文字 + 手动把字号调到明显更大的 B
    #    真实 GUI 拖滑块会走 _on_slider（既设值又记录手动调整），模拟同一条路
    app.text_var.set("替换文字甲")
    app._on_text_key()                       # 模拟击键同步
    fs_b = fs_a + 40
    app.vars["font_size"].set(str(fs_b))
    app._on_slider("font_size", app.vars["font_size"], ".0f")
    app._schedule_preview(0)                 # 等不及 180ms 防抖，立即渲染基准
    _flush(app)
    check("滑块字号 = 用户手动值 B", int(float(app.vars["font_size"].get())) == fs_b,
          f"fs_b={fs_b}")
    h_big = _ink_bbox(app.pending_result)
    check("大字号预览已渲染（有墨迹）", h_big is not None and h_big[3] - h_big[1] > 20,
          f"h_big={None if h_big is None else h_big[3] - h_big[1]}")

    # 3) 真实事件移动选框（向右下拖动 80×60）
    box_now = c._boxes[0]
    cx = (box_now[0] + box_now[2]) // 2
    cy = (box_now[1] + box_now[3]) // 2
    simulate_move_box(c, cx, cy, 80, 60)
    _flush(app)
    moved = c._boxes[0]
    check("选框确实被移动", (moved[0], moved[1]) != (box_now[0], box_now[1]),
          f"from={box_now[:2]} to={moved[:2]}")

    # 4) 核心断言：移动后字号仍 = 用户手动值 B（按下/拖动/松手都不被打回基线 A）
    fs_after = int(float(app.vars["font_size"].get()))
    check("移动选框后字号仍是用户手动值（不被打回 OCR 基线）",
          fs_after == fs_b, f"fs_a={fs_a} fs_b={fs_b} fs_after={fs_after}")

    # 5) 移动后新位置预览：文字仍是大字号（墨迹高度不回缩到基线小字号）。
    #    移动路径会触发字体层重选（同字号不同字体行高可差 ~20%），因此不与
    #    fs_b 的精确高度比对，而是断言：明显高于 fs_a=42 的渲染高度下限
    #    （基线 42 字号按同比例约为 60~65px，fs_a*1.5=63 之上即未被回缩）。
    h_after = _ink_bbox(app.pending_result)
    h_after_v = h_after[3] - h_after[1] if h_after else 0
    h_big_v = h_big[3] - h_big[1] if h_big else 0
    check("移动后新位置渲染文字字号不回缩（墨迹高度仍是大字号量级）",
          h_after is not None and h_after_v > max(fs_a * 1.5, h_big_v * 0.7),
          f"h_after={h_after_v} h_big={h_big_v} fs_a={fs_a}")

    # 6) 再拖一次（第二次移动也不打回）
    simulate_move_box(c, moved[0] + 10, moved[1] + 10, 40, 30)
    _flush(app)
    fs_after2 = int(float(app.vars["font_size"].get()))
    check("再次移动仍保持用户字号", fs_after2 == fs_b,
          f"fs_after2={fs_after2}")

    root.destroy()


def _flush(app):
    try:
        app.root.update()
    except Exception:
        pass


def main():
    test_move_box_keeps_manual_font_size()
    print(f"\n共 {len(PASS)+len(FAIL)} 项，通过 {len(PASS)}，失败 {len(FAIL)}")
    if FAIL:
        print("失败项：", FAIL)
    return len(FAIL) == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
