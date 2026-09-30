"""需求一回归：新建选框后原文字必须立即消失，且识别(/特征/字体匹配)环节崩溃也不能阻断擦除。

验证点：
1. 通过**真实画布事件链路**（_on_left_down → _on_left_drag → _on_left_up）新建选框，
   原文字（红）在 working_bgr 中像素 = 0。
2. 即便 on_box_select（OCR/特征/字体匹配长链路）抛异常，原文字依然被擦除
   —— 这正是「先擦除、再识别」修复所针对的旧顺序致命漏洞。
3. 擦除失败时有明确状态提示，而非静默吞掉。
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
# 开启 OCR，模拟用户真实环境（能看到识别结果）
ocr_mod.is_ocr_supported = lambda: True
ocr_mod.ocr_region = lambda bgr, min_score=0.5: "识别字"

from test_replace_erase_preview_guides import make_image_colored, _setup

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))


def red_in(bgr, box):
    sub = bgr[box[1]:box[3], box[0]:box[2]]
    r = sub[:, :, 2].astype(int); g = sub[:, :, 1].astype(int); b = sub[:, :, 0].astype(int)
    return int(((r > 120) & (g < 90) & (b < 90)).sum())


def _event(x, y):
    return types.SimpleNamespace(x=x, y=y)


def simulate_new_box(canvas, x0, y0, x1, y1):
    """用真实画布事件新建一个选框（覆盖 [x0,y0]-[x1,y1]）。"""
    canvas._on_left_down(_event(x0, y0))
    canvas._on_left_drag(_event(x1, y1))
    canvas._on_left_up(_event(x1, y1))


def test_real_event_flow_erases():
    root, app, full = _setup(text="原始文字六字", size=40, rgb=(255, 0, 0))
    c = app.canvas
    # 让画布坐标 == 图像坐标，便于用图像坐标发事件
    c.fit_scale = 1.0
    c.zoom = 1.0
    c.pan_x = 0.0
    c.pan_y = 0.0
    c.image_cv = app.original_bgr

    x0, y0, x1, y1 = full[0] - 10, full[1] - 10, full[2] + 10, full[3] + 10
    before = red_in(app.original_bgr, (x0, y0, x1, y1))
    simulate_new_box(c, x0, y0, x1, y1)

    check("真实事件链路成功新建选框", len(c._boxes) == 1,
          f"boxes={len(c._boxes)}")
    after = red_in(app.working_bgr, (x0, y0, x1, y1))
    check("新建选框后原文字被立即擦除（红色残留=0）", after == 0,
          f"before={before} after={after}")
    check("主画布显示的是已擦除结果（非原图）",
          red_in(c.image_cv, (x0, y0, x1, y1)) == 0)
    root.destroy()


def test_recognize_crash_still_erases():
    root, app, full = _setup(text="原始文字六字", size=40, rgb=(255, 0, 0))
    c = app.canvas
    c.fit_scale = 1.0
    c.zoom = 1.0
    c.pan_x = 0.0
    c.pan_y = 0.0
    c.image_cv = app.original_bgr

    # 模拟真实环境里 OCR/特征/字体匹配环节崩溃：直接让画布的 on_box_select 抛错
    def boom(*a, **k):
        raise RuntimeError("模拟真实环境识别环节崩溃")

    app.canvas.on_box_select = boom
    x0, y0, x1, y1 = full[0] - 10, full[1] - 10, full[2] + 10, full[3] + 10
    before = red_in(app.original_bgr, (x0, y0, x1, y1))
    leaked = False
    try:
        simulate_new_box(c, x0, y0, x1, y1)
    except Exception as e:
        leaked = True
        check("画布层识别崩溃未外泄（UI 不崩）", False, f"异常外泄: {e}")
        root.destroy()
        return
    check("画布层识别崩溃未外泄（UI 不崩）", not leaked)
    after = red_in(app.working_bgr, (x0, y0, x1, y1))
    check("识别崩溃后原文字仍被擦除（先擦除、再识别的顺序修复生效）",
          after == 0, f"before={before} after={after}")
    root.destroy()


def test_app_level_select_error_reported():
    """App 层 on_box_selected 包裹后：内部实现抛错应被捕获并给出状态提示。"""
    root, app, full = _setup(text="原始文字六字", size=40, rgb=(255, 0, 0))
    app.canvas.on_box_create(0, full)  # 先擦除一次，确保工作图已改动

    def boom(*a, **k):
        raise RuntimeError("模拟内部识别崩溃")

    app._on_box_selected_impl = boom
    leaked = False
    try:
        app.on_box_selected(full)
    except Exception:
        leaked = True
    check("App 层 on_box_selected 异常不外泄", not leaked)
    check("App 层识别失败有明确状态提示",
          "失败" in app.status_var.get(), f"status={app.status_var.get()!r}")
    root.destroy()


def main():
    test_real_event_flow_erases()
    test_recognize_crash_still_erases()
    test_app_level_select_error_reported()
    print(f"\n共 {len(PASS)+len(FAIL)} 项，通过 {len(PASS)}，失败 {len(FAIL)}")
    if FAIL:
        print("失败项：", FAIL)
    return len(FAIL) == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
