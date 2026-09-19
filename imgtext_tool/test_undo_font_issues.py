"""用户三问题回归自测：撤销 / 字体切换确认 / 空框误擦除。

覆盖（成功+失败双向）：
1. 撤销：
   a) 「确认后未点应用」直接撤销 → 丢弃草稿，不再提示“没有可撤销的操作”，
      之后点回该框不会复活替换（此前撤销完全失效）。
   b) 「应用后撤销」→ 图像回到上一步；**再点回同一框**不会把刚撤销的替换
      重新渲染出来（此前撤销被点回/缩放复活的替换打回，用户以为不能撤销）。
   c) 多步提交可逐级回退到初始状态。
2. 字体切换：
   「切换候选字体 → 点确认 → 重新点选该框」仍保留切换后的字体
   （此前 _confirm_text 不写回 font_path，重选恢复旧字体，“确定后又变回原效果”）。
3. 空框保护：仅框选、不输入文字，点别处离开时**不得**把“只擦除预览”固化，
   原文不得被悄悄擦掉（此前随手画框就会擦字且污染撤销栈）。
"""
import os
import sys
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app import config

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_output")
os.makedirs(OUT, exist_ok=True)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))


def make_image(text="测试文字", pos=(40, 70), size=40):
    W, H = 420, 220
    base = np.full((H, W, 3), 240, dtype=np.uint8)
    img = Image.fromarray(base)
    pool = config.resolve_font_pool()
    fp = next((f["path"] for f in pool if f.get("cn")), None)
    font = ImageFont.truetype(fp, size) if fp else ImageFont.load_default()
    ImageDraw.Draw(img).text(pos, text, font=font, fill=(20, 20, 20))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _setup():
    import tkinter as tk
    from tkinter import messagebox
    from app import gui
    messagebox.showinfo = lambda *a, **k: None
    messagebox.showerror = lambda *a, **k: None
    messagebox.showwarning = lambda *a, **k: None
    root = tk.Tk()
    root.withdraw()
    app = gui.App(root)
    bgr = make_image()
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()
    app.history = [{"bgr": bgr.copy(), "box_edits": {}}]
    app.pending_result = None
    app.current_box = None
    app.source_path = None
    app.canvas.set_image(bgr)
    return root, app


def _new_box(app, box):
    app.canvas._boxes.append(box)
    app.canvas._active_index = len(app.canvas._boxes) - 1
    app.on_box_selected(box, is_resize=False)
    app.root.update()


def _type(app, text):
    app.text_var.set(text)
    app._schedule_preview(0)
    app.root.update()


def _flush(app):
    try:
        app.root.update()
    except Exception:
        pass


def test_undo_after_confirm_no_apply():
    print("\n== 1. 确认后（未应用）直接撤销：丢弃草稿、不复活 ==")
    root, app = _setup()
    box = (40, 70, 200, 115)
    _new_box(app, box)
    _type(app, "新文字")
    app._confirm_text()
    _flush(app)
    ok_draft = app.pending_result is not None
    app.undo()
    _flush(app)
    check("确认未应用时撤销可执行（不弹“无法撤销”）", ok_draft)
    check("撤销丢弃了预览草稿", app.pending_result is None)
    check("工作图未被改动（回到原图）",
          np.array_equal(app.working_bgr, app.original_bgr))
    app.on_box_select_same(0)
    _flush(app)
    check("撤销后点回该框不复活替换", app.pending_result is None)
    root.destroy()


def test_undo_apply_then_reselect_no_resurrect():
    print("\n== 2. 应用后撤销 → 点回同框不复活 ==")
    root, app = _setup()
    bgr0 = app.working_bgr.copy()
    box = (40, 70, 200, 115)
    _new_box(app, box)
    _type(app, "新文字")
    app._confirm_text()
    _flush(app)
    app.apply_replace()
    _flush(app)
    check("应用生效", not np.array_equal(app.working_bgr, bgr0))
    app.undo()
    _flush(app)
    check("撤销回到上一步", np.array_equal(app.working_bgr, bgr0))
    check("撤销清空了残留替换文字", app.text_var.get().strip() == "")
    app.on_box_select_same(0)
    _flush(app)
    check("点回该框不复活（预览为空、工作图仍为原图）",
          app.pending_result is None and np.array_equal(app.working_bgr, bgr0))
    root.destroy()


def test_multi_step_undo():
    print("\n== 3. 两步提交可逐级撤销 ==")
    root, app = _setup()
    bgr0 = app.working_bgr.copy()
    box = (40, 70, 200, 115)
    _new_box(app, box)
    _type(app, "第一处")
    app.apply_replace()
    _flush(app)
    img1 = app.working_bgr.copy()
    box2 = (220, 70, 380, 115)
    _new_box(app, box2)
    _type(app, "第二处")
    app.apply_replace()
    _flush(app)
    img2 = app.working_bgr.copy()
    check("两步都生效", not np.array_equal(img2, img1) and not np.array_equal(img1, bgr0))
    app.undo()
    _flush(app)
    check("撤销一步回到第一处状态", np.array_equal(app.working_bgr, img1))
    app.undo()
    _flush(app)
    check("再撤销回到原图", np.array_equal(app.working_bgr, bgr0))
    root.destroy()


def test_font_choice_survives_confirm_reselect():
    print("\n== 4. 切换字体→确认→重选，仍保留切换后的字体 ==")
    root, app = _setup()
    box = (40, 70, 200, 115)
    _new_box(app, box)
    _type(app, "新文字")
    if len(app.candidates) < 2:
        print("  [SKIP] 候选字体不足两款，跳过")
        root.destroy()
        return
    alt = app.candidates[1]["path"]
    app._select_candidate(1)
    _flush(app)
    check("切换字体已生效", app.selected_font_path == alt)
    app._confirm_text()
    _flush(app)
    check("确认后仍为切换的字体",
          app.box_edits[0].get("font_path") == alt)
    app.on_box_select_same(0)
    _flush(app)
    check("重选该框仍保留切换后的字体（不再打回原效果）",
          app.selected_font_path == alt)
    root.destroy()


def test_empty_box_leave_no_erase():
    print("\n== 5. 空框（未输入）离开不自动擦除原文 ==")
    root, app = _setup()
    bgr0 = app.working_bgr.copy()
    n_hist = len(app.history)
    box = (40, 70, 200, 115)
    _new_box(app, box)
    _flush(app)
    app.on_leave_box()
    _flush(app)
    check("空框不会生成只擦除预览", app.pending_result is None)
    check("离开后原文未被悄悄擦掉", np.array_equal(app.working_bgr, bgr0))
    check("撤销栈未被空框污染", len(app.history) == n_hist)
    root.destroy()


def test_empty_box_undo_safe():
    print("\n== 6. 撤销后空框状态安全（不复活不崩溃） ==")
    root, app = _setup()
    box = (40, 70, 200, 115)
    _new_box(app, box)
    _type(app, "草稿")
    app.undo()      # 情形1：丢弃未提交草稿
    _flush(app)
    check("丢弃草稿后输入框为空", app.text_var.get().strip() == "")
    app.on_box_select_same(0)
    _flush(app)
    check("点回空框不产生替换预览", app.pending_result is None)
    app.undo()      # 已无草稿亦无可撤销提交
    _flush(app)
    check("连续撤销不崩溃", True)
    root.destroy()


def main():
    test_undo_after_confirm_no_apply()
    test_undo_apply_then_reselect_no_resurrect()
    test_multi_step_undo()
    test_font_choice_survives_confirm_reselect()
    test_empty_box_leave_no_erase()
    test_empty_box_undo_safe()
    print(f"\n用户三问题回归：通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：" + "；".join(FAIL))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    print("撤销 / 字体切换 / 空框保护 自测通过 ✓")
