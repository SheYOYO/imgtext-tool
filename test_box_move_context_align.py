"""选区长摁移动 / 旧框草稿不丢失 / 替换自动对齐上下文 回归自测。

覆盖佘先生三条要求：
1. 选框「单击框内 = 编辑文字（不移动）；长摁（或明显拖动）框内 = 整体移动选框」
   —— 注：早期版本是「确认替换后位置锁死」，本轮按新要求改为长摁即可移动，
   长摁本身就是防误触门槛，不再区分是否已确认。
2. 「重画新框时上一框的替换文字仍显示」——离开当前框（含点空白画新框）前，
   即使草稿预览还没生成，也先同步渲染再固化，文字不再消失、无需重新点回。
3. 「替换后新内容字间距严格对齐原图」——确认时闭环校准 letter_spacing 使替换文字的
   可见字间距逐像素等于原图实测字间距（与具体字体、字数无关）；字数变化只改整段宽度，绝不收放字距。
"""
import os
import sys
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app import config
from app import font_match as fm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_output")
os.makedirs(OUT, exist_ok=True)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))


def make_image(text, size=40):
    pool = config.resolve_font_pool()
    fp = next((f["path"] for f in pool if f.get("cn")), None)
    font = ImageFont.truetype(fp, size) if fp else ImageFont.load_default()
    tmp = Image.new("RGB", (1, 1), (240, 240, 240))
    bb = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    W = bb[2] + 80
    H = bb[3] + 120
    img = Image.new("RGB", (W, H), (240, 240, 240))
    ImageDraw.Draw(img).text((40 - bb[0], 60 - bb[1]), text, font=font, fill=(20, 20, 20))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), (40 - bb[0], 60 - bb[1],
                                                            40 - bb[0] + bb[2], 60 - bb[1] + bb[3])


def _setup(text="原始文字六字", size=40):
    import tkinter as tk
    from tkinter import messagebox
    from app import gui
    messagebox.showinfo = lambda *a, **k: None
    messagebox.showerror = lambda *a, **k: None
    root = tk.Tk()
    root.withdraw()
    app = gui.App(root)
    bgr, box = make_image(text, size)
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()
    app.history = [{"bgr": bgr.copy(), "box_edits": {}}]
    app.pending_result = None
    app.current_box = None
    app.source_path = None
    app.canvas.set_image(bgr)
    return root, app, box


def _new_box(app, box):
    app.canvas._boxes.append(box)
    app.canvas._active_index = len(app.canvas._boxes) - 1
    app.on_box_selected(box, is_resize=False)
    app.root.update()


def _flush(app):
    try:
        app.root.update()
    except Exception:
        pass


def _click(cvs, x, y):
    """模拟「单击」（按下即松开，不拖动）。"""
    from types import SimpleNamespace
    cvs._on_left_down(SimpleNamespace(x=x, y=y))
    cvs._on_left_up(SimpleNamespace(x=x, y=y))


def _long_press_move(cvs, x, y, dx, dy):
    """模拟「长摁后拖动」：按下 → 长摁定时器到期 → 拖动 → 松开。"""
    from types import SimpleNamespace
    cvs._on_left_down(SimpleNamespace(x=x, y=y))
    cvs._on_long_press()          # 等价于按住超过 LONG_PRESS_MS
    cvs._on_left_drag(SimpleNamespace(x=x + dx, y=y + dy))
    cvs._on_left_up(SimpleNamespace(x=x + dx, y=y + dy))


def test_click_edits_long_press_moves():
    print("\n== 1. 单击框内=编辑文字不移动；长摁框内=整体移动选框（含已确认框）==")
    root, app, box = _setup()
    cvs = app.canvas
    _new_box(app, box)
    _flush(app)
    x0, y0, x1, y1 = cvs.get_active_box()
    cxs, cys = cvs._to_canvas((x0 + x1) / 2, (y0 + y1) / 2)
    mode, _, _ = cvs._hit_test(cxs, cys)
    check("框内命中为 move（长摁即可整体拖动）", mode == "move", f"mode={mode}")

    # ---- 单击：不移动选框，显示四角锚点、收起叉号 ----
    pre = cvs.get_active_box()
    w, h = pre[2] - pre[0], pre[3] - pre[1]
    _click(cvs, cxs, cys)
    _flush(app)
    check("单击框内不移动选框（= 编辑文字）", cvs.get_active_box() == pre,
          f"{pre} → {cvs.get_active_box()}")
    check("单击后显示四角锚点、不显示叉号", cvs._show_close is False)

    # ---- 长摁拖动：整体平移、尺寸不变、松手后显示右上角叉号 ----
    _long_press_move(cvs, cxs, cys, 40, 30)
    _flush(app)
    moved = cvs.get_active_box()
    check("长摁拖动后框整体移动", moved != pre and moved != box, f"{pre} → {moved}")
    check("移动后框尺寸不变", (moved[2] - moved[0]) == w and (moved[3] - moved[1]) == h)
    check("长摁移动松手后显示右上角叉号", cvs._show_close is True)

    # ---- 已确认的框同样能长摁移动（长摁本身即防误触门槛）----
    app.text_var.set("替换文字")
    app._confirm_text()
    _flush(app)
    x0, y0, x1, y1 = cvs.get_active_box()
    c2x, c2y = cvs._to_canvas((x0 + x1) / 2, (y0 + y1) / 2)
    pre2 = cvs.get_active_box()
    _long_press_move(cvs, c2x, c2y, 30, 20)
    _flush(app)
    check("已确认的框也能长摁移动", cvs.get_active_box() != pre2,
          f"{pre2} → {cvs.get_active_box()}")
    root.destroy()


def test_previous_draft_kept_when_drawing_new_box():
    print("\n== 2. 画新框时上一框替换文字仍固化显示（不需重新点回） ==")
    root, app, box1 = _setup("测试文字")
    _new_box(app, box1)
    # 键入文字后**不点确认、不点应用**，立刻去画第二框（模拟防抖预览未跑）
    app.text_var.set("第一处替换")
    # 只写 box_edits（等价于用户刚敲完字、300ms 预览还没生成）
    app.box_edits[0]["text"] = "第一处替换"
    app.pending_result = None
    _flush(app)
    # 点空白画第二框 → canvas 离开回调应先同步渲染并固化第一框草稿
    box2 = (box1[2] + 60, box1[1], box1[2] + 260, box1[3])
    app.canvas._boxes.append(box2)
    app.canvas._active_index = 1
    app.on_leave_box()          # 等价于 _on_left_down 里按空白时的离开固化
    _flush(app)
    bgr_after = app.working_bgr.copy()
    # 第一框区域应已含替换（与原始不同），证明草稿被固化、不会“消失待重选”
    r0 = bgr_after[box1[1]:box1[3], box1[0]:box1[2]]
    r_orig = app.original_bgr[box1[1]:box1[3], box1[0]:box1[2]]
    check("离开后第一框替换文字已固化进工作图",
          not np.array_equal(r0, r_orig))
    check("撤销栈记录了第一框固化的历史帧（可正常撤销）", len(app.history) == 2)
    root.destroy()


def test_replacement_strict_spacing_to_original():
    print("\n== 3. 确认后新文字字间距严格等于原图字间距（闭环校准，不按宽度收放） ==")
    root, app, box = _setup("一二三四五六", size=40)   # 6 字
    _new_box(app, box)
    _flush(app)
    orig_gap = app.features.get("letter_spacing")
    check("已测得原图字间距（>0）", orig_gap and orig_gap > 0, f"orig_gap={orig_gap}")
    app.text_var.set("四字替换")          # 4 字（与 6 字不同）
    app._schedule_preview(0)
    _flush(app)
    app._confirm_text()
    _flush(app)
    locked = app.box_edits[0].get("optimized_params", {})
    ls = locked.get("letter_spacing")
    print(f"  原图字间距={orig_gap}px  optimized letter_spacing={ls}")
    # 用与确认时完全相同的字体与参数重渲染，量出实际可见字间距
    from app import render as render_mod
    cur, _ = render_mod.render_text_with_metrics(
        "四字替换", locked, font_path=app.selected_font_path)
    new_gap = render_mod.measure_blank_gap(cur)
    print(f"  替换文字可见字间距={new_gap}px  与原图差={abs((new_gap or 0) - orig_gap)}px")
    check("闭环校准后替换文字字间距严格对齐原图（差≤1px，量化极限）",
          new_gap is not None and abs(new_gap - orig_gap) <= 1,
          f"new_gap={new_gap} vs orig_gap={orig_gap}")
    # 用户硬性要求：字数变化只改整段宽度，绝不收放字距 —— 字号保持锁定不变
    check("字数变化不改变字号（保持锁定）",
          locked.get("font_size") == app.features.get("font_size"))
    root.destroy()


def main():
    test_click_edits_long_press_moves()
    test_previous_draft_kept_when_drawing_new_box()
    test_replacement_strict_spacing_to_original()
    print(f"\n移动/草稿保持/上下文对齐 自测：通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：" + "；".join(FAIL))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    print("选框移动锁定 / 旧框草稿保持 / 替换对齐上下文 自测通过 ✓")
