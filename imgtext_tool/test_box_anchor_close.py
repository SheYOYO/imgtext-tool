"""选框锚点 / 长摁移动 / 右上角叉号 / 替换结果持久化 回归自测。

覆盖佘先生本轮三条要求：

1. **替换结果持久化**：框选一片区域、原文字被替代后直接消失，
   **删除选框、移走选框、缩放选框都不能让它再出现**，只有「撤销」能撤回。
   （旧实现里预览只是一张贴图，工作图没变，几何一动就基于未擦除的工作图
   重画，原文字当场复活。）
2. **锚点缩放 + 长摁移动**：四角锚点拖动缩放选框；长摁（或明显拖动）选框
   中间整体移动选框；单击框内 = 编辑文字、不移动。
3. **右上角叉号**：只有「长摁移动选框后松手」才出现，此时四角锚点收起；
   单击选框则收起叉号、恢复四角锚点。点叉号 = 删除该选框。
"""
import os
import sys

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


def make_image(text, size=40):
    pool = config.resolve_font_pool()
    fp = next((f["path"] for f in pool if f.get("cn")), None)
    font = ImageFont.truetype(fp, size) if fp else ImageFont.load_default()
    tmp = Image.new("RGB", (1, 1), (240, 240, 240))
    bb = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    W = bb[2] + 160
    H = bb[3] + 160
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


def _evt(x, y):
    from types import SimpleNamespace
    return SimpleNamespace(x=x, y=y)


def _click(cvs, x, y):
    """单击（按下即松开，不拖动）。"""
    cvs._on_left_down(_evt(x, y))
    cvs._on_left_up(_evt(x, y))


def _long_press_move(cvs, x, y, dx, dy):
    """长摁后拖动：按下 → 长摁定时器到期 → 拖动 → 松开。"""
    cvs._on_left_down(_evt(x, y))
    cvs._on_long_press()
    cvs._on_left_drag(_evt(x + dx, y + dy))
    cvs._on_left_up(_evt(x + dx, y + dy))


def _drag_move(cvs, x, y, dx, dy):
    """不等长摁、直接拖动（位移超阈值也应进入移动）。"""
    cvs._on_left_down(_evt(x, y))
    cvs._on_left_drag(_evt(x + dx, y + dy))
    cvs._on_left_up(_evt(x + dx, y + dy))


def _replace_and_apply(app, text="四字替换"):
    """键入替换文字 → 确认 → 应用（走完整五步流程）。"""
    app.text_var.set(text)
    app._refresh_preview()
    app._confirm_text()
    _flush(app)
    app.apply_replace()
    _flush(app)


def _region(img, box):
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]


# ============================================================ 1. 替换结果持久化
def test_delete_box_keeps_replacement():
    print("\n== 1. 删除选框后原文字不复活（替换结果留在图上）==")
    root, app, box = _setup()
    cvs = app.canvas
    _new_box(app, box)
    _flush(app)
    _replace_and_apply(app)
    applied = app.working_bgr.copy()
    check("应用替换后该区域已不是原图",
          not np.array_equal(_region(applied, box), _region(app.original_bgr, box)))
    # 删除选框
    app.delete_selected_box()
    _flush(app)
    check("删除后选框数量归零", len(cvs.get_boxes()) == 0, f"boxes={cvs.get_boxes()}")
    check("删除选框后原文字没有复活（区域仍 ≠ 原图）",
          not np.array_equal(_region(app.working_bgr, box), _region(app.original_bgr, box)))
    check("删除选框后画布显示的就是工作图（未回退）",
          np.array_equal(app.working_bgr, applied))
    root.destroy()


def test_move_box_keeps_erased():
    print("\n== 2. 移走选框后原文字不复活（旧位置保持已擦除）==")
    root, app, box = _setup()
    cvs = app.canvas
    _new_box(app, box)
    _flush(app)
    _replace_and_apply(app)
    x0, y0, x1, y1 = cvs.get_active_box()
    cxs, cys = cvs._to_canvas((x0 + x1) / 2, (y0 + y1) / 2)
    _long_press_move(cvs, cxs, cys, 50, 40)
    _flush(app)
    check("长摁后选框确实移动了", cvs.get_active_box() != box,
          f"{box} → {cvs.get_active_box()}")
    check("旧位置原文字没有复活（≠ 原图）",
          not np.array_equal(_region(app.working_bgr, box), _region(app.original_bgr, box)))
    # 预览/再固化后也应保持：切走再回来不复活
    app.on_leave_box()
    _flush(app)
    check("切走固化后旧位置仍未复活",
          not np.array_equal(_region(app.working_bgr, box), _region(app.original_bgr, box)))
    root.destroy()


def test_resize_box_keeps_erased():
    print("\n== 3. 缩放选框后原文字不露头 ==")
    root, app, box = _setup()
    cvs = app.canvas
    _new_box(app, box)
    _flush(app)
    _replace_and_apply(app)
    x0, y0, x1, y1 = cvs.get_active_box()
    # 拖左上角锚点向内缩（其余角固定）
    hx, hy = cvs._to_canvas(x0, y0)
    cvs._on_left_down(_evt(hx, hy))
    cvs._on_left_drag(_evt(hx + 20, hy + 12))
    cvs._on_left_up(_evt(hx + 20, hy + 12))
    _flush(app)
    check("拖动左上角锚点改变了选框尺寸", cvs.get_active_box() != box,
          f"{box} → {cvs.get_active_box()}")
    check("缩小后原文字区域仍未复活（≠ 原图）",
          not np.array_equal(_region(app.working_bgr, box), _region(app.original_bgr, box)))
    root.destroy()


def test_undo_can_restore_original():
    print("\n== 4. 只有「撤销」能把原文字找回来 ==")
    root, app, box = _setup()
    cvs = app.canvas
    _new_box(app, box)
    _flush(app)
    _replace_and_apply(app)
    # 撤销：回到应用替换之前；再撤销：回到原图
    n0 = len(app.history)
    app.undo()
    _flush(app)
    check("撤销一次后历史栈回退", len(app.history) == n0 - 1,
          f"{n0} → {len(app.history)}")
    while len(app.history) > 1:
        app.undo()
        _flush(app)
    check("撤销到底后工作图回到原图",
          np.array_equal(app.working_bgr, app.original_bgr))
    check("撤销后原文字区域已恢复",
          np.array_equal(_region(app.working_bgr, box), _region(app.original_bgr, box)))
    # 失败路径：没有可撤销操作时不应崩溃
    try:
        app.undo()
        check("撤销到底后再撤销不崩溃", True)
    except Exception as e:
        check("撤销到底后再撤销不崩溃", False, f"{e}")
    root.destroy()


# ============================================================ 2. 锚点 / 长摁移动
def test_anchor_resize_and_long_press_move():
    print("\n== 5. 四角锚点缩放 + 长摁/拖动移动选框 ==")
    root, app, box = _setup()
    cvs = app.canvas
    _new_box(app, box)
    _flush(app)
    x0, y0, x1, y1 = cvs.get_active_box()
    # --- 锚点命中 ---
    hx, hy = cvs._to_canvas(x1, y1)     # 右下角 se 锚点
    mode, midx, hname = cvs._hit_test(hx, hy)
    check("右下角命中 resize 锚点", mode == "resize" and hname == "se",
          f"mode={mode} handle={hname}")
    # --- 拖动锚点缩放（其余角固定）---
    cvs._on_left_down(_evt(hx, hy))
    cvs._on_left_drag(_evt(hx + 30, hy + 18))
    cvs._on_left_up(_evt(hx + 30, hy + 18))
    _flush(app)
    nb = cvs.get_active_box()
    check("拖 se 锚点后左上角固定、右下角变大",
          nb[0] == x0 and nb[1] == y0 and nb[2] > x1 and nb[3] > y1,
          f"{box} → {nb}")
    # --- 单击框内：不移动、显示锚点 ---
    cx, cy = cvs._to_canvas((nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2)
    _click(cvs, cx, cy)
    _flush(app)
    check("单击框内不移动选框", cvs.get_active_box() == nb)
    check("单击后显示锚点、不显示叉号", cvs._show_close is False)
    # --- 明显拖动（不等长摁）也能移动 ---
    w0, h0 = nb[2] - nb[0], nb[3] - nb[1]
    _drag_move(cvs, cx, cy, 40, 25)
    _flush(app)
    mv = cvs.get_active_box()
    check("直接拖动（未等长摁阈值）也移动选框", mv != nb, f"{nb} → {mv}")
    check("移动后尺寸不变", (mv[2] - mv[0]) == w0 and (mv[3] - mv[1]) == h0)
    root.destroy()


# ============================================================ 3. 右上角叉号
def test_close_button_rules():
    print("\n== 6. 右上角叉号：长摁松手才出现，单击即收起 ==")
    root, app, box = _setup()
    cvs = app.canvas
    _new_box(app, box)
    _flush(app)
    x0, y0, x1, y1 = cvs.get_active_box()
    cx, cy = cvs._to_canvas((x0 + x1) / 2, (y0 + y1) / 2)

    # 初始：显示锚点、不显示叉号
    check("新建选框后默认显示锚点（无叉号）", cvs._show_close is False)
    hx, hy = cvs._to_canvas(x1, y0)     # 右上角 ne
    mode, _, hname = cvs._hit_test(hx, hy)
    check("右上角此时命中 resize 锚点（不是叉号）",
          mode == "resize" and hname == "ne", f"mode={mode} handle={hname}")

    # 长摁移动松手 → 显示叉号、锚点收起
    _long_press_move(cvs, cx, cy, 30, 20)
    _flush(app)
    check("长摁移动松手后显示叉号", cvs._show_close is True)
    bx0, by0, bx1, by1 = cvs.get_active_box()
    hx, hy = cvs._to_canvas(bx1, by0)
    mode, _, _ = cvs._hit_test(hx, hy)
    check("右上角命中变为 close（叉号）", mode == "close", f"mode={mode}")
    # 四个锚点位置都不再命中 resize
    modes = []
    for (ix, iy) in ((bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)):
        px, py = cvs._to_canvas(ix, iy)
        modes.append(cvs._hit_test(px, py)[0])
    check("叉号状态下四角锚点全部不响应 resize",
          all(m != "resize" for m in modes), f"modes={modes}")

    # 单击框内 → 收起叉号、恢复锚点
    cx2, cy2 = cvs._to_canvas((bx0 + bx1) / 2, (by0 + by1) / 2)
    _click(cvs, cx2, cy2)
    _flush(app)
    check("单击选框后叉号收起", cvs._show_close is False)
    bx0, by0, bx1, by1 = cvs.get_active_box()
    hx, hy = cvs._to_canvas(bx1, by0)
    mode, _, hname = cvs._hit_test(hx, hy)
    check("单击后右上角恢复为 resize 锚点",
          mode == "resize" and hname == "ne", f"mode={mode} handle={hname}")

    # 点叉号 → 删除该框（替换结果保留）
    app.text_var.set("替换一下")
    app._confirm_text()
    _flush(app)
    app.apply_replace()
    _flush(app)
    applied = app.working_bgr.copy()
    bx0, by0, bx1, by1 = cvs.get_active_box()
    cx3, cy3 = cvs._to_canvas((bx0 + bx1) / 2, (by0 + by1) / 2)
    _long_press_move(cvs, cx3, cy3, 20, 15)     # 让叉号出现
    _flush(app)
    bx0, by0, bx1, by1 = cvs.get_active_box()
    hx, hy = cvs._to_canvas(bx1, by0)
    mode, cidx, _ = cvs._hit_test(hx, hy)
    check("叉号已出现（点它删除选框）", mode == "close", f"mode={mode}")
    if mode == "close":
        cvs._on_left_down(_evt(hx, hy))
        _flush(app)
        check("点叉号后选框被删除", len(cvs.get_boxes()) == 0, f"boxes={cvs.get_boxes()}")
        check("点叉号删除后工作图未回退成原图（替换结果保留）",
              not np.array_equal(app.working_bgr, app.original_bgr))
        check("点叉号删除后原文字没有复活",
              not np.array_equal(_region(app.working_bgr, box), _region(app.original_bgr, box)))
    root.destroy()


def test_empty_box_not_erased():
    print("\n== 7. 失败路径：空框（没输入文字）不会误擦原文 ==")
    root, app, box = _setup()
    cvs = app.canvas
    _new_box(app, box)
    _flush(app)
    app.text_var.set("")            # 明确没有替换文字
    app.pending_result = None
    _flush(app)
    x0, y0, x1, y1 = cvs.get_active_box()
    cx, cy = cvs._to_canvas((x0 + x1) / 2, (y0 + y1) / 2)
    _long_press_move(cvs, cx, cy, 30, 20)
    _flush(app)
    check("空框移动后选框确实移动了", cvs.get_active_box() != box)
    check("空框不会把原文字悄悄擦掉（区域 == 原图）",
          np.array_equal(_region(app.working_bgr, box), _region(app.original_bgr, box)))
    check("空框不污染撤销栈", len(app.history) == 1, f"history={len(app.history)}")
    root.destroy()


def test_edge_midpoint_resize():
    print("\n== 8. 四边中点手柄：单独调宽 / 调高（8 锚点增强）==")
    root, app, box = _setup()
    cvs = app.canvas
    _new_box(app, box)
    _flush(app)
    # 单击框内 → 显示 8 个手柄（四角 + 四边中点），且非叉号态
    bx0, by0 = cvs._to_canvas(box[0], box[1])
    bx1, by1 = cvs._to_canvas(box[2], box[3])
    _click(cvs, int((bx0 + bx1) / 2), int((by0 + by1) / 2))
    _flush(app)
    check("单击后进入锚点态（非叉号态）", cvs._show_close is False)
    hp = cvs._handle_points(bx0, by0, bx1, by1)
    check("手柄已扩为 8 个（四角 + 四边中点）", len(hp) == 8, f"hp={sorted(hp)}")

    # 拖右侧中点 'e' 向右 → 仅加宽、高度不变
    ex, ey = hp["e"]
    cvs._drag = "resize"
    cvs._drag_handle = "e"
    cvs._drag_orig_box = box
    cvs._drag_start_canvas = (ex, ey)
    cvs._on_left_drag(_evt(ex + 40, ey))
    cvs._on_left_up(_evt(ex + 40, ey))
    _flush(app)
    r = cvs.get_active_box()
    check("拖右侧中点仅加宽（宽度增大）", r[2] > box[2], f"{box} → {r}")
    check("拖右侧中点高度不变", (r[3] - r[1]) == (box[3] - box[1]))

    # 拖上侧中点 'n' 向上 → 仅改高度（顶部上移）、宽度不变
    bx0, by0 = cvs._to_canvas(r[0], r[1])
    bx1, by1 = cvs._to_canvas(r[2], r[3])
    hp2 = cvs._handle_points(bx0, by0, bx1, by1)
    nx, ny = hp2["n"]
    cvs._drag = "resize"
    cvs._drag_handle = "n"
    cvs._drag_orig_box = r
    cvs._drag_start_canvas = (nx, ny)
    cvs._on_left_drag(_evt(nx, ny - 40))
    cvs._on_left_up(_evt(nx, ny - 40))
    _flush(app)
    r2 = cvs.get_active_box()
    check("拖上侧中点高度增大（顶部上移）",
          r2[1] < r[1] and (r2[3] - r2[1]) > (r[3] - r[1]), f"{r} → {r2}")
    check("拖上侧中点宽度不变", (r2[2] - r2[0]) == (r[2] - r[0]))
    root.destroy()


def main():
    test_delete_box_keeps_replacement()
    test_move_box_keeps_erased()
    test_resize_box_keeps_erased()
    test_undo_can_restore_original()
    test_anchor_resize_and_long_press_move()
    test_close_button_rules()
    test_empty_box_not_erased()
    test_edge_midpoint_resize()
    print(f"\n锚点/长摁移动/叉号/替换持久化 自测：通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：" + "；".join(FAIL))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    print("选框锚点 / 长摁移动 / 右上角叉号 / 替换结果持久化 自测通过 ✓")
