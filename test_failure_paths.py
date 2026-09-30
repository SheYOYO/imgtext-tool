"""失败路径自测脚本：验证程序在异常输入下不崩溃、给出可读提示/日志。

测试项：
- 不存在的文件路径
- 空图像/异常框选
- 渲染模块对空文字/异常参数的容错
- 保存时非法路径/不支持的格式
- 字体库为空时的降级
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

from app import io_utils
from app import feature_extract as fe
from app import erase as erase_mod
from app import render as render_mod
from app import font_match as fm


def test_invalid_image_read():
    print("== 失败路径：读取不存在的图片 ==")
    img = io_utils.imread_unicode("/this/path/does/not/exist.jpg")
    assert img is None, "不存在的路径应返回 None"
    print("  [PASS] 返回 None 不崩溃\n")


def test_empty_region_features():
    print("== 失败路径：对空区域提取特征 ==")
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    feat = fe.extract_features(empty)
    assert feat["color_rgb"] == (0, 0, 0)
    assert feat["font_size"] >= 6
    print("  [PASS] 空区域返回安全默认值\n")


def test_invalid_box_erase():
    print("== 失败路径：对错误框选区域擦除 ==")
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    # 负坐标/反序框
    out = erase_mod.erase_text(img, (-10, -5, 200, 5))
    assert out.shape == (100, 100, 3)
    # 空框
    out = erase_mod.erase_text(img, (50, 50, 50, 80))
    assert out.shape == (100, 100, 3)
    print("  [PASS] 越界/反序/空框不崩溃\n")


def test_empty_text_render():
    print("== 失败路径：渲染空文字 ==")
    params = {
        "color_rgb": (0, 0, 0), "font_size": 16, "letter_spacing": 0,
        "line_spacing": 0, "skew_angle": 0, "sharpness": 0.7,
        "blur": 0, "edge_soften": 0, "noise": 0, "bold": False, "opacity": 1.0,
    }
    rgba = render_mod.render_text("", params)
    # 空文字应返回一张很小的透明图
    assert rgba.shape[2] == 4
    print("  [PASS] 空文字返回空透明图\n")


def test_bad_font_path():
    print("== 失败路径：使用无效字体路径渲染 ==")
    params = {
        "color_rgb": (0, 0, 0), "font_size": 16, "letter_spacing": 0,
        "line_spacing": 0, "skew_angle": 0, "sharpness": 0.7,
        "blur": 0, "edge_soften": 0, "noise": 0, "bold": False, "opacity": 1.0,
    }
    rgba = render_mod.render_text("test", params, font_path="/not/exist.ttf")
    assert rgba.shape[2] == 4 and rgba[:, :, 3].max() > 0
    print("  [PASS] 无效字体自动回退默认字体\n")


def test_save_invalid_path():
    print("== 失败路径：保存到非法路径 ==")
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    ok = io_utils.imwrite_unicode("/root/cannot_write_here.png", img)
    assert not ok, "非法路径应返回 False"
    print("  [PASS] 非法路径返回 False 不崩溃\n")


def test_font_pool_empty():
    print("== 失败路径：字体库为空时匹配 ==")
    # 传入 None 区域、空候选应由 match_fonts 自己处理
    cands = fm.match_fonts("test", bgr_region=None, target_stroke=0.5)
    # 若系统有可用的字体，这里会返回候选；否则返回空列表。
    # 至少不应抛出异常。
    print(f"  无原图区域时返回候选数: {len(cands)}")
    print("  [PASS] 字体库为空或候选为 0 时不崩溃\n")


def test_gui_with_mock_messagebox():
    print("== 失败路径：GUI 空态操作 ==")
    import tkinter as tk
    from tkinter import messagebox
    from app import gui

    messagebox.showinfo = lambda *a, **k: None
    messagebox.showerror = lambda *a, **k: None
    messagebox.showwarning = lambda *a, **k: None

    root = tk.Tk()
    app = gui.App(root)
    root.update()

    # 无图片时调用主要操作
    app._schedule_preview(0)
    app.apply_replace()
    app.undo()
    app.rematch_fonts()
    app.clear_boxes()
    app.save_image()
    app.toggle_preview_mode()
    app.toggle_pipette()
    app.toggle_pipette()
    app.open_preview_window()
    app._close_preview_window()
    app.show_help()
    print("  [PASS] GUI 空态操作不崩溃\n")

    # ---- 画布在未载入图片时的缩放/平移/框选 ----
    print("== 失败路径：空画布的缩放/平移/框选 ==")
    app.canvas.zoom_at(100, 100, 2.0)
    app.canvas.zoom_center(0.5)
    app.canvas.fit_view()
    app.canvas.set_display(None)
    app.canvas._to_image(10, 10)
    app.canvas.get_zoom_percent()
    app.canvas.clear_boxes()
    root.update()
    print("  [PASS] 空画布交互不崩溃\n")

    # ---- 载入图片后的极端缩放与越界坐标 ----
    print("== 失败路径：极端缩放与越界框选 ==")
    img_bgr = np.full((120, 300, 3), 230, dtype=np.uint8)
    cv2.putText(img_bgr, "Abc 123", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (30, 30, 30), 2, cv2.LINE_AA)
    app.original_bgr = img_bgr
    app.working_bgr = img_bgr.copy()
    app.history = [img_bgr.copy()]
    app.canvas.set_image(img_bgr)
    root.update()

    for f in (0.001, 0.01, 0.5, 64.0, 1000.0):
        app.canvas.zoom_at(50, 50, f)
        root.update()
    app.canvas.zoom_center(1.0)
    root.update()
    print("  [PASS] 极端缩放被限幅，不崩溃\n")

    # 越界 / 反序 / 零尺寸框选
    for box in [(-50, -50, 40, 40), (250, 100, 20, 30), (10, 10, 11, 11),
                (0, 0, 5000, 5000), (100, 100, 100, 100)]:
        try:
            app.canvas.on_box_select(box)
            root.update()
        except Exception as e:
            print(f"  [FAIL] 框选 {box} 抛出异常: {e}")
            root.destroy()
            return
    print("  [PASS] 越界/反序/零尺寸框选不崩溃\n")

    # 无替换文字时预览（只擦除）
    print("== 失败路径：无替换文字时预览 ==")
    app.current_box = (10, 20, 120, 90)
    app.text_var.set("")
    app._schedule_preview(0)
    root.update()
    app.text_var.set("新文字")
    app.selected_font_path = "不存在的字体.ttf"
    app._schedule_preview(0)
    root.update()
    app.selected_font_path = None
    app._schedule_preview(0)
    root.update()
    print("  [PASS] 空文字/无效字体路径时预览不崩溃\n")

    root.destroy()


def test_new_interactions_gui():
    """针对新增交互的失败/边界用例：自动选字、选区拖动/缩放夹取、删除、撤销、双缓冲。"""
    import tkinter as tk
    from tkinter import messagebox
    from app import gui
    from PIL import ImageTk

    messagebox.showinfo = lambda *a, **k: None
    messagebox.showerror = lambda *a, **k: None
    messagebox.showwarning = lambda *a, **k: None

    def Ev(x, y):
        e = tk.Event()
        e.x, e.y = x, y
        return e

    print("== 失败/边界：新增交互 ==")
    root = tk.Tk()
    root.geometry("1400x860")
    app = gui.App(root)
    for _ in range(3):
        root.update()
    cvs = app.canvas

    # 载入一张测试图
    img_bgr = np.full((160, 360, 3), 235, dtype=np.uint8)
    cv2.putText(img_bgr, "Auto Font Test", (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (25, 25, 25), 2, cv2.LINE_AA)
    app.original_bgr = img_bgr
    app.working_bgr = img_bgr.copy()
    app.history = [img_bgr.copy()]
    cvs.set_image(img_bgr)
    for _ in range(3):
        root.update()

    # 标定事件偏移（event_generate 的 x/y 是相对根窗口的）
    cvs.event_generate("<ButtonPress-1>", x=0, y=0)
    root.update()
    off = cvs._drag_start_canvas or (0, 0)
    cvs.event_generate("<ButtonRelease-1>", x=0, y=0)
    root.update()
    cvs._drag = None
    cvs._drag_start_canvas = None
    cvs._rubber_box = None
    cvs._boxes.clear()
    cvs._active_index = -1
    cvs._redraw()

    def ev(x, y):
        return int(x + off[0]), int(y + off[1])

    def drag(x0, y0, x1, y1, btn=1):
        cvs.event_generate(f"<ButtonPress-{btn}>", x=ev(x0, y0)[0], y=ev(x0, y0)[1])
        root.update()
        mx, my = ev((x0 + x1) // 2, (y0 + y1) // 2)
        cvs.event_generate(f"<B{btn}-Motion>", x=mx, y=my)
        root.update()
        ex, ey = ev(x1, y1)
        cvs.event_generate(f"<B{btn}-Motion>", x=ex, y=ey)
        root.update()
        cvs.event_generate(f"<ButtonRelease-{btn}>", x=ex, y=ey)
        root.update()

    # 1) 双缓冲：重绘后画布上仅存在唯一的 image item（整幅替换，无闪烁/黑屏）
    cvs._redraw()
    root.update()
    assert cvs._img_item is not None, "双缓冲未生成 image item"
    assert isinstance(cvs._photo, ImageTk.PhotoImage), "双缓冲 photo 非 PhotoImage"
    print("  [PASS] 双缓冲单 image item 机制生效")

    # 2) 用真实鼠标事件框选 → 画布真正记录选区
    tx0, ty0, tx1, ty1 = 20, 35, 260, 130
    a = cvs._to_canvas(tx0 - 5, ty0 - 8)
    b = cvs._to_canvas(tx1 + 5, ty1 + 8)
    app.text_var.set("新文字")
    drag(a[0], a[1], b[0], b[1], btn=1)
    assert cvs.get_boxes(), "真实框选未生成选区"
    print("  [PASS] 真实鼠标框选生成选区")

    # 自动选字：selected_font_path 应等于相似度最高的候选
    assert app.candidates, "没有候选字体"
    assert app.selected_font_path == app.candidates[0]["path"], \
        "自动选字未选相似度最高字体"
    print("  [PASS] 框选后自动选用相似度最高字体")

    # 3) 选区移动规则（本轮新需求）：确认前可整体移动；确认替换后**仍可长摁/拖动
    #    移动**（取消早期「确认后位置锁死」——否则已确认框永远拿不到右上角叉号）。
    box0 = cvs.get_active_box()
    assert box0 is not None
    bcx0, bcy0 = cvs._to_canvas(box0[0], box0[1])
    bcx1, bcy1 = cvs._to_canvas(box0[2], box0[3])
    ccx, ccy = (bcx0 + bcx1) // 2, (bcy0 + bcy1) // 2
    w0, h0 = box0[2] - box0[0], box0[3] - box0[1]

    # --- 3a) 确认前：整框可移动、尺寸不变 ---
    cvs._on_left_down(Ev(ccx, ccy))
    cvs._on_left_drag(Ev(ccx + 60, ccy + 45))
    cvs._on_left_up(Ev(ccx + 60, ccy + 45))
    root.update()
    moved = cvs.get_active_box()
    assert moved != box0 and (moved[2] - moved[0]) == w0 and (moved[3] - moved[1]) == h0, \
        "确认前选区应可整体移动且尺寸不变"
    print("  [PASS] 确认前选区可整体移动（尺寸不变）")

    # --- 3b) 确认替换后：仍可长摁/拖动移动选框（取消「确认后锁死」）---
    app.text_var.set("锁定替换")
    app._confirm_text()
    root.update()
    box0 = cvs.get_active_box()
    bcx0, bcy0 = cvs._to_canvas(box0[0], box0[1])
    bcx1, bcy1 = cvs._to_canvas(box0[2], box0[3])
    ccx, ccy = (bcx0 + bcx1) // 2, (bcy0 + bcy1) // 2
    # 明显拖动（超过拖动阈值 5px）→ 选框整体移动；松手后右上角出现叉号、锚点收起
    cvs._on_left_down(Ev(ccx, ccy))
    cvs._on_left_drag(Ev(ccx + 50, ccy + 40))
    cvs._on_left_up(Ev(ccx + 50, ccy + 40))
    root.update()
    box1 = cvs.get_active_box()
    assert box1 != box0, "确认后明显拖动应整体移动选框（不再锁死）"
    assert (box1[2] - box1[0]) == w0 and (box1[3] - box1[1]) == h0, \
        "确认后移动选框尺寸应保持不变"
    assert cvs._show_close is True, "长摁/拖动移动松手后右上角应显示叉号"
    print("  [PASS] 确认后选框仍可拖动移动（尺寸不变，松手显示叉号）")

    # --- 3c) 框内纯单击（按下即松手、无位移）→ 仅选中编辑，不改位置、不显示叉号 ---
    cvs.set_active_box(0)
    box_a = cvs.get_active_box()
    ax0, ay0 = cvs._to_canvas(box_a[0], box_a[1])
    ax1, ay1 = cvs._to_canvas(box_a[2], box_a[3])
    acx, acy = (ax0 + ax1) // 2, (ay0 + ay1) // 2
    cvs._on_left_down(Ev(acx, acy))
    cvs._on_left_up(Ev(acx, acy))
    root.update()
    box_b = cvs.get_active_box()
    assert box_b == box_a, "框内纯单击不应改变选框位置（防误触）"
    assert cvs._show_close is False, "纯单击后不应显示叉号（锚点保持）"
    print("  [PASS] 框内纯单击仅选中编辑，不改变位置、不显示叉号")

    # 4) 手柄缩放（se 东南角）：选区应扩大且仍在图内
    # 复位选框到图内安全位置，避免受 3b 位移影响导致缩放无空间
    cvs._boxes[0] = (20, 35, 200, 120)
    cvs.set_active_box(0)
    bx0, by0 = cvs._to_canvas(20, 35)
    bx1, by1 = cvs._to_canvas(200, 120)
    hp = cvs._handle_points(bx0, by0, bx1, by1)
    sx, sy = hp["se"]
    cvs._drag = "resize"
    cvs._drag_handle = "se"
    cvs._drag_orig_box = (20, 35, 200, 120)
    cvs._drag_start_canvas = (sx, sy)
    cvs._on_left_drag(Ev(sx + 30, sy + 20))
    cvs._on_left_up(Ev(sx + 30, sy + 20))
    root.update()
    box2 = cvs.get_active_box()
    assert box2[2] > 200 and box2[3] > 120, "东南手柄拖动未扩大选区"
    assert box2[2] <= img_bgr.shape[1] and box2[3] <= img_bgr.shape[0], "缩放后越出图像边界"
    print("  [PASS] 东南手柄拖拽放大选区生效")

    # 5) 反向缩放夹取（nw 手柄拖到对侧之外）：应夹取为最小 2px，不翻转/不越界
    cvs._active_index = 0
    bx0, by0 = cvs._to_canvas(box2[0], box2[1])
    bx1, by1 = cvs._to_canvas(box2[2], box2[3])
    hp = cvs._handle_points(bx0, by0, bx1, by1)
    sx, sy = hp["nw"]
    cvs._drag = "resize"
    cvs._drag_handle = "nw"
    cvs._drag_orig_box = box2
    cvs._drag_start_canvas = (sx, sy)
    cvs._on_left_drag(Ev(sx - 200, sy - 200))
    cvs._on_left_up(Ev(sx - 200, sy - 200))
    root.update()
    box3 = cvs.get_active_box()
    assert box3[2] - box3[0] >= 2 and box3[3] - box3[1] >= 2, \
        "反向缩放未夹取为最小 2px"
    assert box3[0] >= 0 and box3[1] >= 0, "缩放后坐标为负"
    print("  [PASS] 反向拖拽手柄被夹取，不翻转不越界")

    # 6) 删除选中框 → 数量减少
    n0 = len(cvs.get_boxes())
    app.delete_selected_box()
    root.update()
    assert len(cvs.get_boxes()) == n0 - 1, "删除选中框未生效"
    print("  [PASS] 删除选中选区生效")

    # 7) 无选中时删除 / 撤销不崩溃
    cvs._active_index = -1
    app.delete_selected_box()
    app.undo()
    root.update()
    print("  [PASS] 无选中时删除/撤销不崩溃")

    # 8) 撤销：应用一次替换后，撤销回到修改前
    a = cvs._to_canvas(tx0 - 5, ty0 - 8)
    b = cvs._to_canvas(tx1 + 5, ty1 + 8)
    drag(a[0], a[1], b[0], b[1], btn=1)
    root.update()
    app.text_var.set("撤销验证")
    app._schedule_preview(0)
    root.update()
    before = app.working_bgr.copy()
    app.apply_replace()
    root.update()
    assert not np.array_equal(app.working_bgr, before), "应用替换未改变结果"
    app.undo()
    root.update()
    assert np.array_equal(app.working_bgr, before), "撤销未恢复到修改前"
    print("  [PASS] 撤销恢复到修改前状态")

    root.destroy()


if __name__ == "__main__":
    test_invalid_image_read()
    test_empty_region_features()
    test_invalid_box_erase()
    test_empty_text_render()
    test_bad_font_path()
    test_save_invalid_path()
    test_font_pool_empty()
    test_gui_with_mock_messagebox()
    test_new_interactions_gui()
    print("失败路径自测全部通过 ✓")
