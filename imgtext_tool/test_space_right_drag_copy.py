"""空格 + 右键拖拽复制选框 专项自测（佘先生本轮两条要求）。

要求一：**复制选框时不需要有消除原图文字的效果** —— 复制既不是「新建选框」也不是
「移动选框」，不触发 on_box_create / on_box_edit_start，落点下方盖着的原图内容
一个像素都不许动（判据：落点区域的红字复制前后数量不变）。

要求二：**复制选框的触发条件改为「按住空格 + 鼠标右键在选框中间部分按下并拖动」**。
   · 右键点在选框中间 → 先选中该框；按住右键拖动 → 松手时把副本放到拖到的位置。
   · 不按空格时右键仍是「平移画面」；点在空白/锚点/叉号也不触发复制。

成功路径：
  1. 空格+右键拖到目标位置 → 新增一个框、位置≈落点、文字/字体/特征随原框复制
  2. 复制全程不擦除（落点红字保留、且逐字节不变）
  3. 右键点非活动框 → 先选中它再复制
  4. 落点拖出图像外 → 夹在图像内
失败 / 边界路径：
  5. 不按空格右键拖动 → 只平移画面，不复制、不新增框
  6. 空格 + 右键点空白 → 平移，不复制
  7. 未上传图片 / 无选框 → 不崩
  8. 焦点在输入框按空格 → 不进入复制模式（不误伤打字）
  9. 空格松开后状态复位
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app import config

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))


# ============================================================ 测试图构建
def _font(size=40):
    pool = config.resolve_font_pool()
    fp = next((f["path"] for f in pool if f.get("cn")), None)
    return ImageFont.truetype(fp, size) if fp else ImageFont.load_default()


def _to_bgr(img):
    import cv2
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _ink_bbox(bgr, bg=(240, 240, 240)):
    diff = np.abs(bgr.astype(int) - np.array(bg[::-1], dtype=int)).sum(axis=2)
    ys, xs = np.where(diff > 40)
    if ys.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _red_count(bgr, box=None):
    """统计红色文字像素数（BGR：红 = R 高、G/B 低）。"""
    sub = bgr[box[1]:box[3], box[0]:box[2]] if box else bgr
    b, g, r = sub[:, :, 0].astype(int), sub[:, :, 1].astype(int), sub[:, :, 2].astype(int)
    return int(((r > 120) & (g < 110) & (b < 110)).sum())


def _make_two_blocks(text="原始文字六字", size=40, dx=260, dy=90):
    """画两段**相同**的红字：A（被框选）与 B（拖拽落点处）。

    B 精确摆在 A + (dx, dy) 处，所以「复制若误擦」会立刻抹掉 B 的红字，测试不落空。
    """
    font = _font(size)
    tmp = Image.new("RGB", (1, 1), (240, 240, 240))
    bb = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    W1, H1 = bb[2] + 400, bb[3] + 300
    img1 = Image.new("RGB", (W1, H1), (240, 240, 240))
    ImageDraw.Draw(img1).text((40, 60), text, font=font, fill=(255, 0, 0))
    x0, y0, x1, y1 = _ink_bbox(_to_bgr(img1))
    w, h = x1 - x0, y1 - y0
    W = max(x0 + w + dx + 200, 400)
    H = max(y0 + h + dy + 200, 300)
    img = Image.new("RGB", (W, H), (240, 240, 240))
    d = ImageDraw.Draw(img)
    d.text((40, 60), text, font=font, fill=(255, 0, 0))              # A
    d.text((40 + dx, 60 + dy), text, font=font, fill=(255, 0, 0))    # B（落点）
    box_a = (x0, y0, x1, y1)
    box_b = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
    return _to_bgr(img), box_a, box_b


def _setup(text="原始文字六字", size=40, dx=260, dy=90):
    import tkinter as tk
    from tkinter import messagebox
    from app import gui
    messagebox.showinfo = lambda *a, **k: None
    messagebox.showerror = lambda *a, **k: None
    root = tk.Tk()
    root.withdraw()
    app = gui.App(root)
    bgr, box_a, box_b = _make_two_blocks(text, size, dx, dy)
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()
    app.history = [{"bgr": bgr.copy(), "box_edits": {}}]
    app.pending_result = None
    app.current_box = None
    app.source_path = None
    app.canvas.set_image(bgr)
    app.root.update()
    return root, app, box_a, box_b


def _flush(app):
    try:
        app.root.update()
    except Exception:
        pass


def _evt(x, y):
    from types import SimpleNamespace
    return SimpleNamespace(x=x, y=y)


def _create_box(cvs, box):
    """模拟拖拽框选新建选框（图像坐标 → 画布坐标）。"""
    cx0, cy0 = cvs._to_canvas(box[0], box[1])
    cx1, cy1 = cvs._to_canvas(box[2], box[3])
    cvs._on_left_down(_evt(cx0, cy0))
    cvs._on_left_drag(_evt(cx1, cy1))
    cvs._on_left_up(_evt(cx1, cy1))


def _space_right_drag(app, src_box, target_xy):
    """【需求二】按住空格 + 右键在选框中间按下 → 拖到 target → 松手复制。

    target_xy = 目标框**左上角**图像坐标 (tx, ty)。
    起点必须落在选框中间（需求：右键点中间部分），故位移按画布像素折算。
    """
    cvs = app.canvas
    cx, cy = cvs._to_canvas((src_box[0] + src_box[2]) / 2.0,
                            (src_box[1] + src_box[3]) / 2.0)
    eff = cvs.fit_scale * cvs.zoom
    dix, diy = target_xy[0] - src_box[0], target_xy[1] - src_box[1]
    tx, ty = cx + dix * eff, cy + diy * eff
    cvs._pan_start(_evt(cx, cy))
    cvs._pan_drag(_evt(tx, ty))
    cvs._pan_end(_evt(tx, ty))


# ============================================================ 1. 成功路径：拖到落点即复制
def test_space_right_drag_copies_to_target():
    print("\n== 1. 空格 + 右键拖拽选框中间 → 副本落到拖到的位置 ==")
    root, app, box_a, box_b = _setup()
    cvs = app.canvas
    _create_box(cvs, box_a)
    _flush(app)
    n0 = len(cvs.get_boxes())
    check("新建后已有 1 个选框", n0 == 1, f"boxes={n0}")

    app.text_var.set("替换文字")
    app._confirm_text()
    _flush(app)

    # 【需求二】按下空格 → 右键拖 → 松手
    app.canvas.space_held = True
    before_b = app.working_bgr[box_b[1]:box_b[3], box_b[0]:box_b[2]].copy()
    red_b_before = _red_count(app.working_bgr, box_b)

    _space_right_drag(app, box_a, (box_b[0], box_b[1]))
    _flush(app)

    boxes = cvs.get_boxes()
    check("拖拽复制后新增了一个选框", len(boxes) == n0 + 1, f"boxes={len(boxes)}")
    if len(boxes) != n0 + 1:
        root.destroy()
        return
    new_box = boxes[-1]
    dx_err = abs(new_box[0] - box_b[0])
    dy_err = abs(new_box[1] - box_b[1])
    check("副本落在拖拽目标位置（左上角落点误差 ≤2px）",
          dx_err <= 2 and dy_err <= 2, f"new={new_box} target_tl={(box_b[0], box_b[1])}")
    check("副本尺寸与原框一致",
          (new_box[2] - new_box[0], new_box[3] - new_box[1])
          == (box_a[2] - box_a[0], box_a[3] - box_a[1]))
    # 内容随原框复制
    check("副本文字与原框一致",
          app.box_edits.get(1, {}).get("text") == "替换文字",
          f"text={app.box_edits.get(1, {}).get('text')!r}")
    check("副本带 _copied 标记（允许先拖动定位）",
          app.box_edits.get(1, {}).get("_copied") is True)

    # 【需求一】复制不擦除：落点区域红字数量不变 + 逐字节不变
    after_b = app.working_bgr[box_b[1]:box_b[3], box_b[0]:box_b[2]]
    red_b_after = _red_count(app.working_bgr, box_b)
    check("复制未擦除落点原图文字（红字数量不变）",
          red_b_after == red_b_before, f"{red_b_before} → {red_b_after}")
    check("复制未改动落点任何像素（逐字节一致）",
          np.array_equal(before_b, after_b))
    root.destroy()


# ============================================================ 2. 复制全程不擦除（正控）
def test_copy_never_erases():
    print("\n== 2. 复制选框不消除原图文字（含 Ctrl+D 自动落点路径）==")
    root, app, box_a, box_b = _setup()
    cvs = app.canvas
    _create_box(cvs, box_a)
    _flush(app)
    check("正控：新建选框确实擦掉了框内红字",
          _red_count(app.working_bgr, box_a) == 0,
          f"red={_red_count(app.working_bgr, box_a)}")

    red_before = _red_count(app.working_bgr, box_b)
    before_full = app.working_bgr.copy()
    # 自动落点路径（Ctrl+D / 工具栏）也要不擦除
    app.copy_selected_box()
    _flush(app)
    check("自动落点复制同样不擦除（红字数量不变）",
          _red_count(app.working_bgr, box_b) == red_before,
          f"{red_before} → {_red_count(app.working_bgr, box_b)}")
    # 整图层面：除「新建时擦掉的 A 区」外不应再有别的像素被改
    diff = (app.working_bgr.astype(int) != before_full.astype(int)).any(axis=2)
    outside_a = diff.copy()
    outside_a[box_a[1]:box_a[3], box_a[0]:box_a[2]] = False
    check("复制后框 A 之外没有任何像素被改动", not outside_a.any(),
          f"changed_outside={int(outside_a.sum())}")
    root.destroy()


# ============================================================ 3. 右键点非活动框 → 先选中
def test_right_click_selects_then_copies():
    print("\n== 3. 右键点在非活动选框中间 → 先选中该框再复制 ==")
    root, app, box_a, box_b = _setup(dx=300, dy=120)
    cvs = app.canvas
    ih, iw = app.original_bgr.shape[:2]
    _create_box(cvs, box_a)
    _flush(app)
    app.text_var.set("第一框")
    app._confirm_text()
    _flush(app)
    # 再建一个框（此时它是活动框）
    box2 = (box_a[0], box_a[3] + 40, box_a[2], box_a[3] + 40 + (box_a[3] - box_a[1]))
    if box2[3] >= ih - 1:      # 放不下就往上挪
        box2 = (box_a[0], max(1, box_a[1] - 40 - (box_a[3] - box_a[1])),
                box_a[2], max(1, box_a[1] - 40))
    _create_box(cvs, box2)
    _flush(app)
    app.text_var.set("第二框")
    app._confirm_text()
    _flush(app)
    check("已有 2 个选框", len(cvs.get_boxes()) == 2, f"boxes={len(cvs.get_boxes())}")
    check("当前活动框是第 2 个", cvs.get_active_index() == 1)

    # 空格 + 右键点**第 1 个**框中间（非活动）→ 应先选中它
    cvs.space_held = True
    cx, cy = cvs._to_canvas((box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0)
    cvs._pan_start(_evt(cx, cy))
    _flush(app)
    check("右键点非活动框后，该框被选中（活动框变为第 1 个）",
          cvs.get_active_index() == 0, f"active={cvs.get_active_index()}")
    # 继续拖到空白区松手 → 复制的是第 1 个框
    eff = cvs.fit_scale * cvs.zoom
    cvs._pan_drag(_evt(cx + 10 * eff, cy + 10 * eff))
    cvs._pan_end(_evt(cx + 10 * eff, cy + 10 * eff))
    _flush(app)
    boxes = cvs.get_boxes()
    check("复制出的是被右键选中的第 1 个框（文字 = 第一框）",
          len(boxes) == 3 and app.box_edits.get(2, {}).get("text") == "第一框",
          f"boxes={len(boxes)} text={app.box_edits.get(2, {}).get('text')!r}")
    root.destroy()


# ============================================================ 4. 落点夹在图像内
def test_target_clamped_inside_image():
    print("\n== 4. 拖到图像外 → 落点被夹在图像内 ==")
    root, app, box_a, box_b = _setup()
    cvs = app.canvas
    _create_box(cvs, box_a)
    _flush(app)
    ih, iw = app.original_bgr.shape[:2]
    cvs.space_held = True
    _space_right_drag(app, box_a, (iw + 500, ih + 500))   # 远远拖出右下角
    _flush(app)
    boxes = cvs.get_boxes()
    check("仍成功复制出副本", len(boxes) == 2, f"boxes={len(boxes)}")
    nb = boxes[-1]
    check("副本被夹在图像内（不越界）",
          0 <= nb[0] and 0 <= nb[1] and nb[2] <= iw and nb[3] <= ih,
          f"new={nb} img=({iw}x{ih})")
    root.destroy()


# ============================================================ 5. 失败路径
def test_no_space_keeps_panning():
    print("\n== 5. 不按空格时右键仍是平移画面（不复制）==")
    root, app, box_a, box_b = _setup()
    cvs = app.canvas
    _create_box(cvs, box_a)
    _flush(app)
    n0 = len(cvs.get_boxes())
    px0, py0 = cvs.pan_x, cvs.pan_y
    cx, cy = cvs._to_canvas((box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0)
    cvs._pan_start(_evt(cx, cy))
    cvs._pan_drag(_evt(cx + 30, cy + 20))
    cvs._pan_end(_evt(cx + 30, cy + 20))
    _flush(app)
    check("未按空格：不复制（框数不变）", len(cvs.get_boxes()) == n0,
          f"boxes={len(cvs.get_boxes())}")
    check("未按空格：画面被平移",
          abs(cvs.pan_x - px0) > 0 or abs(cvs.pan_y - py0) > 0,
          f"pan=({cvs.pan_x},{cvs.pan_y}) vs ({px0},{py0})")
    root.destroy()


def test_space_right_on_blank_pans():
    print("\n== 6. 空格 + 右键点空白 → 平移，不复制 ==")
    root, app, box_a, box_b = _setup()
    cvs = app.canvas
    _create_box(cvs, box_a)
    _flush(app)
    n0 = len(cvs.get_boxes())
    px0, py0 = cvs.pan_x, cvs.pan_y
    cvs.space_held = True
    # 找一个远离所有框的空白点
    bx, by = cvs._to_canvas(5, 5)
    cvs._pan_start(_evt(bx, by))
    cvs._pan_drag(_evt(bx + 25, by + 15))
    cvs._pan_end(_evt(bx + 25, by + 15))
    _flush(app)
    check("空白右键：不复制（框数不变）", len(cvs.get_boxes()) == n0,
          f"boxes={len(cvs.get_boxes())}")
    check("空白右键：画面被平移",
          abs(cvs.pan_x - px0) > 0 or abs(cvs.pan_y - py0) > 0)
    root.destroy()


def test_no_image_no_box_no_crash():
    print("\n== 7. 未上传图片 / 无选框 → 不崩溃 ==")
    import tkinter as tk
    from tkinter import messagebox
    from app import gui
    messagebox.showinfo = lambda *a, **k: None
    messagebox.showerror = lambda *a, **k: None
    root = tk.Tk(); root.withdraw()
    app = gui.App(root)
    app.canvas.space_held = True
    try:
        app.canvas._pan_start(_evt(30, 30))
        app.canvas._pan_drag(_evt(80, 60))
        app.canvas._pan_end(_evt(80, 60))
        ok_no_img = True
    except Exception as e:
        ok_no_img = False
        print(f"    exception: {e}")
    check("未上传图片时空格+右键不崩", ok_no_img)

    # 有图但无选框
    bgr, box_a, box_b = _make_two_blocks()
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()
    app.canvas.set_image(bgr)
    app.root.update()
    app.canvas.space_held = True
    try:
        app.canvas._pan_start(_evt(30, 30))
        app.canvas._pan_drag(_evt(80, 60))
        app.canvas._pan_end(_evt(80, 60))
        ok_no_box = (len(app.canvas.get_boxes()) == 0)
    except Exception as e:
        ok_no_box = False
        print(f"    exception: {e}")
    check("无选框时空格+右键不崩、不凭空造框", ok_no_box)
    root.destroy()


def test_space_in_entry_not_captured():
    print("\n== 8. 焦点在输入框时按空格 → 不进入复制模式（不误伤打字）==")
    root, app, box_a, box_b = _setup()
    try:
        # 把焦点放进替换文字输入框（窗口隐藏时 focus_set 不生效，需先显示）
        app.root.deiconify()
        app.root.update()
        app.text_entry.focus_set()
        app.root.update()
        focused = app.root.focus_get()
        check("焦点确实落在替换文字输入框", focused is app.text_entry,
              f"focus={focused}")
        app._on_space_down()
        check("输入框内按空格不进入复制模式", app.canvas.space_held is False,
              f"space_held={app.canvas.space_held}")
        # 焦点回到画布（非输入控件）后应恢复正常——同样在可见状态下切换焦点
        app.canvas.focus_set()
        app.root.update()
        check("焦点确实回到画布", app.root.focus_get() is app.canvas,
              f"focus={app.root.focus_get()}")
        app._on_space_down()
        check("焦点在画布时按空格进入复制模式", app.canvas.space_held is True)
        app._on_space_up()
        check("松开空格后状态复位", app.canvas.space_held is False)
        app.root.withdraw()
        app.root.update()
    except Exception as e:
        check("空格键状态测试", False, f"exception: {e}")
    root.destroy()


def main():
    test_space_right_drag_copies_to_target()
    test_copy_never_erases()
    test_right_click_selects_then_copies()
    test_target_clamped_inside_image()
    test_no_space_keeps_panning()
    test_space_right_on_blank_pans()
    test_no_image_no_box_no_crash()
    test_space_in_entry_not_captured()
    print(f"\n空格+右键拖拽复制 自测：通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：" + "；".join(FAIL))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    print("空格+右键拖拽复制 / 复制不擦除 自测通过 ✓")
