"""擦除严格限制在选框内 + 只在「新建 / 移动」两个时机生效 回归自测。

佘先生本轮明确的三条：

1. **消除原图文字的效果严格限制在选框内**：擦除范围就是选框矩形本身，
   不再取「原文字真实墨迹 ∪ 选框 ∪ 替换文字占位」的并集，框外的上下文
   文字/纹理一个像素都不许动。
2. **只在两个时机生效**：① 新建选框（松手即擦）；② 移动选框。
   缩放（拖 8 锚点改尺寸）、点选、重选、预览、应用等一律不擦除。
3. **复制选框时不擦除**：副本是程序自动摆到旁边的，落点底下盖着的是别处
   的原图内容，沿用「新建即擦除」会凭空抹掉一块与本次替换无关的画面。

验证口径：原文字画成**红色**，用「框内红 = 0 / 框外红保留 / 框外像素逐字节
不变」三重判据把「严格限框内」钉死；用 mode="resize" 与 mode="move" 的 A/B
对比把「只有移动才擦」钉死。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app import config

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_output")
os.makedirs(OUT, exist_ok=True)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))


# ============================================================ 测试图构建
def _font(size=40):
    pool = config.resolve_font_pool()
    fp = next((f["path"] for f in pool if f.get("cn")), None)
    return ImageFont.truetype(fp, size) if fp else ImageFont.load_default()


def _ink_bbox(bgr, bg=(240, 240, 240)):
    diff = np.abs(bgr.astype(int) - np.array(bg[::-1], dtype=int)).sum(axis=2)
    ys, xs = np.where(diff > 40)
    if ys.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _to_bgr(img):
    import cv2
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _make_two_blocks(text="原始文字六字", size=40):
    """画两段**相同**的红字：第二段精确落在「复制选框」的落点（原框右侧 gap 处）。

    复制落点由 `copy_selected_box` 决定：nx0 = x0 + w + gap，gap = max(8, min(w,h)//4)。
    所以先单独画第一段量出墨迹包围盒，再按同一公式摆第二段——这样「复制选框」
    一旦误擦，第二段红字会立刻少掉，测试不会落空。
    """
    font = _font(size)
    # 第一次：只画第一段，量取墨迹包围盒
    tmp = Image.new("RGB", (1, 1), (240, 240, 240))
    bb = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    W1, H1 = bb[2] + 160, bb[3] + 160
    img1 = Image.new("RGB", (W1, H1), (240, 240, 240))
    ImageDraw.Draw(img1).text((40, 60), text, font=font, fill=(255, 0, 0))
    x0, y0, x1, y1 = _ink_bbox(_to_bgr(img1))
    w, h = x1 - x0, y1 - y0
    gap = max(8, min(w, h) // 4)
    # 第二次：两段一起画（第二段起点 = x0 + w + gap）
    W = x0 + 2 * w + gap + 120
    H = y0 + h + 120
    img = Image.new("RGB", (W, H), (240, 240, 240))
    d = ImageDraw.Draw(img)
    d.text((40, 60), text, font=font, fill=(255, 0, 0))
    d.text((40 + w + gap, 60), text, font=font, fill=(255, 0, 0))
    return _to_bgr(img), (x0, y0, x1, y1), (x0 + w + gap, y0, x0 + 2 * w + gap, y1)


def _setup(text="原始文字六字", size=40):
    import tkinter as tk
    from tkinter import messagebox
    from app import gui
    messagebox.showinfo = lambda *a, **k: None
    messagebox.showerror = lambda *a, **k: None
    root = tk.Tk()
    root.withdraw()
    app = gui.App(root)
    bgr, box_a, box_b = _make_two_blocks(text, size)
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()
    app.history = [{"bgr": bgr.copy(), "box_edits": {}}]
    app.pending_result = None
    app.current_box = None
    app.source_path = None
    app.canvas.set_image(bgr)
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
    """模拟一次「拖拽框选」新建选框（图像坐标 → 画布坐标）。"""
    cx0, cy0 = cvs._to_canvas(box[0], box[1])
    cx1, cy1 = cvs._to_canvas(box[2], box[3])
    cvs._on_left_down(_evt(cx0, cy0))
    cvs._on_left_drag(_evt(cx1, cy1))
    cvs._on_left_up(_evt(cx1, cy1))


def _long_press_move(cvs, x, y, dx, dy):
    """长摁（≥260ms 语义）+ 拖动 → 整体移动选框。"""
    cvs._on_left_down(_evt(x, y))
    cvs._on_long_press()
    cvs._on_left_drag(_evt(x + dx, y + dy))
    cvs._on_left_up(_evt(x + dx, y + dy))


def _red_mask(bgr):
    b = bgr[:, :, 0].astype(int); g = bgr[:, :, 1].astype(int); r = bgr[:, :, 2].astype(int)
    return (r > 120) & (g < 90) & (b < 90)


def _red_count(bgr):
    return int(_red_mask(bgr).sum())


def _red_inside(bgr, box):
    x0, y0, x1, y1 = [int(v) for v in box]
    return int(_red_mask(bgr)[max(0, y0):y1, max(0, x0):x1].sum())


def _outside_equal(a, b, box):
    """两张图「选框以外」的像素是否逐字节一致（框外零改动的最硬证据）。"""
    return _outside_equal_multi(a, b, [box])


def _outside_equal_multi(a, b, boxes):
    """同上，但允许一次排除多个矩形（如移动的「旧位置 ∪ 新位置」）。

    移动一次会擦两处：拖动首帧固化旧位置、松手时擦新位置。旧位置本已擦除，
    inpaint 再跑一遍会有像素级微差（属正常），所以「框外零改动」的正确口径是
    **旧框 ∪ 新框 之外**逐字节一致。
    """
    if a is None or b is None or a.shape != b.shape:
        return False
    m = np.ones(a.shape[:2], dtype=bool)
    for box in boxes:
        x0, y0, x1, y1 = [int(v) for v in box]
        m[max(0, y0):y1, max(0, x0):x1] = False
    return bool(np.array_equal(a[m], b[m]))


# ============================================ 1. 擦除范围严格 = 选框矩形
def test_erase_region_is_exactly_box():
    print("\n== 1. _erase_region_for 严格等于选框（不外扩、夹取到图内）==")
    root, app, box_a, box_b = _setup()
    app.current_box = box_a
    r = app._erase_region_for(box_a)
    check("擦除范围 == 选框本身", r == tuple(int(v) for v in box_a), f"{r} vs {box_a}")

    # 即便传入「替换文字占位」也不再参与并集（旧实现会外扩）
    far = (box_a[0] - 200, box_a[1] - 200, box_a[0] - 100, box_a[1] - 100)
    r2 = app._erase_region_for(box_a, far)
    check("替换文字占位不再并入擦除范围", r2 == tuple(int(v) for v in box_a), f"{r2}")

    H, W = app.original_bgr.shape[:2]
    r3 = app._erase_region_for((-50, -60, W + 500, H + 500))
    check("越界框被夹取到图像范围内", r3 == (0, 0, W, H), f"{r3}")
    r4 = app._erase_region_for((300, 200, 100, 80))
    check("反序框被归一化（左上/右下归位）", r4 == (100, 80, 300, 200), f"{r4}")
    check("框为 None 时安全返回 None", app._erase_region_for(None) is None)
    root.destroy()


# ============================================ 2. 新建选框：框内擦 / 框外不动
def test_create_erases_only_inside():
    print("\n== 2. 新建选框：框内原文字清零、框外逐字节不动 ==")
    root, app, box_a, box_b = _setup()
    before_out_red = _red_count(app.original_bgr) - _red_inside(app.original_bgr, box_a)
    check("正控：框外本来就有红字（第二段）", before_out_red > 100, f"框外红={before_out_red}")

    _create_box(app.canvas, box_a)
    _flush(app)
    check("新建后框内原文字已消失（框内红 = 0）",
          _red_inside(app.working_bgr, box_a) == 0,
          f"框内红={_red_inside(app.working_bgr, box_a)}")
    check("新建后框外像素与原图逐字节一致（严格限制在选框内）",
          _outside_equal(app.working_bgr, app.original_bgr, box_a))
    check("第二段红字完好无损（没被顺手擦掉）",
          _red_inside(app.working_bgr, box_b) > 100,
          f"第二段红={_red_inside(app.working_bgr, box_b)}")
    root.destroy()


# ============================================ 3. 移动选框：擦除生效且仍限框内
def test_move_erases_inside_only():
    print("\n== 3. 移动选框：新位置框内被擦除、框外不动 ==")
    root, app, box_a, box_b = _setup()
    _create_box(app.canvas, box_a)
    _flush(app)
    # 向右下长摁拖动，让选框落到「两段文字之间/第二段上」的新位置
    x0, y0, x1, y1 = app.canvas.get_active_box()
    cxs, cys = app.canvas._to_canvas((x0 + x1) / 2, (y0 + y1) / 2)
    before = app.working_bgr.copy()
    _long_press_move(app.canvas, cxs, cys, 70, 55)
    _flush(app)
    new_box = app.canvas.get_active_box()
    check("选框确实移动了", new_box != box_a, f"{box_a} → {new_box}")
    # 旧位置（已擦）之外，只有新框内允许变化
    changed = ~np.array_equal(app.working_bgr, before)
    check("移动后工作图确有变化（新位置被擦除）",
          not np.array_equal(app.working_bgr, before))
    # 移动会擦两处（旧位置 + 新位置），因此「框外零改动」= 两者并集之外逐字节一致
    check("移动后（旧框∪新框）之外逐字节不变（严格限制在选框内）",
          _outside_equal_multi(app.working_bgr, before, [box_a, new_box]))
    check("移动后新位置框内红字被擦除", _red_inside(app.working_bgr, new_box) == 0,
          f"新框内红={_red_inside(app.working_bgr, new_box)}")
    check("changed 掩码非空（正控）", bool(changed))
    root.destroy()


# ============================================ 4. 缩放不擦除 / 移动才擦除（A/B）
def test_resize_does_not_erase_but_move_does():
    print("\n== 4. 缩放选框不擦除，移动选框才擦除（A/B 对比）==")
    root, app, box_a, box_b = _setup()
    _create_box(app.canvas, box_a)
    _flush(app)
    # 覆盖到「第二段文字」的一块从未被擦过的区域
    target = (box_b[0] - 4, box_b[1] - 4, box_b[2] + 4, box_b[3] + 4)
    check("正控：目标区域确实有红字", _red_inside(app.working_bgr, target) > 100,
          f"目标区红={_red_inside(app.working_bgr, target)}")
    # 模拟用户改了字（user_changed=True），移动保护放开
    app.text_var.set("完全不同的替换文字")

    before = app.working_bgr.copy()
    app.on_box_edit_start(0, target, "resize")
    _flush(app)
    check("缩放（mode=resize）不触发擦除",
          np.array_equal(app.working_bgr, before))

    app.on_box_edit_start(0, target, "move")
    _flush(app)
    check("移动（mode=move）触发擦除（正控）",
          not np.array_equal(app.working_bgr, before))
    check("移动擦除后框内红字清零", _red_inside(app.working_bgr, target) == 0,
          f"框内红={_red_inside(app.working_bgr, target)}")
    check("移动擦除后框外仍逐字节不变",
          _outside_equal(app.working_bgr, before, target))
    root.destroy()


# ============================================ 5. 复制选框不擦除
def test_copy_box_does_not_erase():
    print("\n== 5. 复制选框：落点底字不被擦除 ==")
    root, app, box_a, box_b = _setup()
    _create_box(app.canvas, box_a)
    _flush(app)
    before = app.working_bgr.copy()
    check("正控：复制落点处确实盖着第二段红字",
          _red_inside(before, box_b) > 100, f"落点红={_red_inside(before, box_b)}")

    app.copy_selected_box()
    _flush(app)
    boxes = app.canvas.get_boxes()
    check("复制后选框数量 +1", len(boxes) == 2, f"boxes={boxes}")
    new_box = app.canvas.get_active_box()
    check("复制体落点确实压在第二段文字上（测试不落空）",
          _red_inside(before, new_box) > 50, f"复制框内红={_red_inside(before, new_box)}")
    check("复制选框**没有**擦除任何像素（工作图与复制前逐字节一致）",
          np.array_equal(app.working_bgr, before))
    check("复制落点红字完好", _red_inside(app.working_bgr, box_b) > 100,
          f"落点红={_red_inside(app.working_bgr, box_b)}")
    root.destroy()


# ============================================ 6. 复制体被移动时仍按「移动」规则擦除
def test_copy_then_move_erases():
    print("\n== 6. 复制体被用户移动时：仍按「移动」规则擦除（规则不因复制而失效）==")
    root, app, box_a, box_b = _setup()
    _create_box(app.canvas, box_a)
    _flush(app)
    app.copy_selected_box()
    _flush(app)
    new_box = app.canvas.get_active_box()
    app.text_var.set("副本替换文字")        # 与 OCR 原文不同 → 移动保护放开
    x0, y0, x1, y1 = new_box
    cxs, cys = app.canvas._to_canvas((x0 + x1) / 2, (y0 + y1) / 2)
    _long_press_move(app.canvas, cxs, cys, 25, 30)
    _flush(app)
    moved = app.canvas.get_active_box()
    check("复制体确实被移动了", moved != new_box, f"{new_box} → {moved}")
    check("移动后旧位置框内红字被擦除", _red_inside(app.working_bgr, new_box) == 0,
          f"旧位红={_red_inside(app.working_bgr, new_box)}")
    check("移动后新位置框内红字被擦除", _red_inside(app.working_bgr, moved) == 0,
          f"新位红={_red_inside(app.working_bgr, moved)}")
    root.destroy()


# ============================================ 7. 失败路径（绝不崩溃、绝不误擦）
def test_failure_paths():
    print("\n== 7. 失败路径：无图 / 空框 / 反序框 / 纯背景 都不崩溃、不误擦 ==")
    root, app, box_a, box_b = _setup()
    # 7a) 框为 None：两个入口都必须安全返回
    before = app.working_bgr.copy()
    try:
        app.on_box_create(0, None)
        app.on_box_edit_start(0, None, "move")
        ok = True
    except Exception as e:
        ok = False
        detail = str(e)
    check("框为 None 时不崩溃且工作图不变",
          ok and np.array_equal(app.working_bgr, before),
          detail if not ok else "")

    # 7b) 无图状态（original_bgr=None）：不崩溃
    try:
        saved = app.original_bgr
        app.original_bgr = None
        app.on_box_create(0, box_b)
        app.on_box_edit_start(0, box_b, "move")
        ok2 = True
    except Exception:
        ok2 = False
    finally:
        app.original_bgr = saved
    check("无图（original_bgr=None）时不崩溃", ok2)

    # 7c) 反序/零面积框：不崩溃、不动工作图
    try:
        app.on_box_create(0, (300, 300, 100, 100))
        app.on_box_edit_start(0, (200, 200, 200, 200), "move")
        ok3 = np.array_equal(app.working_bgr, before)
    except Exception as e:
        ok3 = False
        detail3 = str(e)
    check("反序/零面积框不崩溃且不改动工作图", ok3, detail3 if not ok3 else "")

    # 7d) 纯背景区域新建：不误擦（与原始图一致且近乎单色）
    h, w = app.original_bgr.shape[:2]
    _create_box(app.canvas, (w - 30, h - 30, w - 6, h - 6))
    _flush(app)
    check("纯背景新建选框不误擦", np.array_equal(app.working_bgr, before))
    root.destroy()


def main():
    test_erase_region_is_exactly_box()
    test_create_erases_only_inside()
    test_move_erases_inside_only()
    test_resize_does_not_erase_but_move_does()
    test_copy_box_does_not_erase()
    test_copy_then_move_erases()
    test_failure_paths()
    print(f"\n擦除严格限框内 / 仅新建·移动生效 自测：通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print("  - " + f)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
