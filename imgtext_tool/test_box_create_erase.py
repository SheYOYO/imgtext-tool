"""新建选框即去除底字 + 四边延长辅助线 回归自测。

覆盖佘先生本轮两条要求：

1. **新建选框后直接去除底字**：框选一片区域，被遮住的原图文字就是想去除的
   文字，**新建后立刻从工作图抹掉**；之后删除选框、移走选框都不再出现
   （只有「撤销」能恢复）。纯背景区域不误擦。
2. **四边延长辅助线**：活动选框的上/下/左/右边各引一条贯穿画布的虚线，
   方便对照周围上下文对齐。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app import config
from app import erase as erase_mod

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


def _flush(app):
    try:
        app.root.update()
    except Exception:
        pass


def _evt(x, y):
    from types import SimpleNamespace
    return SimpleNamespace(x=x, y=y)


def _create_box(cvs, box):
    """在画布上模拟一次「拖拽框选」新建选框（覆盖给定图像坐标区域）。"""
    cx0, cy0 = cvs._to_canvas(box[0], box[1])
    cx1, cy1 = cvs._to_canvas(box[2], box[3])
    cvs._on_left_down(_evt(cx0, cy0))
    cvs._on_left_drag(_evt(cx1, cy1))
    cvs._on_left_up(_evt(cx1, cy1))


def _long_press_move(cvs, x, y, dx, dy):
    cvs._on_left_down(_evt(x, y))
    cvs._on_long_press()
    cvs._on_left_drag(_evt(x + dx, y + dy))
    cvs._on_left_up(_evt(x + dx, y + dy))


def _region(img, box):
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]


def _erase_only(app, box, strength=0.5):
    """纯擦除对照基准。

    必须与生产代码用**同一个擦除范围**（原文字真实墨迹 ∪ 选框）：
    inpaint 的修复结果依赖邻域，范围一大，即使框内像素也会不同，
    若这里仍按「仅选框」构造基准就会误判成「写了 OCR 文字」。
    """
    return erase_mod.erase_text(app.original_bgr.copy(),
                                app._erase_region_for(box), strength=strength)


# ============================================================ 1. 新建即去除
def test_create_erases_immediately():
    print("\n== 1. 新建选框后底字立即被去除（无需确认/输入）==")
    root, app, box = _setup()
    cvs = app.canvas
    _create_box(cvs, box)
    _flush(app)
    check("新建后选框已存在", len(cvs.get_boxes()) == 1, f"boxes={cvs.get_boxes()}")
    check("新建后底字区域已 ≠ 原图（原文字被抹掉）",
          not np.array_equal(_region(app.working_bgr, box),
                              _region(app.original_bgr, box)))
    # 此时尚未输入任何替换文字，区域应只是「擦除」而非「被 OCR 文字填充」
    erased_ref = _erase_only(app, box)
    check("新建后区域 == 纯擦除结果（未写入 OCR/替换文字）",
          np.array_equal(_region(app.working_bgr, box), _region(erased_ref, box)))
    root.destroy()


def test_create_then_delete_keeps_erased():
    print("\n== 2. 删除选框后原文字仍不出现（区域保持已擦除）==")
    root, app, box = _setup()
    cvs = app.canvas
    _create_box(cvs, box)
    _flush(app)
    app.delete_selected_box()
    _flush(app)
    check("删除后选框数量归零", len(cvs.get_boxes()) == 0, f"boxes={cvs.get_boxes()}")
    check("删除后原文字没有复活（区域仍 ≠ 原图）",
          not np.array_equal(_region(app.working_bgr, box),
                              _region(app.original_bgr, box)))
    erased_ref = _erase_only(app, box)
    check("删除后区域仍 == 纯擦除结果（没冒出 OCR 文字）",
          np.array_equal(_region(app.working_bgr, box), _region(erased_ref, box)))
    root.destroy()


def test_create_then_move_keeps_erased():
    print("\n== 3. 移走选框后：原位置与新位置底字都不出现 ==")
    root, app, box = _setup()
    cvs = app.canvas
    _create_box(cvs, box)
    _flush(app)
    # 长摁拖动选框到新位置
    x0, y0, x1, y1 = cvs.get_active_box()
    cxs, cys = cvs._to_canvas((x0 + x1) / 2, (y0 + y1) / 2)
    _long_press_move(cvs, cxs, cys, 60, 45)
    _flush(app)
    new_box = cvs.get_active_box()
    check("长摁后选框确实移动了", new_box != box, f"{box} → {new_box}")
    check("旧位置（原框处）原文字没有复活",
          not np.array_equal(_region(app.working_bgr, box),
                              _region(app.original_bgr, box)))
    check("新位置（移走后框处）原文字也被去除",
          not np.array_equal(_region(app.working_bgr, new_box),
                              _region(app.original_bgr, new_box)))
    root.destroy()


def test_create_undo_restores():
    print("\n== 4. 撤销可恢复被去除的原文字 ==")
    root, app, box = _setup()
    cvs = app.canvas
    _create_box(cvs, box)
    _flush(app)
    check("新建撤销前：区域 ≠ 原图",
          not np.array_equal(_region(app.working_bgr, box),
                              _region(app.original_bgr, box)))
    app.undo()
    _flush(app)
    check("撤销后：区域 == 原图（原文字恢复）",
          np.array_equal(_region(app.working_bgr, box),
                          _region(app.original_bgr, box)))
    root.destroy()


def test_create_over_background_skips():
    print("\n== 5. 框在纯背景（与原文一致）区域不误擦 ==")
    root, app, box = _setup()
    cvs = app.canvas
    # 取图像右下角一块「没有文字」的纯背景区域（远离文字 bbox）
    h, w = app.original_bgr.shape[:2]
    bg_box = (w - 30, h - 30, w - 5, h - 5)
    before = app.working_bgr.copy()
    _create_box(cvs, bg_box)
    _flush(app)
    check("纯背景框选后工作图无变化（不误擦背景）",
          np.array_equal(app.working_bgr, before))
    root.destroy()


# ============================================================ 2. 四边延长辅助线
def test_guides_drawn_for_active_box():
    print("\n== 6. 活动选框绘制四边延长辅助线（贯穿画布）不报错 ==")
    root, app, box = _setup()
    cvs = app.canvas
    _create_box(cvs, box)
    _flush(app)
    check("新建后该框为活动框", cvs.get_active_index() == 0)
    # 直接调用绘制，确认 _draw_guides 在活动框上被触发且不抛异常
    try:
        cvs._redraw(regen=True)
        ok = True
    except Exception as e:
        ok = False
        detail = str(e)
    check("活动框重绘（含四边延长线）无异常", ok, detail if not ok else "")
    # 非活动框不应绘制辅助线：清空活动 → 重绘应同样正常
    cvs._active_index = -1
    try:
        cvs._redraw(regen=True)
        ok2 = True
    except Exception as e:
        ok2 = False
        detail2 = str(e)
    check("非活动态重绘无异常", ok2, detail2 if not ok2 else "")
    root.destroy()


def main():
    test_create_erases_immediately()
    test_create_then_delete_keeps_erased()
    test_create_then_move_keeps_erased()
    test_create_undo_restores()
    test_create_over_background_skips()
    test_guides_drawn_for_active_box()
    print(f"\n自测：通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print("  - " + f)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
