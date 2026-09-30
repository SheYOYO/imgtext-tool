"""复制选框（含编辑文字 + 字体特征）回归自测。

覆盖新增功能「复制已有选框」：
1. 复制后选框数量 +1、尺寸与原框相同、位置有偏移且夹在图像内；
2. 复制体**完整带走**原框的编辑文字（text）、字体（font_path）、
   已校准特征（features）与确认冻结的排版参数（optimized_params）；
3. 深拷贝：改动副本不影响原框（两框各自独立）；
4. 副本可先拖动定位（_copied 标记），点「确认替换文字」后与其它确认框一样锁死；
5. 四向参照按**新位置**重新采集；
6. 原功能不受影响：原框内容/状态保持不变。
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


def make_image():
    """一张含两处文字的图（第一处上方、第二处下方），便于复制后放到空位。"""
    pool = config.resolve_font_pool()
    fp = next((f["path"] for f in pool if f.get("cn")), None)
    font = ImageFont.truetype(fp, 40) if fp else ImageFont.load_default()
    W, H = 640, 460
    img = Image.new("RGB", (W, H), (240, 240, 240))
    d = ImageDraw.Draw(img)
    d.text((40, 60), "原始文字内容", font=font, fill=(20, 20, 20))
    d.text((40, 300), "下方上下文文字", font=font, fill=(20, 20, 20))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), (36, 52, 250, 110)


def setup():
    import tkinter as tk
    from tkinter import messagebox
    from app import gui
    messagebox.showinfo = messagebox.showerror = lambda *a, **k: None
    root = tk.Tk()
    root.withdraw()
    app = gui.App(root)
    bgr, box = make_image()
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()
    app.history = [{"bgr": bgr.copy(), "box_edits": {}}]
    app.pending_result = None
    app.current_box = None
    app.source_path = None
    app.canvas.set_image(bgr)
    return root, app, box


def flush(app):
    try:
        app.root.update()
    except Exception:
        pass


def select(app, box):
    app.canvas._boxes.clear()
    app.canvas._active_index = -1
    app.canvas.add_box(box, activate=True)
    app.on_box_selected(box, is_resize=False)
    flush(app)


def test_copy_carries_text_and_font():
    print("\n== 1. 复制选框带走编辑文字与字体特征 ==")
    root, app, box = setup()
    select(app, box)

    # 输入替换文字，并手动切到另一款候选字体（验证字体也被复制）
    app.text_var.set("复制测试文字")
    app._schedule_preview(0)
    flush(app)
    alt_font = app.candidates[1]["path"] if len(app.candidates) > 1 else app.selected_font_path
    app._select_candidate(1 if len(app.candidates) > 1 else 0)
    flush(app)
    app._confirm_text()
    flush(app)

    src_state = dict(app.box_edits.get(0, {}))
    n_before = len(app.canvas.get_boxes())
    app.copy_selected_box()
    flush(app)
    boxes = app.canvas.get_boxes()
    check("复制后选框数量 +1", len(boxes) == n_before + 1, f"{n_before} → {len(boxes)}")

    new_idx = app.canvas.get_active_index()
    check("复制体成为当前活动框", new_idx == len(boxes) - 1, f"active={new_idx}")

    x0, y0, x1, y1 = boxes[0]
    nx0, ny0, nx1, ny1 = boxes[new_idx]
    check("复制体尺寸与原框相同",
          (nx1 - nx0) == (x1 - x0) and (ny1 - ny0) == (y1 - y0),
          f"原{x1-x0}×{y1-y0} 新{nx1-nx0}×{ny1-ny0}")
    check("复制体位置发生偏移（不与原框重合）", (nx0, ny0) != (x0, y0),
          f"原({x0},{y0}) 新({nx0},{ny0})")
    ih, iw = app.original_bgr.shape[:2]
    check("复制体夹在图像内",
          0 <= nx0 and 0 <= ny0 and nx1 <= iw and ny1 <= ih,
          f"新框={boxes[new_idx]} 图={iw}×{ih}")

    new_state = app.box_edits.get(new_idx, {})
    check("复制体带走编辑文字", new_state.get("text") == src_state.get("text"),
          f"新={new_state.get('text')!r} 原={src_state.get('text')!r}")
    check("复制体带走字体", new_state.get("font_path") == alt_font == src_state.get("font_path"),
          f"新={os.path.basename(str(new_state.get('font_path')))}")
    check("复制体带走已校准特征",
          (new_state.get("features") or {}).get("font_size")
          == (src_state.get("features") or {}).get("font_size"),
          f"fs 新={(new_state.get('features') or {}).get('font_size')} "
          f"原={(src_state.get('features') or {}).get('font_size')}")
    check("复制体带走确认冻结的排版参数",
          (new_state.get("optimized_params") or {}).get("font_size")
          == (src_state.get("optimized_params") or {}).get("font_size"))
    check("输入框显示复制来的文字", app.text_var.get().strip() == src_state.get("text"),
          f"text_var={app.text_var.get()!r}")
    check("当前字体为复制来的字体", app.selected_font_path == alt_font)
    root.destroy()


def test_copy_is_independent():
    print("\n== 2. 复制体与原框相互独立（深拷贝） ==")
    root, app, box = setup()
    select(app, box)
    app.text_var.set("原框文字")
    app._confirm_text()
    flush(app)
    app.copy_selected_box()
    flush(app)
    new_idx = app.canvas.get_active_index()
    # 改副本的文字，不应影响原框
    app.text_var.set("副本改字")
    app._on_text_key()
    flush(app)
    check("改副本文字不影响原框",
          app.box_edits[0].get("text") == "原框文字",
          f"原框 text={app.box_edits[0].get('text')!r}")
    check("副本文字已更新", app.box_edits[new_idx].get("text") == "副本改字")
    root.destroy()


def test_copy_movable_before_and_after_confirm():
    print("\n== 3. 副本可先定位，确认后仍可长摁移动（取消锁死） ==")
    root, app, box = setup()
    select(app, box)
    app.text_var.set("定位后确认")
    app._confirm_text()
    flush(app)
    app.copy_selected_box()
    flush(app)
    idx = app.canvas.get_active_index()
    check("副本可移动（先拖动定位）", app._box_can_move(idx) is True)
    app._confirm_text()
    flush(app)
    # 本轮新需求：所有框（含已确认副本）都可长摁移动，不再锁死
    check("确认后副本仍可长摁移动（取消锁死）", app._box_can_move(idx) is True)
    check("确认后 _copied 标记已清除", "_copied" not in app.box_edits[idx])
    root.destroy()


def test_copy_refreshes_neighbor_reference():
    print("\n== 4. 四向参照按新位置重新采集 ==")
    root, app, box = setup()
    select(app, box)
    app.text_var.set("参照刷新")
    app._confirm_text()
    flush(app)
    app.copy_selected_box()
    flush(app)
    idx = app.canvas.get_active_index()
    st = app.box_edits[idx]
    check("复制体已写入四向参照字段",
          "neighbor_ref" in st and "neighbor_boxes" in st,
          f"keys={sorted(st.keys())}")
    # 位置变了 → 邻居矩形集合应当按新位置重算（至少字段存在且与当前 self 同步）
    check("当前四向参照指向复制体位置",
          app.neighbor_boxes == (st.get("neighbor_boxes") or []))
    root.destroy()


def test_original_untouched():
    print("\n== 5. 原功能不受影响（原框状态保持） ==")
    root, app, box = setup()
    select(app, box)
    app.text_var.set("保持原样")
    app._confirm_text()
    flush(app)
    before = dict(app.box_edits[0])
    app.copy_selected_box()
    flush(app)
    after = app.box_edits[0]
    check("原框文字不变", after.get("text") == before.get("text"))
    check("原框字体不变", after.get("font_path") == before.get("font_path"))
    check("原框坐标不变", app.canvas.get_boxes()[0] == box,
          f"{app.canvas.get_boxes()[0]} vs {box}")
    check("原框未带 _copied 标记", "_copied" not in after)
    root.destroy()


def test_copy_without_selection_safe():
    print("\n== 6. 无选中/无图片时复制安全不崩溃（失败路径） ==")
    import tkinter as tk
    from tkinter import messagebox
    from app import gui
    messagebox.showinfo = messagebox.showerror = lambda *a, **k: None
    root = tk.Tk()
    root.withdraw()
    app = gui.App(root)
    app.copy_selected_box()          # 未上传图片
    bgr, box = make_image()
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()
    app.canvas.set_image(bgr)
    app.copy_selected_box()          # 已上传但无选中框
    flush(app)
    check("无图片/无选中时复制不崩溃", True)
    root.destroy()


def main():
    test_copy_carries_text_and_font()
    test_copy_is_independent()
    test_copy_movable_before_and_after_confirm()
    test_copy_refreshes_neighbor_reference()
    test_original_untouched()
    test_copy_without_selection_safe()
    print(f"\n复制选框自测：通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：" + "；".join(FAIL))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    print("复制选框（含文字与字体特征）自测通过 ✓")
