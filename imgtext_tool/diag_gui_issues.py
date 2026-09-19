"""复现佘先生三问题：1 撤销无效/不可靠 2 切字体后点确认回到原效果 3 字体识别精度。
以 App 状态机驱动（复用 test_gui_smoke 手法），输出事实，不改任何生产代码。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))


def make_img(text="测试文字", pos=(40, 70), size=40, fill=(20, 20, 20)):
    from app import config
    pool = config.resolve_font_pool()
    fp = next((f["path"] for f in pool if f.get("cn")), None)
    W, H = 420, 220
    base = np.full((H, W, 3), 240, dtype=np.uint8)
    img = Image.fromarray(base)
    font = ImageFont.truetype(fp, size) if fp else ImageFont.load_default()
    ImageDraw.Draw(img).text(pos, text, font=font, fill=fill)
    rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), fp


def setup():
    import tkinter as tk
    from tkinter import messagebox
    from app import gui
    messagebox.showinfo = lambda *a, **k: print("   [msgbox-info]", a[0] if a else "")
    root = tk.Tk()
    root.withdraw()
    app = gui.App(root)
    bgr, fp = make_img()
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()
    app.history = [bgr.copy()]
    app.pending_result = None
    app.current_box = None
    app.source_path = None
    app.canvas.set_image(bgr)
    return root, app, fp


def new_box(app, box):
    """模拟在图上完成一次新框选（画布 on_box_select 全链路）。"""
    app.canvas._boxes.append(box)
    app.canvas._active_index = len(app.canvas._boxes) - 1
    app.on_box_selected(box, is_resize=False)


def flush(app):
    try:
        app.root.update()
    except Exception:
        pass


def main():
    print("=" * 60)
    print("复现 1：确认替换后（未点应用）直接撤销 → 是否提示无法撤销")
    root, app, fp = setup()
    bgr = app.original_bgr
    box = (40, 70, 200, 115)
    new_box(app, box)
    app.text_var.set("新文字测试")
    app._schedule_preview(0); flush(app)
    n_hist_before = len(app.history)
    app._confirm_text(); flush(app)
    print(f"   history={len(app.history)} (载入后=1)  pending_result is not None={app.pending_result is not None}")
    app.undo(); flush(app)
    print(f"   undo 后 history={len(app.history)}  working 是否回到原图={np.array_equal(app.working_bgr, bgr)}")
    root.destroy()

    print("=" * 60)
    print("复现 2：应用替换后撤销，再点回该框 → 替换是否复活")
    root, app, fp = setup()
    bgr0 = app.working_bgr.copy()
    box = (40, 70, 200, 115)
    new_box(app, box)
    app.text_var.set("新文字测试")
    app._schedule_preview(0); flush(app)
    app._confirm_text(); flush(app)
    app.apply_replace(); flush(app)
    img_applied = app.working_bgr.copy()
    print(f"   应用后 working!=原图: {not np.array_equal(img_applied, bgr0)}")
    app.undo(); flush(app)
    print(f"   撤销后 working==原图: {np.array_equal(app.working_bgr, bgr0)}  pending={app.pending_result is not None}")
    # 点回同一框
    idx = app.canvas.get_active_index()
    app.on_box_select_same(idx); flush(app)
    print(f"   点回该框后 pending_result is not None={app.pending_result is not None}（非None=替换复活）")
    if app.pending_result is not None:
        print(f"   点回该框后画布显示图与撤销后的 working 不同={not np.array_equal(app.pending_result, app.working_bgr)}")
    root.destroy()

    print("=" * 60)
    print("复现 3：切换候选字体 → 点确认 → 渲染是否仍是切换后的字体")
    root, app, fp = setup()
    box = (40, 70, 200, 115)
    new_box(app, box)
    app.text_var.set("新文字测试")
    app._schedule_preview(0); flush(app)
    cands = list(app.candidates)
    names = [os.path.basename(c["path"]) for c in cands]
    print(f"   候选: {names}")
    if len(cands) >= 2:
        auto_font = app.selected_font_path
        alt_font = cands[1]["path"]
        print(f"   自动选中={os.path.basename(auto_font)}  将切到={os.path.basename(alt_font)}")
        app._select_candidate(1); flush(app)
        print(f"   切换后 selected_font_path={os.path.basename(app.selected_font_path)}")
        # 用两字体分别强制渲染同一 params 对比像素，确定“字体现效果”可分辨
        from app import render as render_mod
        idx = app.canvas.get_active_index()
        base = app.working_bgr
        params = app._collect_params()
        for path, tag in ((auto_font, "A自动"), (alt_font, "B切换")):
            cur, _ = render_mod.render_text_with_metrics("新文字测试", params, font_path=path)
            ink = render_mod.text_ink_bbox(cur)
            bx, by = app._base_ink_position(cur, ink, box)
            out = render_mod.composite_text(base.copy(), cur, (bx, by))
            print(f"   {tag} 渲染差异(相对原图) mean={np.abs(out.astype(int)-base.astype(int)).mean():.3f}")
        # 记录切到 B 的 pending
        app._confirm_text(); flush(app)
        print(f"   确认后 selected_font_path={os.path.basename(app.selected_font_path)} 仍为B? {app.selected_font_path==alt_font}")
    root.destroy()
    print("=" * 60)
    print("复现 4：撤销一个“确认过但从未应用”的框 → 该框编辑状态是否还残留")
    root, app, fp = setup()
    bgr0 = app.working_bgr.copy()
    box = (40, 70, 200, 115)
    new_box(app, box)
    app.text_var.set("新文字测试")
    app._confirm_text(); flush(app)
    print(f"   box_edits 键: {list(app.box_edits.keys())}  active_idx={app.canvas.get_active_index()}")
    app.undo(); flush(app)
    st = app.box_edits.get(app.canvas.get_active_index(), {})
    print(f"   undo 后该框 box_edits: { {k: (v if not isinstance(v,(dict,list)) else '...') for k,v in st.items()} }")
    app.on_box_select_same(app.canvas.get_active_index()); flush(app)
    print(f"   点回该框后 text_var='{app.text_var.get()}' pending={'有' if app.pending_result is not None else '无'}  → 是否复活={'复活' if app.pending_result is not None else '未复活'}")
    root.destroy()


if __name__ == "__main__":
    main()
