"""手动字号尊重 + 紧凑字 em + 不再自动缩小字体 回归自测。

佘先生三现象：
1. 微调字号后点确认被拉回识别大小 —— 二次优化不再参考邻居字号取平均/覆盖；
2. 结构紧凑的字（赢羸蠃臝…）被识别偏小 —— 行段小间隙合并，em 不再被切行切小；
3. 不要自动缩小字体（放大后文字跑到选框外/被逐档缩小）—— 渲染不再 1.0→0.7
   逐级缩小字号；防重叠只做距离优先的位置避让。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app import config
from app import feature_extract as fe

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_output")
os.makedirs(OUT, exist_ok=True)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))


def render_single(text, path, size=48):
    font = ImageFont.truetype(path, size)
    tmp = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    bb = d.textbbox((0, 0), text, font=font)
    w = bb[2] - bb[0] + 20
    h = bb[3] - bb[1] + 20
    img = Image.new("RGBA", (int(w), int(h)), (245, 245, 245, 255))
    ImageDraw.Draw(img).text((-bb[0] + 10, -bb[1] + 10), text, font=font,
                             fill=(20, 20, 20, 255))
    a = np.array(img)
    al = a[:, :, 3] / 255.0
    return (a[:, :, :3].astype(float) * al[:, :, None]
            + 245.0 * (1 - al[:, :, None])).astype(np.uint8)


def _cn():
    return [f for f in config.resolve_font_pool() if f.get("cn")]


def _setup_multi_line():
    """三行文字图，返回 (bgr, 中间行 box)。"""
    import tkinter as tk
    from tkinter import messagebox
    from app import gui
    messagebox.showinfo = messagebox.showerror = lambda *a, **k: None
    cn = _cn()
    fp = cn[0]["path"]
    size = 48
    font = ImageFont.truetype(fp, size)
    lines = ["上一行参考字", "替换目标文字", "下一行上下文"]
    bb = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), lines[0], font=font)
    W = bb[2] - bb[0] + 80
    H = size * 3 + 90
    img = Image.new("RGB", (W, H), (240, 240, 240))
    d = ImageDraw.Draw(img)
    ys = []
    for i, t in enumerate(lines):
        y0 = 30 + i * (size + 16)
        d.text((30, y0), t, font=font, fill=(20, 20, 20))
        ys.append(y0)
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    box = (20, ys[1] - 8, W - 20, ys[1] + size + 8)
    root = tk.Tk()
    root.withdraw()
    app = gui.App(root)
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()
    app.history = [{"bgr": bgr.copy(), "box_edits": {}}]
    app.canvas.set_image(bgr)
    return root, app, box


def test_compact_chars_em_not_collapsed():
    print("\n== 1. 结构紧凑字 em 不塌缩（赢羸蠃臝 ≈48 不再是 22~37） ==")
    cn = _cn()
    bad = []
    for f in cn:
        r = render_single("赢羸蠃臝", f["path"], 48)
        est = fe.estimate_em_size(r)[0]
        if est < 43:  # 修复前 Deng 22 / simhei 27 / msyh 37
            bad.append((os.path.basename(f["path"]), est))
    check("紧凑字 em ≥ 43（修复前最低 22）", not bad, f"bad={bad}")


def test_manual_font_size_survives_confirm_with_neighbors():
    print("\n== 2. 手动放大字号 → 确认后仍是手动值（不被邻居/识别值拉回） ==")
    root, app, box = _setup_multi_line()
    app.canvas._boxes.append(box)
    app.canvas._active_index = 0
    app.on_box_selected(box, is_resize=False)
    app.root.update()
    check("该场景存在上下行邻居参照", app.neighbor_ref is not None)
    app.text_var.set("文字替换")
    app._schedule_preview(0)
    app.root.update()
    auto_fs = float(app.vars["font_size"].get())
    manual = int(round(auto_fs * 1.6))
    app.vars["font_size"].set(manual)
    app._confirm_text()
    app.root.update()
    frozen = app.box_edits[0].get("optimized_params", {})
    check("确认冻结字号 == 手动放大值",
          int(frozen.get("font_size", 0)) == manual,
          f"frozen={frozen.get('font_size')} manual={manual} auto={auto_fs}")
    root.destroy()


def test_render_does_not_shrink_font():
    print("\n== 3. 渲染不再 1.0→0.7 逐档缩小字号 ==")
    from app import gui, render as render_mod
    orig_render = render_mod.render_text_with_metrics
    calls = []

    def spy(text, params, font_path=None):
        calls.append(int(params.get("font_size", 0)))
        return orig_render(text, params, font_path=font_path)

    render_mod.render_text_with_metrics = spy
    try:
        root, app, box = _setup_multi_line()
        app.canvas._boxes.append(box)
        app.canvas._active_index = 0
        app.on_box_selected(box, is_resize=False)
        app.root.update()
        app.text_var.set("超长替换文字内容测试")
        app._confirm_text()
        app.root.update()
        base_fs = app.box_edits[0]["optimized_params"].get("font_size")
        check("确认渲染调用未出现小于原字号的档位",
              calls and all(c == base_fs for c in calls),
              f"render sizes={calls} frozen={base_fs}")
        root.destroy()
    finally:
        render_mod.render_text_with_metrics = orig_render


def main():
    test_compact_chars_em_not_collapsed()
    test_manual_font_size_survives_confirm_with_neighbors()
    test_render_does_not_shrink_font()
    print(f"\n手动字号/紧凑字/不缩小字号 自测：通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：" + "；".join(FAIL))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    print("手动字号尊重 + 紧凑字 em + 不再自动缩小字体 自测通过 ✓")
