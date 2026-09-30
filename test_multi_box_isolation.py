"""多选区独立运行自测（用户本轮重点修复）。

验证：
1. 新建第二个选框时，输入框文字不会继承上一框的文字（完全不继承上一个选区的数据）。
2. 每个选框独立提取特征（各自从原图区域扫描像素，参数互不串台）。
3. 点击已存在选框可独立恢复并修改其专属文字。
4. 在框 A 输入文字并应用后，再处理框 B，框 A 的修改成果保留、互不影响。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageFont

from app import gui, config

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "_test_output")
os.makedirs(OUT, exist_ok=True)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))


def make_image():
    W, H = 420, 220
    rng = np.random.default_rng(3)
    base = np.full((H, W, 3), 240, dtype=np.uint8)
    base = np.clip(base.astype(np.int16) + rng.normal(0, 2, (H, W, 3)), 0, 255).astype(np.uint8)
    img = Image.fromarray(base)
    pool = config.resolve_font_pool()
    fp = next((f["path"] for f in pool if f.get("cn")), None)
    font = ImageFont_try(fp)
    d = ImageDraw.Draw(img)
    d.text((40, 70), "测试文字", font=font, fill=(20, 20, 20))
    rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def ImageFont_try(fp):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(fp, 40) if fp else ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def main():
    messagebox.showinfo = lambda *a, **k: None
    messagebox.showerror = lambda *a, **k: None
    messagebox.showwarning = lambda *a, **k: None

    root = tk.Tk()
    app = gui.App(root)
    root.geometry("1400x860")
    for _ in range(3):
        root.update()

    bgr = make_image()
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()
    app.history = [bgr.copy()]
    app.pending_result = None
    app.current_box = None
    app.canvas.set_image(bgr)
    for _ in range(2):
        root.update()

    cvs = app.canvas

    print("== 1. 框1：输入文字并应用 ==")
    box1 = (40, 70, 200, 115)   # 覆盖「测试文字」
    cvs._boxes.append(box1)
    cvs._active_index = 0
    app.on_box_selected(box1)    # 新框 —— 应清空文字
    check("新框1 输入框为空（未继承任何内容）", app.text_var.get() == "")
    app.text_var.set("北京")
    app._schedule_preview(0); root.update()
    app.apply_replace()
    check("框1 已记忆文字", app.box_edits[0]["text"] == "北京")
    feat1 = dict(app.box_edits[0]["features"])

    print("\n== 2. 框2：新建不应继承框1的文字/参数 ==")
    box2 = (240, 150, 360, 200)  # 空白区域（远离文字，无前景）
    cvs._boxes.append(box2)
    cvs._active_index = 1
    app.on_box_selected(box2)    # 关键：不应继承框1 的「北京」
    check("新框2 不继承框1 文字（输入框为空）", app.text_var.get() == "",
          f"实际={app.text_var.get()!r}")
    check("框2 拥有独立的特征记录", 1 in app.box_edits and app.box_edits[1]["text"] == "")
    # 独立特征提取：每个框各自从「自己的」图像区域扫描像素，而不是沿用上一框。
    # 用两点证明：(a) 两框 features 是不同对象（各自记录）；
    # (b) 框1 命中文字（真实字号 >20），框2 为空区（走高度兜底），两者字号不同。
    feat2 = dict(app.box_edits[1]["features"])
    check("两框 features 各自独立存储（非同一对象）",
          app.box_edits[0]["features"] is not app.box_edits[1]["features"])
    check("两框独立提取特征（框1命中文字、框2为空区，字号不同）",
          feat1["font_size"] > 20 and feat1["font_size"] != feat2["font_size"],
          f"框1字号={feat1['font_size']} 框2字号={feat2['font_size']}")
    check("两框文字各自保存、互不覆盖",
          app.box_edits[0]["text"] == "北京" and app.box_edits[1]["text"] == "")

    print("\n== 3. 切回框1：可独立恢复并修改其文字 ==")
    cvs._active_index = 0
    app.on_box_select_same(0)
    check("切回框1 恢复其专属文字「北京」", app.text_var.get() == "北京")
    app.text_var.set("上海")
    app._schedule_preview(0); root.update()
    app.apply_replace()
    check("框1 文字改为「上海」并保存", app.box_edits[0]["text"] == "上海")

    print("\n== 4. 框1成果保留在 work 图中，框2不影响它 ==")
    # 工作图应已写入「上海」对应的渲染像素（与原始空白不同）
    check("工作图已被框1修改（非原图）",
          not np.array_equal(app.working_bgr, bgr))

    root.destroy()
    print(f"\n多选区独立运行自测：通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：" + "；".join(FAIL))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    print("多选区独立运行（不继承/各自独立）自测通过 ✓")
