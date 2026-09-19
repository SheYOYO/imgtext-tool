"""Bug2 修复自测：新建选区识别稳定 + 重新选取已有选框不再次识别 + 各框独立编辑。

验证：
1. 新建选区 → 执行识别（extract_features 被调用），输入框清空（不继承）。
2. 重新选取已有选框 → 不再次调用 extract_features（不重新识别），
   但恢复该框专属文字与特征（字号/颜色等），保证渲染稳定不串台。
3. 各选区文字各自独立保存、互不覆盖。
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
    rng = np.random.default_rng(7)
    base = np.clip(np.full((H, W, 3), 240, np.float32) + rng.normal(0, 2, (H, W, 3)), 0, 255).astype(np.uint8)
    img = Image.fromarray(base)
    pool = config.resolve_font_pool()
    fp = next((f["path"] for f in pool if f.get("cn")), None)
    font = ImageFont.truetype(fp, 40) if fp else ImageFont.load_default()
    d = ImageDraw.Draw(img)
    d.text((40, 70), "测试文字", font=font, fill=(20, 20, 20))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    messagebox.showinfo = lambda *a, **k: None
    messagebox.showerror = lambda *a, **k: None
    messagebox.showwarning = lambda *a, **k: None

    # 以间谍函数统计「特征识别（extract_features）」调用次数
    real_extract = gui.fe.extract_features
    calls = {"n": 0}

    def spy(region):
        calls["n"] += 1
        return real_extract(region)

    gui.fe.extract_features = spy

    root = tk.Tk()
    app = gui.App(root)
    root.geometry("1400x860")
    for _ in range(3):
        root.update()

    bgr = make_image()
    app.original_bgr = bgr.copy()
    app.working_bgr = bgr.copy()
    app.history = [bgr.copy()]
    app.current_box = None
    app.canvas.set_image(bgr)
    for _ in range(2):
        root.update()

    cvs = app.canvas

    print("== 1. 新建框1：应触发识别，输入框清空 ==")
    calls["n"] = 0
    box1 = (40, 70, 200, 115)
    cvs._boxes.append(box1)
    cvs._active_index = 0
    app.on_box_selected(box1)
    check("新建框1 执行了特征识别", calls["n"] >= 1, f"识别调用={calls['n']}")
    check("新建框1 输入框为空（不继承任何内容）", app.text_var.get() == "")
    check("框1 已保存独立的特征记录", 0 in app.box_edits and app.box_edits[0].get("features"))
    feat1 = dict(app.box_edits[0]["features"])

    app.text_var.set("北京")
    app._on_text_key(); root.update()
    app.apply_replace()
    check("框1 已记忆文字「北京」", app.box_edits[0]["text"] == "北京")

    print("\n== 2. 新建框2：再次识别、不继承框1文字 ==")
    box2 = (240, 150, 360, 200)
    cvs._boxes.append(box2)
    cvs._active_index = 1
    app.on_box_selected(box2)
    check("新建框2 再次执行特征识别（独立识别）", calls["n"] >= 2, f"累计识别调用={calls['n']}")
    check("新建框2 输入框清空（不继承框1 的「北京」）", app.text_var.get() == "",
          f"实际={app.text_var.get()!r}")
    check("框2 拥有独立特征记录", 1 in app.box_edits and app.box_edits[1]["text"] == "")
    check("两框特征各自独立（非同一对象）",
          app.box_edits[0]["features"] is not app.box_edits[1]["features"])

    print("\n== 3. 重新选取框1：不再次识别，但恢复专属文字/特征 ==")
    n_before = calls["n"]
    cvs._active_index = 0
    app.on_box_select_same(0)
    check("重新选取框1 不再次触发特征识别（不用再次识别）", calls["n"] == n_before,
          f"识别调用 {n_before}→{calls['n']}")
    check("重新选取框1 恢复其专属文字「北京」", app.text_var.get() == "北京",
          f"实际={app.text_var.get()!r}")
    check("重新选取框1 恢复其专属特征（字号一致）",
          app.features.get("font_size") == feat1.get("font_size"),
          f"现={app.features.get('font_size')} 原={feat1.get('font_size')}")
    check("恢复的特征与框1 保存的一致（非框2）",
          app.features.get("font_size") == feat1.get("font_size"))

    # 独立编辑：在框1 改文字、应用，框2 不受影响
    app.text_var.set("上海")
    app._on_text_key(); root.update()
    app.apply_replace()
    check("框1 改为「上海」并独立保存", app.box_edits[0]["text"] == "上海")
    check("框2 文字仍为独立记录（未覆盖）", app.box_edits[1]["text"] == "")

    # 切回框2（同样不重新识别），验证两框各自特征独立
    n2 = calls["n"]
    cvs._active_index = 1
    app.on_box_select_same(1)
    check("重新选取框2 同样不再次识别", calls["n"] == n2)
    check("框2 恢复其专属特征", app.features is not None)

    root.destroy()
    print(f"\nBug2（新建识别稳定/重选不重识别/各框独立）自测：通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：" + "；".join(FAIL))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    print("Bug2 多选区重选不重识别 + 独立编辑 自测通过 ✓")
