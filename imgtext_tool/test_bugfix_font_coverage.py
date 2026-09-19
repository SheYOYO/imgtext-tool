"""Bug1 修复自测：常用词必须能显示，不能优先出现生僻字/方块。

验证：
1. _font_text_coverage / font_covers_text 能正确判定「字体缺字」。
2. match_fonts 选出的首选项一定能完整渲染用户实际文字（覆盖率满分置顶）。
3. 交互层：在选区键入中文后，自动选用的字体可完整渲染该文字（不再出方块）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageFont

from app import gui, config, font_match as fm

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "_test_output")
os.makedirs(OUT, exist_ok=True)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))


def make_image():
    W, H = 420, 220
    rng = np.random.default_rng(5)
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

    pool = config.resolve_font_pool()
    cn = next((f for f in pool if f.get("cn")), None)
    check("存在可用中文字体池", cn is not None, cn["name"] if cn else "无")
    if cn is None:
        print("无中文字体，跳过覆盖率断言")
        return

    print("== 1. 覆盖率判定：字体对缺字应判为不覆盖 ==")
    # 常见中文字「中」：CJK 字体必含 → 覆盖
    cov_cn = fm._font_text_coverage("中", cn["path"], 40)
    check("CJK 字体覆盖常用字「中」", cov_cn >= 0.999, f"coverage={cov_cn:.3f}")
    check("font_covers_text 对常用中文字返回 True",
          fm.font_covers_text("中文字体测试", cn["path"]) is True)

    # 用「英文字体（不含中文）」作为确定性反例，验证「缺字 → 判为不覆盖」。
    # 微软雅黑等超级字体几乎覆盖全部码点，不适合做反例；Arial 仅含拉丁字形，
    # 必缺中文，可稳定验证覆盖率逻辑。
    en = next((f for f in pool if not f.get("cn")), None)
    check("字体池中存在英文（非中文）字体", en is not None, en["name"] if en else "无")
    if en is not None:
        cov_cn_in_en = fm._font_text_coverage("中文测试", en["path"], 40)
        check("英文字体不覆盖中文（缺字→判为不覆盖）", cov_cn_in_en < 0.999,
              f"cov={cov_cn_in_en:.3f}")
        check("font_covers_text 对中文返回 False（英文字体）",
              fm.font_covers_text("中文测试", en["path"]) is False)
        cov_en_in_en = fm._font_text_coverage("Hello", en["path"], 40)
        check("英文字体覆盖 ASCII", cov_en_in_en >= 0.999, f"cov={cov_en_in_en:.3f}")
        check("font_covers_text 对 ASCII 返回 True（英文字体）",
              fm.font_covers_text("Hello", en["path"]) is True)

    print("\n== 2. match_fonts 首选字体必能完整渲染用户文字 ==")
    bgr = make_image()
    region = bgr[70:115, 40:200]
    text = "常用词显示测试"
    cands = fm.match_fonts(text, region, target_stroke=0.5, target_size=40, target_stroke_px=4.0)
    check("返回候选字体", len(cands) > 0)
    top = cands[0]
    check("首选字体覆盖率满分（能渲染全部常用字）", top.get("coverage", 0) >= 0.999,
          f"top={top['name']} cov={top.get('coverage')}")
    check("首选字体确实能渲染用户文字", fm.font_covers_text(text, top["path"]) is True)

    print("\n== 3. 交互层：键入中文后自动选取可渲染字体 ==")
    root = tk.Tk()
    app = gui.App(root)
    root.geometry("1400x860")
    for _ in range(3):
        root.update()
    app.original_bgr = bgr.copy()
    app.working_bgr = bgr.copy()
    app.history = [bgr.copy()]
    app.current_box = None
    app.canvas.set_image(bgr)
    for _ in range(2):
        root.update()

    cvs = app.canvas
    box = (40, 70, 200, 115)
    cvs._boxes.append(box)
    cvs._active_index = 0
    app.on_box_selected(box)
    # 真实用户键入中文（此前匹配用的是占位「示例」，需据真实文字重选字体）
    app.text_var.set("常用词显示测试")
    app._on_text_key()
    root.update()
    check("键入后已自动选用字体", app.selected_font_path is not None,
          str(app.selected_font_path))
    check("自动选用的字体可完整渲染键入的中文（Bug1 修复 · 不再出方块）",
          fm.font_covers_text("常用词显示测试", app.selected_font_path) is True)
    root.destroy()

    print(f"\nBug1（常用词显示/字体覆盖）自测：通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：" + "；".join(FAIL))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    print("Bug1 字体字形覆盖修复自测通过 ✓")
