"""OCR 流程自测（OCR 自动识别 + 上下行参照校准 迭代）。

覆盖成功 + 失败双向：
1. 真实 OCR 能力：ocr.ocr_region 能离线识别合成中文图（RapidOCR）。
2. GUI 三步执行顺序（用户硬性规定）：第一步 OCR 识别文字填入 →
   第二步 上下行参照校准 → 第三步 渲染。
3. 上下行参照提取：含上下行的图能提取参照、纯背景返回 None、边缘框不越界。
4. OCR 不可用降级（失败路径）：is_ocr_supported=False 时不崩溃、保持手动输入、流程继续。
5. 重选已有框不重新识别：on_box_select_same 不调用 OCR / 不重新提取参照。
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
import app.feature_extract as fe_real

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "_test_output")
os.makedirs(OUT, exist_ok=True)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))


def _cn_font():
    pool = config.resolve_font_pool()
    fp = next((f["path"] for f in pool if f.get("cn")), None)
    return fp


def make_text_image(text="测试文字", size=40, w=320, h=120):
    img = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(img)
    fp = _cn_font()
    font = ImageFont.truetype(fp, size) if fp else ImageFont.load_default()
    d.text((20, 40), text, font=font, fill=(15, 15, 15))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def make_three_line_image():
    """上 / 中 / 下 三行文字（紧凑布局，行距≈字号1.3倍，符合真实段落），
    用于上下行参照提取。选中中间行时，参照窗口（约1.6×字号）能覆盖到上下行文字。"""
    img = Image.new("RGB", (420, 200), (245, 245, 245))
    d = ImageDraw.Draw(img)
    fp = _cn_font()
    font = ImageFont.truetype(fp, 30) if fp else ImageFont.load_default()
    d.text((30, 20), "上行参照文字", font=font, fill=(15, 15, 15))
    d.text((30, 60), "中行目标文字", font=font, fill=(15, 15, 15))
    d.text((30, 100), "下行参照文字", font=font, fill=(15, 15, 15))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# 保存原始函数，便于 monkeypatch 后干净还原
_ORIG_OCR_SUPPORTED = gui.ocr.is_ocr_supported
_ORIG_OCR_REGION = gui.ocr.ocr_region
_ORIG_EXTRACT_REF = gui.fe.extract_neighbor_reference
_ORIG_MATCH = gui.fm.match_fonts


def _restore():
    gui.ocr.is_ocr_supported = _ORIG_OCR_SUPPORTED
    gui.ocr.ocr_region = _ORIG_OCR_REGION
    gui.fe.extract_neighbor_reference = _ORIG_EXTRACT_REF
    gui.fm.match_fonts = _ORIG_MATCH


def main():
    messagebox.showinfo = lambda *a, **k: None
    messagebox.showerror = lambda *a, **k: None
    messagebox.showwarning = lambda *a, **k: None

    # ----------------------------------------------------------------
    print("== 1. 真实 OCR 能力（离线 RapidOCR）==")
    bgr = make_text_image("测试文字")
    txt = gui.ocr.ocr_region(bgr)
    check("ocr.ocr_region 识别出文字（非空）", bool(txt), f"识别={txt!r}")
    check("识别结果长度 >= 2", len(txt) >= 2, f"识别={txt!r}")

    # ----------------------------------------------------------------
    print("\n== 2. GUI 三步执行顺序 + OCR 自动填入 + 校准生效 ==")
    root = tk.Tk()
    app = gui.App(root)
    bgr = make_text_image("测试文字")
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()
    app.history = [bgr.copy()]
    app.pending_result = None
    app.current_box = None
    app.canvas.set_image(bgr)
    root.update()

    box = (20, 40, 220, 95)
    region = bgr[40:95, 20:220]
    region_fs = fe_real.extract_features(region)["font_size"]  # 校准前选区字号

    calls = []
    orig_match = gui.fm.match_fonts

    def _fake_ocr(bgr_region, min_score=0.5):
        calls.append("ocr")
        return "识别字"

    def _fake_ref(*a, **k):
        calls.append("ref")
        return {"font_size": 30, "letter_spacing": 2, "stroke_px": 3.0,
                "stroke_px_skeleton": 3.0, "stroke_weight": 0.5, "skew_angle": 0.0,
                "blur_level": 0.3, "noise_level": 0.0, "sharpness": 0.7,
                "color_rgb": (10, 10, 10), "_ref_count": 2}

    def _fake_match(text, region, *args, **kwargs):
        # 第三步「自动渲染匹配样式」= 字体候选自动选择；OCR 原文不再自动
        # 光栅化回图上（那是 _do_render，现已改为用户改字/确认后才触发）。
        calls.append("render")
        return orig_match(text, region, *args, **kwargs)

    gui.ocr.is_ocr_supported = lambda: True
    gui.ocr.ocr_region = _fake_ocr
    gui.fe.extract_neighbor_reference = _fake_ref
    gui.fm.match_fonts = _fake_match

    app.canvas._boxes.append(box)
    app.canvas._active_index = 0
    app.on_box_selected(box)

    check("OCR 识别文字已自动填入输入框", app.text_var.get() == "识别字",
          f"text={app.text_var.get()!r}")
    check("执行顺序严格为 OCR → 参照 → 渲染",
          calls[:3] == ["ocr", "ref", "render"], f"calls={calls}")
    expect_fs = int(round((region_fs + 30) / 2))
    check(f"第二步校准生效（字号 = 选区{region_fs}与参照30的平均={expect_fs}）",
          app.features.get("font_size") == expect_fs,
          f"got={app.features.get('font_size')}")
    _restore()  # 关键：还原补丁，避免污染后续测试
    root.destroy()

    # ----------------------------------------------------------------
    print("\n== 3. 上下行参照提取（真实函数）==")
    img = make_three_line_image()
    box = (30, 60, 260, 95)  # 选中中间行
    ref = gui.fe.extract_neighbor_reference(img, box, font_size_hint=30)
    check("含上下行的图能提取参照（非 None）", ref is not None)
    if ref:
        check("参照字号合理（20~80）", 20 < ref["font_size"] < 80, f"fs={ref['font_size']}")
        check("参照记录来自上下行（_ref_count >= 1）",
              ref.get("_ref_count", 0) >= 1, f"n={ref.get('_ref_count')}")

    blank = np.full((100, 200, 3), 240, np.uint8)
    ref2 = gui.fe.extract_neighbor_reference(blank, (10, 10, 50, 50), font_size_hint=16)
    check("纯背景框（无上下文字）返回 None", ref2 is None, f"ref2={ref2}")

    edge = make_three_line_image()
    edge_box = (30, 0, 260, 40)  # 顶部边缘，上方无内容
    ref3 = gui.fe.extract_neighbor_reference(edge, edge_box, font_size_hint=30)
    check("顶部边缘框不越界且不崩溃", ref3 is None or isinstance(ref3, dict))

    # ----------------------------------------------------------------
    print("\n== 4. OCR 不可用降级（失败路径）==")
    root = tk.Tk()
    app = gui.App(root)
    bgr = make_text_image("测试文字")
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()
    app.history = [bgr.copy()]
    app.canvas.set_image(bgr)
    root.update()

    gui.ocr.is_ocr_supported = lambda: False  # 模拟 OCR 不可用
    box = (20, 40, 220, 95)
    app.canvas._boxes.append(box)
    app.canvas._active_index = 0
    ok = True
    err = ""
    try:
        app.on_box_selected(box)
    except Exception as e:
        ok = False
        err = str(e)
    check("OCR 不可用时 on_box_selected 不崩溃", ok, err)
    check("OCR 不可用时不自动填字（保持手动输入，输入框为空）",
          app.text_var.get() == "")
    check("OCR 不可用时仍生成候选字体（流程继续）", len(app.candidates) > 0,
          f"candidates={len(app.candidates)}")
    _restore()
    root.destroy()

    # ----------------------------------------------------------------
    print("\n== 5. 重选已有框不重新识别 ==")
    root = tk.Tk()
    app = gui.App(root)
    bgr = make_text_image("测试文字")
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()
    app.history = [bgr.copy()]
    app.canvas.set_image(bgr)
    root.update()

    called = []
    gui.ocr.is_ocr_supported = lambda: True

    def _spy_ocr(*a, **k):
        called.append("ocr")
        return "识别字"

    def _spy_ref(*a, **k):
        called.append("ref")
        return {"font_size": 30, "letter_spacing": 2, "stroke_px": 3.0,
                "stroke_px_skeleton": 3.0, "stroke_weight": 0.5, "skew_angle": 0.0,
                "blur_level": 0.3, "noise_level": 0.0, "sharpness": 0.7,
                "color_rgb": (10, 10, 10), "_ref_count": 2}

    gui.ocr.ocr_region = _spy_ocr
    gui.fe.extract_neighbor_reference = _spy_ref

    box = (20, 40, 220, 95)
    app.canvas._boxes.append(box)
    app.canvas._active_index = 0
    app.on_box_selected(box)  # 首次：会识别（called 加 ocr/ref）
    called.clear()             # 清空，只观察「重选」阶段

    # 模拟用户改字并应用，建立该框专属记录
    app.text_var.set("北京")
    app._schedule_preview(0)
    root.update()
    app.apply_replace()

    app.canvas._active_index = 0
    app.on_box_select_same(0)  # 重选：应恢复，不重新 OCR / 不重新提取参照

    check("重选已有框不调用 OCR（不重新识别文字）", "ocr" not in called,
          f"called={called}")
    check("重选已有框不重新提取上下行参照", "ref" not in called,
          f"called={called}")
    check("重选后恢复该框专属文字「北京」", app.text_var.get() == "北京",
          f"text={app.text_var.get()!r}")
    _restore()
    root.destroy()

    # ----------------------------------------------------------------
    print(f"\nOCR 流程自测：通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：" + "；".join(FAIL))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    print("OCR 流程（识别→校准→渲染）自测通过 ✓")
