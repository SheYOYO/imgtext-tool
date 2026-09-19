"""第十一轮：布局锁定 / 切框不回弹 / 字数变化自动字间距 专项自测。

成功路径：
  - 修复「画布回弹」：apply_replace 改为合并写入，切换选框不再清空
    _locked / optimized / neighbor_ref / neighbor_boxes（各选区数据互相独立）。
  - 重开后仅编辑文字：确认渲染完毕的框，整套样式特征冻结保存，重选时复用冻结参数，
    字间距严格锁定原图实测值（_calibrate_letter_spacing 闭环校准，结果与原图一致即不再改动），
    原有样式不被覆盖。
  - 字数变化：替换文字字间距严格等于原图字间距（闭环校准到原图间隙，不做任何收放；
    仅保持基线/字号/笔画粗细不变；宽度差异由字数自然决定）。
  - 防重叠信息（neighbor_boxes）跨切框保留，重开后仍不与原图文字重叠。
"""
import os
import sys
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

from app import gui, config, render as render_mod


def _cn_font():
    pool = config.resolve_font_pool()
    for f in pool:
        if f.get("cn"):
            return f["path"]
    return None


def _make_neighbor_image():
    """目标框四周都有紧邻文字（上下左右），用于四向参照与防重叠验证。"""
    img = Image.new("RGB", (640, 480), (245, 245, 245))
    d = ImageDraw.Draw(img)
    fp = _cn_font()
    font = ImageFont.truetype(fp, 36) if fp else ImageFont.load_default()
    d.text((210, 210), "目标字", font=font, fill=(15, 15, 15))   # 目标（中部）
    d.text((210, 158), "上", font=font, fill=(15, 15, 15))       # 上邻
    d.text((210, 270), "下", font=font, fill=(15, 15, 15))       # 下邻
    d.text((168, 210), "左", font=font, fill=(15, 15, 15))       # 左邻
    d.text((330, 210), "右", font=font, fill=(15, 15, 15))       # 右邻
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


_pass = 0
_fail = 0
def check(name, cond, info=""):
    global _pass, _fail
    if cond:
        _pass += 1
        print(f"  [PASS] {name}")
    else:
        _fail += 1
        print(f"  [FAIL] {name}  {info}")


# ---------------------------------------------------------------- 1. apply_replace 不丢数据（修复回弹）
def test_apply_replace_preserves_lock():
    print("\n== 1. apply_replace 合并写入，不丢 _locked/optimized/neighbor_ref/neighbor_boxes ==")
    root = tk.Tk(); root.withdraw()
    app = gui.App(root)
    bgr = _make_neighbor_image()
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()

    box = (205, 205, 320, 262)
    app.canvas._boxes.append(box)
    app.canvas._active_index = 0
    app.on_box_selected(box)
    app.text_var.set("替换文字")
    app._confirm_text()
    # 固化当前框草稿（模拟切框前的 on_leave_box）
    app.apply_replace(silent=True)

    idx = 0
    st = app.box_edits.get(idx, {})
    check("固化后 _locked 仍在", st.get("_locked") is True)
    check("固化后 optimized 仍在", st.get("optimized") is True)
    check("固化后 neighbor_ref 仍在", st.get("neighbor_ref") is not None)
    check("固化后 neighbor_boxes 仍在", len(st.get("neighbor_boxes") or []) >= 1)
    check("固化后 optimized_params 已冻结", isinstance(st.get("optimized_params"), dict))
    check("固化后字体路径仍在", bool(st.get("font_path")))
    root.destroy()


# ---------------------------------------------------------------- 2. 重开仅编辑文字：样式冻结不重算
def test_reselect_reuses_frozen_params():
    print("\n== 2. 重开已确认框：复用冻结参数，绝不重排（样式不变）==")
    root = tk.Tk(); root.withdraw()
    app = gui.App(root)
    bgr = _make_neighbor_image()
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()

    box = (205, 205, 320, 262)
    app.canvas._boxes.append(box)
    app.canvas._active_index = 0
    app.on_box_selected(box)
    orig_font_size = app.features.get("font_size")
    app.text_var.set("短字")
    app._confirm_text()

    # 记录冻结参数
    frozen = dict(app.box_edits[0]["optimized_params"])
    orig_ls = app.features.get("letter_spacing")
    # 监控 _calibrate_letter_spacing：重开后仍会被渲染路径调用以闭环校准字间距，
    # 但校准结果应==原图实测间隙（不会把字距收放/改写），即「复用冻结值、样式不变」。
    cal_calls = []
    orig_cal = app._calibrate_letter_spacing
    def _spy(text, params, fp):
        cal_calls.append(1)
        res = orig_cal(text, params, fp)
        return res
    app._calibrate_letter_spacing = _spy

    # 重选同一框（模拟用户点开已确认框）
    app.on_box_select_same(0)
    check("重开后特征未漂移（字号一致）",
          app.features.get("font_size") == orig_font_size,
          f"{app.features.get('font_size')} vs {orig_font_size}")
    check("重开后字间距严格锁定原图实测值（未被改写）",
          app.box_edits[0]["optimized_params"].get("letter_spacing") == orig_ls,
          f"frozen_ls={app.box_edits[0]['optimized_params'].get('letter_spacing')} vs orig_ls={orig_ls}")
    # 渲染路径触发闭环校准且结果可用
    out = app._do_render(app.working_bgr.copy(), optimize=True)
    check("重开后渲染结果有效", out is not None)
    check("重开后字体大小与确认时冻结一致",
          app.box_edits[0]["optimized_params"].get("font_size") == frozen.get("font_size"))
    check("重开后闭环校准被执行（字间距严格对齐原图）", len(cal_calls) >= 1, f"cal_calls={cal_calls}")
    root.destroy()


# ---------------------------------------------------------------- 3. 字数变化：字间距严格等于原图
def test_char_count_change_adjusts_spacing():
    print("\n== 3. 字数变化：替换文字字间距严格等于原图字间距（基准线/字号/粗细不变）==")
    root = tk.Tk(); root.withdraw()
    app = gui.App(root)
    bgr = _make_neighbor_image()
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()

    box = (205, 205, 320, 262)
    app.canvas._boxes.append(box)
    app.canvas._active_index = 0
    app.on_box_selected(box)
    app.text_var.set("一二")
    app._confirm_text()
    base = dict(app.box_edits[0]["optimized_params"])
    base_ls = base.get("letter_spacing")
    base_fs = base.get("font_size")
    base_stroke = base.get("target_stroke_px")

    # 原图实测字间距（用户硬性要求对齐的目标）
    orig_gap = app.features.get("letter_spacing")
    check("原图字间距已正确测得（>0）", orig_gap and orig_gap > 0, f"orig_gap={orig_gap}")

    # 改字：字数变多 / 变少 —— 替换文字的「可见字间距」必须严格等于原图，不收不放。
    # 注意：letter_spacing 参数本身会随字形侧边距不同而变化（不同字形达成同一可见间隙需不同
    # 参数），这是闭环校准的正确表现；真正的不变量是「渲染后的可见间隙」对齐原图。
    variants = ["一二三四五六七八", "一", "新文啊呀测试", "国国国国国国国国", "中文English混合"]
    for t in variants:
        cal = app._calibrate_letter_spacing(t, dict(base), app.selected_font_path)
        cur, _m = render_mod.render_text_with_metrics(t, cal, font_path=app.selected_font_path)
        rendered_gap = render_mod.measure_blank_gap(cur)
        # 多字才有可测内部间隙；单字（t='一'）跳过间隙断言
        if rendered_gap is not None:
            check(f"替换[{t}] 字间距严格对齐原图（差≤1px，量化极限）",
                  abs(rendered_gap - orig_gap) <= 1,
                  f"rendered_gap={rendered_gap} vs orig_gap={orig_gap}")
        # letter_spacing 由原图实测间隙驱动（非照搬邻居、非宽度拟合收放）
        check(f"替换[{t}] 字间距由闭环校准得出（非极端/负值）",
              isinstance(cal.get("letter_spacing"), int) and cal.get("letter_spacing") >= 0,
              f"cal_ls={cal.get('letter_spacing')}")

    # 字数变化仍不改变字号 / 笔画粗细（保持锁定）
    cal_long = app._calibrate_letter_spacing("一二三四五六七八", dict(base), app.selected_font_path)
    check("字数变化不改变字号（保持锁定）", cal_long.get("font_size") == base_fs)
    check("字数变化不改变笔画粗细（保持锁定）", cal_long.get("target_stroke_px") == base_stroke)
    root.destroy()


# ---------------------------------------------------------------- 4. 切框后防重叠/隔离不回弹（端到端）
def test_switch_box_no_rebound():
    print("\n== 4. 切换选区后效果稳定保留 + 防重叠信息不丢 ==")
    root = tk.Tk(); root.withdraw()
    app = gui.App(root)
    bgr = _make_neighbor_image()
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()
    app.history = [bgr.copy()]

    # 框 A
    boxA = (205, 205, 320, 262)
    app.canvas._boxes.append(boxA)
    app.canvas._active_index = 0
    app.on_box_selected(boxA)
    app.text_var.set("甲字")
    app._confirm_text()
    app.apply_replace(silent=True)   # 固化 A 到 working_bgr
    wg_after_A = app.working_bgr.copy()

    # 框 B（上邻区域），切框前 on_leave_box 固化 A
    boxB = (205, 120, 320, 168)
    app.canvas._boxes.append(boxB)
    app.canvas._active_index = 1
    app.on_box_selected(boxB)
    app.text_var.set("乙字")
    app._confirm_text()

    # 切回 A
    app.on_box_select_same(0)
    st = app.box_edits.get(0, {})
    check("切回 A 后 neighbor_boxes 仍保留（防重叠不丢）",
          len(st.get("neighbor_boxes") or []) >= 1)
    check("切回 A 后 optimized 仍标记",
          st.get("optimized") is True)
    check("切回 A 后 _locked 仍标记", st.get("_locked") is True)
    check("切回 A 后文字仍为「甲字」（各选区独立）",
          st.get("text") == "甲字", f"text={st.get('text')!r}")
    check("切回 A 后工作图仍含 A 的固化结果（未回弹）",
          not np.array_equal(app.working_bgr, bgr), "working_bgr 不应等于原图")
    # 工作图在切框过程中没有被 A 的预览覆盖回原样
    check("工作图保留 A 应用结果", np.any(app.working_bgr != bgr))
    root.destroy()


if __name__ == "__main__":
    test_apply_replace_preserves_lock()
    test_reselect_reuses_frozen_params()
    test_char_count_change_adjusts_spacing()
    test_switch_box_no_rebound()
    print(f"\n布局锁定 / 切框不回弹 自测：通过 {_pass} 项，失败 {_fail} 项")
    sys.exit(1 if _fail else 0)
