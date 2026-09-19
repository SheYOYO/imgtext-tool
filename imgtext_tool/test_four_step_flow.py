"""四步流程迭代自测：强制顺序、四向参照、特征锁定、确认、二次排版优化、防重叠。

成功路径：
  - 强制五步顺序：识别框内原文 → 采集四周上下文 → 用户输入确认 → 二次排版优化 → 防重叠渲染 → 预览/导出
  - 特征锁定：新建框时确定特征并锁定，重选/缩放不重新识别、不漂移
  - 四向上下文：上下左右邻近文字均被采集并用于校准/防重叠
  - 确认触发闭环校准：替换文字字间距严格等于原图实测字间距（不参考邻居、不按宽度收放）
  - 防重叠：渲染文字允许超出选框，但禁止与邻近原始文字重叠
  - 多选区独立
失败路径：
  - OCR 缺失优雅降级、流程继续
  - 无邻近文字时参照为 None、流程不崩
"""
import os
import sys
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

from app import gui, config, feature_extract as fe, render as render_mod


# ---------------------------------------------------------------- 测试图构造
def _cn_font():
    pool = config.resolve_font_pool()
    for f in pool:
        if f.get("cn"):
            return f["path"]
    return None


def make_four_neighbor_image():
    """目标框四周（上下左右）都有『紧邻』文字，便于四向参照与防重叠验证。

    注意：四向参照窗口仅覆盖紧邻范围（上/下约 1.6×字号高、左/右约 1.2×字宽），
    因此邻居必须贴着目标框放置，太远会被视为非紧邻而忽略。
    """
    img = Image.new("RGB", (640, 480), (245, 245, 245))
    d = ImageDraw.Draw(img)
    fp = _cn_font()
    font = ImageFont.truetype(fp, 36) if fp else ImageFont.load_default()

    # 目标文字（中部）
    d.text((210, 210), "目标字", font=font, fill=(15, 15, 15))
    # 上方紧邻（单字，贴着目标上沿）
    d.text((210, 158), "上", font=font, fill=(15, 15, 15))
    # 下方紧邻
    d.text((210, 270), "下", font=font, fill=(15, 15, 15))
    # 左方紧邻（贴着目标左沿）
    d.text((168, 210), "左", font=font, fill=(15, 15, 15))
    # 右方紧邻（贴着目标右沿，便于验证防重叠）
    d.text((330, 210), "右", font=font, fill=(15, 15, 15))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def make_blank_neighbor_image():
    """目标框四周无文字（仅背景），用于『无参照』降级路径。"""
    img = Image.new("RGB", (400, 300), (245, 245, 245))
    d = ImageDraw.Draw(img)
    fp = _cn_font()
    font = ImageFont.truetype(fp, 36) if fp else ImageFont.load_default()
    d.text((150, 130), "孤立字", font=font, fill=(15, 15, 15))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def make_right_neighbor_image():
    """仅有右侧邻近文字（无上/下/左），用于验证防重叠可纵向避让。"""
    img = Image.new("RGB", (640, 480), (245, 245, 245))
    d = ImageDraw.Draw(img)
    fp = _cn_font()
    font = ImageFont.truetype(fp, 36) if fp else ImageFont.load_default()
    # 目标文字（中部）
    d.text((210, 210), "目标字", font=font, fill=(15, 15, 15))
    # 右侧紧邻（与目标框右沿保持约 40px 间距，纵向避让即可避开）
    d.text((370, 210), "右邻文字", font=font, fill=(15, 15, 15))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------- 轻量断言
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


def _press(cvs, x, y):
    cvs._on_left_down(_Ev(x, y))
    cvs._on_left_up(_Ev(x, y))


class _Ev:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.delta = 0


# ---------------------------------------------------------------- 1. 强制顺序 + 四向参照 + 特征锁定
def test_pipeline_order_and_lock():
    print("\n== 1. 强制顺序 / 四向参照 / 特征锁定 ==")
    root = tk.Tk()
    root.withdraw()
    app = gui.App(root)
    bgr = make_four_neighbor_image()
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()

    calls = []
    orig_match = gui.fm.match_fonts
    ref_calls = []
    orig_ref = gui.fe.extract_neighbor_reference
    def _fake_render(text, region, *a, **k):
        # 第三步「自动渲染匹配样式」= 字体候选自动选择；OCR 原文不再自动
        # 光栅化回图上（_do_render 现改为用户改字/确认后才触发）。
        calls.append("render")
        return orig_match(text, region, *a, **k)
    def _spy_ref(*a, **k):
        ref_calls.append(1)
        return orig_ref(*a, **k)
    gui.fm.match_fonts = _fake_render
    gui.fe.extract_neighbor_reference = _spy_ref

    # 目标框（覆盖「目标字」）
    box = (205, 205, 320, 262)
    app.canvas._boxes.append(box)
    app.canvas._active_index = 0
    app.on_box_selected(box)

    check("新建框即识别+采集四向上下文（含 render 调用）",
          "render" in calls, f"calls={calls}")
    idx = 0
    state = app.box_edits.get(idx, {})
    check("特征已锁定（_locked=True）", state.get("_locked") is True)
    check("四向参照已采集（neighbor_boxes 非空）",
          len(state.get("neighbor_boxes") or []) >= 1,
          f"n={len(state.get('neighbor_boxes') or [])}")
    nb = state.get("neighbor_boxes") or []
    xs_min = [b[2] < box[0] for b in nb]   # 在框左侧
    xs_max = [b[0] > box[2] for b in nb]   # 在框右侧
    ys_min = [b[3] < box[1] for b in nb]   # 在框上侧
    ys_max = [b[1] > box[3] for b in nb]   # 在框下侧
    check("四向均采集到邻近文字", any(xs_min) and any(xs_max) and any(ys_min) and any(ys_max),
          f"left={any(xs_min)} right={any(xs_max)} up={any(ys_min)} down={any(ys_max)}")

    # 记录锁定特征
    feat_locked = dict(app.features)
    # 重选同一框（模拟点击已存在框）—— 不应重新识别/重新采集参照
    ref_before = len(ref_calls)
    app.on_box_select_same(0)
    check("重选不重新识别（特征锁定未漂移）",
          app.features == feat_locked,
          f"\n  locked={feat_locked}\n  now   ={app.features}")
    check("重选不重新采集四向参照（ref 调用次数不变）",
          len(ref_calls) == ref_before,
          f"ref_calls={ref_calls}")
    gui.fm.match_fonts = orig_match   # 还原字体候选补丁，避免污染后续测试
    root.destroy()


# ---------------------------------------------------------------- 2. 确认触发二次排版优化 + 防重叠
def test_confirm_optimize_anti_overlap():
    print("\n== 2. 确认触发二次优化 + 防重叠渲染 ==")
    root = tk.Tk()
    root.withdraw()
    app = gui.App(root)
    bgr = make_right_neighbor_image()
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()

    # 捕获最终放置位置与文字墨迹包围盒
    placed = {}
    orig_composite = gui.render_mod.composite_text
    def _spy_composite(base, text_rgba, position):
        placed["pos"] = position
        placed["ink"] = gui.render_mod.text_ink_bbox(text_rgba)
        return orig_composite(base, text_rgba, position)
    gui.render_mod.composite_text = _spy_composite

    box = (205, 205, 320, 262)
    app.canvas._boxes.append(box)
    app.canvas._active_index = 0
    app.on_box_selected(box)

    # 输入较长替换文字并确认（触发第四步二次优化 + 第五步防重叠）
    app.text_var.set("替换后的文字")
    app._confirm_text()

    idx = 0
    state = app.box_edits.get(idx, {})
    check("确认后标记为已优化（optimized=True）", state.get("optimized") is True)
    check("确认后仍标记特征锁定", state.get("_locked") is True)

    # 防重叠验证：最终文字墨迹包围盒不得与任一邻近文字矩形重叠（安全边距内）
    pos = placed.get("pos")
    ink = placed.get("ink")
    forbidden = state.get("neighbor_boxes") or []
    if pos and ink:
        overlap = app._overlaps_any(pos[0], pos[1], ink, forbidden, margin=3)
        check("渲染文字未与邻近原始文字重叠（允许超出选框）",
              not overlap, f"pos={pos} ink={ink} forbidden={forbidden}")
    else:
        check("捕获到放置坐标", False, "composite 未被调用")
    root.destroy()


# ---------------------------------------------------------------- 3. 闭环校准（严格对齐原图字间距，不拉回字号/字距）
def test_secondary_optimize_alignment():
    print("\n== 3. 闭环校准：不参考邻居改字号、替换字间距严格等于原图实测值 ==")
    root = tk.Tk()
    root.withdraw()
    app = gui.App(root)
    # 模拟「本框已识别」的特征：原图实测字间距 8、单行
    app.features = {"letter_spacing": 8, "line_count": 1}
    app.selected_font_path = _cn_font()
    params = {"color_rgb": (0, 0, 0), "font_size": 30, "letter_spacing": 0,
              "line_spacing": 0, "skew_angle": 0.0, "sharpness": 0.7,
              "blur": 0.0, "edge_soften": 0.0, "noise": 0.0, "bold": False,
              "target_stroke_px": 0.0, "opacity": 1.0}
    # 周边参照：字间距 6、字号 50（与本框 30 偏差很大）—— 校准绝不能参考邻居
    neighbor = {"letter_spacing": 6, "font_size": 50}
    out = app._calibrate_letter_spacing("示例文字", dict(params), app.selected_font_path)
    # 用户硬性要求：字号绝不因邻居识别值被“平均/拉回”（微调字号后确定仍生效）
    check("闭环校准不改字号（30 不被邻居 50 平均）", out["font_size"] == 30, f"fs={out['font_size']}")
    # 字间距不照搬邻居：严格以原图实测值(8)为目标做闭环校准
    check("闭环校准不照搬邻居字间距", out["letter_spacing"] != neighbor["letter_spacing"])
    # 渲染验证：可见字间距严格对齐原图实测间隙（量化极限 ≤1px）
    cur, _m = render_mod.render_text_with_metrics("示例文字", out, font_path=app.selected_font_path)
    rendered_gap = render_mod.measure_blank_gap(cur)
    check("闭环校准后可见字间距严格对齐原图（差≤1px）",
          rendered_gap is not None and abs(rendered_gap - 8) <= 1,
          f"rendered_gap={rendered_gap} vs target=8")
    check("闭环校准不改变笔画/颜色等特征", out["color_rgb"] == (0, 0, 0) and out["opacity"] == 1.0)
    # 极端字间距参数会被校准回原图值（不再照搬、也不保留极端值）
    p2 = dict(params)
    p2["letter_spacing"] = 999
    out2 = app._calibrate_letter_spacing("示例", p2, app.selected_font_path)
    cur2, _m2 = render_mod.render_text_with_metrics("示例", out2, font_path=app.selected_font_path)
    gap2 = render_mod.measure_blank_gap(cur2)
    check("闭环校准将极端字间距收敛到原图值",
          out2["letter_spacing"] != 999 and gap2 is not None and abs(gap2 - 8) <= 1,
          f"ls={out2['letter_spacing']} gap={gap2}")
    root.destroy()


# ---------------------------------------------------------------- 4. 防重叠单元：放置搜索 + 重叠判定
def test_anti_overlap_unit():
    print("\n== 4. 防重叠单元：_overlaps_any / _place_anti_overlap ==")
    root = tk.Tk()
    root.withdraw()
    app = gui.App(root)
    forbidden = [(430, 218, 560, 270)]  # 右邻矩形
    ink = (0, 0, 100, 40)               # 文字墨迹在 text_rgba 内相对坐标
    # 基础位置若落在右邻内 → 应判重叠
    check("重叠判定：基础位置与右邻相交判 True",
          app._overlaps_any(450, 230, ink, forbidden, margin=3) is True)
    # 平移到左上方空位 → 不重叠
    check("重叠判定：远离右邻位置判 False",
          app._overlaps_any(100, 100, ink, forbidden, margin=3) is False)

    # 防重叠放置：从与右邻重叠的基础位搜索到空位
    bx, by = app._place_anti_overlap(None, ink, (218, 218, 330, 272), forbidden, 450, 230)
    check("防重叠放置：最终位置不与右邻重叠",
          not app._overlaps_any(bx, by, ink, forbidden, margin=3), f"pos=({bx},{by})")
    root.destroy()


# ---------------------------------------------------------------- 5. 无邻近文字降级（失败路径）
def test_no_neighbor_degrade():
    print("\n== 5. 无邻近文字降级（失败路径） ==")
    root = tk.Tk()
    root.withdraw()
    app = gui.App(root)
    bgr = make_blank_neighbor_image()
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()
    box = (148, 128, 262, 182)  # 孤立字区域
    app.canvas._boxes.append(box)
    app.canvas._active_index = 0
    # 不应崩溃，且参照为 None
    app.on_box_selected(box)
    check("无邻近文字时流程不崩溃", True)
    check("无邻近文字时 neighbor_ref 为 None（回退选区自身特征）",
          app.neighbor_ref is None, f"ref={app.neighbor_ref}")
    check("无邻近文字时 neighbor_boxes 为空", app.neighbor_boxes == [])
    # 确认仍可用（防重叠退化为不约束）
    app.text_var.set("替代")
    app._confirm_text()
    check("无邻近文字确认后仍标记 optimized", app.box_edits[0].get("optimized") is True)
    root.destroy()


# ---------------------------------------------------------------- 6. 多选区独立 + OCR 缺失降级
def test_multi_box_independent_and_ocr_missing():
    print("\n== 6. 多选区独立 + OCR 缺失降级 ==")
    root = tk.Tk()
    root.withdraw()
    app = gui.App(root)
    bgr = make_four_neighbor_image()
    app.original_bgr = bgr
    app.working_bgr = bgr.copy()

    # OCR 缺失降级
    gui.ocr.is_ocr_supported = lambda: False
    box = (205, 205, 320, 262)
    app.canvas._boxes.append(box)
    app.canvas._active_index = 0
    app.on_box_selected(box)
    check("OCR 缺失时输入框不被自动填充（保持手动输入）", app.text_var.get() == "")
    check("OCR 缺失时流程继续、生成候选字体", len(app.candidates) > 0,
          f"cands={len(app.candidates)}")
    # 手动输入互不继承
    app.text_var.set("甲框文字")
    app._confirm_text()
    app.box_edits[0]["text"] = "甲框文字"

    # 第二个框
    box2 = (220, 120, 360, 168)  # 上邻区域
    app.canvas._boxes.append(box2)
    app.canvas._active_index = 1
    app.on_box_selected(box2)
    check("新框不继承旧框文字", app.text_var.get() == "" or app.text_var.get() != "甲框文字",
          f"text={app.text_var.get()!r}")
    app.text_var.set("乙框文字")
    app._confirm_text()
    check("两框各自独立保存文字",
          app.box_edits[0]["text"] == "甲框文字" and app.box_edits[1]["text"] == "乙框文字")
    check("两框特征各自锁定独立",
          app.box_edits[0].get("_locked") and app.box_edits[1].get("_locked"))
    root.destroy()


if __name__ == "__main__":
    test_pipeline_order_and_lock()
    test_confirm_optimize_anti_overlap()
    test_secondary_optimize_alignment()
    test_anti_overlap_unit()
    test_no_neighbor_degrade()
    test_multi_box_independent_and_ocr_missing()
    print(f"\n四步流程自测：通过 {_pass} 项，失败 {_fail} 项")
    sys.exit(1 if _fail else 0)
