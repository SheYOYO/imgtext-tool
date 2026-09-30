"""确认替换后原文字彻底消失 + 预览「井」字辅助线 回归自测。

覆盖佘先生本轮两条要求：

1. **擦除严格限制在选框内**（本轮需求变更）：
   擦除范围就是**选框矩形本身**，不再取「原文字真实墨迹 ∪ 选框 ∪ 替换文字占位」
   的并集——框外的上下文文字/纹理一个像素都不许动。
   因此本文件的断言口径同步改为：
     · **框内**：原文字必须清零（残留红 = 0）；
     · **框外**：原文字必须原样保留（不允许被顺手抹掉）。
   检测手法：原文字画成**红色**、替换文字渲染成**蓝色**，
   于是「原文字残留」= 残留红色像素，可按框内/框外分别精确计数。

2. **预览「井」字辅助线**：预览界面里用四根两端延长的直线围住替换文字，
   相邻直线相交形成井字形，方便对齐上下文；但**只画在显示副本上**，
   确定替换后辅助线绝不能出现在效果图 / 导出图上。
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


def _measure_ink_bbox(bgr, bg=(240, 240, 240)):
    """实测图中「非背景」像素的包围盒 (x0, y0, x1, y1)。

    注意：不能用 textbbox 的返回值推算——它给出的是**绘制原点**相关的框，
    与真正的墨迹相差一个字体 ascent 偏移（本例 CJK 字体实测差 7px），
    会让断言基准整体偏移，误判成「擦除范围没盖住原文字」。
    """
    diff = np.abs(bgr.astype(int) - np.array(bg[::-1], dtype=int)).sum(axis=2)
    ys, xs = np.where(diff > 40)
    if ys.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def make_image_colored(text, size=40, rgb=(255, 0, 0)):
    """生成浅灰底 + 指定颜色文字的测试图，返回 (BGR, 文字**真实墨迹**包围盒)。"""
    pool = config.resolve_font_pool()
    fp = next((f["path"] for f in pool if f.get("cn")), None)
    font = ImageFont.truetype(fp, size) if fp else ImageFont.load_default()
    tmp = Image.new("RGB", (1, 1), (240, 240, 240))
    bb = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    W, H = bb[2] + 160, bb[3] + 160
    img = Image.new("RGB", (W, H), (240, 240, 240))
    x, y = 40 - bb[0], 60 - bb[1]
    ImageDraw.Draw(img).text((x, y), text, font=font, fill=rgb)
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return bgr, _measure_ink_bbox(bgr)


def _setup(text="原始文字六字", size=40, rgb=(255, 0, 0)):
    import tkinter as tk
    from tkinter import messagebox
    from app import gui
    messagebox.showinfo = lambda *a, **k: None
    messagebox.showerror = lambda *a, **k: None
    root = tk.Tk()
    root.withdraw()
    app = gui.App(root)
    bgr, box = make_image_colored(text, size, rgb)
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


def _new_box(app, box):
    app.canvas._boxes.append(box)
    app.canvas._active_index = len(app.canvas._boxes) - 1
    app.on_box_selected(box, is_resize=False)
    app.root.update()


def _red_mask(bgr):
    """「原文字（红色）」像素掩码。"""
    b = bgr[:, :, 0].astype(int); g = bgr[:, :, 1].astype(int); r = bgr[:, :, 2].astype(int)
    return (r > 120) & (g < 90) & (b < 90)


def _red_count(bgr):
    """残留「原文字（红色）」像素数（整图）。"""
    return int(_red_mask(bgr).sum())


def _red_inside(bgr, box):
    """**选框内**的残留红色像素数——新需求下必须为 0。"""
    x0, y0, x1, y1 = [int(v) for v in box]
    return int(_red_mask(bgr)[max(0, y0):y1, max(0, x0):x1].sum())


def _red_outside(bgr, box):
    """**选框外**的红色像素数——新需求下必须原样保留（不许被顺手擦掉）。"""
    return int(_red_mask(bgr).sum()) - _red_inside(bgr, box)


def _outside_equal(a, b, box):
    """两张图「选框以外」的像素是否逐字节一致（框外零改动的最硬证据）。"""
    if a is None or b is None or a.shape != b.shape:
        return False
    x0, y0, x1, y1 = [int(v) for v in box]
    m = np.ones(a.shape[:2], dtype=bool)
    m[max(0, y0):y1, max(0, x0):x1] = False
    return bool(np.array_equal(a[m], b[m]))


def _blue_count(bgr):
    """「替换文字（蓝色）」像素数。"""
    b = bgr[:, :, 0].astype(int); g = bgr[:, :, 1].astype(int); r = bgr[:, :, 2].astype(int)
    return int(((b > 120) & (r < 90) & (g < 90)).sum())


# ============================================ 1. 擦除范围覆盖原文字真实墨迹
def test_text_extent_region_covers_real_ink():
    print("\n== 1. _text_extent_region 覆盖原文字真实墨迹范围（诊断工具，已不参与擦除）==")
    root, app, full = _setup()
    # 选框比文字小一圈（文字溢出选框 6px）——正是「只擦选框会留残字」的场景
    box = (full[0] + 6, full[1] + 6, full[2] - 6, full[3] - 6)
    _new_box(app, box)
    _flush(app)
    region = app._text_extent_region(box)
    check("返回了扩展后的擦除区域", region is not None, f"region={region}")
    check("擦除范围覆盖到文字左边界之外（含溢出部分）", region[0] <= full[0],
          f"region[0]={region[0]} vs 文字左={full[0]}")
    check("擦除范围覆盖到文字上边界之外", region[1] <= full[1],
          f"region[1]={region[1]} vs 文字上={full[1]}")
    check("擦除范围覆盖到文字右边界之外", region[2] >= full[2],
          f"region[2]={region[2]} vs 文字右={full[2]}")
    check("擦除范围覆盖到文字下边界之外", region[3] >= full[3],
          f"region[3]={region[3]} vs 文字下={full[3]}")
    root.destroy()


# ============================================ 2. 替换渲染：框内清零 / 框外保留
def test_replace_erases_original_completely():
    print("\n== 2. 替换渲染后框内原文字清零、框外原文字保留 ==")
    root, app, full = _setup()
    box = (full[0] + 6, full[1] + 6, full[2] - 6, full[3] - 6)
    _new_box(app, box)
    _flush(app)
    base_red = _red_count(app.original_bgr)
    base_out = _red_outside(app.original_bgr, box)
    check("原始图中确有红色原文字", base_red > 100, f"红色像素={base_red}")
    check("原文字确实溢出选框（框外还有红色，正控）", base_out > 50,
          f"框外红={base_out}")

    # 替换文字设定为蓝色，长度故意比选框长（会超出选框放置）
    app.text_var.set("替换后文字更长一串")
    app.color_var = (0, 0, 255)
    result = app._do_render(app.original_bgr.copy(), optimize=False)
    check("替换渲染返回了图像", result is not None)
    check("框内原文字被擦除（框内红色 = 0）", _red_inside(result, box) == 0,
          f"框内残留红={_red_inside(result, box)}")
    check("框外原文字未被抹掉（擦除严格限制在选框内）",
          _red_outside(result, box) > base_out * 0.5,
          f"框外红={_red_outside(result, box)}（原始框外 {base_out}）")
    check("替换文字已渲染（蓝色像素 > 0）", _blue_count(result) > 50,
          f"蓝色像素={_blue_count(result)}")
    root.destroy()


def test_replacement_never_overlaps_original():
    print("\n== 3. 替换文字落在选框内的部分：无原文字残留 ==")
    root, app, full = _setup()
    box = (full[0] + 6, full[1] + 6, full[2] - 6, full[3] - 6)
    _new_box(app, box)
    _flush(app)
    app.text_var.set("替换后文字更长一串")
    app.color_var = (0, 0, 255)
    result = app._do_render(app.original_bgr.copy(), optimize=False)
    tb = app._last_text_bbox
    check("已记录替换文字在整图中的包围盒", tb is not None, f"bbox={tb}")
    if tb is not None:
        # 只校验「替换文字 ∩ 选框」的区域：框外不属于本次擦除的职责范围
        ix0, iy0 = max(tb[0], box[0]), max(tb[1], box[1])
        ix1, iy1 = min(tb[2], box[2]), min(tb[3], box[3])
        if ix1 - ix0 >= 1 and iy1 - iy0 >= 1:
            sub = result[iy0:iy1, ix0:ix1]
            check("替换文字落在选框内的部分无原文字残留（不与替换文字重合）",
                  _red_count(sub) == 0, f"占位区∩框内 残留红色像素={_red_count(sub)}")
        else:
            check("替换文字落在选框内的部分无原文字残留（不与替换文字重合）",
                  True, "替换文字完全在框外，跳过")
    root.destroy()


def test_no_text_still_erases_extent():
    print("\n== 4. 无替换文字时：只擦选框内，框外逐字节不动 ==")
    root, app, full = _setup()
    box = (full[0] + 6, full[1] + 6, full[2] - 6, full[3] - 6)
    _new_box(app, box)
    _flush(app)
    app.text_var.set("")            # 不输入替换文字 → 只擦除
    result = app._do_render(app.original_bgr.copy(), optimize=False)
    check("无替换文字时仍返回擦除结果", result is not None)
    check("框内原文字被擦除（框内红色 = 0）", _red_inside(result, box) == 0,
          f"框内残留红={_red_inside(result, box)}")
    check("框外像素与原图逐字节一致（严格限制在选框内）",
          _outside_equal(result, app.original_bgr, box))
    check("无替换文字时不记录辅助线包围盒", app._last_text_bbox is None)
    root.destroy()


# ============================================ 5. 预览「井」字辅助线
def test_preview_guides_display_only():
    print("\n== 5. 预览「井」字辅助线只画在显示副本，绝不进入效果图 ==")
    root, app, full = _setup()
    box = (full[0] + 6, full[1] + 6, full[2] - 6, full[3] - 6)
    _new_box(app, box)
    _flush(app)
    app.text_var.set("替换文字")
    app.color_var = (0, 0, 255)
    app._refresh_preview()
    _flush(app)
    check("预览已生成 pending_result", app.pending_result is not None)
    check("已记录替换文字包围盒（辅助线定位用）", app._last_text_bbox is not None)

    pending_snap = app.pending_result.copy()
    working_snap = app.working_bgr.copy()

    # 5a) 显示副本上确实画了辅助线
    crop = app._preview_crop(app.pending_result)
    check("预览裁剪区域已记录（供坐标换算）",
          getattr(app, "_last_crop_rect", None) is not None)
    crop_snap = crop.copy()
    disp = app._preview_display_image(crop)
    check("显示副本上叠加了辅助线（与源图不同）", not np.array_equal(disp, crop_snap))
    check("辅助线未污染裁剪源图（crop 视图保持不变）",
          np.array_equal(crop, crop_snap))

    # 5b) 刷新预览面板不会把辅助线写回 pending_result / 工作图
    app._update_preview_panel()
    _flush(app)
    check("刷新预览面板后 pending_result 未被写入辅助线",
          np.array_equal(app.pending_result, pending_snap))
    check("刷新预览面板后工作图未被写入辅助线",
          np.array_equal(app.working_bgr, working_snap))

    # 5c) 不触发二次排版优化时，确定替换后工作图与预览结果**字节级一致**
    #     —— 这是「辅助线没被一起提交」最强的证据
    pre_apply = app.pending_result.copy()
    app.apply_replace()
    _flush(app)
    check("确定替换后工作图 == 无辅助线的预览结果（字节级一致）",
          np.array_equal(app.working_bgr, pre_apply))

    # 5d) 完整「确认」流程（会触发二次排版优化，成品**允许**与预览略有差异），
    #     但辅助线绝不能出现在成品图里。用「显示副本 vs 源图」的差分掩码精确定位
    #     辅助线像素（与颜色无关，不受抗锯齿影响）。
    app.text_var.set("替换文字")
    app._refresh_preview()
    _flush(app)
    crop2 = app._preview_crop(app.pending_result)
    disp2 = app._preview_display_image(crop2)
    mask = np.abs(disp2.astype(int) - crop2.astype(int)).sum(axis=2) > 12
    guide_px = int(mask.sum())
    check("辅助线差分掩码非空（正控：确实画上了）", guide_px > 50, f"辅助线像素={guide_px}")

    cx0, cy0, cx1, cy1 = app._last_crop_rect
    app._confirm_text()
    _flush(app)
    app.apply_replace()
    _flush(app)
    final = app.working_bgr[cy0:cy1, cx0:cx1]
    same = (np.abs(final.astype(int) - disp2.astype(int)).sum(axis=2) <= 12) & mask
    check("确认后成品图中不含辅助线像素", int(same.sum()) == 0,
          f"残留辅助线像素={int(same.sum())}")
    check("确认后框内原文字已消失", _red_inside(app.working_bgr, box) == 0,
          f"框内残留红色像素={_red_inside(app.working_bgr, box)}")
    check("确认后替换文字仍在（成品内容正确）", _blue_count(app.working_bgr) > 50,
          f"蓝色像素={_blue_count(app.working_bgr)}")
    root.destroy()


def test_preview_guides_hidden_when_original_or_no_text():
    print("\n== 6. 显示原图 / 无替换文字时不画辅助线 ==")
    root, app, full = _setup()
    box = (full[0] + 6, full[1] + 6, full[2] - 6, full[3] - 6)
    _new_box(app, box)
    _flush(app)
    app.text_var.set("替换文字")
    app.color_var = (0, 0, 255)
    app._refresh_preview()
    _flush(app)
    crop = app._preview_crop(app.pending_result)
    crop_snap = crop.copy()

    # 切到「显示原图」模式 → 不画辅助线
    app._preview_show_original = True
    disp_orig = app._preview_display_image(crop)
    check("显示原图模式下不叠加辅助线", np.array_equal(disp_orig, crop_snap))
    app._preview_show_original = False

    # 无替换文字（包围盒为空）→ 不画辅助线
    app._last_text_bbox = None
    disp_none = app._preview_display_image(crop)
    check("无替换文字时不叠加辅助线", np.array_equal(disp_none, crop_snap))

    # 异常情况（crop 为 None）不应崩溃
    check("crop 为 None 时安全返回", app._preview_display_image(None) is None)
    root.destroy()


# ============================================ 7. 颜色通道不互换（红蓝不能反）
def test_color_channels_not_swapped():
    """渲染层是 PIL 的 RGB，底图是 OpenCV 的 BGR —— 合成时必须翻转通道。

    一旦漏掉翻转：要蓝字出来是红字、要红字出来是蓝字（黑白灰三通道相同才看不出）。
    这里用红/蓝两个方向双向夹住，防止回归。
    """
    print("\n== 7. 替换文字颜色通道不互换（RGB 渲染层 → BGR 底图）==")
    # 方向一：请求蓝色
    root, app, full = _setup()
    box = (full[0] + 6, full[1] + 6, full[2] - 6, full[3] - 6)
    _new_box(app, box)
    _flush(app)
    app.text_var.set("替换文字")
    app.color_var = (0, 0, 255)          # RGB 蓝
    r_blue = app._do_render(app.original_bgr.copy(), optimize=False)
    # 通道一旦互换：要蓝 → 框内会渲染出红色，故用「框内红 = 0」反向夹住
    check("请求蓝色 → 成品里是蓝色（不是红色）",
          _blue_count(r_blue) > 50 and _red_inside(r_blue, box) == 0,
          f"蓝={_blue_count(r_blue)} 框内红={_red_inside(r_blue, box)}")
    root.destroy()

    # 方向二：请求红色
    root, app, full = _setup()
    box = (full[0] + 6, full[1] + 6, full[2] - 6, full[3] - 6)
    _new_box(app, box)
    _flush(app)
    app.text_var.set("替换文字")
    app.color_var = (255, 0, 0)          # RGB 红
    r_red = app._do_render(app.original_bgr.copy(), optimize=False)
    check("请求红色 → 成品里是红色（不是蓝色）",
          _red_inside(r_red, box) > 50 and _blue_count(r_red) <= 10,
          f"框内红={_red_inside(r_red, box)} 蓝={_blue_count(r_red)}")
    root.destroy()


def main():
    test_text_extent_region_covers_real_ink()
    test_replace_erases_original_completely()
    test_replacement_never_overlaps_original()
    test_no_text_still_erases_extent()
    test_preview_guides_display_only()
    test_preview_guides_hidden_when_original_or_no_text()
    test_color_channels_not_swapped()
    print(f"\n自测：通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print("  - " + f)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
