"""需求二（PPT 式智能参考线）回归：常驻辅助框 + 智能对齐线（对齐闪现 + 吸附）。

验证点：
1. _alignment_references 提供画布中线（无其它框时）。
2. _smart_alignments：替换文字对齐基准接近参考线 → 返回对齐线 + 吸附偏移。
3. _apply_smart_snap：渲染前把文字平移到对齐位置（仅平移、不改样式）。
4. _draw_smart_text_guides：对齐时在对应参考线画贯穿整幅品红实线；不对齐则不画；
   只画在显示副本（不写回工作图/导出图）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tkinter as tk
from tkinter import messagebox
from app import gui

messagebox.showinfo = lambda *a, **k: None
messagebox.showerror = lambda *a, **k: None

from test_replace_erase_preview_guides import _setup

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))


def _magenta_count(img):
    b = img[:, :, 0].astype(int); g = img[:, :, 1].astype(int); r = img[:, :, 2].astype(int)
    return int(((b > 200) & (r > 200) & (g < 60)).sum())


def test_references_has_canvas_center():
    root, app, full = _setup()
    app.original_bgr = np.full((200, 300, 3), 240, np.uint8)
    refs = app._alignment_references()
    vs = [p for a, p in refs if a == "v"]
    hs = [p for a, p in refs if a == "h"]
    check("参考线含画布垂直中线 x=150", 150.0 in vs, f"vs={vs}")
    check("参考线含画布水平中线 y=100", 100.0 in hs, f"hs={hs}")
    root.destroy()


def test_alignment_detection_and_snap():
    root, app, full = _setup()
    app.original_bgr = np.full((200, 300, 3), 240, np.uint8)
    app.working_bgr = app.original_bgr.copy()
    cur = np.zeros((40, 100, 4), np.uint8)
    cur[10:30, 20:80, :] = 255          # 墨迹块 → 墨迹框 (20,10,80,30)
    # 放置到 (97,80) → 整图墨迹框中心 (147,100)，距画布中线(150,100)横向 3px（<阈值6）
    placed = (97, 80)
    aligned, sdx, sdy = app._smart_alignments(
        app._abs_text_bbox(cur, placed), app._alignment_references(), thr=6)
    check("横向对齐被检测到", any(a == "v" for a, _ in aligned))
    check("吸附横向偏移 = +3（趋近中线）", abs(sdx - 3) < 1.0, f"sdx={sdx}")

    snapped = app._apply_smart_snap(cur, placed)
    tb = app._abs_text_bbox(cur, snapped)
    check("吸附后文字中心对齐到画布水平中线 x=150",
          abs(((tb[0] + tb[2]) / 2.0) - 150) < 1.5, f"cx={((tb[0]+tb[2])/2.0)}")
    check("吸附后文字中心对齐到画布垂直中线 y=100",
          abs(((tb[1] + tb[3]) / 2.0) - 100) < 1.5, f"cy={((tb[1]+tb[3])/2.0)}")
    # 远离中线时不应吸附
    far = (10, 10)
    sf = app._apply_smart_snap(cur, far)
    check("远离参考线时不吸附（位置不变）", sf == far, f"sf={sf}")
    root.destroy()


def test_smart_guide_draw_only_when_aligned():
    root, app, full = _setup()
    app.original_bgr = np.full((200, 300, 3), 240, np.uint8)
    W, H = 300, 200
    # 对齐场景：墨迹框中心恰好在画布中线 (150,100)
    tb_align = (140, 90, 160, 110)
    img = np.full((H, W, 3), 200, np.uint8)
    img_copy = img.copy()
    app._draw_smart_text_guides(img_copy, tb_align, (0, 0), W, H)
    m_align = _magenta_count(img_copy)
    check("对齐时画出贯穿整幅品红智能线", m_align > W, f"品红像素={m_align}")
    # 原图未被污染（画在副本上）
    check("绘制不写回传入图（仅显示副本）", np.array_equal(img, np.full((H, W, 3), 200, np.uint8)))

    # 不对齐场景：墨迹框在角落
    tb_far = (10, 10, 30, 30)
    img2 = np.full((H, W, 3), 200, np.uint8)
    app._draw_smart_text_guides(img2, tb_far, (0, 0), W, H)
    check("不对齐时不画智能线", _magenta_count(img2) == 0)
    root.destroy()


def main():
    test_references_has_canvas_center()
    test_alignment_detection_and_snap()
    test_smart_guide_draw_only_when_aligned()
    print(f"\n共 {len(PASS)+len(FAIL)} 项，通过 {len(PASS)}，失败 {len(FAIL)}")
    if FAIL:
        print("失败项：", FAIL)
    return len(FAIL) == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
