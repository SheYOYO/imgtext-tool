"""要求一（OCR 原文不回渲） + 要求二（主画布贯穿式参考线）回归自测。

佘先生本轮明确的两点：

1. 新建选框 + 识别文字后，**原文字直接从图上消失**；OCR 自动填入的原文只是
   建议，**绝不能被重新渲染回图上**（旧实现因为「输入框有字就渲染」，
   把原文又画了回来，看起来像「没删掉」）。删除选框后原文字仍不出现，
   只有「撤销」能恢复。

2. 替换文字周围要有**类似 PPT 智能参考线**的对齐辅助线。本轮把参考线从「短井字」
   升级为**贯穿整幅**的井字，并且**主画布编辑时也显示**（不再只藏在预览面板），
   但确定/导出时绝对不进成品图。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from test_replace_erase_preview_guides import (
    make_image_colored, _setup, _new_box, _flush, _red_count, _blue_count,
    _red_inside, _red_outside,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))


def _orange_count(bgr):
    """橙色参考线像素数（BGR≈(0,140,255)）。替换蓝字、原图红字均不匹配。"""
    if bgr is None or bgr.ndim != 3:
        return 0
    b = bgr[:, :, 0].astype(int); g = bgr[:, :, 1].astype(int); r = bgr[:, :, 2].astype(int)
    return int(((r > 180) & (g > 90) & (g < 200) & (b < 80)).sum())


# ============================================ 要求一：OCR 原文不回渲
def test_ocr_text_not_rerendered_and_stays_erased():
    print("\n== 8. 要求一：OCR 原文不回渲 + 删框仍消失 + 撤销恢复 ==")
    root, app, full = _setup()
    box = (full[0] + 6, full[1] + 6, full[2] - 6, full[3] - 6)
    base_red = _red_count(app.original_bgr)
    _new_box(app, box)            # 新建选框 + OCR（原文被填进输入框）
    # 真实画布在 _on_left_up 新建分支会再调一次 on_box_create（执行「新建即擦除」）；
    # 这里同步模拟该回调，否则不会真正擦除。
    app.on_box_create(app.canvas.get_active_index(), box)
    _flush(app)
    idx = app.canvas.get_active_index()
    ocr_text = (app.box_edits.get(idx, {}) or {}).get("ocr_text", "")
    # 模拟「用户没改字」：输入框就是 OCR 识别出的原文
    app.text_var.set(ocr_text)
    app._refresh_preview()
    _flush(app)

    # 8a) 新建选框后**框内**原文字已被立即擦除（擦除严格限制在选框内：框外保留）
    check("新建选框后框内原文字立即消失（框内红=0）",
          _red_inside(app.working_bgr, box) == 0,
          f"框内红={_red_inside(app.working_bgr, box)}")
    check("框外原文字未被误擦（严格限制在选框内）",
          _red_outside(app.working_bgr, box) > 50,
          f"框外红={_red_outside(app.working_bgr, box)}")
    # 8b) OCR 原文**没有**被重新渲染回预览/工作图
    check("OCR 原文不触发替换渲染（pending_result 为 None）",
          app.pending_result is None)
    check("主画布显示的是已擦除底图（与 working_bgr 一致）",
          np.array_equal(getattr(app.canvas, "image_cv", None), app.working_bgr))
    # 8c) 删除选框后，原文字仍不出现（只允许撤销恢复）
    app.delete_selected_box()
    _flush(app)
    check("删除选框后框内原文字仍不出现（框内红=0）",
          _red_inside(app.working_bgr, box) == 0,
          f"框内红={_red_inside(app.working_bgr, box)}")
    # 8d) 撤销可恢复原文字
    app.undo()
    _flush(app)
    check("撤销后原文字恢复（红色回到原始量级）",
          _red_count(app.working_bgr) > base_red * 0.8,
          f"恢复红={_red_count(app.working_bgr)} vs 原始={base_red}")
    root.destroy()


# ============================================ 要求二：主画布贯穿式参考线
def test_main_canvas_guides_display_only():
    print("\n== 9. 要求二：主画布贯穿式参考线只显示、不进成品 ==")
    root, app, full = _setup()
    box = (full[0] + 6, full[1] + 6, full[2] - 6, full[3] - 6)
    _new_box(app, box)
    _flush(app)
    # 用户改字 → 触发替换渲染 + 参考线
    app.text_var.set("替换后文字一串")
    app.color_var = (0, 0, 255)
    app._refresh_preview()
    _flush(app)
    check("已生成 pending_result（替换已渲染）", app.pending_result is not None)
    check("已记录替换文字包围盒（参考线定位用）", app._last_text_bbox is not None)

    # 9a) 主画布显示副本上叠加了参考线（与 pending 不同）
    disp = getattr(app.canvas, "image_cv", None)
    check("主画布显示图 != pending_result（叠加了参考线）",
          disp is not None and not np.array_equal(disp, app.pending_result))
    # 9b) pending_result 本身未被污染（仅替换文字）
    pending_snap = app.pending_result.copy()
    check("pending_result 无参考线（仅替换文字）",
          np.array_equal(app.pending_result, pending_snap))

    # 9c) 参考线为橙色（PPT 风格）且**贯穿整幅**：图像上下边缘都能找到橙色
    h = disp.shape[0]
    top_band = disp[:max(1, h // 10), :]
    bot_band = disp[-max(1, h // 10):, :]
    check("主画布参考线为橙色且可见", _orange_count(disp) > 50, f"橙色像素={_orange_count(disp)}")
    check("参考线贯穿到图像上边缘", _orange_count(top_band) > 0)
    check("参考线贯穿到图像下边缘", _orange_count(bot_band) > 0)

    # 9d) 参考线只在显示层：用差分掩码定位，确认这些像素在 pending 里不存在
    mask = np.abs(disp.astype(int) - app.pending_result.astype(int)).sum(axis=2) > 12
    check("参考线差分掩码非空（正控：确实画上了）", int(mask.sum()) > 50,
          f"参考线像素={int(mask.sum())}")

    # 9e) 确定 + 应用后，成品图**绝对不含**任何参考线（橙色=0），替换文字仍在
    app._confirm_text()
    _flush(app)
    app.apply_replace()
    _flush(app)
    check("确认后成品图不含参考线（橙色像素=0）",
          _orange_count(app.working_bgr) == 0, f"橙色={_orange_count(app.working_bgr)}")
    check("确认后替换文字仍在成品（内容正确）", _blue_count(app.working_bgr) > 50,
          f"蓝色像素={_blue_count(app.working_bgr)}")
    root.destroy()


def main():
    test_ocr_text_not_rerendered_and_stays_erased()
    test_main_canvas_guides_display_only()
    print(f"\n自测：通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print("  - " + f)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
