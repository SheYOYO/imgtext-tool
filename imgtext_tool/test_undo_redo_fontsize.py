"""「恢复（redo）」新功能 + 「手调字号确认后跳变」修复 回归自测。

佘先生本轮两条：

1. **加一个和撤销相反的「恢复」功能**：撤销下来的（图像 + 各框编辑状态 + 未提交
   草稿）要能原样还原回去；产生新提交 / 框集合变化后恢复栈作废；空栈只提示不崩。
2. **修复「手调过的替代文字大小，点确定后会变化」**：根因是已确认框的预览直接沿用
   `optimized_params`（冻结值），用户手调字号后预览毫无反应，一点「确认替换文字」
   才突然跳到新大小。修复后：手调字号优先级最高，预览当场反映、确认冻结同一个
   值，且预览与确认成品走同一套间距拟合 → 确定前后完全一致，不再跳变。

判据用 `em_size`（渲染实际使用的 em 字号），比墨迹包围盒更稳（墨迹高会受字间距
与倾斜旋转影响，不能当字号度量）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from test_replace_erase_preview_guides import (
    make_image_colored, _setup, _new_box, _flush, _blue_count,
)
from test_box_create_erase import _create_box, _long_press_move, _evt

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))


def _em(app):
    """最近一次渲染实际使用的 em 字号（None 表示还没渲染过）。"""
    return (getattr(app, "_last_metrics", {}) or {}).get("em_size")


def _tweak_font_size(app, value):
    """模拟用户拖动「字号」滑杆（写滑块 + 记进 _user_tweaks，与 _on_slider 同效）。"""
    idx = app.canvas.get_active_index()
    app.vars["font_size"].set(value)
    if 0 <= idx:
        app._user_tweaks.setdefault(idx, {})["font_size"] = float(value)


def _make_app(text="原始文字六字", size=40):
    root, app, full = _setup(text, size)
    return root, app, full


# ============================================ 1. 恢复：应用替换 → 撤销 → 恢复
def test_redo_after_undo_apply():
    print("\n== 1. 应用替换后：撤销 → 恢复（图像 + 框状态一起回来）==")
    root, app, full = _make_app()
    box = (full[0] + 4, full[1] + 4, full[2] - 4, full[3] - 4)
    _create_box(app.canvas, box)
    _flush(app)
    app.text_var.set("替换文字")
    app.color_var = (0, 0, 255)
    app._confirm_text()
    _flush(app)
    app.apply_replace()
    _flush(app)
    applied = app.working_bgr.copy()
    blue_applied = _blue_count(applied)
    check("正控：应用后成品里有替换文字", blue_applied > 50, f"蓝色像素={blue_applied}")

    app.undo()
    _flush(app)
    check("撤销后替换文字消失", _blue_count(app.working_bgr) <= 5,
          f"蓝色像素={_blue_count(app.working_bgr)}")
    check("撤销后有可恢复的步骤", app.has_redo())

    app.redo()
    _flush(app)
    check("恢复后图像与撤销前**逐字节一致**", np.array_equal(app.working_bgr, applied))
    check("恢复后替换文字回来了", _blue_count(app.working_bgr) == blue_applied,
          f"蓝色像素={_blue_count(app.working_bgr)}")
    idx = app.canvas.get_active_index()
    st = app.box_edits.get(idx, {}) or {}
    check("恢复后框状态也回来了（文字/冻结参数）",
          st.get("text") == "替换文字" and bool(st.get("optimized_params")),
          f"text={st.get('text')} op={'有' if st.get('optimized_params') else '无'}")
    check("恢复后还可以继续撤销", app.undo() is None and _blue_count(app.working_bgr) <= 5)
    root.destroy()


# ============================================ 2. 多步撤销 / 多步恢复
def test_redo_multi_step():
    print("\n== 2. 多步：应用两处 → 撤销两步 → 恢复两步 ==")
    root, app, full = _make_app()
    box = (full[0] + 4, full[1] + 4, full[2] - 4, full[3] - 4)
    _create_box(app.canvas, box)
    _flush(app)
    app.text_var.set("第一处")
    app.color_var = (0, 0, 255)
    app._confirm_text()
    _flush(app)
    app.apply_replace()
    _flush(app)
    snap1 = app.working_bgr.copy()

    # 第二处：放到第一处**下方空白处**（互不重叠，避免新建擦除把第一处成果盖掉）
    h, w = app.original_bgr.shape[:2]
    bh = box[3] - box[1]
    by0 = min(h - bh - 2, box[3] + 30)
    box2 = (box[0], by0, box[2], by0 + bh)
    _create_box(app.canvas, box2)
    _flush(app)
    app.text_var.set("第二处")
    app.color_var = (0, 0, 255)
    app._confirm_text()
    _flush(app)
    app.apply_replace()
    _flush(app)
    snap2 = app.working_bgr.copy()
    check("正控：两处应用后图像各不相同",
          not np.array_equal(snap1, snap2) and not np.array_equal(snap1, app.original_bgr))

    # 帧序列：初始 → 擦除1 → 应用1 → 应用2（第二处落在空白处，新建不产生擦除帧）
    app.undo(); _flush(app)
    check("撤销一步 == 第一处应用后的图像", np.array_equal(app.working_bgr, snap1),
          f"蓝={_blue_count(app.working_bgr)}")
    app.undo(); _flush(app)
    check("再撤销一步 == 第一处替换之前（替换文字已消失）",
          _blue_count(app.working_bgr) == 0, f"蓝={_blue_count(app.working_bgr)}")
    app.redo(); _flush(app)
    check("恢复一步 == 第一处应用后", np.array_equal(app.working_bgr, snap1))
    check("恢复一步后仍有可恢复步骤", app.has_redo())
    app.redo(); _flush(app)
    check("再恢复一步 == 第二处应用后", np.array_equal(app.working_bgr, snap2))
    check("恢复栈已空", not app.has_redo())
    root.destroy()


# ============================================ 3. 撤销「未提交草稿」也能恢复
def test_redo_discarded_draft():
    print("\n== 3. 确认后未应用就撤销（丢弃草稿）→ 恢复能把草稿找回来 ==")
    root, app, full = _make_app()
    # 直接建框（不触发新建擦除的历史帧），让 history 只有初始帧 → 走「情形 1」
    box = (full[0] + 4, full[1] + 4, full[2] - 4, full[3] - 4)
    _new_box(app, box)
    _flush(app)
    app.text_var.set("草稿文字")
    app.color_var = (0, 0, 255)
    app._refresh_preview()
    _flush(app)
    check("正控：已生成未提交的草稿预览", app.pending_result is not None)
    draft_blue = _blue_count(app.pending_result)
    check("正控：草稿里有替换文字", draft_blue > 20, f"蓝色={draft_blue}")

    app.undo()
    _flush(app)
    check("撤销后草稿被丢弃", app.pending_result is None)
    check("撤销后草稿可恢复", app.has_redo())

    app.redo()
    _flush(app)
    check("恢复后草稿回来了（pending_result 复原）", app.pending_result is not None)
    if app.pending_result is not None:
        check("恢复的草稿内容与撤销前一致",
              _blue_count(app.pending_result) == draft_blue,
              f"蓝色={_blue_count(app.pending_result)}")
    root.destroy()


# ============================================ 4. 新提交 / 框集合变化 → 恢复栈作废
def test_redo_stack_invalidated():
    print("\n== 4. 产生新提交 / 删除·复制选框后，恢复栈作废（标准 redo 语义）==")
    root, app, full = _make_app()
    box = (full[0] + 4, full[1] + 4, full[2] - 4, full[3] - 4)
    _create_box(app.canvas, box)
    _flush(app)
    app.text_var.set("替换文字")
    app.color_var = (0, 0, 255)
    app._confirm_text()
    _flush(app)
    app.apply_replace()
    _flush(app)
    app.undo()
    _flush(app)
    check("撤销后存在可恢复步骤（正控）", app.has_redo())

    # 4a) 新的应用提交 → 作废
    app.text_var.set("再来一次")
    app._refresh_preview()
    _flush(app)
    app.apply_replace()
    _flush(app)
    check("新提交后恢复栈被清空", not app.has_redo())

    # 4b) 删除选框 → 作废
    app.undo(); _flush(app)
    had = app.has_redo()
    app.delete_selected_box(); _flush(app)
    check("删除选框后恢复栈被清空", had and not app.has_redo(), f"删除前={had}")

    # 4c) 复制选框 → 作废
    root.destroy()
    root, app, full = _make_app()
    _create_box(app.canvas, box)
    _flush(app)
    app.text_var.set("替换文字")
    app.color_var = (0, 0, 255)
    app._confirm_text(); _flush(app)
    app.apply_replace(); _flush(app)
    app.undo(); _flush(app)
    had2 = app.has_redo()
    app.copy_selected_box(); _flush(app)
    check("复制选框后恢复栈被清空", had2 and not app.has_redo(), f"复制前={had2}")

    # 4d) 清空框选 → 作废
    app.undo(); _flush(app)
    had3 = app.has_redo()
    app.clear_boxes(); _flush(app)
    check("清空框选后恢复栈被清空", had3 and not app.has_redo(), f"清空前={had3}")
    root.destroy()


# ============================================ 5. 失败路径：空栈恢复不崩溃
def test_redo_empty_stack():
    print("\n== 5. 失败路径：空栈 / 反复恢复 都不崩溃 ==")
    root, app, full = _make_app()
    try:
        app.redo()
        app.redo()
        ok = True
        detail = app.status_var.get()
    except Exception as e:
        ok = False
        detail = str(e)
    check("空栈点恢复不崩溃（只提示）", ok, detail)

    # 只做一次撤销 → 只能恢复一次，第二次应为空栈且不崩
    box = (full[0] + 4, full[1] + 4, full[2] - 4, full[3] - 4)
    _create_box(app.canvas, box)
    _flush(app)
    app.undo(); _flush(app)
    try:
        app.redo(); _flush(app)
        app.redo(); _flush(app)
        ok2 = True
    except Exception as e:
        ok2 = False
        detail2 = str(e)
    check("恢复次数超过可恢复步数时不崩溃", ok2, detail2 if not ok2 else "")
    check("恢复栈用尽后为空", not app.has_redo())
    root.destroy()


# ============================================ 6. 手调字号：未确认框（预览=手调值，确认不跳）
def test_manual_font_size_before_confirm():
    print("\n== 6. 未确认框：手调字号预览即生效，确认后不跳变 ==")
    root, app, full = _make_app()
    box = (full[0] + 4, full[1] + 4, full[2] - 4, full[3] - 4)
    _create_box(app.canvas, box)
    _flush(app)
    app.text_var.set("替换文字四个")
    _flush(app)
    auto_slider = int(app.vars["font_size"].get())

    _tweak_font_size(app, 70)
    app._refresh_preview(); _flush(app)
    em_preview = _em(app)
    check("手调后预览使用手调字号", em_preview == 70, f"em={em_preview}")

    app._confirm_text(); _flush(app)
    app._refresh_preview(); _flush(app)
    em_after = _em(app)
    check("确认后字号与手调预览**完全一致**（不跳变）", em_after == em_preview,
          f"确认前={em_preview} 确认后={em_after}")
    check("手调值确实 != 自动识别值（正控）", auto_slider != 70, f"自动字号={auto_slider}")
    root.destroy()


# ============================================ 7. 手调字号：已确认框（核心修复）
def test_manual_font_size_after_confirm_no_jump():
    print("\n== 7. 已确认框：手调字号预览**当场**生效，确定前后零跳变（本轮核心修复）==")
    root, app, full = _make_app()
    box = (full[0] + 4, full[1] + 4, full[2] - 4, full[3] - 4)
    _create_box(app.canvas, box)
    _flush(app)
    app.text_var.set("替换文字四个")
    _flush(app)
    app._confirm_text()
    _flush(app)
    app._refresh_preview(); _flush(app)
    em_confirmed = _em(app)
    check("正控：确认后已冻结字号", em_confirmed is not None, f"em={em_confirmed}")

    # 用户在已确认的框上把字号手动调小
    _tweak_font_size(app, 30)
    app._refresh_preview(); _flush(app)
    em_preview = _em(app)
    check("【核心】已确认框手调后，预览**立即**反映手调值（不再沿用冻结值）",
          em_preview == 30, f"预览 em={em_preview}（修复前会是 {em_confirmed}）")

    # 点「确认替换文字」→ 必须与预览一致，不许再跳
    app._confirm_text(); _flush(app)
    app._refresh_preview(); _flush(app)
    em_after = _em(app)
    check("【核心】确定后字号 == 确定前预览字号（零跳变）", em_after == em_preview,
          f"确定前={em_preview} 确定后={em_after}")

    # 应用后进工作图，再重开该框，仍应是手调值
    app.apply_replace(); _flush(app)
    idx = app.canvas.get_active_index()
    app.on_box_select_same(idx); _flush(app)
    app._refresh_preview(); _flush(app)
    check("重开该框后仍是手调字号", _em(app) == 30, f"em={_em(app)}")
    check("滑块显示的就是手调值", int(app.vars["font_size"].get()) == 30,
          f"slider={app.vars['font_size'].get()}")
    root.destroy()


# ============================================ 8. 手调字号：移动选框后不打回
def test_manual_font_size_survives_move():
    print("\n== 8. 移动选框后手调字号不被打回 ==")
    root, app, full = _make_app()
    box = (full[0] + 4, full[1] + 4, full[2] - 4, full[3] - 4)
    _create_box(app.canvas, box)
    _flush(app)
    app.text_var.set("替换文字四个")
    app._confirm_text(); _flush(app)
    _tweak_font_size(app, 55)
    app._refresh_preview(); _flush(app)
    check("手调生效（正控）", _em(app) == 55, f"em={_em(app)}")

    x0, y0, x1, y1 = app.canvas.get_active_box()
    cxs, cys = app.canvas._to_canvas((x0 + x1) / 2, (y0 + y1) / 2)
    _long_press_move(app.canvas, cxs, cys, 40, 30)
    _flush(app)
    app._refresh_preview(); _flush(app)
    check("真实事件移动选框后仍是手调字号", _em(app) == 55, f"em={_em(app)}")
    root.destroy()


# ============================================ 9. 回归：未手调时仍走自动值
def test_auto_font_size_unchanged():
    print("\n== 9. 回归：没手调过时，确认前后都按自动识别字号（不被新逻辑带偏）==")
    root, app, full = _make_app()
    box = (full[0] + 4, full[1] + 4, full[2] - 4, full[3] - 4)
    _create_box(app.canvas, box)
    _flush(app)
    app.text_var.set("替换文字四个")
    _flush(app)
    auto_slider = int(app.vars["font_size"].get())
    app._refresh_preview(); _flush(app)
    em_preview = _em(app)
    check("未手调：预览用自动值 == 滑块值", em_preview == auto_slider,
          f"em={em_preview} slider={auto_slider}")
    app._confirm_text(); _flush(app)
    app._refresh_preview(); _flush(app)
    check("未手调：确认后仍是自动值", _em(app) == auto_slider, f"em={_em(app)}")
    root.destroy()


def main():
    test_redo_after_undo_apply()
    test_redo_multi_step()
    test_redo_discarded_draft()
    test_redo_stack_invalidated()
    test_redo_empty_stack()
    test_manual_font_size_before_confirm()
    test_manual_font_size_after_confirm_no_jump()
    test_manual_font_size_survives_move()
    test_auto_font_size_unchanged()
    print(f"\n恢复（redo）+ 手调字号 自测：通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print("  - " + f)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
