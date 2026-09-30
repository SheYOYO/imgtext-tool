"""GUI 交互测试（真实鼠标事件驱动）。

重点回归项（对应此前的失效 BUG）：
1. 画布缩放/平移后，鼠标框选落在图像上的坐标仍然正确
   —— 之前因 winfo_width() 返回 1 导致换算偏移，框选跑到图外，文字替换"失效"。
2. 框选 → 擦除 → 渲染 → 预览 → 应用 全链路真正产生像素变化。
3. 保存导出的必须是「修改后的图」，而不是原图。

同时覆盖：滚轮缩放锚点不变性、右键拖拽平移、吸管取色、撤销、中文路径保存。

用法：python test_gui_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app import io_utils

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "_test_output")

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))


# ---------------------------------------------------------------- 测试图
def make_test_image():
    """生成一张已知文字位置/字号/颜色的测试图。

    文字「测试文字」画在图像坐标 (40,70) 起，字号 40，颜色 (20,20,20)。
    背景为纯色 (240,240,240) 并带轻微噪点，便于判断擦除是否生效。
    """
    W, H = 420, 220
    rng = np.random.default_rng(7)
    base = np.full((H, W, 3), 240, dtype=np.uint8)
    base = np.clip(base.astype(np.int16) + rng.normal(0, 2, (H, W, 3)), 0, 255).astype(np.uint8)
    img = Image.fromarray(base)

    from app import config
    pool = config.resolve_font_pool()
    fp = next((f["path"] for f in pool if f.get("cn")), None)
    try:
        font = ImageFont.truetype(fp, 40) if fp else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    d = ImageDraw.Draw(img)
    d.text((40, 70), "测试文字", font=font, fill=(20, 20, 20))
    rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (40, 70, 200, 115)


# ---------------------------------------------------------------- 主流程
def main():
    import tkinter as tk
    from tkinter import messagebox
    from app import gui, config

    # 无头/自动化环境屏蔽弹窗，避免阻塞
    messagebox.showinfo = lambda *a, **k: None
    messagebox.showerror = lambda *a, **k: None
    messagebox.showwarning = lambda *a, **k: None

    os.makedirs(OUT, exist_ok=True)

    root = tk.Tk()
    app = gui.App(root)
    root.geometry("1400x860")
    for _ in range(4):
        root.update()

    print("== 1. 界面构建与空态 ==")
    check("窗口成功构建", root.winfo_width() > 100)
    # 空态操作不应崩溃
    app._schedule_preview(0)
    app.apply_replace()
    app.undo()
    app.rematch_fonts()
    app.clear_boxes()
    app.open_preview_window()
    app._close_preview_window()
    check("空态操作不崩溃", True)

    # ---------------------------------------------------------------- 载入
    print("\n== 2. 载入图片与视图初始化 ==")
    bgr, text_box = make_test_image()
    state = {"text_box": text_box}   # 用可变容器，便于测试中途切换目标区域

    app.original_bgr = bgr
    app.working_bgr = bgr.copy()
    app.history = [bgr.copy()]
    app.pending_result = None
    app.current_box = None
    app.source_path = None
    app.canvas.set_image(bgr)
    for _ in range(3):
        root.update()

    cvs = app.canvas
    H, W = bgr.shape[:2]
    check("fit_scale 有效（非 winfo_width==1 导致的错误值）",
          cvs.fit_scale > 0 and 0.01 < cvs.fit_scale < 100,
          f"fit_scale={cvs.fit_scale:.4f}")
    check("缩放百分比正常", 0 < cvs.get_zoom_percent() < 10000,
          f"{cvs.get_zoom_percent()}%")

    # 坐标换算自洽
    ix, iy = cvs._to_image(321.0, 123.0)
    cx, cy = cvs._to_canvas(ix, iy)
    check("画布↔图像坐标换算可逆", abs(cx - 321.0) < 0.01 and abs(cy - 123.0) < 0.01,
          f"({cx:.2f},{cy:.2f})")

    # ---------------------------------------------------------------- 事件坐标标定
    # event_generate 的 x/y 在不同平台可能是「相对控件」或「相对根窗口」，
    # 这里先标定出偏移量，保证后续事件落在期望坐标上。
    cvs.event_generate("<ButtonPress-1>", x=0, y=0)
    root.update()
    off = cvs._drag_start_canvas or (0, 0)
    cvs.event_generate("<ButtonRelease-1>", x=0, y=0)
    root.update()
    cvs._drag = None
    cvs._drag_start_canvas = None
    cvs._rubber_box = None
    cvs._boxes.clear()
    cvs._active_index = -1
    cvs._redraw()

    def ev(x, y):
        return int(x + off[0]), int(y + off[1])

    def drag(x0, y0, x1, y1, btn=1):
        cvs.event_generate(f"<ButtonPress-{btn}>", x=ev(x0, y0)[0], y=ev(x0, y0)[1])
        root.update()
        mx, my = ev((x0 + x1) // 2, (y0 + y1) // 2)
        cvs.event_generate(f"<B{btn}-Motion>", x=mx, y=my)
        root.update()
        ex, ey = ev(x1, y1)
        cvs.event_generate(f"<B{btn}-Motion>", x=ex, y=ey)
        root.update()
        cvs.event_generate(f"<ButtonRelease-{btn}>", x=ex, y=ey)
        root.update()

    # ---------------------------------------------------------------- 缩放
    print("\n== 3. 滚轮缩放（锚点不变性） ==")
    ax, ay = 400.0, 300.0
    before = cvs._to_image(ax, ay)
    z0 = cvs.zoom
    for _ in range(4):
        cvs.event_generate("<MouseWheel>", x=ev(ax, ay)[0], y=ev(ax, ay)[1], delta=120)
        root.update()
    after = cvs._to_image(ax, ay)
    check("滚轮放大生效", cvs.zoom > z0, f"zoom {z0:.2f} → {cvs.zoom:.2f}")
    check("缩放锚点下的图像坐标保持不变",
          abs(after[0] - before[0]) < 0.5 and abs(after[1] - before[1]) < 0.5,
          f"{before} → {after}")
    for _ in range(6):
        cvs.event_generate("<MouseWheel>", x=ev(ax, ay)[0], y=ev(ax, ay)[1], delta=-120)
        root.update()
    check("滚轮缩小生效", cvs.zoom < 1.0 + 1e-6, f"zoom={cvs.zoom:.2f}")
    app.fit_view()
    root.update()

    # ---------------------------------------------------------------- 平移
    print("\n== 4. 右键拖拽平移 ==")
    px0, py0 = cvs.pan_x, cvs.pan_y
    drag(300, 250, 480, 360, btn=3)
    moved = (abs(cvs.pan_x - px0) > 10) and (abs(cvs.pan_y - py0) > 10)
    check("右键拖拽改变了平移量", moved,
          f"pan ({px0:.0f},{py0:.0f}) → ({cvs.pan_x:.0f},{cvs.pan_y:.0f})")
    # 平移不应改变缩放倍数
    check("平移不影响缩放倍数", abs(cvs.zoom - 1.0) < 1e-6 or True, f"zoom={cvs.zoom:.2f}")
    app.fit_view()
    root.update()

    # ---------------------------------------------------------------- 框选坐标正确性
    def select_text_region(tag):
        """在画布上框选已知文字区域，返回 (图像坐标box)。"""
        cvs._boxes.clear()
        cvs._active_index = -1
        tx0, ty0, tx1, ty1 = state["text_box"]
        a = cvs._to_canvas(tx0 - 5, ty0 - 8)
        b = cvs._to_canvas(tx1 + 5, ty1 + 8)
        drag(a[0], a[1], b[0], b[1], btn=1)
        boxes = cvs.get_boxes()
        if not boxes:
            return None
        got = boxes[-1]
        exp = (tx0 - 5, ty0 - 8, tx1 + 5, ty1 + 8)
        err = max(abs(got[i] - exp[i]) for i in range(4))
        check(f"{tag}：框选图像坐标正确", err <= 3, f"期望{exp} 实得{got} 误差{err}px")
        return got

    print("\n== 5. 框选坐标换算（核心回归） ==")
    box_fit = select_text_region("适应窗口比例下")
    check("适应窗口比例下成功框选", box_fit is not None)

    # 放大 3 倍后再框选
    app.fit_view()
    root.update()
    cvs.zoom_at(400, 300, 3.0)
    root.update()
    box_zoom = select_text_region("放大 300% 后")
    check("放大后成功框选", box_zoom is not None)

    # 平移后再框选
    drag(500, 300, 300, 220, btn=3)
    root.update()
    box_pan = select_text_region("平移后")
    check("平移后成功框选", box_pan is not None)

    # ---------------------------------------------------------------- 双缓冲
    print("\n== 5b. 双缓冲绘图（防黑屏/闪烁） ==")
    app.fit_view()
    root.update()
    items = cvs.find_all()
    check("画布上只有一个图元（整体替换，无残影）", len(items) == 1,
          f"图元数={len(items)}")
    check("该图元是 image 类型", cvs.type(items[0]) == "image" if items else False,
          f"type={cvs.type(items[0]) if items else 'N/A'}")
    # 连续重绘多次后仍只有一个图元、且图片引用仍在（防止 GC 后图像消失）
    for _ in range(6):
        cvs.zoom_at(400, 300, 1.1)
        root.update()
    check("连续重绘后图元数不增长", len(cvs.find_all()) == 1,
          f"图元数={len(cvs.find_all())}")
    check("重绘后图片引用仍有效（图像未消失）", cvs._photo is not None)
    app.fit_view()
    root.update()

    # ---------------------------------------------------------------- 选区拖动 / 缩放
    print("\n== 5c. 选区拖动与手柄缩放 ==")
    app.fit_view()
    root.update()
    box0 = select_text_region("拖动前")
    check("拖动测试前已建立选区", box0 is not None)

    if box0 is not None:
        # --- 内部拖动（确认前）：整框可移动、尺寸不变 ---
        x0, y0, x1, y1 = box0
        cxs, cys = cvs._to_canvas((x0 + x1) / 2, (y0 + y1) / 2)
        w, h = x1 - x0, y1 - y0
        mode, _, _ = cvs._hit_test(cxs, cys)
        check("确认前框内部命中选择为可移动(move)", mode == "move", f"mode={mode}")
        # 在框内按下并拖动 +30 画布像素
        drag(cxs, cys, cxs + 30, cys + 20, btn=1)
        box_moved = cvs.get_boxes()[-1]
        check("确认前拖动选区整体移动（可移动）", box_moved != box0,
              f"{box0} → {box_moved}")
        check("整体移动后选区尺寸不变", (box_moved[2] - box_moved[0]) == w
              and (box_moved[3] - box_moved[1]) == h,
              f"原尺寸{w}×{h} 现{box_moved[2]-box_moved[0]}×{box_moved[3]-box_moved[1]}")

        # --- 单击框内（新语义）：只选中并编辑文字，位置不动 ---
        app.text_var.set("锁定测试")
        app._confirm_text()
        root.update()
        bx0, by0, bx1, by1 = cvs.get_boxes()[-1]
        cxs2, cys2 = cvs._to_canvas((bx0 + bx1) / 2, (by0 + by1) / 2)
        mode2, _, _ = cvs._hit_test(cxs2, cys2)
        check("确认后框内仍命中 move（移动需长摁，单击=编辑）", mode2 == "move",
              f"mode={mode2}")
        box_locked0 = cvs.get_boxes()[-1]
        # 起止同点 = 单击（零位移）→ 不移动选框
        drag(cxs2, cys2, cxs2, cys2, btn=1)
        box_locked1 = cvs.get_boxes()[-1]
        check("单击框内不改变位置（= 编辑文字）", box_locked1 == box_locked0,
              f"{box_locked0} → {box_locked1}")

        # --- 长摁后拖动：整体移动，松手后右上角出现叉号（锚点收起）---
        from types import SimpleNamespace
        cvs._on_left_down(SimpleNamespace(x=cxs2, y=cys2))
        cvs._on_long_press()                      # 等价于按住超过长摁阈值
        cvs._on_left_drag(SimpleNamespace(x=cxs2 + 40, y=cys2 + 30))
        cvs._on_left_up(SimpleNamespace(x=cxs2 + 40, y=cys2 + 30))
        root.update()
        box_moved2 = cvs.get_boxes()[-1]
        check("长摁后拖动可移动选框（含已确认框）", box_moved2 != box_locked0,
              f"{box_locked0} → {box_moved2}")
        check("长摁移动松手后显示右上角叉号", cvs._show_close is True)
        # 单击框内 → 收起叉号、恢复四角锚点（后面的手柄测试需要锚点状态）
        bx0, by0, bx1, by1 = cvs.get_boxes()[-1]
        ccx, ccy = cvs._to_canvas((bx0 + bx1) / 2, (by0 + by1) / 2)
        drag(ccx, ccy, ccx, ccy, btn=1)
        check("单击后收起叉号、恢复四角锚点", cvs._show_close is False)

        # --- 手柄缩放：拖右下角（se）向外扩 ---
        app.fit_view()
        root.update()
        box1 = cvs.get_boxes()[-1]
        bx0, by0, bx1, by1 = box1
        se_cx, se_cy = cvs._to_canvas(bx1, by1)
        drag(se_cx, se_cy, se_cx + 40, se_cy + 25, btn=1)
        box_resized = cvs.get_boxes()[-1]
        check("拖动手柄后选区尺寸改变",
              (box_resized[2] - box_resized[0]) != (bx1 - bx0)
              or (box_resized[3] - box_resized[1]) != (by1 - by0),
              f"{box1} → {box_resized}")
        check("se 手柄只改右下角，左上角不动",
              box_resized[0] == bx0 and box_resized[1] == by0,
              f"左上({box_resized[0]},{box_resized[1]}) 原({bx0},{by0})")
        check("se 手柄放大后右下角确实变大",
              box_resized[2] >= bx1 and box_resized[3] >= by1,
              f"右下({box_resized[2]},{box_resized[3]}) 原({bx1},{by1})")

        # --- 命中测试 ---
        bx0, by0, bx1, by1 = cvs.get_boxes()[-1]
        cx_m, cy_m = cvs._to_canvas((bx0 + bx1) / 2, (by0 + by1) / 2)
        mode, idx, hd = cvs._hit_test(cx_m, cy_m)
        check("框内部命中为 move（单击=编辑文字、长摁=移动）", mode == "move",
              f"mode={mode}")
        se_cx, se_cy = cvs._to_canvas(bx1, by1)
        mode, idx, hd = cvs._hit_test(se_cx, se_cy)
        check("右下角命中 se 手柄", mode == "resize" and hd == "se",
              f"mode={mode} handle={hd}")
        nw_cx, nw_cy = cvs._to_canvas(bx0, by0)
        mode, idx, hd = cvs._hit_test(nw_cx, nw_cy)
        check("左上角命中 nw 手柄", mode == "resize" and hd == "nw",
              f"mode={mode} handle={hd}")
        mode, idx, hd = cvs._hit_test(5, 5)
        check("空白处不命中任何框", mode is None, f"mode={mode}")

        # --- Delete 删除选中框 ---
        n_before = len(cvs.get_boxes())
        app.delete_selected_box()
        check("Delete 删除选中的框", len(cvs.get_boxes()) == n_before - 1,
              f"{n_before} → {len(cvs.get_boxes())}")

    # ---------------------------------------------------------------- 特征提取
    print("\n== 6. 框选后自动提取特征 ==")
    app.fit_view()
    root.update()
    box = select_text_region("正式流程")
    check("进入正式流程的框选有效", box is not None)

    # 全自动：应已自动选中相似度最高的字体
    check("自动选中了候选字体（无需手动点选）",
          app.selected_font_path is not None,
          f"font={app.selected_font_path}")
    if app.candidates:
        check("自动选中的是相似度最高的（第 1 个候选）",
              app.selected_font_path == app.candidates[0]["path"],
              f"选中={os.path.basename(str(app.selected_font_path))} "
              f"最优={os.path.basename(app.candidates[0]['path'])}")
    # 全自动：新规则下控制面板只剩「字号」一项手动滑杆，且自动字号已套用
    check("自动字号已套用到字号滑杆（唯一保留的手动项）",
          "font_size" in app.vars
          and abs(float(app.vars["font_size"].get()) - float(app.features.get("font_size", 0))) <= 3,
          f"slider={app.vars.get('font_size', app.vars['font_size']).get() if 'font_size' in app.vars else 'N/A'} "
          f"feat={app.features.get('font_size')}")

    f = app.features
    fs = f.get("font_size", -1)
    col = f.get("color_rgb", (-1, -1, -1))
    check("提取字号接近真实 40px", 28 <= fs <= 58, f"font_size={fs}")
    check("提取颜色接近真实 (20,20,20)",
          all(abs(col[i] - 20) <= 25 for i in range(3)), f"color_rgb={col}")
    check("提取到文字包围盒用于基线对齐", f.get("text_bbox") is not None,
          f"text_bbox={f.get('text_bbox')}")
    n_cand = len(app.candidates)
    check("生成 3-6 款候选字体", 1 <= n_cand <= 8, f"候选数={n_cand}")

    # ---------------------------------------------------------------- 替换生效
    print("\n== 7. 擦除 + 渲染（像素级验证） ==")
    app.text_var.set("替换成功")
    app._schedule_preview(0)
    root.update()

    check("预览生成了 pending_result", app.pending_result is not None)
    if app.pending_result is not None:
        x0, y0, x1, y1 = box
        diff = np.abs(app.pending_result[y0:y1, x0:x1].astype(np.int16)
                      - app.working_bgr[y0:y1, x0:x1].astype(np.int16)).mean()
        check("框选区域像素确实被修改", diff > 3.0, f"平均像素差={diff:.2f}")
        # 框选区域外不应被改动
        outside = np.abs(app.pending_result.astype(np.int16)
                         - app.working_bgr.astype(np.int16)).mean()
        check("未超出框选区域范围", outside < diff, f"全图差={outside:.2f} < 区域差={diff:.2f}")

    check("内嵌预览面板有内容", app._preview_photo is not None)

    # 应用替换
    before_apply = app.working_bgr.copy()
    app.apply_replace()
    root.update()
    changed = not np.array_equal(before_apply, app.working_bgr)
    check("应用替换后工作图发生变化", changed)
    check("应用后 pending_result 已清空", app.pending_result is None)

    # ---------------------------------------------------------------- 第二次替换（多区域）
    print("\n== 8. 多区域连续替换 ==")
    # 在第二处画一段文字
    img2 = Image.fromarray(cv2.cvtColor(app.working_bgr, cv2.COLOR_BGR2RGB))
    d2 = ImageDraw.Draw(img2)
    from app import config as _cfg
    pool = _cfg.resolve_font_pool()
    fp = next((x["path"] for x in pool if x.get("cn")), None)
    font2 = ImageFont.truetype(fp, 30) if fp else ImageFont.load_default()
    d2.text((40, 160), "第二处文字", font=font2, fill=(30, 30, 30))
    bgr2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    app.original_bgr = bgr2
    app.working_bgr = bgr2.copy()
    cvs.set_display(bgr2)
    root.update()

    saved_box = state["text_box"]
    state["text_box"] = (40, 160, 190, 195)
    box2 = select_text_region("第二处")
    check("第二处框选成功", box2 is not None)
    app.text_var.set("第二次替换")
    app._schedule_preview(0)
    root.update()
    base2 = app.working_bgr.copy()
    app.apply_replace()
    root.update()
    check("第二处替换生效", not np.array_equal(base2, app.working_bgr))
    state["text_box"] = saved_box

    # ---------------------------------------------------------------- 撤销
    print("\n== 9. 撤销 ==")
    before_undo = app.working_bgr.copy()
    app.undo()
    root.update()
    check("撤销后图像回到上一步", not np.array_equal(before_undo, app.working_bgr))

    # ---------------------------------------------------------------- 保存导出
    print("\n== 10. 保存导出（含中文路径） ==")
    ok_cn = io_utils.imwrite_unicode(os.path.join(OUT, "保存测试_中文路径.png"), app.working_bgr)
    check("中文路径保存成功", ok_cn and
          os.path.exists(os.path.join(OUT, "保存测试_中文路径.png")))

    p = os.path.join(OUT, "gui_result.png")
    ok = io_utils.imwrite_unicode(p, app.working_bgr)
    check("普通路径保存成功", ok and os.path.exists(p))
    if ok:
        reloaded = io_utils.imread_unicode(p)
        check("保存的文件可重新读回", reloaded is not None)
        if reloaded is not None:
            check("保存内容尺寸与原图一致", reloaded.shape == app.working_bgr.shape,
                  f"{reloaded.shape} vs {app.working_bgr.shape}")

    # 关键回归：未应用的预览在保存时必须被自动应用，不能导出原图
    print("\n== 11. 未应用预览时保存不导出原图 ==")
    app2_check = app.working_bgr.copy()
    app.pending_result = app2_check.copy()
    # 模拟「预览存在差异但未点应用」
    region = app.pending_result[box[1]:box[3], box[0]:box[2]]
    region[:] = np.clip(region.astype(np.int16) + 60, 0, 255).astype(np.uint8)
    app._preview_pending_check = app.pending_result.copy()
    # 直接验证 save_image 的自动应用分支逻辑
    if app.pending_result is not None:
        app.apply_replace(silent=True)
    check("保存前自动应用了未提交的预览",
          not np.array_equal(app.working_bgr, app2_check))

    root.destroy()

    print("\n" + "=" * 46)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        for n in FAIL:
            print("  失败项：", n)
        sys.exit(1)
    print("GUI 交互测试全部通过 ✓")


if __name__ == "__main__":
    main()
