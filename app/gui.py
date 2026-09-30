"""tkinter GUI 界面。

交互：
- 鼠标滚轮        放大 / 缩小整张图片（以光标为锚点）
- 按住右键/中键拖动 平移画面（放大后查看细节）
- 鼠标左键拖拽     框选要替换的文字区域

流程：
框选 → 自动擦除旧文字 + 提取参数 → 生成候选字体小样 → 实时预览 → 应用替换 → 保存

历史修复要点：
1. 画布尺寸未渲染时 winfo_width() 返回 1，导致缩放比与坐标换算错误，
   框选区域偏移到图外，文字替换失效 —— 现统一走 _canvas_size() 兜底。
2. 预览结果此前只更新显示、未写入 working_bgr，导致保存导出的是原图
   —— 现引入 pending_result，保存前自动应用未提交的预览。
3. preview 此前以 original_bgr 为基底，多次替换会丢失已应用结果
   —— 现统一以 working_bgr 为基底。
"""
import math
import os
import copy
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser
from typing import List, Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from . import config
from . import feature_extract as fe
from . import erase as erase_mod
from . import render as render_mod
from . import font_match as fm
from . import ocr as ocr
from . import io_utils


# 画布未被布局渲染时的兜底尺寸，避免 winfo_width() 返回 1 造成缩放错误
_FALLBACK_W = 900
_FALLBACK_H = 620


class ImageCanvas(tk.Canvas):
    """支持滚轮缩放、右键平移、左键框选/拖动/缩放选区的图像画布。

    渲染策略（双缓冲 + 视口裁剪）：
    所有内容（缩放后的图像、已有选区框、8 个缩放手柄、正在拖拽的框）
    先画到一张「与画布等大」的离屏 PIL 图上，再一次性转成 PhotoImage
    贴到画布上**唯一一个** image item。

    - 双缓冲：画面永远一次性整体替换，不会出现半张图、闪烁、黑屏或残留。
    - 视口裁剪：离屏图尺寸恒等于画布尺寸（而非「原图 × 缩放倍数」），
      因此即使放大到 64 倍也不会生成 GB 级位图而卡死。
    """

    MIN_ZOOM = 0.05
    MAX_ZOOM = 64.0

    # 手柄（画布坐标下的边长）与命中容差
    HANDLE_SIZE = 8
    HANDLE_TOL = 6
    # 8 个手柄：四角 + 四边中点。角落斜向缩放；上下边中点调高度、左右边中点调宽度。
    # 四边中点对应 col/row=1（见 _on_left_drag 的 resize 分支：col=1 不改横向、
    # row=1 不改纵向），因此拖边中点只改一侧尺寸 —— 可单独调宽/高。
    HANDLES = ("nw", "n", "ne", "e", "se", "s", "sw", "w")
    _HANDLE_POS = {
        "nw": (0, 0), "n": (1, 0), "ne": (2, 0),
        "w": (0, 1),                  "e": (2, 1),
        "sw": (0, 2), "s": (1, 2), "se": (2, 2),
    }
    # 不同手柄对应的鼠标指针样式
    _HANDLE_CURSOR = {
        "nw": "size_nw_se", "se": "size_nw_se",
        "ne": "size_ne_sw", "sw": "size_ne_sw",
        "n": "size_ns", "s": "size_ns",
        "e": "size_ew", "w": "size_ew",
    }
    # ---- 长摁移动 / 右上角叉号（本轮新增交互）----
    LONG_PRESS_MS = 260      # 框内按住多久算「长摁」，进入整体移动模式
    DRAG_TRIGGER_PX = 5      # 未达长摁阈值但已明显拖动，也立即进入移动（更跟手）
    CLOSE_SIZE = 14          # 右上角叉号（圆形）直径
    CLOSE_TOL = 4            # 叉号命中额外容差

    def __init__(self, master, on_box_select=None, on_click=None,
                 on_view_change=None, on_box_change=None,
                 on_leave_box=None, on_box_select_same=None,
                 on_can_move=None, on_box_close=None,
                 on_box_edit_start=None, on_box_create=None,
                 on_copy_box_to=None, **kw):
        super().__init__(master, **kw)
        self.on_box_select = on_box_select       # 新建框选完成
        self.on_click = on_click                 # 左键单击（吸管）
        self.on_view_change = on_view_change     # 缩放/平移变化
        self.on_box_change = on_box_change       # 已有框位置/尺寸变化（移动或缩放）
        self.on_leave_box = on_leave_box         # 即将切换/新建框前（用于固化当前框草稿）
        self.on_box_select_same = on_box_select_same  # 点击已存在框（选中并编辑文字）
        self.on_can_move = on_can_move           # 查询第 i 个框当前是否可整体平移
        self.on_box_close = on_box_close         # 点击选框右上角叉号 → 删除该框
        self.on_box_edit_start = on_box_edit_start   # 即将移动/缩放某框（几何改变前）
        self.on_box_create = on_box_create       # 新建框选完成（OCR 前）→ 立即擦除底字
        self.on_copy_box_to = on_copy_box_to     # 空格+右键拖拽复制：把副本落到 target_box

        self.image_cv = None          # 当前显示的 BGR 图
        self.fit_scale = 0.0          # 适应窗口的基础缩放
        self.zoom = 1.0               # 用户缩放倍数
        self.pan_x = 0.0              # 平移偏移（画布坐标）
        self.pan_y = 0.0

        self._boxes: List[Tuple[int, int, int, int]] = []   # 已框选区域（图像坐标）
        self._active_index = -1       # 当前编辑的框索引
        self._photo = None            # 必须持有引用，否则图像会消失
        self._img_item = None
        self._pan_last = None

        # 拖拽状态机：None / "new" / "move" / "resize"
        self._drag = None
        self._drag_start_canvas = None   # 拖拽起始画布坐标
        self._drag_orig_box = None       # 拖拽前的框（图像坐标）
        self._drag_handle = None         # resize 时的手柄标识
        self._rubber_box = None          # 正在新建的框（画布坐标）

        # 框内按下 → 长摁待定态（未达长摁阈值前不移动，松手即视为单击）
        self._press_start = None         # 框内按下的画布坐标（待定态非空）
        self._press_idx = -1             # 待定态对应的框索引
        self._press_job = None           # 长摁定时器
        # 右上角叉号：长摁移动松手后显示（此时收起四角锚点）；单击框内则反之
        self._show_close = False
        self._geometry_committed = False   # 本次按下是否已固化过几何（移动/缩放首帧）

        # 空格 + 右键拖拽复制选框（本轮新增交互）
        self.space_held = False      # 空格键是否按住（由 App 的键盘回调维护）
        self._copy_drag = None       # 复制拖拽状态 dict：src_idx / src_box / start / target_box
        self._ghost_box = None       # 拖拽中的落点预览框（图像坐标）

        # 左键：新建框选 / 拖动已有框 / 拖拽锚点缩放框
        self.bind("<ButtonPress-1>", self._on_left_down)
        self.bind("<B1-Motion>", self._on_left_drag)
        self.bind("<ButtonRelease-1>", self._on_left_up)
        # 右键 / 中键平移
        self.bind("<ButtonPress-2>", self._pan_start)
        self.bind("<B2-Motion>", self._pan_drag)
        self.bind("<ButtonRelease-2>", self._pan_end)
        self.bind("<ButtonPress-3>", self._pan_start)
        self.bind("<B3-Motion>", self._pan_drag)
        self.bind("<ButtonRelease-3>", self._pan_end)
        # 滚轮缩放
        self.bind("<MouseWheel>", self._on_mousewheel)      # Windows / macOS
        self.bind("<Button-4>", self._on_zoom_in_linux)     # Linux 上滚
        self.bind("<Button-5>", self._on_zoom_out_linux)    # Linux 下滚
        # 尺寸变化时保持视图
        self.bind("<Configure>", self._on_configure)
        # 光标随命中位置变化（未拖拽时）
        self.bind("<Motion>", self._on_hover)

    # ---------------------------------------------------------------- 尺寸
    def _canvas_size(self) -> Tuple[int, int]:
        """返回画布尺寸；未渲染时用兜底尺寸，避免缩放比计算错误。"""
        cw = self.winfo_width()
        ch = self.winfo_height()
        if cw < 10:
            cw = _FALLBACK_W
        if ch < 10:
            ch = _FALLBACK_H
        return cw, ch

    def _on_configure(self, e):
        # 首次拿到真实尺寸时，若还没初始化过视图，做一次适应窗口
        if self.image_cv is not None and self.fit_scale <= 0:
            self.fit_view()

    # ---------------------------------------------------------------- 视图
    def set_image(self, bgr: np.ndarray):
        """载入新图：清空框选并适应窗口。"""
        self.image_cv = bgr
        self._boxes.clear()
        self._active_index = -1
        self.zoom = 1.0
        self.fit_scale = 0.0
        self.fit_view()

    def set_display(self, bgr: np.ndarray):
        """只更新图像内容，保留框选、缩放、平移状态（预览用）。"""
        self.image_cv = bgr
        self._redraw(regen=True)

    def fit_view(self):
        """缩放至适应窗口并居中。"""
        if self.image_cv is None:
            return
        h, w = self.image_cv.shape[:2]
        cw, ch = self._canvas_size()
        self.fit_scale = min(cw / w, ch / h)
        self.zoom = 1.0
        self.pan_x = (cw - w * self.fit_scale) / 2.0
        self.pan_y = (ch - h * self.fit_scale) / 2.0
        self._redraw(regen=True)

    def zoom_at(self, cx: float, cy: float, factor: float):
        """以画布坐标 (cx, cy) 为锚点缩放，保持该点下的图像内容不动。"""
        if self.image_cv is None:
            return
        if self.fit_scale <= 0:
            self.fit_view()
        # 缩放前，光标位置对应的图像坐标
        ix, iy = self._to_image(cx, cy)
        new_zoom = self.zoom * factor
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, new_zoom))
        if abs(new_zoom - self.zoom) < 1e-9:
            return
        self.zoom = new_zoom
        eff = self.fit_scale * self.zoom
        # 反推平移，使同一图像坐标仍落在同一画布位置
        self.pan_x = cx - ix * eff
        self.pan_y = cy - iy * eff
        self._redraw(regen=True)

    def zoom_center(self, factor: float):
        """以画布中心为锚点缩放（工具栏按钮用）。"""
        cw, ch = self._canvas_size()
        self.zoom_at(cw / 2.0, ch / 2.0, factor)

    # ---------------------------------------------------------------- 绘制
    def _redraw(self, regen: bool = True):
        """双缓冲重绘：先把图像 + 选区 + 手柄画到离屏图，再一次性贴到画布。

        画布上始终只有**一个** image item，且每次都是整幅替换，
        因此不会出现「半张图 / 闪烁 / 黑屏 / 残影 / 图像消失」。
        """
        cw, ch = self._canvas_size()
        cw = max(1, int(cw))
        ch = max(1, int(ch))

        # ---- 1. 准备离屏缓冲（尺寸恒等于画布尺寸，内存可控） ----
        buf = Image.new("RGB", (cw, ch), (32, 34, 38))
        draw = ImageDraw.Draw(buf)

        if self.image_cv is None:
            # 无图时也要刷新一次，避免残留上一张图
            self._blit(buf)
            return

        h, w = self.image_cv.shape[:2]
        if self.fit_scale <= 0:
            self.fit_scale = min(cw / w, ch / h)
        eff = self.fit_scale * self.zoom

        # ---- 2. 视口裁剪：只光栅化可见的源区域 ----
        sx0 = (0.0 - self.pan_x) / eff
        sy0 = (0.0 - self.pan_y) / eff
        sx1 = (cw - self.pan_x) / eff
        sy1 = (ch - self.pan_y) / eff
        ix0 = int(max(0, math.floor(sx0)))
        iy0 = int(max(0, math.floor(sy0)))
        ix1 = int(min(w, math.ceil(sx1) + 1))
        iy1 = int(min(h, math.ceil(sy1) + 1))

        if ix1 > ix0 and iy1 > iy0:
            crop = self.image_cv[iy0:iy1, ix0:ix1]
            dw = max(1, int(round((ix1 - ix0) * eff)))
            dh = max(1, int(round((iy1 - iy0) * eff)))
            # 放大用最近邻保留像素边界便于精准框选，缩小用区域插值避免摩尔纹
            interp = cv2.INTER_NEAREST if eff >= 3.0 else cv2.INTER_AREA
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            try:
                small = cv2.resize(rgb, (dw, dh), interpolation=interp)
                dx0 = int(round(ix0 * eff + self.pan_x))
                dy0 = int(round(iy0 * eff + self.pan_y))
                buf.paste(Image.fromarray(small), (dx0, dy0))
            except Exception:
                # 极端缩放下的 resize 异常不应让画面黑掉，降级为纯背景
                pass

        # ---- 3. 画已完成的选区框 ----
        for i, box in enumerate(self._boxes):
            x0, y0 = self._to_canvas(box[0], box[1])
            x1, y1 = self._to_canvas(box[2], box[3])
            active = (i == self._active_index)
            color = (255, 140, 26) if active else (0, 208, 255)
            self._draw_dash_rect(draw, x0, y0, x1, y1, color,
                                 width=2 if active else 1,
                                 dash=() if active else (5, 3))
            if active:
                # 活动框：先画贯穿画布的四边延长辅助线，再画框与锚点
                self._draw_guides(draw, x0, y0, x1, y1, cw, ch)
                self._draw_handles(draw, x0, y0, x1, y1)

        # ---- 4. 画正在拖拽的新建框（橡皮筋） ----
        if self._rubber_box:
            rx0, ry0, rx1, ry1 = self._rubber_box
            self._draw_dash_rect(draw, rx0, ry0, rx1, ry1, (255, 85, 0),
                                 width=2, dash=(4, 3))

        # ---- 4.5 画「空格+右键拖拽复制」的落点预览（幽灵框，青绿色虚线） ----
        if self._ghost_box:
            gx0, gy0 = self._to_canvas(self._ghost_box[0], self._ghost_box[1])
            gx1, gy1 = self._to_canvas(self._ghost_box[2], self._ghost_box[3])
            self._draw_dash_rect(draw, gx0, gy0, gx1, gy1, (110, 255, 170),
                                 width=2, dash=(6, 4))

        # ---- 5. 一次性贴到画布 ----
        self._blit(buf)

    def _blit(self, buf: Image.Image):
        """把离屏缓冲贴到画布上唯一的 image item（整体替换，无闪烁）。"""
        self._photo = ImageTk.PhotoImage(buf)
        if self._img_item is None:
            self.delete("all")
            self._img_item = self.create_image(0, 0, anchor="nw",
                                               image=self._photo)
        else:
            self.itemconfig(self._img_item, image=self._photo)
            self.coords(self._img_item, 0, 0)

    @staticmethod
    def _draw_dash_line(draw, x0, y0, x1, y1, color, width=1, dash=(4, 4)):
        """画一条（可选虚线的）直线，坐标规整为整数。"""
        x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        pos, on = 0.0, True
        seg_on, seg_off = dash
        while pos < length:
            seg = seg_on if on else seg_off
            end = min(pos + seg, length)
            if on:
                draw.line([x0 + ux * pos, y0 + uy * pos,
                           x0 + ux * end, y0 + uy * end],
                          fill=color, width=width)
            pos = end
            on = not on

    def _draw_guides(self, draw, x0, y0, x1, y1, cw, ch):
        """活动选框的四边延长线，贯穿整个画布，方便对照上下文对齐。

        从选框的上/下边与左/右边各引一条贯穿画布的虚线（对齐辅助线），
        便于把选框的行列与周围文字对齐。淡蓝灰、细虚线，避免喧宾夺主。
        """
        color = (120, 165, 215)
        dash = (6, 6)
        for ax, ay, bx, by in (
            (0, y0, cw, y0),   # 上边 → 水平贯穿
            (0, y1, cw, y1),   # 下边 → 水平贯穿
            (x0, 0, x0, ch),   # 左边 → 垂直贯穿
            (x1, 0, x1, ch),   # 右边 → 垂直贯穿
        ):
            self._draw_dash_line(draw, ax, ay, bx, by, color, width=1, dash=dash)

    @staticmethod
    def _draw_dash_rect(draw, x0, y0, x1, y1, color, width=1, dash=()):
        """在 PIL 上画（可选虚线的）矩形。

        PIL 的 ImageDraw.rectangle 不支持虚线，这里按 dash 模式逐段画直线，
        同时把坐标规整为整数，避免浮点坐标导致的绘制异常。
        """
        x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        if not dash:
            for k in range(width):
                draw.rectangle([x0 - k, y0 - k, x1 + k, y1 + k], outline=color)
            return

        # 四条边分别按虚线模式绘制，保证闭合
        edges = [
            (x0, y0, x1, y0), (x1, y0, x1, y1),
            (x1, y1, x0, y1), (x0, y1, x0, y0),
        ]
        for (ax, ay, bx, by) in edges:
            dx, dy = bx - ax, by - ay
            length = math.hypot(dx, dy)
            if length < 1:
                continue
            ux, uy = dx / length, dy / length
            pos, on = 0.0, True
            seg_on, seg_off = dash
            while pos < length:
                seg = seg_on if on else seg_off
                end = min(pos + seg, length)
                if on:
                    draw.line([ax + ux * pos, ay + uy * pos,
                               ax + ux * end, ay + uy * end],
                              fill=color, width=width)
                pos = end
                on = not on

    def _draw_handles(self, draw, x0, y0, x1, y1):
        """活动选区的四角锚点；长摁移动松手后改为只画右上角叉号（锚点收起）。

        两种状态互斥（用户硬性要求）：
          · 单击框内 → 显示四角锚点、不显示叉号（用于缩放选框）
          · 长摁移动松手 → 显示右上角叉号、收起锚点（用于删除该选框）
        """
        if self._show_close:
            self._draw_close_button(draw, x0, y0, x1, y1)
            return
        s = self.HANDLE_SIZE
        half = s // 2
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        pts = {
            "nw": (x0, y0), "n": (mx, y0), "ne": (x1, y0),
            "w": (x0, my),                  "e": (x1, my),
            "sw": (x0, y1), "s": (mx, y1), "se": (x1, y1),
        }
        for name, (px, py) in pts.items():
            draw.rectangle([int(round(px - half)), int(round(py - half)),
                            int(round(px + half)), int(round(py + half))],
                           fill=(255, 255, 255), outline=(255, 140, 26), width=1)

    def _draw_close_button(self, draw, x0, y0, x1, y1):
        """在选框右上角画一个圆形叉号（点击即删除该选框）。"""
        cx, cy = self._close_point(x0, y0, x1, y1)
        cx, cy = int(round(cx)), int(round(cy))
        r = self.CLOSE_SIZE // 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=(214, 60, 50), outline=(255, 255, 255), width=1)
        d = max(2, r - 3)
        draw.line([cx - d, cy - d, cx + d, cy + d], fill=(255, 255, 255), width=2)
        draw.line([cx + d, cy - d, cx - d, cy + d], fill=(255, 255, 255), width=2)

    @staticmethod
    def _close_point(x0, y0, x1, y1) -> Tuple[float, float]:
        """右上角叉号中心（与 ne 锚点同轴，位于选框右上角）。"""
        return (max(x0, x1), min(y0, y1))

    # ---------------------------------------------------------------- 命中测试
    def _handle_points(self, x0, y0, x1, y1) -> Dict[str, Tuple[float, float]]:
        """返回八个手柄（四角 + 四边中点）的中心画布坐标。"""
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        return {
            "nw": (x0, y0), "n": (mx, y0), "ne": (x1, y0),
            "w": (x0, my),                  "e": (x1, my),
            "sw": (x0, y1), "s": (mx, y1), "se": (x1, y1),
        }

    def _hit_test(self, cx: float, cy: float):
        """判定光标位置命中了什么。

        返回 (mode, index, handle)：
          ("close",  i, None)        命中活动框右上角的叉号（此时锚点已收起）
          ("resize", i, handle_name) 命中第 i 个框的某个角落锚点（此时不显示叉号）
          ("move",   i, None)        落在第 i 个框内部、且该框当前**允许移动**
                                     （需长摁或明显拖动才真正移动）
          ("select", i, None)        落在第 i 个框内部（仅选中，不可平移）
          (None,     -1, None)       空白处
        活动框优先判定，且倒序遍历（后画的框在上层）。
        """
        if not self._boxes:
            return (None, -1, None)

        order = list(range(len(self._boxes)))
        # 活动框放最后判定（优先级最高）
        if 0 <= self._active_index < len(self._boxes):
            order.remove(self._active_index)
            order.append(self._active_index)
        else:
            order.reverse()

        tol = self.HANDLE_TOL + self.HANDLE_SIZE // 2
        for i in reversed(order):
            bx0, by0 = self._to_canvas(self._boxes[i][0], self._boxes[i][1])
            bx1, by1 = self._to_canvas(self._boxes[i][2], self._boxes[i][3])
            rx0, rx1 = min(bx0, bx1), max(bx0, bx1)
            ry0, ry1 = min(by0, by1), max(by0, by1)
            # 命中容差外扩，便于抓到细边框
            if not (rx0 - tol <= cx <= rx1 + tol and ry0 - tol <= cy <= ry1 + tol):
                continue
            # 先判叉号 / 锚点（活动框才显示，且二者互斥）
            if i == self._active_index:
                if self._show_close:
                    px, py = self._close_point(bx0, by0, bx1, by1)
                    ctol = self.CLOSE_SIZE // 2 + self.CLOSE_TOL
                    if abs(cx - px) <= ctol and abs(cy - py) <= ctol:
                        return ("close", i, None)
                else:
                    for name, (px, py) in self._handle_points(bx0, by0, bx1, by1).items():
                        if abs(cx - px) <= tol and abs(cy - py) <= tol:
                            return ("resize", i, name)
            if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
                movable = (i == self._active_index and self.on_can_move is not None
                           and self.on_can_move(i))
                return (("move", i, None) if movable else ("select", i, None))
        return (None, -1, None)

    # ---------------------------------------------------------------- 坐标
    def _to_image(self, cx: float, cy: float) -> Tuple[float, float]:
        eff = self.fit_scale * self.zoom
        if eff <= 0:
            return 0.0, 0.0
        return ((cx - self.pan_x) / eff, (cy - self.pan_y) / eff)

    def _to_canvas(self, ix: float, iy: float) -> Tuple[float, float]:
        eff = self.fit_scale * self.zoom
        return (ix * eff + self.pan_x, iy * eff + self.pan_y)

    # ---------------------------------------------------------------- 事件
    def _on_mousewheel(self, e):
        if e.delta > 0:
            self.zoom_at(e.x, e.y, 1.15)
        elif e.delta < 0:
            self.zoom_at(e.x, e.y, 1.0 / 1.15)

    def _on_zoom_in_linux(self, e):
        self.zoom_at(e.x, e.y, 1.15)

    def _on_zoom_out_linux(self, e):
        self.zoom_at(e.x, e.y, 1.0 / 1.15)

    # ---------------------------------------------------------------- 空格+右键拖拽复制
    def _offset_box(self, box, dix: float, diy: float):
        """把框整体平移 dix/diy（图像坐标），夹在图像内并保持原尺寸。"""
        if self.image_cv is None:
            return tuple(box)
        ih, iw = self.image_cv.shape[:2]
        w, h = box[2] - box[0], box[3] - box[1]
        nx0 = int(round(box[0] + dix))
        ny0 = int(round(box[1] + diy))
        nx0 = int(max(0, min(nx0, max(0, iw - w))))
        ny0 = int(max(0, min(ny0, max(0, ih - h))))
        return (nx0, ny0, nx0 + w, ny0 + h)

    def _pan_start(self, e):
        # 【按住空格 + 右键点在选框中间】= 拖拽复制该选框 → 不走画面平移
        if self.space_held and self.image_cv is not None:
            mode, idx, _h = self._hit_test(e.x, e.y)
            if mode in ("move", "select") and 0 <= idx < len(self._boxes):
                # 需求：右键点选框中间部分即「选中」该选框（非活动框先切过去）。
                # 顺序沿用左键点击已有框：先离开固化旧框 → 再置活动 → 再恢复该框状态。
                if idx != self._active_index:
                    if self.on_leave_box is not None:
                        self.on_leave_box()
                    self._active_index = idx
                    self._show_close = False
                    if self.on_box_select_same is not None:
                        self.on_box_select_same(idx)
                src = self._active_index if self._active_index >= 0 else idx
                box = tuple(self._boxes[src])
                self._copy_drag = {"src_idx": src, "src_box": box,
                                   "start": (e.x, e.y), "target_box": box}
                self._ghost_box = box
                self.configure(cursor="plus")
                self._redraw(regen=False)
                return
        self._pan_last = (e.x, e.y)
        self.configure(cursor="fleur")

    def _pan_drag(self, e):
        # 复制拖拽中：只更新落点预览，不动画面
        if self._copy_drag:
            dx = e.x - self._copy_drag["start"][0]
            dy = e.y - self._copy_drag["start"][1]
            eff = self.fit_scale * self.zoom
            if eff <= 0:
                return
            tb = self._offset_box(self._copy_drag["src_box"], dx / eff, dy / eff)
            self._copy_drag["target_box"] = tb
            self._ghost_box = tb
            self._redraw(regen=False)
            return
        if not self._pan_last:
            return
        dx = e.x - self._pan_last[0]
        dy = e.y - self._pan_last[1]
        self._pan_last = (e.x, e.y)
        self.pan_x += dx
        self.pan_y += dy
        # 因为采用视口裁剪光栅化，平移会改变可见源区域，必须重新生成位图
        self._redraw(regen=True)
        if self.on_view_change:
            self.on_view_change()

    def _pan_end(self, e):
        if self._copy_drag:
            tb = self._copy_drag.get("target_box") or self._copy_drag["src_box"]
            src_idx = self._copy_drag.get("src_idx", -1)
            self._copy_drag = None
            self._ghost_box = None
            self.configure(cursor="crosshair")
            if self.on_copy_box_to and src_idx >= 0:
                # 由 App 创建副本（复制不擦除任何原图内容）
                self.on_copy_box_to(src_idx, tb)
            else:
                self._redraw(regen=False)
            return
        self._pan_last = None
        self.configure(cursor="crosshair")

    # ---------------------------------------------------------------- 左键交互
    def _on_hover(self, e):
        """未拖拽时，根据命中位置切换光标，提示可拖动/可缩放。"""
        if self._drag or self._pan_last:
            return
        if self.image_cv is None:
            return
        mode, _, handle = self._hit_test(e.x, e.y)
        if mode == "close":
            self.configure(cursor="hand2")      # 右上角叉号：点击删除选框
        elif mode == "resize":
            self.configure(cursor=self._HANDLE_CURSOR.get(handle, "crosshair"))
        elif mode == "move":
            self.configure(cursor="fleur")  # 长摁（或明显拖动）可整框移动
        elif mode == "select":
            self.configure(cursor="arrow")  # 框内仅可选中，不可平移
        else:
            self.configure(cursor="crosshair")

    # ------------------------------------------------------------ 长摁状态机
    def _cancel_press_job(self):
        """取消尚未触发的长摁定时器。"""
        if self._press_job is not None:
            try:
                self.after_cancel(self._press_job)
            except Exception:
                pass
            self._press_job = None

    def _begin_move(self):
        """进入「整体移动」模式（长摁达成或拖动超过阈值时调用）。"""
        if not (0 <= self._press_idx < len(self._boxes)) or self._press_start is None:
            return False
        self._drag = "move"
        self._drag_start_canvas = self._press_start
        self._drag_orig_box = self._boxes[self._press_idx]
        self.configure(cursor="fleur")
        return True

    def _commit_geometry(self, idx):
        """几何**真正开始改变**时（移动/缩放的首帧拖动），通知 App 固化「已擦除」。

        只在真正拖动了才调用，且每次按下只调用一次：
          · 单击框内（编辑文字）不触发 → 替换文字不会被临时抹掉造成闪烁；
          · 长摁达成但没拖动就松手也不触发 → 只显示叉号，画面不受影响。
        """
        if self._geometry_committed:
            return
        self._geometry_committed = True
        if self.on_box_edit_start is not None and 0 <= idx < len(self._boxes):
            # 把「移动 / 缩放」模式一并通知：App 侧只在 mode=="move" 时擦除
            self.on_box_edit_start(idx, self._boxes[idx],
                                   "move" if self._drag == "move" else "resize")

    def _on_long_press(self):
        """长摁达成：框内按下转为「整体移动」模式（松手后右上角显示叉号）。"""
        # 走正规取消流程（after_cancel + 置空），直接置 None 会让在途定时器
        # 无法取消，销毁窗口后触发 Tk 报「invalid command name」。
        self._cancel_press_job()
        if self._press_start is None or self._drag is not None:
            return
        if self.image_cv is None:
            return
        self._begin_move()

    def _on_left_down(self, e):
        if self.image_cv is None:
            return
        self._geometry_committed = False   # 本次按下尚未固化过几何
        mode, idx, handle = self._hit_test(e.x, e.y)

        # ---- 右上角叉号：删除该选框（替换结果保留，仍可撤销）----
        if mode == "close":
            if self.on_box_close is not None:
                self.on_box_close(idx)
            return

        # 离开当前框之前，先把当前框的草稿固化进工作图（多选区互不覆盖的前提）。
        # 【修复：点空白画新框也必须先固化】旧判据只在命中其它框时离开，导致
        # 「重画新框 → 上一框替换文字消失，需重新点回才显示」。现在只要按下的
        # 位置不在活动框自身（新建/点其它框/点空白）就先固化活动框草稿。
        has_active = 0 <= self._active_index < len(self._boxes)
        # 几何即将改变（移动/缩放**当前**活动框）：不走 on_leave_box。
        # on_leave_box 固化的是「擦除 + 新文字」，而几何马上就变、新文字要按新
        # 位置重画，等于白压一帧历史；这里改由 on_box_edit_start 在真正拖动的
        # 首帧只固化「已擦除」，既防原文字复活、又不在旧位置留下残字。
        # 注意：命中的若是**别的**框，活动框仍要照常离开固化。
        geometry_pending = mode in ("move", "resize") and idx == self._active_index
        staying_here = geometry_pending or (mode == "select" and idx == self._active_index)
        if has_active and not staying_here and self.on_leave_box is not None:
            self.on_leave_box()

        if mode in ("move", "select"):
            # ---- 框内按下：进入「长摁待定」态 ----
            # 短按松手 = 单击（选中 + 编辑文字 + 显示四角锚点、收起叉号）；
            # 长摁（LONG_PRESS_MS）或明显拖动（DRAG_TRIGGER_PX）才整体移动选框。
            self._active_index = idx
            self._drag = None
            self._drag_orig_box = self._boxes[idx]
            self._press_start = (e.x, e.y)
            self._press_idx = idx
            self._show_close = False      # 点击即收起叉号、恢复锚点
            self._redraw()
            if self.on_box_select_same is not None:
                self.on_box_select_same(idx)
            self._press_job = self.after(self.LONG_PRESS_MS, self._on_long_press)
            return
        elif mode == "resize":
            # 拖拽角落锚点调整选区大小（其余角固定，不改变原始定位）
            self._active_index = idx
            self._drag = "resize"
            self._drag_handle = handle
            self._drag_orig_box = self._boxes[idx]
        else:
            # 空白处按下 → 准备新建框选
            self._drag = "new"
        self._drag_start_canvas = (e.x, e.y)
        self._redraw()

    def _on_left_drag(self, e):
        if self.image_cv is None:
            return
        # 框内按下、尚未达长摁阈值，但已经明显拖动 → 立即进入移动（更跟手）
        if self._drag is None and self._press_start is not None:
            sx0, sy0 = self._press_start
            if (abs(e.x - sx0) >= self.DRAG_TRIGGER_PX
                    or abs(e.y - sy0) >= self.DRAG_TRIGGER_PX):
                self._cancel_press_job()
                if not self._begin_move():
                    return
        if not self._drag:
            return
        sx, sy = self._drag_start_canvas
        if self._drag == "new":
            # 橡皮筋：记录画布坐标，松手时再换算成图像坐标
            self._rubber_box = (sx, sy, e.x, e.y)
            self._redraw()
            return

        dx = e.x - sx
        dy = e.y - sy
        ox0, oy0, ox1, oy1 = self._drag_orig_box
        h, w = self.image_cv.shape[:2]
        eff = self.fit_scale * self.zoom
        if eff <= 0:
            return
        # 几何真的要变了：先让 App 把「已擦除」固化进工作图，
        # 否则移走/缩小选框后，预览会基于未擦除的工作图重画，原文字当场复活。
        self._commit_geometry(self._active_index)

        if self._drag == "move":
            # 整框平移：换算成图像像素位移，平移后整框夹在图内、保持原尺寸
            dix, diy = dx / eff, dy / eff
            bw, bh = ox1 - ox0, oy1 - oy0
            nx0 = int(round(max(0.0, min(ox0 + dix, w - bw))))
            ny0 = int(round(max(0.0, min(oy0 + diy, h - bh))))
            self._boxes[self._active_index] = (nx0, ny0, nx0 + bw, ny0 + bh)
        elif self._drag == "resize":
            # 按手柄方向移动对应边；其余边固定
            dix, diy = dx / eff, dy / eff
            nx0, ny0, nx1, ny1 = ox0, oy0, ox1, oy1
            col, row = self._HANDLE_POS[self._drag_handle]
            if col == 0:    # 西边（左边）移动
                nx0 = int(round(max(0, min(ox0 + dix, ox1 - 2))))
            elif col == 2:  # 东边（右边）移动
                nx1 = int(round(max(ox0 + 2, min(ox1 + dix, w))))
            if row == 0:    # 北边（上边）移动
                ny0 = int(round(max(0, min(oy0 + diy, oy1 - 2))))
            elif row == 2:  # 南边（下边）移动
                ny1 = int(round(max(oy0 + 2, min(oy1 + diy, h))))
            self._boxes[self._active_index] = (nx0, ny0, nx1, ny1)
        self._redraw()

    def _on_left_up(self, e):
        # ---- 框内短按（未进入移动）：视为单击 ----
        # 语义：选中该框 + 进入文字编辑 + 显示四角锚点 + 收起叉号。
        if self._drag is None:
            self._cancel_press_job()
            if self._press_start is not None:
                self._press_start = None
                self._press_idx = -1
                self._show_close = False
                self._redraw()
            return

        mode = self._drag
        sx, sy = self._drag_start_canvas
        self._drag = None
        self._drag_handle = None
        self._press_start = None
        self._press_idx = -1
        self._cancel_press_job()

        # --- 新建框选 ---
        if mode == "new":
            self._rubber_box = None
            self._show_close = False
            if self.image_cv is None:
                self._redraw()
                return

            # 位移过小视为单击（供吸管取色）。
            # 必须用 and：否则一次「很扁但很宽」的框选会被误判成单击。
            if abs(e.x - sx) < 5 and abs(e.y - sy) < 5:
                if self.on_click:
                    ix, iy = self._to_image(e.x, e.y)
                    h, w = self.image_cv.shape[:2]
                    ix = int(max(0, min(ix, w - 1)))
                    iy = int(max(0, min(iy, h - 1)))
                    self.on_click(ix, iy)
                self._redraw()
                return

            ix0, iy0 = self._to_image(min(sx, e.x), min(sy, e.y))
            ix1, iy1 = self._to_image(max(sx, e.x), max(sy, e.y))
            h, w = self.image_cv.shape[:2]
            ix0 = int(max(0, min(ix0, w))); iy0 = int(max(0, min(iy0, h)))
            ix1 = int(max(0, min(ix1, w))); iy1 = int(max(0, min(iy1, h)))
            if ix1 - ix0 < 2 or iy1 - iy0 < 2:
                self._redraw()
                return

            box = (ix0, iy0, ix1, iy1)
            self._boxes.append(box)
            self._active_index = len(self._boxes) - 1
            self._redraw()
            # 新建框选完成：被遮住的原文字就是想去除的文字 → **先**立即擦除并固化。
            # 必须排在 on_box_select（OCR/特征/字体匹配等长链路）之前：
            # 那串流程任一步在真实环境抛异常时，本行若在其后就会永远执行不到，
            # 导致「原文字完全没消失」。先擦除能保证「删字」这一核心结果一定达成。
            if self.on_box_create is not None:
                self.on_box_create(self._active_index, box)
            # OCR/特征/字体匹配等长链路：任一步异常都**不能**阻断已完成的擦除。
            # on_box_select 自身已做异常兜底（不会外泄），这里再兜一层，
            # 即便它意外抛错也不让画布事件回调崩溃、已擦除结果得以保留。
            if self.on_box_select:
                try:
                    self.on_box_select(box)
                except Exception:
                    pass
            return

        # --- 移动 / 缩放已有框 ---
        # 【右上角叉号】只有「长摁移动选框后松手」才显示叉号并收起四角锚点；
        # 缩放锚点松手仍显示锚点（用户要求：点击即恢复锚点、不显示叉号）。
        self._show_close = (mode == "move")
        orig = self._drag_orig_box
        cur = self._boxes[self._active_index] if 0 <= self._active_index < len(self._boxes) else None
        self._drag_orig_box = None
        self._redraw()
        # 移动完成后，把**最终位置**覆盖的原文字擦除并固化——
        # 否则把已擦除的框移走后，目标位置的原文字会露在底下
        # （用户要求「移走选框原文字也不要出现」）。
        # 【本轮需求】只有移动（mode=="move"）才擦除；缩放不触发。
        if mode in ("move", "resize") and self.on_box_edit_start is not None:
            aidx = self._active_index
            if 0 <= aidx < len(self._boxes):
                self.on_box_edit_start(aidx, self._boxes[aidx],
                                       "move" if mode == "move" else "resize")
        # 位置确实变了才通知，避免单纯点选也触发重新提取
        if cur is not None and orig is not None and cur != orig:
            if self.on_box_change:
                self.on_box_change(self._active_index, cur)

    # ---------------------------------------------------------------- 接口
    def get_boxes(self):
        return list(self._boxes)

    def get_active_index(self):
        """返回当前活动框索引（-1 表示无）。"""
        return self._active_index

    def get_active_box(self):
        """返回当前活动框（图像坐标），无则返回 None。"""
        if 0 <= self._active_index < len(self._boxes):
            return self._boxes[self._active_index]
        return None

    def add_box(self, box, activate: bool = True) -> int:
        """追加一个选框（图像坐标），返回新框索引。

        供「复制选框」使用：画布是新框的唯一归属，App 侧只负责同步编辑状态。
        """
        self._boxes.append(tuple(int(v) for v in box))
        if activate:
            self._active_index = len(self._boxes) - 1
        self._show_close = False       # 新框显示四角锚点，不带叉号
        self._redraw()
        return len(self._boxes) - 1

    def set_active_box(self, index: int):
        """设置活动框并刷新（锚点随之显示，叉号收起）。"""
        if 0 <= index < len(self._boxes):
            self._active_index = index
            self._show_close = False
            self._redraw()

    def delete_active_box(self):
        """删除当前活动框，返回被删除的框；无活动框返回 None。"""
        if not (0 <= self._active_index < len(self._boxes)):
            return None
        box = self._boxes.pop(self._active_index)
        self._active_index = len(self._boxes) - 1
        self._show_close = False
        self._redraw()
        return box

    def clear_boxes(self):
        self._boxes.clear()
        self._active_index = -1
        self._drag = None
        self._rubber_box = None
        self._press_start = None
        self._press_idx = -1
        self._show_close = False
        self._cancel_press_job()
        self._redraw(regen=False)

    def get_zoom_percent(self) -> int:
        return int(round(self.fit_scale * self.zoom * 100))


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("本地离线图片文字替换工具")
        root.geometry("1400x860")

        self.original_bgr = None       # 原始图片（永不修改，用于特征提取）
        self.working_bgr = None        # 工作图（已应用的替换结果）
        self.pending_result = None     # 未应用的预览结果
        # 撤销栈：每个元素是一个「历史帧」dict {"bgr": 该次提交后的图像,
        # "box_edits": 该时刻各框编辑状态的快照}。撤销时把图像与框编辑状态
        # **一并**还原——单还原图像的话，已撤销的替换会因重新点选/缩放该框而
        # 复活（用户报「撤销了又变回来」），这正是把框状态纳入撤销帧的原因。
        self.history: List[Dict] = []
        self.current_box = None
        self.features: Dict = {}
        self.candidates: List[Dict] = []
        self.selected_font_path = None
        self.source_path = None

        self._preview_job = None       # 防抖定时器
        self._cand_photos = []         # 保持候选字体小样引用，防止被 GC
        self._preview_photo = None
        self.preview_window = None     # 独立预览窗口
        self._pipette_active = False
        self._preview_show_original = False
        # 替换文字在**整图**中的墨迹包围盒（绝对图像坐标），供预览「井」字辅助线定位；
        # 无替换文字 / 未渲染时为 None。仅用于显示层，绝不写入图像数据。
        self._last_text_bbox = None
        # 最近一次预览裁剪区域 (cx0, cy0, cx1, cy1)，用于把整图坐标换算到裁剪图坐标
        self._last_crop_rect = None
        self.box_edits: Dict[int, Dict] = {}   # 按 canvas 索引记忆每框的编辑状态（文字/字体/特征）
        self.neighbor_ref = None               # 当前框的四向上下文参照特征（第二步采集用）
        self.neighbor_boxes = []               # 邻近文字矩形（图像坐标），供第四步防重叠
        self._ocr_filled = False               # 当前输入框文字是否由 OCR 自动填入（用于状态提示）
        # 用户**手动**调整过的滑块参数快照：{框索引: {滑块键: 数值}}。
        # 「恢复该框基线特征 / optimized_params」时，这些手动项要盖回滑块，
        # 否则用户调好的字号一移动选框（或一点选框）就被 OCR 基线打回。
        self._user_tweaks: Dict[int, Dict[str, float]] = {}
        # 「恢复（redo）」栈：撤销下来的帧按后进先出暂存，元素与 history 帧同构
        # {"bgr": 图像, "box_edits": 各框编辑状态}，另可带 "pending"（撤销掉的
        # 未提交草稿）。一旦产生**新的**历史帧（应用替换 / 新建擦除 / 移动擦除）
        # 或框集合变化（清空/删除/复制），立即清空——与通用 redo 语义一致。
        self.redo_stack: List[Dict] = []

        self._build_ui()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        toolbar = tk.Frame(self.root, padx=6, pady=4)
        toolbar.pack(side="top", fill="x")
        tk.Button(toolbar, text="上传图片", command=self.upload_image).pack(side="left", padx=2)
        tk.Button(toolbar, text="适应窗口", command=self.fit_view).pack(side="left", padx=2)
        tk.Button(toolbar, text="放大 +", command=lambda: self.canvas.zoom_center(1.25)).pack(side="left", padx=2)
        tk.Button(toolbar, text="缩小 -", command=lambda: self.canvas.zoom_center(0.8)).pack(side="left", padx=2)
        tk.Button(toolbar, text="清空框选", command=self.clear_boxes).pack(side="left", padx=2)
        tk.Button(toolbar, text="撤销", command=self.undo).pack(side="left", padx=2)
        tk.Button(toolbar, text="恢复", command=self.redo).pack(side="left", padx=2)
        tk.Button(toolbar, text="复制选框", command=self.copy_selected_box).pack(side="left", padx=2)
        tk.Button(toolbar, text="保存图片", command=self.save_image).pack(side="left", padx=2)
        tk.Button(toolbar, text="使用说明", command=self.show_help).pack(side="right", padx=4)
        self.status_var = tk.StringVar(value="请上传图片")
        tk.Label(toolbar, textvariable=self.status_var, fg="#666").pack(side="right", padx=8)

        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True)

        # ---- 左侧：画布 + 缩放信息 ----
        left = tk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = ImageCanvas(left,
                                  on_box_select=self.on_box_selected,
                                  on_click=self.on_canvas_click,
                                  on_view_change=self._notify_zoom,
                                  on_box_change=self.on_box_changed,
                                  on_leave_box=self.on_leave_box,
                                  on_box_select_same=self.on_box_select_same,
                                  on_can_move=self._box_can_move,
                                  on_box_close=self.on_box_close,
                                  on_box_edit_start=self.on_box_edit_start,
                                  on_box_create=self.on_box_create,
                                  on_copy_box_to=self.copy_selected_box_to,
                                  bg="#2b2b2b", cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=4, pady=(4, 0))

        # 快捷键：Ctrl+Z 撤销、Ctrl+Y / Ctrl+Shift+Z 恢复、Delete 删除、Ctrl+D 复制
        self.root.bind_all("<Control-z>", lambda e: self.undo())
        self.root.bind_all("<Control-Z>", lambda e: self.undo())
        self.root.bind_all("<Control-y>", lambda e: self.redo())
        self.root.bind_all("<Control-Y>", lambda e: self.redo())
        self.root.bind_all("<Control-Shift-Z>", lambda e: self.redo())
        self.root.bind_all("<Control-Shift-z>", lambda e: self.redo())
        self.root.bind_all("<Control-d>", lambda e: self.copy_selected_box())
        self.root.bind_all("<Control-D>", lambda e: self.copy_selected_box())
        self.root.bind_all("<Delete>", lambda e: self.delete_selected_box())
        # 按住空格 + 右键拖拽选框中间 = 复制选框（在输入框里打字时不拦截空格）
        self.root.bind_all("<KeyPress-space>", self._on_space_down)
        self.root.bind_all("<KeyRelease-space>", self._on_space_up)
        # 删除键要作用到画布，先让它拿到焦点
        self.canvas.bind("<ButtonPress-1>",
                         lambda e: self.canvas.focus_set(), add="+")
        self.zoom_var = tk.StringVar(value="缩放 100%")
        tk.Label(left, textvariable=self.zoom_var, fg="#888").pack(anchor="w", padx=6)

        # ---- 右侧：控制面板 ----
        right = tk.Frame(main, width=400)
        right.pack(side="right", fill="y", padx=6, pady=4)
        right.pack_propagate(False)

        outer = tk.Canvas(right, highlightthickness=0)
        vsb = ttk.Scrollbar(right, orient="vertical", command=outer.yview)
        outer.configure(yscrollcommand=vsb.set)
        outer.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        panel = tk.Frame(outer)
        win_id = outer.create_window((0, 0), window=panel, anchor="nw")
        panel.bind("<Configure>", lambda e: outer.configure(scrollregion=outer.bbox("all")))
        outer.bind("<Configure>", lambda e: outer.itemconfig(win_id, width=e.width))
        self._panel = panel
        self._build_control_panel(panel)

    def _build_control_panel(self, panel):
        # ---- 替换文字 ----
        grp = tk.LabelFrame(panel, text="替换文字", padx=6, pady=6)
        grp.pack(fill="x", pady=4)
        self.text_var = tk.StringVar()
        e = tk.Entry(grp, textvariable=self.text_var, font=("", 11))
        e.pack(fill="x")
        e.bind("<KeyRelease>", lambda ev: self._on_text_key())
        # 单击选框即聚焦到这里（用户要求「单击选框中间时编辑文字」）
        self.text_entry = e
        tk.Label(grp, text="输入后自动实时预览；确定替换文字请点「确认」", fg="#888", font=("", 8)).pack(anchor="w")
        fr = tk.Frame(grp)
        fr.pack(fill="x", pady=(2, 0))
        tk.Button(fr, text="确认替换文字", command=self._confirm_text,
                  bg="#3367d6", fg="white").pack(side="left", fill="x", expand=True, padx=(0, 2))
        tk.Button(fr, text="清空文字", command=self._clear_text).pack(side="left", fill="x", expand=True, padx=(2, 0))

        # ---- 候选字体小样 ----
        grp = tk.LabelFrame(panel, text="候选字体（自动选用最相似的，可点击改选）", padx=6, pady=6)
        grp.pack(fill="x", pady=4)
        self.cand_frame = tk.Frame(grp)
        self.cand_frame.pack(fill="x")
        fr = tk.Frame(grp)
        fr.pack(fill="x", pady=3)
        tk.Button(fr, text="重新匹配字体", command=self.rematch_fonts).pack(side="left", fill="x", expand=True, padx=(0, 2))
        tk.Button(fr, text="恢复自动参数", command=self.reapply_auto).pack(side="left", fill="x", expand=True, padx=(2, 0))

        # ---- 参数微调（可折叠：全自动模式下通常无需展开，手动覆盖时再点开）----
        grp = tk.LabelFrame(panel, text="", padx=6, pady=6)
        grp.pack(fill="x", pady=4)
        self.vars: Dict[str, tk.DoubleVar] = {}
        self.sliders: Dict[str, tk.Scale] = {}
        self._tune_collapsed = False  # False = 展开

        head = tk.Frame(grp)
        head.pack(fill="x")
        self._tune_title_var = tk.StringVar(value="▾ 参数微调（手动覆盖用 · 点击折叠/展开）")
        tk.Button(head, textvariable=self._tune_title_var, anchor="w", relief="flat",
                  command=self._toggle_tune_panel).pack(fill="x")
        self._tune_body = tk.Frame(grp)
        self._tune_body.pack(fill="x", pady=(2, 0))

        def add_slider(key, label, lo, hi, init, res=1, fmt=".0f"):
            fr = tk.Frame(self._tune_body)
            fr.pack(fill="x", pady=1)
            tk.Label(fr, text=label, width=10, anchor="w").pack(side="left")
            val = tk.DoubleVar(value=init)
            var = tk.StringVar(value=f"{init:{fmt}}")
            sc = tk.Scale(fr, from_=lo, to=hi, resolution=res, orient="horizontal",
                          variable=val, showvalue=False, length=150,
                          command=lambda v, k=key, sv=var, f=fmt: self._on_slider(k, sv, f))
            sc.pack(side="left", fill="x", expand=True)
            tk.Label(fr, textvariable=var, width=6).pack(side="left")
            self.vars[key] = val
            self.sliders[key] = sc

        # ---- 手动调节（按需求精简：只保留「字号」一项）----
        # 其余参数（字间距 / 行间距 / 倾斜 / 清晰度 / 模糊 / 边缘柔化 / 噪点 /
        # 擦除强度 / 颜色 / 加粗 / 透明度）全部由「四向上下文自动采集」决定，
        # 不再提供手动滑杆——手动项过多正是「改一处带出新 bug」的温床。
        add_slider("font_size", "字号", 6, 300, 16)

        # 以下变量保留在内存中（供自动流程读写与既有测试兼容），但不再暴露 UI
        self.color_var = (0, 0, 0)
        self.bold_var = tk.BooleanVar(value=False)
        self.opacity_var = tk.DoubleVar(value=1.0)

        # ---- 实时预览面板 ----
        grp = tk.LabelFrame(panel, text="实时预览（当前框选区域）", padx=6, pady=6)
        grp.pack(fill="x", pady=4)
        self.preview_label = tk.Label(grp, text="框选文字区域后显示预览",
                                      bg="#f0f0f0", height=8)
        self.preview_label.pack(fill="x")
        fr = tk.Frame(grp)
        fr.pack(fill="x", pady=2)
        tk.Button(fr, text="原图 / 效果对比", command=self.toggle_preview_mode).pack(side="left", padx=2)
        tk.Button(fr, text="打开独立预览窗口", command=self.open_preview_window).pack(side="left", padx=2)

        # ---- 操作 ----
        grp = tk.Frame(panel)
        grp.pack(fill="x", pady=6)
        tk.Button(grp, text="应用替换（可继续框选下一处）", command=self.apply_replace,
                  bg="#2e8b57", fg="white").pack(fill="x", pady=2)
        tk.Button(grp, text="重新预览", command=lambda: self._schedule_preview(0)).pack(fill="x", pady=2)

        # ---- 限制说明 ----
        grp = tk.LabelFrame(panel, text="限制说明", padx=6, pady=6)
        grp.pack(fill="x", pady=4)
        tk.Label(grp, text=config.LIMITATIONS_TEXT, justify="left", fg="#888",
                 wraplength=340, font=("", 8)).pack(anchor="w")

    # ---------------------------------------------------------------- 交互
    def _toggle_tune_panel(self):
        """折叠 / 展开「参数微调」面板（全自动模式下通常无需展开）。"""
        self._tune_collapsed = not self._tune_collapsed
        if self._tune_collapsed:
            self._tune_body.pack_forget()
            self._tune_title_var.set("▸ 参数微调（手动覆盖用 · 点击折叠/展开）")
        else:
            self._tune_body.pack(fill="x", pady=(2, 0))
            self._tune_title_var.set("▾ 参数微调（手动覆盖用 · 点击折叠/展开）")

    def _on_slider(self, key, var, fmt):
        var.set(f"{float(self.vars[key].get()):{fmt}}")
        # 记录用户**手动**调整（滑块键 → 显示值），供「恢复框基线参数」时不被打回。
        # 自动提取（_set_slider）不走这里，因此不会被误记为手动调整。
        idx = self.canvas.get_active_index()
        if 0 <= idx:
            self._user_tweaks.setdefault(idx, {})[key] = float(self.vars[key].get())
        self._schedule_preview()

    def _schedule_preview(self, delay: int = 180):
        """防抖调度实时预览。"""
        if self._preview_job is not None:
            try:
                self.root.after_cancel(self._preview_job)
            except Exception:
                pass
            self._preview_job = None
        if delay <= 0:
            self._refresh_preview()
        else:
            self._preview_job = self.root.after(delay, self._refresh_preview)

    def _on_text_key(self):
        """输入框文字变化时：记忆当前框的专属文字，并依新文字重新挑选可渲染字体。

        这一步是 Bug1 的关键修复：此前只在「新建选区」时用占位文字「示例」
        做一次字体匹配，用户随后键入的真实文字若命中字体缺失字符，就会渲染成
        方块（常用词不显示）。现在每次键入都据真实文字重选「能完整渲染」的字体，
        - 且只在字体层重匹配，不重新扫描原图像素（不动既有识别结果）。
        """
        idx = self.canvas.get_active_index()
        if 0 <= idx:
            self.box_edits.setdefault(idx, {})
            self.box_edits[idx]["features"] = self.features
            self._refresh_font_for_text()
            self.box_edits[idx]["text"] = self.text_var.get().strip()
            self.box_edits[idx]["font_path"] = self.selected_font_path
        else:
            self._refresh_font_for_text()
        self._schedule_preview(300)

    def _clear_text(self):
        """清空当前框的替换文字（不继承、不自动填 OCR）。"""
        idx = self.canvas.get_active_index()
        self.text_var.set("")
        if 0 <= idx:
            self.box_edits.setdefault(idx, {})["text"] = ""
        self._schedule_preview(0)
        self.status_var.set("已清空替换文字（仍保留已锁定特征，可重新输入）")

    def _confirm_text(self):
        """第三步：用户输入自定义替换文字并点击「确认」。

        确认后触发：
        - 记忆当前框专属文字（多选区独立、互不覆盖）；
        - 标记该框进入「二次排版优化 + 防重叠渲染」阶段（optimized=True）；
        - 字体层重选（覆盖校验，不重新识别原图）；
        - 立即刷新预览（第四步自动二次优化排版、第五步防重叠渲染）。
        这是用户硬性流程里明确的「确认」动作，确认前预览为过渡态，确认后方为成品态。
        """
        idx = self.canvas.get_active_index()
        text = self.text_var.get().strip()
        if not text:
            self.status_var.set("请先输入或识别替换文字，再点确认")
            return
        if 0 <= idx:
            self.box_edits.setdefault(idx, {})
            self.box_edits[idx]["text"] = text
            self.box_edits[idx]["_confirmed"] = True
            self.box_edits[idx]["optimized"] = True
            # 冻结本次「二次排版优化」后的最终参数，使之后重渲染（重开/改字）样式
            # 保持稳定、不再随上下文重算而漂移（用户硬性要求：重开后仅编辑文字、
            # 原有样式不被覆盖）。optimized_params 含全部渲染所需键。
            # 字间距严格对齐原图（闭环校准，见 _calibrate_letter_spacing）：
            # 替换文字「字与字距离」逐像素等于原图，不因字数增减、不因换字体而漂移。
            # 字号走手动优先逻辑（手调值优先，且冻结的就是用户所见的值，
            # 保证「确认前预览」与「确认后成品」字号完全一致）。
            final_params = self._collect_params()
            manual_fs = self._manual_font_size(idx)
            if manual_fs is not None:
                final_params["font_size"] = manual_fs
            final_params = self._calibrate_letter_spacing(
                text, final_params, self.selected_font_path)
            if manual_fs is not None:
                final_params["font_size"] = manual_fs
            self.box_edits[idx]["optimized_params"] = final_params
            self.box_edits[idx]["confirmed_text"] = text
            # 复制出来的副本在此“落地”：确认后与其它确认框一样锁死位置
            self.box_edits[idx].pop("_copied", None)
            # 【修复：切换字体后点确认被“打回原形”】确认时同步持久化当前字体，
            # 保证之后重选/缩放该框恢复的是用户确认时的字体，而不是创建时的旧字体。
            self.box_edits[idx]["font_path"] = self.selected_font_path
        self._refresh_font_for_text()
        # 强制立即以「二次优化 + 防重叠」路径渲染（delay=0）
        self._schedule_preview(0)
        self.status_var.set("已确认替换文字 → 自动二次排版优化 + 防重叠渲染")

    def _refresh_font_for_text(self):
        """按当前输入框文字，重新挑选「能正确渲染该文字」的字体（字形覆盖优先）。

        仅在已有选区、且已做过特征识别时调用；传入 bgr_region=None，复用
        已保存的 self.features 作为字重/字号依据，**不重新扫描原图像素**
        （即「重新选取已有选框时不用再次识别」在字体层同样成立）。
        若已选字体已能完整渲染当前文字，则保持其不被改动（尊重用户手动改选）。
        """
        if self.current_box is None or self.original_bgr is None or not self.features:
            return
        text = self.text_var.get().strip()
        if not text:
            return
        # 已选字体已能完整渲染当前文字 → 不动（尊重用户已选/已保存的字体）
        if self.selected_font_path and fm.font_covers_text(text, self.selected_font_path, 40):
            return
        f = self.features
        try:
            cands = fm.match_fonts(
                text, None,
                target_stroke=f["stroke_weight"],
                target_size=f["font_size"],
                target_stroke_px=f.get("stroke_px", 0.0),
            )
        except Exception:
            return
        if cands:
            self.candidates = cands
            self.selected_font_path = cands[0]["path"]
            self._render_candidates()

    def _notify_zoom(self):
        self.zoom_var.set(f"缩放 {self.canvas.get_zoom_percent()}%")

    def fit_view(self):
        self.canvas.fit_view()
        self._notify_zoom()

    # ---------------------------------------------------------------- 图片
    def upload_image(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.webp"), ("所有文件", "*.*")])
        if not path:
            return
        bgr = io_utils.imread_unicode(path)
        if bgr is None:
            messagebox.showerror("错误", "无法读取图片，请确认格式正确。")
            return
        self.source_path = path
        self.original_bgr = bgr
        self.working_bgr = bgr.copy()
        self.history = [{"bgr": bgr.copy(), "box_edits": {}}]
        self._clear_redo()      # 换图 = 全新历史，恢复栈一并作废
        self.pending_result = None
        self.current_box = None
        self.canvas.set_image(bgr)
        self._notify_zoom()
        h, w = bgr.shape[:2]
        self.status_var.set(f"已加载 {os.path.basename(path)}（{w}×{h}）— 滚轮缩放，左键拖拽框选")

    def clear_boxes(self):
        # 与「删除选框」同一套规则：先固化当前框已产生的替换，原文字不回退
        self.on_leave_box()
        self.canvas.clear_boxes()
        self.box_edits.clear()
        self._user_tweaks.clear()      # 手动调整快照随选框一并清空
        self._clear_redo()             # 框集合被清空，旧帧已无法对齐 → 恢复栈作废
        self.current_box = None
        self.pending_result = None
        self.candidates = []
        self._render_candidates()
        if self.working_bgr is not None:
            self.canvas.set_display(self.working_bgr)
        self._update_preview_panel()
        self.status_var.set("已清空框选")

    # ---------------------------------------------------------------- 框选
    def on_box_selected(self, box, is_resize=False):
        """框选完成（或已有框被缩放后）对外入口：内部实现包一层异常兜底。

        任一步（OCR/特征/字体匹配/候选渲染）出错都**只提示、绝不让异常外泄**
        到画布事件回调 —— 否则异常会打断「先擦除」之后、又让 UI 崩溃。
        这里兜底后，即使识别失败，新建选框时已擦除的原文字也稳稳保留。
        """
        try:
            self._on_box_selected_impl(box, is_resize)
        except Exception as e:
            self.status_var.set(f"选区识别/校准失败：{e}")

    def _on_box_selected_impl(self, box, is_resize=False):
        """框选完成（或已有框被缩放后）：严格三步流水线。

        ===== 用户硬性执行顺序 =====
        第一步：识别本选区的文字（OCR 自动识别，填入输入框）
        第二步：读取紧邻上下一行做排版/质感校准（融合参照特征）
        第三步：自动渲染匹配样式（自动选字体 + 实时预览）

        is_resize=False 表示全新选区：OCR 识别 + 校准 + 清空继承文字。
        is_resize=True  表示缩放已有框：保留旧文字、重新校准上下行参照（不重新 OCR）。
        """
        self.current_box = box
        # 换框/新建框：清掉上一次替换文字的包围盒，避免预览辅助线画在旧位置
        self._last_text_bbox = None
        # 新选区：清空文字输入框，杜绝跨选区继承
        if not is_resize:
            self.text_var.set("")
            self._ocr_filled = False
        x0, y0, x1, y1 = box
        # 特征始终从未经修改的原图提取，保证参数准确
        region = self.original_bgr[y0:y1, x0:x1]
        try:
            region_feats = fe.extract_features(region)
        except Exception as e:
            # 极小/异常区域不应导致崩溃，退回安全默认值
            region_feats = fe.extract_features(np.zeros((0, 0, 3), np.uint8))
            self.status_var.set(f"特征提取异常，已使用默认值：{e}")

        # ============ 第一步：识别本选区的文字（OCR） ============
        # 仅「全新选区」才做 OCR 识别（多选区各自独立识别、不继承、不覆盖手动改字）。
        # 缩放已有框（is_resize=True）保留旧文字，不重新识别（重识别可能不稳定）。
        idx = self.canvas.get_active_index()
        if not is_resize:
            if ocr.is_ocr_supported():
                try:
                    # 小图放大后再识别，显著提升 OCR 准确率（修复「文字识别不准」）
                    ocr_region = region
                    if ocr_region.shape[0] < 40:
                        f = 40.0 / max(1, ocr_region.shape[0])
                        ocr_region = cv2.resize(
                            ocr_region, None, fx=f, fy=f,
                            interpolation=cv2.INTER_CUBIC)
                    recognized = ocr.ocr_region(ocr_region)
                except Exception:
                    recognized = ""
                if recognized:
                    self.text_var.set(recognized)
                    self._ocr_filled = True
                    if 0 <= idx:
                        self.box_edits.setdefault(idx, {})["ocr_text"] = recognized

        # ============ 第二步：采集选区上下左右紧邻文字做排版/质感参照 ============
        ref = fe.extract_neighbor_reference(
            self.original_bgr, box,
            font_size_hint=region_feats.get("font_size", 16))
        self.neighbor_ref = ref
        self.neighbor_boxes = ref.get("neighbor_boxes") if ref else []
        if ref is not None:
            # 用上下行参照校准：优先对齐字号/字间距/笔画粗细，质感融合颜色/倾斜/模糊/噪声
            self.features = self._calibrate_features(region_feats, ref)
        else:
            self.features = dict(region_feats)

        # ---- 四向参照兜底：本区 em 不可信时，用邻近文字的 em 重算 ----
        # 典型退化场景：整行只有「一」「丶」这类扁平字符，或只有标点。
        # 此时本区自身的行高/字高/字宽三个通道会同时崩溃
        # （实测「一」被估成真实值的 0.59 倍），必须靠周边文字救命。
        self._fallback_em_from_neighbors(region, ref)

        # 全自动：把（校准后的）特征一次性套用到滑块与开关
        self._auto_apply_features()

        # ============ 第三步：自动渲染匹配样式 ============
        text = self.text_var.get().strip() or "示例"
        self.candidates = fm.match_fonts(
            text, region,
            target_stroke=self.features["stroke_weight"],
            target_size=self.features["font_size"],
            target_stroke_px=self.features.get("stroke_px", 0.0),
            # 用「原图原文」作候选测量样本，使候选与原图测的是同一批字符，
            # 消除约 8% 的样本口径偏差（实测可让字体自匹配率由 75% 升至 100%）
            reference_text=(self.box_edits.get(idx, {}).get("ocr_text")
                            if 0 <= idx else None),
        )
        self._render_candidates()
        if self.candidates:
            # 已有选区缩放时：若之前用户手动选过某款字体且该字体仍在候选里，
            # 则保留用户选择；否则（全新选区）取相似度最高的第一款。
            cand_paths = [c["path"] for c in self.candidates]
            if not (is_resize and self.selected_font_path in cand_paths):
                self.selected_font_path = self.candidates[0]["path"]

        hexc = "#%02x%02x%02x" % self.features["color_rgb"]
        top = self.candidates[0]["name"] if self.candidates else "无"
        ocr_hint = "OCR识别" if self._ocr_filled else "手动输入"
        self.status_var.set(
            f"已框选 {x1-x0}×{y1-y0}｜{ocr_hint}｜字号≈{self.features['font_size']}px｜"
            f"颜色{hexc}｜倾斜{self.features['skew_angle']:.1f}°｜自动选字：{top}")

        # 记录该框状态（供切换回来时恢复）。
        # 已有框缩放：保留之前记忆的文字；全新框：文字为本次 OCR 识别结果。
        # _locked=True 标记「本框文字特征已确定，后续重选/缩放均不再重新识别/重提取」
        # （用户硬性要求：确定文字的特征后不再修改，和该选框的位置一同固定）。
        if 0 <= idx:
            prev_text = self.box_edits.get(idx, {}).get("text", "")
            self.box_edits[idx] = {
                "text": self.text_var.get().strip() or prev_text,
                "ocr_text": self.box_edits.get(idx, {}).get("ocr_text", ""),
                "font_path": self.selected_font_path,
                "features": self.features,
                "neighbor_ref": self.neighbor_ref,
                "neighbor_boxes": self.neighbor_boxes,
                "_locked": True,
            }

        # 输入过替换文字则立即预览，否则只做擦除预览
        self._schedule_preview(0)

    def _fallback_em_from_neighbors(self, region, ref: Dict | None) -> None:
        """四向参照兜底：本区 em 不可信时，采信邻近文字的字号并重算基线。

        退化场景：整行只有「一」「丶」这类扁平字符，或只有标点。此时本区
        自身的行高 / 字高 / 字宽三个推定通道会**同时崩溃**（实测「一」只估到
        真实值的 0.59 倍），渲染出的新文字会小到几乎不可见。

        正确的应对不是继续在单区域内硬凑补丁，而是采信「四向邻近文字」的
        字号——同一段落的排版通常一致。这也是「自动采集周边文字作为参考」
        这条规则真正的价值所在。

        基线依赖 em（baseline = 墨迹底边 − K_INK_BELOW × em），
        所以 em 换了之后基线必须一并重算。
        """
        if not self.features or self.features.get("em_reliable"):
            return
        if ref is None:
            return
        ref_em = ref.get("font_size")
        if not ref_em or ref_em <= 0:
            return
        try:
            self.features["font_size"] = int(ref_em)
            self.features["em_reliable"] = True
            self.features["baseline_y"] = fe.estimate_baseline_y(region, int(ref_em))
        except Exception:
            # 兜底失败也不能让主流程崩掉
            pass

    def _calibrate_features(self, region_feats: Dict, ref: Dict) -> Dict:
        """用上下行参照特征校准选区自身特征（执行顺序第二步的核心）。

        对齐范围严格遵循用户要求：『排版间距、字号、笔画粗细』+ 『质感』。
        - 字号 / 字间距 / 笔画粗细（含骨架口径、归一化字重）：对齐到上下行
          （参照更稳定、更代表段落排版风格），采用选区与参照的融合值。
        - 模糊 / 噪声 / 清晰度：同属「段落质感」，用参照与选区融合，使替换后
          文字的失焦/颗粒质感与上下文一致。
        - 颜色 / 倾斜：保留选区自身（替换的是「该位置」的文字，应与原位置同色、
          同倾；上下行可能只是恰好同段但颜色/角度略有差异，沿用参照反而引入误差）。
        复制输入字典，不原地修改 region_feats。
        """
        f = dict(region_feats)
        if ref.get("font_size", 0) > 0:
            f["font_size"] = int(round((f["font_size"] + ref["font_size"]) / 2))
        # 字间距：严格对齐「本框原图自身的字间距」（用户硬性要求：替换字距与原图一致）。
        # 仅当本框自身无法测出间距（如单字框 → estimate_letter_spacing 返回 0）时，
        # 才借上下行邻居的字间距兜底，避免退化为 0 导致替换字粘连。多字框一律用自身值。
        _reg_ls = int(round(region_feats.get("letter_spacing", 0)))
        _ref_ls = int(round(ref.get("letter_spacing", 0))) if ref else 0
        if _reg_ls > 0:
            f["letter_spacing"] = _reg_ls
        elif _ref_ls > 0:
            f["letter_spacing"] = _ref_ls
        else:
            f["letter_spacing"] = 0
        if ref.get("stroke_px", 0) > 0:
            # 以选区自身实测笔画粗细为主（被替换文字即真值），参照为辅，避免邻居把
            # 笔画拉离原文字真实粗细，导致「笔画粗细不一致」。
            f["stroke_px"] = float(0.7 * f["stroke_px"] + 0.3 * ref["stroke_px"])
        if ref.get("stroke_px_skeleton", 0) > 0:
            f["stroke_px_skeleton"] = float(0.7 * f["stroke_px_skeleton"] + 0.3 * ref["stroke_px_skeleton"])
        if ref.get("stroke_weight", 0) > 0:
            f["stroke_weight"] = float(0.7 * f["stroke_weight"] + 0.3 * ref["stroke_weight"])
        # 质感（段落级，与上下文对齐）：模糊/噪点/清晰度取融合值
        if ref.get("blur_level", 0) >= 0:
            f["blur_level"] = float((f["blur_level"] + ref["blur_level"]) / 2)
        if ref.get("noise_level", 0) >= 0:
            f["noise_level"] = float((f["noise_level"] + ref["noise_level"]) / 2)
        if ref.get("sharpness", 0) > 0:
            f["sharpness"] = float((f["sharpness"] + ref["sharpness"]) / 2)
        # 颜色与倾斜：保持选区自身（见函数说明）
        return f

    def _auto_apply_features(self):
        """全自动模式：把自动提取的特征一次性套用到所有滑块与开关。

        映射关系（提取值 → 滑块取值）都做了量程缩放，
        因为滑块的量程与特征的 0~1 归一化区间并不一致。
        """
        f = self.features
        if not f:
            return

        self._set_slider("font_size", f["font_size"])
        self._set_slider("letter_spacing", f["letter_spacing"])
        self._set_slider("skew_angle", f["skew_angle"])
        self._set_slider("sharpness", f["sharpness"])

        # 行间距：单行时提取为 0，按字号给一个常规行距
        line_sp = f.get("line_spacing", 0)
        if not line_sp:
            line_sp = max(0, int(round(f["font_size"] * 0.2)))
        self._set_slider("line_spacing", max(0, line_sp - f["font_size"]))

        # 模糊：blur_level 0~1 → 滑块 0~3（sigma）
        self._set_slider("blur", round(float(f.get("blur_level", 0.3)) * 1.5, 2))
        # 边缘柔化：抗锯齿强度 0~1 → 滑块 0~2
        self._set_slider("edge_soften", round(float(f.get("antialias", 0.5)) * 0.8, 2))
        # 噪点：noise_level 0~1 → 滑块 0~3
        self._set_slider("noise", round(float(f.get("noise_level", 0.0)) * 1.5, 2))

        # 字色 / 粗细 / 透明度
        self.color_var = self._normalize_color(f.get("color_rgb", (0, 0, 0)))
        self._update_color_button()
        # 关键修复：全自动模式下不再「强行勾选加粗」。字重由 feature_extract
        # 实测的原图笔画厚度（stroke_px）在渲染时自动对齐，绝不越描越粗。
        # 「加粗」复选框仅作为用户主动叠加的手动覆盖项，默认关闭。
        self.bold_var.set(False)
        self.opacity_var.set(1.0)

    def reapply_auto(self):
        """一键回到自动值：手动调乱后，重新套用自动提取的全部参数。"""
        if self.current_box is None or not self.features:
            messagebox.showinfo("提示", "请先框选一个文字区域。")
            return
        self._auto_apply_features()
        if self.candidates:
            self.selected_font_path = self.candidates[0]["path"]
            self._render_candidates()
        self._schedule_preview(0)
        self.status_var.set("已恢复为自动匹配的参数")

    def on_box_changed(self, index, box):
        """已有选区被移动/缩放后：刷新几何相关状态，仅重新采集四向上下文参照。

        关键（用户硬性要求「移动/缩放选框不改变已调好的文字参数」）：
        几何操作**绝不**重新识别本框文字、**绝不**重新提取本框特征、也**绝不**
        把字号/间距/笔画等滑块参数打回 OCR 基线或冻结值 —— UI 当前参数就是该框
        正在编辑的有效参数，移动选框后原样保留（用户手动调过的字号不会因拖动
        而变回去）。这里只同步内部 features 缓存、重新采集四周邻近文字参照
        （仅参照、非本框特征），供防重叠判定与智能对齐线使用。
        """
        self.current_box = box
        state = self.box_edits.get(index)
        if state:
            if state.get("text"):
                self.text_var.set(state["text"])
            if state.get("font_path"):
                self.selected_font_path = state["font_path"]
            if state.get("features"):
                self.features = dict(state["features"])
                # 【修复：移动/缩放选框绝不改动用户已调参数（字号/间距/笔画…）】
                # 旧代码在此 _auto_apply_features()，会把滑块写回 OCR 时的基线字号。
                # 用户刚手动把字号调成新值（滑块已是新值，state.features 仍是旧基线）
                # 时，一移动选框滑块就被打回 → 预览文字大小当场变化（用户反馈）。
                # 几何操作只改变位置：UI 当前参数就是该框正在编辑的有效参数，保留。
                # 同理不用 optimized_params 覆盖滑块——用户此刻就在编辑该框，
                # 若冻结参数与滑块不一致，以滑块（用户当前所见）为准。
            self.neighbor_ref = state.get("neighbor_ref")
            self.neighbor_boxes = list(state.get("neighbor_boxes", []) or [])
        # 重新采集四向上下文参照（仅参照、不动本框已锁定特征），更新防重叠矩形
        ref = fe.extract_neighbor_reference(
            self.original_bgr, box,
            font_size_hint=self.features.get("font_size", 16))
        if ref is not None:
            self.neighbor_ref = ref
            self.neighbor_boxes = ref.get("neighbor_boxes") or []
            self.box_edits.setdefault(index, {})["neighbor_ref"] = ref
            self.box_edits[index]["neighbor_boxes"] = self.neighbor_boxes
        # 字体层重选（覆盖校验，不重新扫描原图）
        self._refresh_font_for_text()
        self._schedule_preview(0)
        self.status_var.set(f"已缩放第 {index + 1} 个选区（特征已锁定，仅刷新四周参照）")

    def on_leave_box(self):
        """即将离开当前框（切换/新建/点空白）时，先把当前框的草稿固化进工作图。

        这是「多选区编辑互不覆盖、上一处修改不自动消失」的关键：用户在一个框里
        输入了替换文字、正在实时预览，但还没点「应用替换」就去画下一个框，此时
        若直接开始新框，旧框的预览会被新框覆盖而丢失。改为先 silent-apply 固化，
        新框的预览便基于「已含旧框结果的工作图」，旧框成果自然保留显示。

        【修复：框内有草稿但预览尚未生成时也要先渲染再固化】此前只在
        pending_result 存在时才固化；若用户键入文字后立刻去画新框（300ms 防抖
        预览还没跑），草稿只存在于 box_edits，离开后画布上完全不显示该框的
        替换文字，必须重新点回该框才出现。现在离开前先同步补一次渲染。
        """
        if self.current_box is None:
            return
        # 【去重】本次编辑内容与上次提交完全一致（文字/字体/字号/位置都没变）
        # → 无需再渲染、再压历史帧，直接收工。
        idx_now = self.canvas.get_active_index()
        if 0 <= idx_now:
            st_now = self.box_edits.get(idx_now, {}) or {}
            if st_now.get("_committed_fp") == self._edit_fingerprint():
                self.pending_result = None
                return
        if self.pending_result is None:
            text = self.text_var.get().strip()
            idx = self.canvas.get_active_index()
            st = self.box_edits.get(idx, {}) if 0 <= idx else {}
            has_draft = bool(text) or bool(st.get("_confirmed") or st.get("optimized"))
            if has_draft:
                try:
                    self._refresh_preview()   # 同步渲染，产生 pending_result
                except Exception:
                    pass
        if self.pending_result is not None:
            # 【去重保护】预览结果与工作图完全一致时不再压历史帧——
            # 单击/长摁选框都会先走一次固化，若无脑提交会把撤销栈撑爆
            # （点十次框就要按十次撤销才能回到上一步）。
            if self._same_image(self.pending_result, self.working_bgr):
                self.pending_result = None
                self.canvas.set_display(self.working_bgr)
                return
            self.apply_replace(silent=True)

    def _box_can_move(self, idx: int) -> bool:
        """该框当前是否允许整体拖动？——**所有框都可以**，但必须长摁。

        早期版本是「确认替换后位置锁死」：一旦确认就完全不能动，副作用是已确认
        的框永远拿不到右上角的叉号。按本轮要求改为一视同仁——移动需要**长摁**
        （260ms）或明显拖动（>5px）才触发，这本身就是防误触门槛，因此不再区分
        是否已确认：单击框内 = 编辑文字，长摁 = 移动选框。
        """
        return idx >= 0

    @staticmethod
    def _same_image(a, b) -> bool:
        """两张图内容是否完全一致（用于避免重复提交历史帧）。"""
        if a is None or b is None:
            return False
        try:
            return a.shape == b.shape and bool(np.array_equal(a, b))
        except Exception:
            return False

    def _edit_fingerprint(self):
        """当前框的「编辑指纹」= 文字 + 字体 + 字号 + 框位置。

        用来判断「这次的编辑内容是不是已经提交过了」：完全一致就不必再渲染、
        再压一帧历史——否则用户反复点选同一个框，撤销栈会被无意义的重复帧
        撑爆（点十次框就得按十次撤销），且每次重画都会带来像素微漂。
        """
        try:
            fs = int(float(self.vars["font_size"].get()))
        except Exception:
            fs = -1
        box = tuple(int(v) for v in self.current_box) if self.current_box else None
        return (self.text_var.get().strip(), self.selected_font_path, fs, box)

    def on_box_edit_start(self, index, box, mode="move"):
        """即将移动/缩放某选框（几何改变**之前**）：先把「已擦除」固化进工作图。

        用户要求：原文字一旦被替代就永久消失，之后**删除或移走选框都不再出现**
        （只有「撤销」才能让它回来）。

        此前预览只是一张贴图（pending_result 覆盖显示，工作图并没有变），几何一
        变就基于**未擦除的工作图**重新渲染，于是原文字原地复活。这里在几何改变前
        先把擦除结果写进工作图，原文字就再也不会回来了。

        【本轮需求变更：只有「移动」才擦除】
        消除原图文字只在两个时机生效——① 新建选框（`on_box_create`）；
        ② 移动选框（本函数 mode=="move"）。缩放（拖 8 锚点改尺寸）**不再触发**，
        撤销/重选/点选等非几何操作更不会触发。擦除范围严格 = 选框矩形本身。

        只固化「擦除」而**不写入替换文字**：否则移动选框后旧位置会残留一份替换
        文字，看起来像凭空多出一份。新位置的文字由随后的预览/应用照常渲染。

        「空框」绝不固化（保留「随手框一下不许误擦原文」的保护）。判据两条，
        命中任意一条才算「这里确实已经被替代过」：
          ① 该区域在工作图上**已经与原始图不同**（已擦除/已替换，最硬的证据）；
          ② 用户**确实改了字**（输入框内容 ≠ OCR 识别出的原文）或已点过确认
             （草稿还没落盘，但替换意图明确）。
        两条都不满足 = 只是随手框了一下、或 OCR 自动填了原文没改 → 不动工作图。
        """
        if self.working_bgr is None or box is None or self.original_bgr is None:
            return
        # 【本轮需求】只有「移动选框」才执行消除；缩放（锚点拖拽）不擦除。
        if mode != "move":
            return
        x0, y0, x1, y1 = [int(v) for v in box]
        h, w = self.working_bgr.shape[:2]
        x0, y0 = max(0, min(x0, w)), max(0, min(y0, h))
        x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
        if x1 - x0 < 1 or y1 - y0 < 1:
            return

        already_touched = not np.array_equal(self.working_bgr[y0:y1, x0:x1],
                                             self.original_bgr[y0:y1, x0:x1])
        if not already_touched:
            st = self.box_edits.get(index, {}) or {}
            if index == self.canvas.get_active_index():
                cur_text = self.text_var.get().strip()
            else:
                cur_text = (st.get("text") or "").strip()
            ocr_text = (st.get("ocr_text") or "").strip()
            user_changed = bool(cur_text) and cur_text != ocr_text
            confirmed = bool(st.get("_confirmed") or st.get("optimized")
                             or st.get("optimized_params"))
            if not (user_changed or confirmed):
                return      # 空框/未改字 → 不许误擦原文
        try:
            strength = (float(self.vars["erase_strength"].get())
                        if "erase_strength" in self.vars else 0.5)
            # 擦除范围严格 = 选框本身（移动后原文字彻底消失，框外内容不动）
            erased = erase_mod.erase_text(self.working_bgr,
                                          self._erase_region_for(box),
                                          strength=strength)
        except Exception:
            return
        # 内容没有实质变化就不压历史帧（避免反复点选把撤销栈撑爆）
        if self._same_image(erased, self.working_bgr):
            self.pending_result = None
            return
        self.history.append({
            "bgr": erased.copy(),
            "box_edits": copy.deepcopy(self.box_edits),
        })
        self.working_bgr = erased
        self.pending_result = None
        self._clear_redo()      # 新的擦除提交 → 恢复栈作废
        # 固化后该框「已提交指纹」失效（旧位置已被擦掉），下次离开需重新渲染
        st = self.box_edits.get(index)
        if isinstance(st, dict):
            st.pop("_committed_fp", None)
        self.canvas.set_display(self.working_bgr)

    def on_box_create(self, index, box):
        """新建选框完成（OCR 识别之前）：立即把覆盖的原图文字擦除并固化进工作图。

        用户需求一：框选一片区域后，被遮住的原文字就是想去除的文字，
        **新建后直接去除**；之后无论删除还是移走选框都不再出现
        （只有「撤销」才能恢复）。

        与 on_box_edit_start 的区别：这里**无条件擦除**（只要该区域确实与原始图
        不同，即下面盖着文字/内容），不要求用户已经改字或确认。纯背景区域
        （与原始图完全一致）则跳过，避免无谓地涂抹干净背景、也避免误压历史帧。

        【本轮需求】本函数是「消除原图文字」的两个入口之一（另一个是「移动选框」
        on_box_edit_start）。擦除范围严格 = 选框矩形本身，框外像素一律不动。
        """
        if self.working_bgr is None or box is None or self.original_bgr is None:
            return
        x0, y0, x1, y1 = [int(v) for v in box]
        h, w = self.working_bgr.shape[:2]
        x0, y0 = max(0, min(x0, w)), max(0, min(y0, h))
        x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
        if         x1 - x0 < 1 or y1 - y0 < 1:
            return
        # 纯背景（与原始图一致且近乎单色、无文字）→ 本就没什么可去除的，跳过，
        # 不顺手擦背景、也不压历史帧。只要区域里盖着文字（明暗有变化）就照常擦除。
        # 注意：刚新建时 working_bgr 与 original_bgr 在文字处本就相等（都含文字），
        # 所以必须用「近乎单色」二次判定，不能只看「是否等于原图」否则会漏擦文字。
        region = self.working_bgr[y0:y1, x0:x1]
        if (np.array_equal(region, self.original_bgr[y0:y1, x0:x1])
                and int(region.max()) - int(region.min()) < 8):
            self.pending_result = None
            return
        try:
            strength = (float(self.vars["erase_strength"].get())
                        if "erase_strength" in self.vars else 0.5)
            # 擦除范围取「原文字真实墨迹范围∪选框」，确保新建后原文字彻底消失
            erased = erase_mod.erase_text(self.working_bgr,
                                          self._erase_region_for(box),
                                          strength=strength)
        except Exception as e:
            # 擦除失败必须明示，绝不能静默吞掉——否则用户会看到「原文完全没消失」
            # 却无任何提示，无从排查。
            self.status_var.set(f"擦除原文字失败：{e}")
            return
        # 擦除后内容无变化（极少见）→ 不压历史帧
        if self._same_image(erased, self.working_bgr):
            self.pending_result = None
            return
        self.history.append({
            "bgr": erased.copy(),
            "box_edits": copy.deepcopy(self.box_edits),
        })
        self.working_bgr = erased
        self.pending_result = None
        self._clear_redo()      # 新建擦除提交 → 恢复栈作废
        self.canvas.set_display(self.working_bgr)

    def _box_has_real_content(self, idx: int) -> bool:
        """该框是否含有「用户真正要写入的替换内容」（而非仅 OCR 自动填充）。

        用于删除选框时判断：仅有 OCR 自动识别文字、用户未改也未确认的框，
        删除时不应把 OCR 文字自动落盘（那样看起来像「原文字又出现了」），
        只保留 on_box_create 已固化的「已擦除」即可，删除后该区域保持干净。
        """
        if idx < 0:
            return False
        st = self.box_edits.get(idx) or {}
        if st.get("_confirmed") or st.get("optimized") or st.get("optimized_params"):
            return True
        text = self.text_var.get().strip()
        ocr = (st.get("ocr_text") or "").strip()
        return bool(text) and text != ocr

    def on_box_close(self, idx):
        """点击选框右上角的叉号 → 删除该选框。

        删除前会先把本框已产生的替换固化进工作图，所以「原文字」不会回来；
        想退回只能用「撤销」。
        """
        boxes = self.canvas.get_boxes()
        if not (0 <= idx < len(boxes)):
            return
        if idx != self.canvas.get_active_index():
            self.canvas.set_active_box(idx)
            self.current_box = boxes[idx]
        self.delete_selected_box()

    def _focus_text_entry(self):
        """把焦点交给「替换文字」输入框（单击选框 = 编辑文字）。"""
        e = getattr(self, "text_entry", None)
        if e is None:
            return
        try:
            e.focus_set()
            e.icursor(tk.END)
            e.xview_moveto(1.0)
        except Exception:
            pass

    def on_box_select_same(self, idx):
        """点击已存在的框（选中并**进入文字编辑**）：恢复其专属文字/字体/特征。

        单击语义（本轮新增）：恢复该框文字与特征、显示四角锚点、收起右上角叉号，
        并把焦点交给「替换文字」输入框，用户可以直接打字改字。

        关键（用户硬性要求「重新选取已有选框时不用再次识别」）：
        仅从 box_edits 恢复该框在「新建识别」时保存的文字与已校准特征，
        **绝不**重新 OCR 识别原图、也**绝不**重新提取上下行参照。
        只在字体层按该框文字重选「能完整渲染」的字体（Bug1/Bug2 共修），
        这一步是字体重匹配，不是文字重新识别。
        """
        boxes = self.canvas.get_boxes()
        if not (0 <= idx < len(boxes)):
            return
        self.current_box = boxes[idx]
        state = self.box_edits.get(idx)
        if state:
            saved = state.get("text")
            if saved:
                self.text_var.set(saved)
            # 恢复该框专属的已校准特征（避免沿用其它框特征导致对齐/字号错乱、识别「不稳定」）
            if state.get("features"):
                self.features = dict(state["features"])
                self._auto_apply_features()
            # 已确认框：字号滑杆显示实际使用字号（不是识别值），微调不再被恢复值打回
            op = state.get("optimized_params")
            if op and op.get("font_size"):
                self._set_slider("font_size", int(op["font_size"]))
            if state.get("font_path"):
                self.selected_font_path = state["font_path"]
                self._render_candidates()
            # 恢复四向参照（特征锁定，不重新采集/识别）
            self.neighbor_ref = state.get("neighbor_ref")
            self.neighbor_boxes = list(state.get("neighbor_boxes", []) or [])
            # 同步 OCR 状态标记（不重新识别，只是恢复历史标记）
            self._ocr_filled = bool(state.get("ocr_text"))
            # 【修复：移动/点选框不覆盖用户手动调整】基线恢复之后，把该框用户
            # 手动调过的滑块项（字号/间距/笔画…）重新盖回滑块——否则 OCR 基线
            # 会把用户刚调好的字号打回原值（「一按选框/一移动字号就变」的根因）。
            for k, v in self._user_tweaks.get(idx, {}).items():
                self._set_slider(k, v)
        # 依据该框文字重选「能完整渲染」的字体（仅字体层，不重新识别原图）
        self._refresh_font_for_text()
        self._schedule_preview(0)
        # 单击 = 编辑文字：把光标交给输入框。延后一拍执行——画布上还有一条
        # 「按下即 focus_set」的追加绑定，会抢走焦点，必须排在它后面。
        self.root.after(10, self._focus_text_entry)
        self.status_var.set(f"已选中第 {idx + 1} 个选区，可直接编辑文字（长摁可移动选框）")

    def copy_selected_box(self, target_box=None):
        """复制当前选中的选框 —— **连同编辑文字与字体特征**一起复制。

        复制内容（整份 box_edits 深拷贝，故与原框互不影响）：
          · 编辑文字（text / ocr_text）
          · 字体（font_path）与全部已校准特征（features：字号/笔画/颜色/倾斜…）
          · 确认后的冻结排版参数（optimized_params）与确认状态
            → 复制框与原框的替换效果完全一致。

        target_box：可选落点 (x0,y0,x1,y1)。传入时副本直接落到该处（保持原尺寸、
        夹在图像内），用于「空格 + 右键拖拽复制」——由用户拖到哪就放哪；
        不传时沿用自动落点（右侧 → 下方 → 右下微调 → 原位微调）。

        位置：优先放到原框右侧，放不下则下方，再放不下则右下微调；始终夹在图像内。
        复制体默认带 `_copied` 标记：**允许先拖动定位**（确认前可移动的同一套规则），
        用户点一次「确认替换文字」后即与其它确认框一样锁死位置。

        特征锁定：不重新 OCR、不重新提取本框特征；但因为位置变了，四周邻居不再是
        原来那几个，所以按**新位置**重新采集四向参照（供防重叠判定使用）。

        原功能保持不变：只追加一个新框 + 一份状态，不改动任何既有流程。
        """
        if self.original_bgr is None:
            messagebox.showinfo("提示", "请先上传图片。")
            return
        src = self.canvas.get_active_index()
        boxes = self.canvas.get_boxes()
        if not (0 <= src < len(boxes)):
            messagebox.showinfo("提示", "请先选中一个要复制的选框。")
            return
        # 复制前先把当前框的草稿固化（与切换/新建框同一套规则，避免草稿丢失）
        self.on_leave_box()

        x0, y0, x1, y1 = boxes[src]
        w, h = x1 - x0, y1 - y0
        ih, iw = self.original_bgr.shape[:2]
        gap = max(8, min(w, h) // 4)

        def _fits(bx0, by0):
            return bx0 >= 0 and by0 >= 0 and bx0 + w <= iw and by0 + h <= ih

        if target_box is not None:
            # 空格 + 右键拖拽复制：落点由用户拖拽决定（保持原尺寸、下面统一夹取）
            nx0, ny0 = int(round(target_box[0])), int(round(target_box[1]))
        else:
            # 候选落点：右侧 → 下方 → 右下微调 → 原位微调
            for cand in ((x0 + w + gap, y0), (x0, y0 + h + gap),
                         (x0 + gap, y0 + gap), (x0 + gap, y0)):
                if _fits(*cand):
                    nx0, ny0 = cand
                    break
            else:
                nx0, ny0 = x0 + gap, y0 + gap
        # 夹取在图像内（保持原尺寸）
        nx0 = int(max(0, min(nx0, max(0, iw - w))))
        ny0 = int(max(0, min(ny0, max(0, ih - h))))
        new_box = (nx0, ny0, nx0 + w, ny0 + h)

        new_idx = self.canvas.add_box(new_box, activate=True)
        # 【本轮需求】复制选框**不擦除**：副本是程序自动摆到旁边的（右侧/下方），
        # 落点底下盖着的是别处的原图内容，若沿用「新建即擦除」会凭空抹掉一块
        # 与本次替换无关的画面。消除原图文字只在「手动新建选框」与「移动选框」
        # 两个时机生效，复制不在其列。
        # 框集合新增了一个 → 恢复栈里的旧帧缺少该框状态，作废
        self._clear_redo()
        # 整份状态深拷贝 → 两框各自独立，后续修改互不干扰
        src_state = self.box_edits.get(src)
        self.box_edits[new_idx] = copy.deepcopy(src_state) if src_state else {}
        self.box_edits[new_idx]["_copied"] = True   # 副本允许先拖动定位
        # 用户手动调整快照一并复制：副本再移动/点选也不会被基线打回
        self._user_tweaks[new_idx] = dict(self._user_tweaks.get(src, {}))

        self.current_box = new_box
        self.text_var.set(self.box_edits[new_idx].get("text", ""))
        if self.box_edits[new_idx].get("font_path"):
            self.selected_font_path = self.box_edits[new_idx]["font_path"]
        if self.box_edits[new_idx].get("features"):
            self.features = copy.deepcopy(self.box_edits[new_idx]["features"])
            self._auto_apply_features()
        op = self.box_edits[new_idx].get("optimized_params")
        if op and op.get("font_size"):
            self._set_slider("font_size", int(op["font_size"]))

        # 位置变了 → 按新位置重新采集四向参照（仅参照，不动已锁定的本框特征）
        ref = fe.extract_neighbor_reference(
            self.original_bgr, new_box,
            font_size_hint=self.features.get("font_size", 16))
        if ref is not None:
            self.neighbor_ref = ref
            self.neighbor_boxes = ref.get("neighbor_boxes") or []
            self.box_edits[new_idx]["neighbor_ref"] = ref
            self.box_edits[new_idx]["neighbor_boxes"] = self.neighbor_boxes

        self._refresh_font_for_text()
        self._render_candidates()
        self._schedule_preview(0)
        self.status_var.set(
            f"已复制第 {src + 1} 个选区 → 第 {new_idx + 1} 个"
            f"（文字/字体/特征已复制，可拖动定位，点「确认替换文字」后锁定）")

    def copy_selected_box_to(self, src_idx: int, target_box):
        """【空格 + 右键拖拽复制】把第 src_idx 个选框复制到 target_box 落点。

        与 `copy_selected_box()`（Ctrl+D / 工具栏按钮）的区别：
          · 落点由用户拖拽决定，而不是自动排到原框右侧；
          · 全程**不擦除任何原图内容** —— 复制既不是「新建选框」也不是「移动选框」，
            不触发 `on_box_create` / `on_box_edit_start`，因此绝不会凭空抹掉
            落点下方与本次替换无关的画面（用户硬性要求）。
        """
        if self.original_bgr is None:
            return
        boxes = self.canvas.get_boxes()
        if not (0 <= src_idx < len(boxes)):
            return
        # 右键按下时已选中该框；这里再保险一次，确保复制的是用户拖的那个
        if self.canvas.get_active_index() != src_idx:
            self.on_box_select_same(src_idx)
        if self.canvas.get_active_index() != src_idx:
            return
        self.copy_selected_box(target_box=target_box)
        self.status_var.set(
            f"已拖拽复制第 {src_idx + 1} 个选区（未擦除原图内容）")

    # ---- 空格键状态：按住空格 + 右键拖拽选框中间 = 复制选框 ----
    _TEXT_WIDGETS = ("Entry", "TEntry", "Text", "Spinbox", "TSpinbox",
                     "TCombobox", "Combobox")

    def _focus_in_text_widget(self) -> bool:
        """焦点是否在文本输入控件里 —— 此时空格是打字，不能触发复制模式。"""
        try:
            w = self.root.focus_get()
        except Exception:
            return False
        if w is None:
            return False
        try:
            return w.winfo_class() in self._TEXT_WIDGETS
        except Exception:
            return False

    def _on_space_down(self, e=None):
        if self._focus_in_text_widget():
            return          # 输入框里正常打字
        self.canvas.space_held = True
        self.canvas.configure(cursor="plus")
        return "break"      # 阻止空格触发按钮/滚动

    def _on_space_up(self, e=None):
        self.canvas.space_held = False
        # 正在拖拽复制时不抢光标（由拖拽结束统一复位）
        if not getattr(self.canvas, "_copy_drag", None):
            self.canvas.configure(cursor="crosshair")
        return "break"

    def _reindex_box_edits(self, removed_idx: int):
        """删除某个框后，后续框索引前移，box_edits 的 key 需同步重建。"""
        if removed_idx < 0:
            return
        new_edits = {}
        for k, v in self.box_edits.items():
            if k < removed_idx:
                new_edits[k] = v
            elif k > removed_idx:
                new_edits[k - 1] = v
        self.box_edits = new_edits
        # 用户手动调整快照的 key 与选框索引一一对应，删除后同样平移
        if self._user_tweaks:
            new_tw = {}
            for k, v in self._user_tweaks.items():
                if k < removed_idx:
                    new_tw[k] = v
                elif k > removed_idx:
                    new_tw[k - 1] = v
            self._user_tweaks = new_tw

    def delete_selected_box(self):
        """删除当前选中的框（Delete 键 / 右上角叉号）。

        【替换结果不回退】删除前先把本框已产生的替换固化进工作图——用户要求
        「原文字被替代后直接消失，删除选框后也不要出现」，所以删框只是撤掉选框
        这个工具，替换结果留在图上；只有「撤销」才能把原文字找回来。
        """
        removed_idx = self.canvas.get_active_index()
        if removed_idx < 0:
            return
        # 仅当该框确有「用户真正的替换内容」时才先固化（写入替换文字）；
        # 若只是 OCR 自动填充、用户没改也没确认，则不落盘——保留 on_box_create
        # 已固化的「已擦除」状态，删除后该区域保持干净，不会冒出 OCR 识别出的
        # 「原样文字」让人误以为没去掉。
        if self._box_has_real_content(removed_idx):
            self.on_leave_box()
        removed = self.canvas.delete_active_box()
        if removed is None:
            return
        if removed_idx >= 0:
            self._reindex_box_edits(removed_idx)
        # 框索引已重排，恢复栈里的旧帧与现状不再对应 → 作废
        self._clear_redo()
        self.current_box = None
        self.pending_result = None
        self.candidates = []
        self._render_candidates()
        if self.working_bgr is not None:
            self.canvas.set_display(self.working_bgr)
        self._update_preview_panel()
        self.status_var.set("已删除选中的框选区域")

    def _set_slider(self, key, value):
        if key in self.vars:
            self.vars[key].set(float(value))

    def _manual_font_size(self, idx) -> int | None:
        """该框用户**手动**拖过的字号；没手动调过返回 None。

        手调字号是「用户显式意图」，优先级高于冻结参数 / 自动识别值：
        预览、确认冻结、重渲染一律以它为准，杜绝「手调后预览不变、点确定才跳变」。
        """
        if idx is None or idx < 0:
            return None
        tw = self._user_tweaks.get(idx) or {}
        if "font_size" not in tw:
            return None
        try:
            return max(6, int(round(float(tw["font_size"]))))
        except Exception:
            return None

    def _manual_letter_spacing(self, idx) -> int | None:
        """该框用户**手动**拖过的字间距；没手动调过返回 None。

        手动项是「用户显式意图」，优先级高于原图测量值（与字号同处理）。
        当前面板未暴露字间距滑杆，故正常流程恒为 None —— 替换字间距严格等于
        原图测量值（用户硬性要求：字与字距离与原图一致）。若日后加入字间距滑杆，
        这里即成为手动覆盖入口，无需改动调用方。
        """
        if idx is None or idx < 0:
            return None
        tw = self._user_tweaks.get(idx) or {}
        if "letter_spacing" not in tw:
            return None
        try:
            return int(round(float(tw["letter_spacing"])))
        except Exception:
            return None

    def on_canvas_click(self, ix, iy):
        """画布单击：吸管模式下取色。"""
        if not self._pipette_active or self.original_bgr is None:
            return
        h, w = self.original_bgr.shape[:2]
        ix = max(0, min(ix, w - 1)); iy = max(0, min(iy, h - 1))
        bgr = self.original_bgr[iy, ix]
        self.color_var = self._normalize_color((bgr[2], bgr[1], bgr[0]))
        self._update_color_button()
        self._pipette_active = False
        # 精简面板后吸管按钮已移除，取色逻辑保留供程序内部调用
        btn = getattr(self, "pipette_btn", None)
        if btn is not None:
            btn.configure(relief="raised", bg=self.root.cget("bg"))
        hexc = "#%02x%02x%02x" % self.color_var
        self.status_var.set(f"已取色 {hexc}")
        self._schedule_preview(0)

    def toggle_pipette(self):
        self._pipette_active = not self._pipette_active
        btn = getattr(self, "pipette_btn", None)
        if self._pipette_active:
            if btn is not None:
                btn.configure(relief="sunken", bg="#ffe08a")
            self.status_var.set("吸管模式：在图上单击要取色的文字像素")
        else:
            if btn is not None:
                btn.configure(relief="raised", bg=self.root.cget("bg"))
            self.status_var.set("已退出吸管模式")

    def pick_color(self):
        rgb, _ = colorchooser.askcolor(initialcolor=self._normalize_color(self.color_var),
                                       title="选择文字颜色")
        if rgb:
            self.color_var = self._normalize_color(rgb)
            self._update_color_button()
            self._schedule_preview(0)

    @staticmethod
    def _normalize_color(c) -> Tuple[int, int, int]:
        """把任意来源的颜色值规整为 3 个 0-255 的整数。

        极小框选区域下颜色提取可能返回异常长度/类型的值，
        这里统一兜底，避免 "%02x" 格式化或渲染时崩溃。
        """
        try:
            vals = [int(round(float(v))) for v in list(c)[:3]]
        except Exception:
            vals = []
        while len(vals) < 3:
            vals.append(0)
        return tuple(max(0, min(255, v)) for v in vals[:3])

    def _update_color_button(self):
        self.color_var = self._normalize_color(self.color_var)
        # 精简面板后取色按钮已移除，颜色仍由自动提取写入 color_var
        btn = getattr(self, "color_btn", None)
        if btn is None:
            return
        hexc = "#%02x%02x%02x" % self.color_var
        fg = "#ffffff" if sum(self.color_var) < 380 else "#000000"
        btn.configure(bg=hexc, fg=fg)

    # ---------------------------------------------------------------- 候选字体
    def _render_candidates(self):
        """渲染候选字体小样（点击可选中）。"""
        for w in self.cand_frame.winfo_children():
            w.destroy()
        self._cand_photos.clear()

        if not self.candidates:
            tk.Label(self.cand_frame, text="（未找到可用字体，请把字体放入 fonts/ 目录）",
                     fg="#888", wraplength=340).pack(anchor="w")
            return

        sample = self.text_var.get().strip() or "示例 Aa"
        for i, c in enumerate(self.candidates):
            try:
                pil = fm.render_candidate_preview(sample, c["path"], size=26)
                bg = Image.new("RGBA", pil.size, (250, 250, 250, 255))
                bg.alpha_composite(pil)
                photo = ImageTk.PhotoImage(bg.convert("RGB"))
            except Exception:
                photo = None
            self._cand_photos.append(photo)
            btn = tk.Button(
                self.cand_frame,
                image=photo if photo else None,
                text=f"{c['name']}  ({c['similarity']:.2f})",
                compound="left", anchor="w",
                command=lambda idx=i: self._select_candidate(idx))
            btn.pack(fill="x", pady=1)
            if c["path"] == self.selected_font_path:
                btn.configure(bg="#d6eaff", relief="solid")

    def _select_candidate(self, idx):
        if 0 <= idx < len(self.candidates):
            self.selected_font_path = self.candidates[idx]["path"]
            # 【修复：手动切字体后换框/重选不丢失】此前只有 apply / 键入才会把
            # font_path 写回 box_edits；一旦「切换字体 → 确认」后重选该框，恢复路径
            # (on_box_select_same/on_box_changed) 读到的仍是旧字体，效果被“打回原形”。
            # 切换即持久化，确认/应用/重选都不再回退。
            bi = self.canvas.get_active_index()
            if 0 <= bi:
                self.box_edits.setdefault(bi, {})["font_path"] = self.selected_font_path
            self._render_candidates()
            self._schedule_preview(0)

    def rematch_fonts(self):
        if self.current_box is None:
            messagebox.showinfo("提示", "请先在图上框选文字区域。")
            return
        x0, y0, x1, y1 = self.current_box
        region = self.original_bgr[y0:y1, x0:x1]
        text = self.text_var.get().strip() or "示例"
        self.candidates = fm.match_fonts(
            text, region,
            target_stroke=self.features.get("stroke_weight", 0.5),
            target_size=self.features.get("font_size", 16),
            target_stroke_px=self.features.get("stroke_px", 0.0),
        )
        if self.candidates:
            self.selected_font_path = self.candidates[0]["path"]
        self._render_candidates()
        self._schedule_preview(0)

    # ---------------------------------------------------------------- 渲染
    def _collect_params(self) -> Dict:
        """收集渲染参数。

        按需求精简：**只有「字号」一项可手动调节**，其余全部取自
        四向上下文自动采集的特征（不再有手动滑杆覆盖）。
        这样既避免手动项互相干扰，也保证新文字与原图质感自动一致。
        """
        f = self.features or {}
        return {
            "color_rgb": self.color_var,
            # 唯一保留的手动项
            "font_size": max(6, int(self.vars["font_size"].get())),
            # 以下全部来自自动特征
            "letter_spacing": int(f.get("letter_spacing", 0)),
            "line_spacing": int(f.get("line_spacing", 0)),
            "skew_angle": float(f.get("skew_angle", 0.0)),
            "sharpness": float(f.get("sharpness", 0.7)),
            "blur": float(f.get("blur_level", 0.0)),
            "edge_soften": 0.0,
            "noise": float(f.get("noise_level", 0.0)),
            "bold": bool(self.bold_var.get()),
            "target_stroke_px": float(f.get("stroke_px", 0.0)),
            "opacity": float(self.opacity_var.get()),
        }

    # ---------------------------------------------------------------- 二次排版 / 防重叠
    def _calibrate_letter_spacing(self, text: str, params: Dict,
                                  font_path: str | None) -> Dict:
        """让替换文字的字间距**严格等于原图字间距**（闭环校准，与原字体侧边距无关）。

        原图字间距 = features['letter_spacing']（estimate_letter_spacing 测得的相邻字符
        墨迹间空白列宽）。用户若手动拖过字间距滑块，则以手动值为准；否则严格等于原图。

        【为什么必须闭环校准】渲染引擎的 letter_spacing 是「字符 advance 之后追加的
        间距」，而可见的『字与字距离』= letter_spacing + 替换字体自身的左右侧边距，
        可能 ≠ 原图字体侧边距。直接把原图空白列宽塞进 letter_spacing，渲染出的可见
        间距会偏一个侧边距差。这里渲染一次、量一次真实空白列宽、按差值反推修正
        letter_spacing（blank_gap 对 letter_spacing 是 1 次线性关系，一次修正即精确收敛），
        从而无论换成哪款字体，替换文字的『字与字距离』都与原图逐像素一致。

        【与旧实现的区别】旧 `_auto_balance_layout` 按「原文字像素总宽」反推字间距，
        使新文字整段宽度去贴合原图跨度——字数一变就把字距收放，替换字距与原图
        根本对不齐（用户报「识别原图文字时没有识别字与字之间的距离」的根因）。
        现改为：**字间距严格锁定原图实测值**，宽度差异由字数自然决定，不做任何收放。
        """
        p = dict(params)
        idx = self.canvas.get_active_index()
        manual = self._manual_letter_spacing(idx)
        target = manual if manual is not None else int(
            round(float((self.features or {}).get("letter_spacing", 0))))
        p["letter_spacing"] = int(target)
        # 仅单行且字数 ≥ 2 才有可测的字符间空白；多行 / 单字跳过校准
        # （行距由 line_spacing 管，单字无字距可言）。
        if (font_path and "\n" not in text and len(text) >= 2
                and (self.features or {}).get("line_count", 1) == 1):
            try:
                seen = set()
                # 闭环修正：可见字间距 ≈ letter_spacing + 恒定侧边距偏移（offset）。
                # 由当前点反推：offset = meas - ls；要达到目标间隙 target 需
                #   ls' = target - offset = ls + (target - meas)
                # 故修正量应加在当前 letter_spacing 上（ls + diff），而非 target + diff——
                # 后者会在「整数像素量化台阶」处放大振荡（ls 在两点间来回跳），永不收敛。
                # 最多留 8 遍；若 new_ls 回到已访问值，说明目标间隙落在相邻整数之间无精确解，
                # 停在最接近的一档（误差 ≤1px，属整数像素渲染的量化极限）。
                for _ in range(8):
                    cur, _m = render_mod.render_text_with_metrics(
                        text, p, font_path=font_path)
                    meas = render_mod.measure_blank_gap(cur)
                    if meas is None:
                        break
                    if meas == target:
                        break
                    new_ls = int(round(p["letter_spacing"] + (target - meas)))
                    if new_ls == p["letter_spacing"] or new_ls in seen:
                        break
                    seen.add(p["letter_spacing"])
                    p["letter_spacing"] = new_ls
            except Exception:
                pass
        return p

    @staticmethod
    def _overlaps_any(px: int, py: int, ink, forbidden, margin: int = 3) -> bool:
        """判断文字在 (px,py) 处的墨迹包围盒是否与任一 forbidden 矩形重叠。"""
        if not forbidden or ink is None:
            return False
        bx0, by0, bx1, by1 = px + ink[0], py + ink[1], px + ink[2], py + ink[3]
        for (fx0, fy0, fx1, fy1) in forbidden:
            if (bx1 <= fx0 - margin or bx0 >= fx1 + margin
                    or by1 <= fy0 - margin or by0 >= fy1 + margin):
                continue  # 不相交
            return True
        return False

    # 最近一次渲染的排版度量（由 _do_render 写入），供基线对齐使用
    _last_metrics: Dict = {}

    def _target_baseline_y(self, box) -> int | None:
        """原图文字基线的 **整图** y 坐标。优先本区推定，不可靠时用四向参照。

        返回 None 表示无从推定，由调用方降级为垂直中心对齐。
        """
        x0, y0, x1, y1 = box
        f = self.features or {}
        by = f.get("baseline_y")
        if by is not None:
            return int(y0) + int(by)
        return None

    def _base_ink_position(self, text_rgba, ink, box):
        """【本轮核心修复】按 **基线** 对齐放置，而非墨迹包围盒中心。

        旧实现把新文字的墨迹中心对齐到原文字墨迹中心。但墨迹中心到基线的
        距离随字符剧烈变化 —— 实测 em=64 时该偏移跨度达 **23px**（「一」24px
        vs「。」1px），所以只要替换文字与原文字符构成不同，就会明显忽高忽低。

        改为纵向按「基线」对齐（基线是排版的唯一正确锚点），
        横向仍按墨迹中心对齐（水平方向无基线概念，中心对齐是合理的）。
        允许超出选框，由上层做防重叠处理。
        """
        x0, y0, x1, y1 = box
        tb = self.features.get("text_bbox")

        # ---- 横向：墨迹中心对齐原文字中心 ----
        if tb is not None and ink is not None:
            orig_cx = x0 + (tb[0] + tb[2]) / 2.0
            ink_cx = (ink[0] + ink[2]) / 2.0
            px = int(round(orig_cx - ink_cx))
        else:
            th, tw = text_rgba.shape[:2]
            px = x0 + max(0, ((x1 - x0) - tw) // 2)

        # ---- 纵向：基线对齐 ----
        baseline_in_render = (self._last_metrics or {}).get("baseline_y")
        orig_baseline = self._target_baseline_y(box)
        if baseline_in_render is not None and orig_baseline is not None:
            py = int(round(orig_baseline - baseline_in_render))
        elif tb is not None and ink is not None:
            # 退化：拿不到基线时，才退回墨迹中心对齐
            orig_cy = y0 + (tb[1] + tb[3]) / 2.0
            ink_cy = (ink[1] + ink[3]) / 2.0
            py = int(round(orig_cy - ink_cy))
        else:
            th, tw = text_rgba.shape[:2]
            py = y0 + max(0, ((y1 - y0) - th) // 2)
        return px, py

    def _place_anti_overlap(self, text_rgba, ink, box, forbidden, base_px, base_py):
        """第五步：防重叠放置 —— 允许超出选框，但优先避开邻近原始文字。

        策略：以基础位置为中心，按**位移由小到大**搜索最近的不重叠空位
        （用户报「放大字体后文字跑到选框外面」：旧实现先垂直盲扫 ±80px，可能
        把整块文字挪到很远；按距离优先可让文字尽量留在选框附近）。
        若全部空位都被邻近文字占满，则退回基础位置尽力放置（**不再缩小字号**，
        用户硬性要求：不要自动缩小字体）。
        """
        if not forbidden:
            return base_px, base_py
        if not self._overlaps_any(base_px, base_py, ink, forbidden):
            return base_px, base_py
        # 按 |dx|+|dy| 由小到大枚举候选（步长 4，范围 ±80 像素，距离优先）
        cands = []
        for dx in range(-80, 81, 4):
            for dy in range(-80, 81, 4):
                cands.append((dx, dy))
        cands.sort(key=lambda p: (abs(p[0]) + abs(p[1]),
                                  0.01 * max(abs(p[0]), abs(p[1]))))
        for dx, dy in cands:
            if dx == 0 and dy == 0:
                continue
            if not self._overlaps_any(base_px + dx, base_py + dy, ink, forbidden):
                return base_px + dx, base_py + dy
        return base_px, base_py  # 尽力而为（不缩小字号）

    # ------------------------------------------------------- 擦除范围（原文字彻底消失）
    def _text_extent_region(self, box):
        """返回该框处「原文字真实墨迹范围」的绝对图像坐标区域（与选框取并集）。

        【注意】本函数**已不再参与擦除范围的决定**——按本轮需求，擦除严格限制在
        选框内（见 `_erase_region_for`）。此函数仅作为「原文字真实占用范围」的
        探查工具保留（诊断/测试仍在使用），不参与任何写图路径。

        早期设计（已废弃）为什么要这样：原文字常常**超出选框**（框选偏紧、有上伸/下伸笔画），
        只按选框擦除会留下残字；一旦移动选框，这些残字就露出来，与新渲染的
        替换文字**重合**。这里从**原图**检测该处文字的实际墨迹包围盒，
        再与选框取并集作为擦除范围，保证「确定替换后原文字彻底消失」。

        检测失败或结果明显不合理（几乎等于整块探查区）时**回退为原选框**，
        保证最坏情况下行为不劣于从前。
        """
        if self.original_bgr is None or box is None:
            return box
        x0, y0, x1, y1 = [int(v) for v in box]
        H, W = self.original_bgr.shape[:2]
        x0, y0 = max(0, min(x0, W)), max(0, min(y0, H))
        x1, y1 = max(0, min(x1, W)), max(0, min(y1, H))
        if x1 - x0 < 1 or y1 - y0 < 1:
            return box
        # 向外探查余量：覆盖超出选框的笔画（至少 10px，兼容小幅溢出）
        pad = max(10, int(min(x1 - x0, y1 - y0) * 0.35))
        ex0, ey0 = max(0, x0 - pad), max(0, y0 - pad)
        ex1, ey1 = min(W, x1 + pad), min(H, y1 + pad)
        if ex1 - ex0 < 1 or ey1 - ey0 < 1:
            return box
        try:
            bb = fe.foreground_bbox(self.original_bgr[ey0:ey1, ex0:ex1])
        except Exception:
            return box
        if not bb:
            return box
        tx0, ty0 = ex0 + int(bb[0]), ey0 + int(bb[1])
        tx1, ty1 = ex0 + int(bb[2]), ey0 + int(bb[3])
        bw, bh = x1 - x0, y1 - y0
        # 合理性 1：不能比选框大太多（否则多半是把背景/大片纹理误判成文字）
        if (tx1 - tx0) > bw * 3 or (ty1 - ty0) > bh * 3:
            return box
        # 合理性 2：必须与选框有交集，避免误检到远处内容
        if not (tx0 < x1 and tx1 > x0 and ty0 < y1 and ty1 > y0):
            return box
        m = 2   # 边缘再留 2px，避免抗锯齿边没擦干净
        ux0 = max(0, min(x0, tx0) - m)
        uy0 = max(0, min(y0, ty0) - m)
        ux1 = min(W, max(x1, tx1) + m)
        uy1 = min(H, max(y1, ty1) + m)
        return (ux0, uy0, ux1, uy1)

    def _erase_region_for(self, box, text_bbox=None):
        """擦除范围 = **选框矩形本身**（严格限制在选框内，再夹取到图像边界）。

        【本轮需求变更】旧实现是「原文字真实墨迹范围 ∪ 选框 ∪ 替换文字占位」，
        为防「残字与替换文字重合」而**越出选框**擦除，把框外的上下文文字/纹理
        也一并抹掉。用户明确要求：**消除原图文字的效果必须严格限制在选框内**，
        框外一个像素都不许动。因此这里不再取并集、不再外扩，只用选框本身。

        text_bbox（替换文字占位）形参保留仅为兼容旧调用点，**不再参与范围计算**。
        """
        if box is None:
            return None
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        H = W = 0
        if self.original_bgr is not None:
            H, W = self.original_bgr.shape[:2]
        if W <= 0 or H <= 0:
            return (x0, y0, x1, y1)
        return (max(0, min(x0, W)), max(0, min(y0, H)),
                max(0, min(x1, W)), max(0, min(y1, H)))

    @staticmethod
    def _abs_text_bbox(text_rgba, placed):
        """替换文字在**整图**中的墨迹包围盒（绝对图像坐标）；无墨迹返回 None。"""
        try:
            ink = render_mod.text_ink_bbox(text_rgba)
        except Exception:
            return None
        if not ink:
            return None
        px, py = placed
        return (int(px + ink[0]), int(py + ink[1]),
                int(px + ink[2]), int(py + ink[3]))

    # ------------------------------------------------------- 预览「井」字辅助线
    @staticmethod
    def _cv_dashed_line(img, p1, p2, color, dash=5, gap=4, thick=1):
        """在 BGR 图上画虚线（按段绘制，端点超出画布由 cv2 自动裁剪）。"""
        x0, y0 = p1
        x1, y1 = p2
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        pos, on = 0.0, True
        while pos < length:
            seg = dash if on else gap
            end = min(pos + seg, length)
            if on:
                cv2.line(img,
                         (int(round(x0 + ux * pos)), int(round(y0 + uy * pos))),
                         (int(round(x0 + ux * end)), int(round(y0 + uy * end))),
                         color, thick, cv2.LINE_AA)
            pos = end
            on = not on

    def _draw_preview_guides(self, img: np.ndarray, text_bbox, crop_origin, extend_to=None):
        """在图上画「井」字参考线：四根直线围住替换文字，相邻直线相交成井字，
        用于把替换文字与周围上下文对齐（类似 PPT 的智能参考线）。

        四根线分别是替换文字墨迹的上 / 下 / 左 / 右边界：
          - extend_to=(W,H) 时，四根线**贯穿整幅**（到图像/裁剪区边缘），
            像 PPT 对齐辅助线一样帮助看清与四周上下文的行/列对齐关系；
          - 为 None 时仅向两端各延伸一小段（紧凑井字）。

        只画在显示副本上，绝不写入工作图 / 预览结果 / 导出图。
        """
        cx0, cy0 = crop_origin
        tx0, ty0, tx1, ty1 = [int(v) for v in text_bbox]
        # 整图坐标 → 显示图坐标
        x0, y0, x1, y1 = tx0 - cx0, ty0 - cy0, tx1 - cx0, ty1 - cy0
        if x1 - x0 < 1 or y1 - y0 < 1:
            return
        color = (0, 140, 255)      # BGR：橙色，与 UI 主色一致
        dash, gap, thick = 5, 4, 1
        if extend_to is not None:
            ew, eh = int(extend_to[0]), int(extend_to[1])
            # 上/下线贯穿整宽，左/右线贯穿整高 → 完整井字，便于对照上下文对齐
            self._cv_dashed_line(img, (0, y0), (ew, y0), color, dash, gap, thick)
            self._cv_dashed_line(img, (0, y1), (ew, y1), color, dash, gap, thick)
            self._cv_dashed_line(img, (x0, 0), (x0, eh), color, dash, gap, thick)
            self._cv_dashed_line(img, (x1, 0), (x1, eh), color, dash, gap, thick)
        else:
            ext = max(6, int(min(x1 - x0, y1 - y0) * 0.45))
            self._cv_dashed_line(img, (x0 - ext, y0), (x1 + ext, y0), color, dash, gap, thick)
            self._cv_dashed_line(img, (x0 - ext, y1), (x1 + ext, y1), color, dash, gap, thick)
            self._cv_dashed_line(img, (x0, y0 - ext), (x0, y1 + ext), color, dash, gap, thick)
            self._cv_dashed_line(img, (x1, y0 - ext), (x1, y1 + ext), color, dash, gap, thick)

    def _alignment_references(self):
        """对齐参考线（图像坐标）：画布中线 + 其它选框 + 邻近文字框 的 左/中/右/上/中/下。

        供 PPT 式智能参考线判定——替换文字的对应边一旦接近这些参考线即视为对齐。
        """
        refs = []
        if self.original_bgr is not None:
            h, w = self.original_bgr.shape[:2]
            refs.append(("v", w / 2.0))   # 画布垂直中线
            refs.append(("h", h / 2.0))   # 画布水平中线
        try:
            aidx = self.canvas.get_active_index()
        except Exception:
            aidx = -1
        for i, b in enumerate(self.canvas._boxes):
            if i == aidx:
                continue
            x0, y0, x1, y1 = [float(v) for v in b]
            refs += [("v", x0), ("v", (x0 + x1) / 2.0), ("v", x1),
                     ("h", y0), ("h", (y0 + y1) / 2.0), ("h", y1)]
        for b in (self.neighbor_boxes or []):
            x0, y0, x1, y1 = [float(v) for v in b]
            refs += [("v", x0), ("v", (x0 + x1) / 2.0), ("v", x1),
                     ("h", y0), ("h", (y0 + y1) / 2.0), ("h", y1)]
        return refs

    def _smart_alignments(self, tb_abs, refs, thr=6):
        """判定替换文字墨迹框 tb_abs 与各参考线的对齐情况。

        返回 (aligned, snap_dx, snap_dy)：
          - aligned: 已对齐的参考线列表 [("v", x) / ("h", y)]（图像坐标），供画智能线；
          - snap_dx/dy: 让文字整体平移到最近对齐处的最小偏移（像素）。
        """
        tx0, ty0, tx1, ty1 = [float(v) for v in tb_abs]
        ax = [tx0, (tx0 + tx1) / 2.0, tx1]
        ay = [ty0, (ty0 + ty1) / 2.0, ty1]
        best_v = best_h = None
        for axis, pos in refs:
            if axis == "v":
                for a in ax:
                    d = pos - a
                    if abs(d) <= thr and (best_v is None or abs(d) < abs(best_v[0])):
                        best_v = (d, pos)
            else:
                for a in ay:
                    d = pos - a
                    if abs(d) <= thr and (best_h is None or abs(d) < abs(best_h[0])):
                        best_h = (d, pos)
        aligned = []
        snap_dx = snap_dy = 0.0
        if best_v is not None:
            snap_dx, pv = best_v
            aligned.append(("v", pv))
        if best_h is not None:
            snap_dy, ph = best_h
            aligned.append(("h", ph))
        return aligned, snap_dx, snap_dy

    def _apply_smart_snap(self, text_rgba, placed):
        """PPT 式智能吸附：把替换文字整体平移，使其对齐基准贴近最近参考线。

        仅平移、不改字号/样式；异常时退回原放置，绝不中断渲染。
        """
        try:
            tb = self._abs_text_bbox(text_rgba, placed)
            refs = self._alignment_references()
            _, sdx, sdy = self._smart_alignments(tb, refs, thr=6)
        except Exception:
            return placed
        # placed 必须是整数像素（composite_text 用它做切片），吸附仅整数位移
        return (int(round(placed[0] + sdx)), int(round(placed[1] + sdy)))

    def _draw_smart_text_guides(self, img, tb_abs, origin, disp_w, disp_h):
        """PPT 式智能参考线：替换文字对齐到某参考线时，在对应位置画一条**贯穿整幅**
        的实线（品红，与常驻橙色井字区分），表示已对齐、可吸附；无对齐则不画。
        只画在显示副本上，绝不写入工作图/导出图。
        """
        try:
            refs = self._alignment_references()
            if not refs:
                return
            aligned, _, _ = self._smart_alignments(tb_abs, refs, thr=6)
        except Exception:
            return
        if not aligned:
            return
        cx0, cy0 = origin
        color = (255, 0, 255)      # BGR 品红，与橙色井字区分
        for axis, pos in aligned:
            if axis == "v":
                x = int(round(pos - cx0))
                if 0 <= x <= disp_w:
                    cv2.line(img, (x, 0), (x, disp_h), color, 1, cv2.LINE_AA)
            else:
                y = int(round(pos - cy0))
                if 0 <= y <= disp_h:
                    cv2.line(img, (0, y), (disp_w, y), color, 1, cv2.LINE_AA)

    def _preview_display_image(self, crop: np.ndarray) -> np.ndarray:
        """预览显示用图：在**副本**上叠加「井」字参考线（绝不污染工作图）。

        crop 是 pending_result / working_bgr 的**视图**，直接在其上绘图会顺着
        视图写回底层图像 —— 那样辅助线就会被「确定替换」一起提交、
        甚至被导出，属于严重事故。所以这里必须先 copy() 再画。
        """
        rect = getattr(self, "_last_crop_rect", None)
        tb = getattr(self, "_last_text_bbox", None)
        if crop is None or rect is None or tb is None or self._preview_show_original:
            return crop
        try:
            disp = crop.copy()
            cw = rect[2] - rect[0]
            ch = rect[3] - rect[1]
            # 贯穿整幅预览区（到裁剪区边缘），更像 PPT 对齐辅助线
            self._draw_preview_guides(disp, tb, (rect[0], rect[1]), extend_to=(cw, ch))
            # PPT 式智能参考线（对齐到画布中线/其它选框/邻近文字时闪现）
            self._draw_smart_text_guides(disp, tb, (rect[0], rect[1]), cw, ch)
        except Exception:
            return crop
        return disp

    def _canvas_display_image(self, img: np.ndarray) -> np.ndarray:
        """主画布显示用图：在**副本**上叠加「井」字参考线（绝不污染工作图/导出图）。

        set_display 仅写入显示缓冲 image_cv，与「应用替换」提交的 pending_result、
        以及导出用的 working_bgr 完全分离；所以这里叠加的参考线**不可能**进入成品图。
        主画布显示整图，参考线贯穿整幅图像边缘（类似 PPT 智能参考线）。
        """
        tb = getattr(self, "_last_text_bbox", None)
        if img is None or tb is None or self._preview_show_original:
            return img
        try:
            disp = img.copy()
            ih, iw = img.shape[:2]
            self._draw_preview_guides(disp, tb, (0, 0), extend_to=(iw, ih))
            # PPT 式智能参考线（对齐到画布中线/其它选框/邻近文字时闪现）
            self._draw_smart_text_guides(disp, tb, (0, 0), iw, ih)
        except Exception:
            return img
        return disp

    def _do_render(self, base_bgr: np.ndarray, optimize: bool = False) -> np.ndarray | None:
        """在 base_bgr 上执行「擦除旧文字 + 写入新文字」，返回结果图。

        optimize=True（确认后）：执行第四步二次排版优化 + 第五步防重叠渲染；
        optimize=False（实时预览过渡态）：仅做基础对齐放置（仍避免与邻近文字重叠）。
        """
        if self.current_box is None or base_bgr is None:
            return None
        x0, y0, x1, y1 = self.current_box
        idx = self.canvas.get_active_index()
        params = self._collect_params()

        # 擦除强度：面板精简后不再暴露，固定使用自动擦除强度
        strength = float(self.vars["erase_strength"].get()) if "erase_strength" in self.vars else 0.5

        # 1) 有替换文字才渲染新文字。**先算出放置位置**（放置只依赖选框与邻居，
        #    与擦除结果无关），这样后面才能把「替换文字占位」一并纳入擦除范围，
        #    确保新文字不会压在未擦净的原文字上造成重合。
        text = self.text_var.get().strip()
        if not text:
            self._last_text_bbox = None
            return erase_mod.erase_text(base_bgr,
                                        self._erase_region_for(self.current_box),
                                        strength=strength)

        forbidden = self.neighbor_boxes or []

        # 2) 字间距严格对齐原图（闭环校准，与原字体侧边距无关）
        #    所有路径统一执行（预览 / 确认后成品一致）：替换文字「字与字距离」
        #    逐像素等于原图——无论字数增减、无论换成哪款字体都稳定。
        manual_fs = self._manual_font_size(idx)
        if optimize:
            # 已确认框：优先复用「冻结的二次优化参数」——样式稳定，绝不随上下文重算漂移
            # （用户硬性要求：重开后仅编辑文字，原有样式不被覆盖）。
            locked = self.box_edits.get(idx, {}).get("optimized_params") if 0 <= idx else None
            if locked is not None:
                params = dict(locked)
                if manual_fs is not None:
                    params["font_size"] = manual_fs
        if manual_fs is not None:
            params["font_size"] = manual_fs    # 兜底：任何分支都以手调值为准
        params = self._calibrate_letter_spacing(text, params, self.selected_font_path)

        # 3) 渲染（保持字号原样，**不再逐级自动缩小**——用户硬性要求
        #    「不要自动缩小字体」）+ 第五步防重叠放置（允许超出选框）
        try:
            cur, metrics = render_mod.render_text_with_metrics(
                text, params, font_path=self.selected_font_path)
        except Exception as e:
            self.status_var.set(f"文字渲染失败：{e}")
            self._last_text_bbox = None
            return erase_mod.erase_text(base_bgr,
                                        self._erase_region_for(self.current_box),
                                        strength=strength)
        # 记录本次渲染的基线位置，供 _base_ink_position 做基线对齐
        self._last_metrics = metrics
        ink = render_mod.text_ink_bbox(cur)
        bx, by = self._base_ink_position(cur, ink, self.current_box)
        if not self._overlaps_any(bx, by, ink, forbidden):
            text_rgba, placed = cur, (bx, by)
        else:
            # 尝试在选框附近平移避开（距离优先，不缩小字号）
            px, py = self._place_anti_overlap(cur, ink, self.current_box, forbidden, bx, by)
            text_rgba, placed = cur, (px, py)
        # 4) PPT 式智能吸附：替换文字的对齐基准（左/中/右/上/中/下）接近画布中线、
        #    其它选框或邻近文字的对应边时，自动平移吸附到该参考线（仅平移、不改样式）。
        placed = self._apply_smart_snap(text_rgba, placed)
        text_bbox_abs = self._abs_text_bbox(text_rgba, placed)

        # 5) 擦除旧文字：范围 = 原文字真实墨迹范围 ∪ 选框 ∪ 替换文字占位（已含吸附后位置）。
        #    第三项是「原文字不与替换文字重合」的物理保证——新文字允许超出选框
        #    放置（防重叠避让会平移），若只擦选框，超出部分就会压在未擦净的
        #    原文字上，看起来就是两个字叠在一起。
        result = erase_mod.erase_text(
            base_bgr, self._erase_region_for(self.current_box, text_bbox_abs),
            strength=strength)

        # 6) 合成新文字
        result = render_mod.composite_text(result, text_rgba, placed)
        # 记录替换文字在整图中的包围盒：仅供预览「井」字辅助线定位，不写入图像
        self._last_text_bbox = text_bbox_abs
        return result

    def _refresh_preview(self):
        """生成预览：更新主画布 + 预览面板。

        判定「是否有替换内容」是**需求一**的关键：OCR 自动填入的原文只作建议，
        不算替换 —— 否则原文会被「重新渲染回去」，看起来就像「没删掉」。只有
        用户改过字或点过确认，才渲染替换文字（并叠加参考线）。
        """
        self._preview_job = None
        if self.working_bgr is None:
            return
        if self.current_box is None:
            # 未框选时显示当前工作图
            self._last_text_bbox = None
            self.canvas.set_display(self.working_bgr)
            self.pending_result = None
            self._update_preview_panel()
            return
        # 是否有替换内容：
        #   - 已确认 / 已二次优化（optimized / optimized_params）→ 有
        #   - 用户在 OCR 原文基础上**改过字** → 有
        #   - 仅 OCR 自动填入、未改动也未确认 → **无**（直接显示已擦除的空白底图，
        #     绝不能把原文渲染回去，否则「新建选框识别文字后原文字复活」= 需求一失败）
        idx = self.canvas.get_active_index()
        st = self.box_edits.get(idx, {}) if 0 <= idx else {}
        user_text = self.text_var.get().strip()
        ocr_text = (st.get("ocr_text") or "").strip()
        confirmed = bool(st.get("_confirmed") or st.get("optimized")
                         or st.get("optimized_params"))
        has_replacement = confirmed or (bool(user_text) and user_text != ocr_text)
        if not has_replacement:
            # 仅擦除、无替换：显示空白底图，并清掉过期参考线包围盒
            self._last_text_bbox = None
            self.canvas.set_display(self.working_bgr)
            self.pending_result = None
            self._update_preview_panel()
            self._notify_zoom()
            return
        try:
            optimized = (0 <= idx) and bool(self.box_edits.get(idx, {}).get("optimized"))
            result = self._do_render(self.working_bgr, optimize=optimized)
        except Exception as e:
            self.status_var.set(f"预览生成失败：{e}")
            return
        if result is None:
            return
        self.pending_result = result            # 无参考线，供「应用替换」提交 / 导出
        # 主画布显示：在副本上叠加「井」字参考线（仅显示，绝不写回工作图/导出图）
        self.canvas.set_display(self._canvas_display_image(result))
        self._update_preview_panel()            # 预览面板同样叠加参考线
        self._notify_zoom()

    def _preview_crop(self, img: np.ndarray):
        """裁剪当前框选区域（带边距）用于预览面板。

        同时把裁剪区域记进 `self._last_crop_rect`，供预览辅助线把「整图坐标」
        换算成「裁剪图坐标」。
        """
        if img is None or self.current_box is None:
            self._last_crop_rect = None
            return None
        x0, y0, x1, y1 = self.current_box
        h, w = img.shape[:2]
        pad = max(12, int(max(x1 - x0, y1 - y0) * 0.3))
        cx0 = max(0, x0 - pad); cy0 = max(0, y0 - pad)
        cx1 = min(w, x1 + pad); cy1 = min(h, y1 + pad)
        if cx1 <= cx0 or cy1 <= cy0:
            self._last_crop_rect = None
            return None
        self._last_crop_rect = (cx0, cy0, cx1, cy1)
        return img[cy0:cy1, cx0:cx1]

    def _update_preview_panel(self):
        """更新内嵌预览面板与独立预览窗口。"""
        src = self.pending_result if self.pending_result is not None else self.working_bgr
        if self._preview_show_original:
            src = self.working_bgr
        crop = self._preview_crop(src)
        if crop is None:
            self.preview_label.configure(image="", text="框选文字区域后显示预览")
            self._preview_photo = None
            if self.preview_window is not None:
                self._update_preview_window(None)
            return
        # 辅助线只画在显示副本上（_preview_display_image 内部会先 copy）
        photo = self._to_photo(self._preview_display_image(crop), max_w=360)
        self._preview_photo = photo
        self.preview_label.configure(image=photo, text="")
        if self.preview_window is not None:
            self._update_preview_window(crop)

    def _to_photo(self, bgr: np.ndarray, max_w: int, max_h: int = 900):
        """BGR → 适配尺寸的 PhotoImage。"""
        h, w = bgr.shape[:2]
        scale = min(max_w / w, max_h / h, 4.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        rgb = cv2.cvtColor(cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def toggle_preview_mode(self):
        self._preview_show_original = not self._preview_show_original
        self._update_preview_panel()
        self.status_var.set("预览显示：" + ("原图（未替换）" if self._preview_show_original else "替换效果"))

    # ---------------------------------------------------------------- 独立预览窗口
    def open_preview_window(self):
        if self.preview_window is not None:
            try:
                self.preview_window.lift()
                return
            except Exception:
                self.preview_window = None
        win = tk.Toplevel(self.root)
        win.title("预览窗口 — 实时同步")
        win.geometry("720x560")
        lab = tk.Label(win, bg="#f0f0f0")
        lab.pack(fill="both", expand=True, padx=6, pady=6)
        self.preview_window = win
        self._preview_window_label = lab
        win.protocol("WM_DELETE_WINDOW", self._close_preview_window)
        self._update_preview_panel()

    def _close_preview_window(self):
        if self.preview_window is not None:
            try:
                self.preview_window.destroy()
            except Exception:
                pass
        self.preview_window = None
        self._preview_window_label = None

    def _update_preview_window(self, crop):
        if self.preview_window is None:
            return
        lab = getattr(self, "_preview_window_label", None)
        if lab is None:
            return
        if crop is None:
            lab.configure(image="", text="请先框选文字区域")
            return
        # 独立预览窗口同样只在显示副本上叠加辅助线
        photo = self._to_photo(self._preview_display_image(crop), max_w=680, max_h=480)
        lab.configure(image=photo, text="")
        lab.image = photo  # 防 GC

    # ---------------------------------------------------------------- 应用/撤销/保存
    def apply_replace(self, silent: bool = False):
        """把当前预览结果写入工作图（提交一个可撤销的「历史帧」）。

        静默提交（离开当前框 / 保存前的自动固化）仅在框内确有可替换内容时执行：
        用户随手框选一下、一个字都没输入就点别处，绝不能把「只擦除预览」固化下去
        把原文永久擦掉（这是此前的一个隐性破坏）。
        """
        if self.working_bgr is None:
            messagebox.showinfo("提示", "请先上传图片。")
            return
        # 记忆当前框的编辑状态（文字/字体/特征），供切回该框时恢复。
        # 注意：必须是「合并」而非「整体覆盖」，否则会清掉该框已锁定的
        # _locked / optimized / neighbor_ref / neighbor_boxes 等字段，导致切换选区后
        # 上一框的特征/防重叠信息丢失、效果回弹（用户明确报的 BUG）。
        idx = self.canvas.get_active_index()
        state = None
        if 0 <= idx:
            state = self.box_edits.setdefault(idx, {})
            state["text"] = self.text_var.get().strip()
            state["font_path"] = self.selected_font_path
            state["features"] = self.features
        if self.pending_result is None:
            if not silent:
                messagebox.showinfo("提示", "没有可应用的修改，请先框选文字区域并输入替换文字。")
            return
        # 静默提交保护：没有替换文字且未确认过的空框「只擦除预览」绝不自动固化
        text = self.text_var.get().strip()
        has_content = bool(text) or bool(state and (
            state.get("_confirmed") or state.get("optimized") or state.get("optimized_params")))
        if silent and not has_content:
            self.pending_result = None
            self.canvas.set_display(self.working_bgr)
            self._update_preview_panel()
            return
        # 提交：记录「图像 + 各框编辑状态」历史帧，供撤销一并还原（防复活）
        self.history.append({
            "bgr": self.pending_result.copy(),
            "box_edits": copy.deepcopy(self.box_edits),
        })
        self._clear_redo()      # 产生新提交 → 之前的「可恢复」分支作废
        self.working_bgr = self.pending_result
        self.pending_result = None
        if state is not None:
            # 记下本次提交的内容指纹：下次离开该框时若没改动就跳过重复提交
            state["_committed_fp"] = self._edit_fingerprint()
        self.canvas.set_display(self.working_bgr)
        self._update_preview_panel()
        if not silent:
            self.status_var.set("已应用替换 — 可继续框选下一处文字")

    # ------------------------------------------------------------ 撤销 / 恢复
    def _push_redo(self, frame: Dict):
        """把「被撤销掉的那一帧」压入恢复栈（redo 用）。

        只保存**引用安全的深拷贝**：图像 copy + box_edits 深拷贝，
        保证后续对工作图的任何修改都不会污染栈里的内容。
        """
        try:
            self.redo_stack.append({
                "bgr": frame["bgr"].copy() if frame.get("bgr") is not None else None,
                "box_edits": copy.deepcopy(frame.get("box_edits", {}) or {}),
                "pending": (frame["pending"].copy()
                            if frame.get("pending") is not None else None),
            })
        except Exception:
            # 记栈失败绝不能影响撤销本身（用户底线：出错也要能撤销、绝不崩）
            pass

    def _clear_redo(self):
        """产生**新的**历史帧 / 框集合变化后清空恢复栈（标准 redo 语义）。"""
        self.redo_stack.clear()

    def redo(self):
        """恢复（redo）：重做最近一次被撤销的操作，与「撤销」严格互逆。

        栈里每一帧都是 undo 时完整保存的「图像 + 各框编辑状态」（外加被丢弃的
        未提交草稿），因此恢复后：替换文字回到图上、各框的文字/字体/冻结参数
        一起回来、输入框文字同步，不会出现「恢复了但框状态是空的」这种半吊子状态。

        栈空时只给提示，绝不抛异常。
        """
        if not self.redo_stack:
            messagebox.showinfo("提示", "没有可恢复的操作。")
            return
        frame = self.redo_stack.pop()
        if frame.get("bgr") is None:
            messagebox.showinfo("提示", "没有可恢复的操作。")
            return
        # 恢复帧重新入历史栈（之后还可以继续撤销）
        self.history.append({
            "bgr": frame["bgr"].copy(),
            "box_edits": copy.deepcopy(frame.get("box_edits", {}) or {}),
        })
        self.working_bgr = frame["bgr"].copy()
        self.box_edits = copy.deepcopy(frame.get("box_edits", {}) or {})
        self.pending_result = (frame["pending"].copy()
                               if frame.get("pending") is not None else None)
        # 输入框文字与恢复后的帧同步（避免残留旧文字导致「恢复后文字不对」）
        idx = self.canvas.get_active_index()
        if 0 <= idx:
            restored = (self.box_edits.get(idx, {}) or {}).get("text", "")
            if self.text_var.get().strip() != restored:
                self.text_var.set(restored or "")
        self.canvas.set_display(self.pending_result
                                if self.pending_result is not None else self.working_bgr)
        self._update_preview_panel()
        self.status_var.set("已恢复上一步（可继续撤销）")

    def has_redo(self) -> bool:
        """是否还有可恢复的步骤（供 UI/自测判断）。"""
        return bool(self.redo_stack)

    def undo(self):
        """撤销上一步操作（按用户报障重构）。

        两层语义：
        1) 有「已提交」步骤（history > 1）：回退最近一次提交——图像与各框编辑状态
           **一并**还原到上一帧。仅回退图像会让已撤销的替换在重新点选/缩放该框时
           复活，是用户报「不能撤销 / 撤销后又回来」的根因之一。
        2) 只有「未提交」的可见预览（确认后未点应用就按撤销）：丢弃该预览并清除
           本框未确认的编辑草稿，等价于撤销一次「确认/预览」动作（此前直接提示
           「没有可撤销的操作」，用户感觉撤销完全失效）。

        两种情形都会把「被撤销掉的状态」压入恢复栈，供 `redo` 原样还原。
        """
        idx = self.canvas.get_active_index()
        # ---- 情形 1：未提交的预览草稿（尚未形成历史帧）----
        if len(self.history) <= 1 and self.pending_result is not None:
            # 先把「当前可见的草稿」原样存进恢复栈（含 pending 预览图），
            # 这样撤销后再点「恢复」能把刚才那版替换完整找回来。
            self._push_redo({"bgr": self.working_bgr,
                             "box_edits": self.box_edits,
                             "pending": self.pending_result})
            self.pending_result = None
            self.canvas.set_display(self.working_bgr)
            if 0 <= idx and idx in self.box_edits:
                e = self.box_edits[idx]
                for k in ("text", "_confirmed", "optimized", "optimized_params",
                          "confirmed_text", "font_path"):
                    e.pop(k, None)
                # 输入框文字同步清空，避免重选该框时草稿复活
                if self.text_var.get().strip():
                    self.text_var.set("")
            self._update_preview_panel()
            self.status_var.set("已撤销（丢弃未确认的替换草稿）— 可点「恢复」找回")
            return
        # ---- 情形 2：回退最近一次已提交步骤 ----
        if len(self.history) > 1:
            undone = self.history[-1]
            src = undone if isinstance(undone, dict) else {"bgr": undone, "box_edits": {}}
            # 连同「尚未提交的草稿」一起入栈：撤销后再恢复，那版预览也原样回来
            self._push_redo({"bgr": src.get("bgr"),
                             "box_edits": src.get("box_edits", {}),
                             "pending": self.pending_result})
            self.pending_result = None
            self.history.pop()
            top = self.history[-1]
            if isinstance(top, dict) and "bgr" in top:
                self.working_bgr = top["bgr"].copy()
                self.box_edits = copy.deepcopy(top.get("box_edits", {})) or {}
            else:
                # 兼容旧格式（纯图像）：仅还原图像，框状态保持现状
                self.working_bgr = top.copy()
            # 【关键：输入框文字与恢复后的帧同步】若活动框在恢复帧里没有文字
            # （即刚被撤销的那次替换属于本框），必须清空 text_var——否则残留文字
            # 会让任何一次预览/重选把它重新渲染出来，造成「撤销后替换复活」。
            if 0 <= idx:
                restored = (self.box_edits.get(idx, {}) or {}).get("text", "")
                if self.text_var.get().strip() != restored:
                    self.text_var.set(restored or "")
            self.canvas.set_display(self.working_bgr)
            self._update_preview_panel()
            self.status_var.set("已撤销上一步 — 可点「恢复」找回")
        else:
            messagebox.showinfo("提示", "没有可撤销的操作。")

    def save_image(self):
        if self.working_bgr is None:
            messagebox.showinfo("提示", "请先上传并处理图片。")
            return
        # 关键修复：未应用的预览也要导出，避免保存出原图
        if self.pending_result is not None:
            self.apply_replace(silent=True)

        initial = "edited_image.png"
        if self.source_path:
            base = os.path.splitext(os.path.basename(self.source_path))[0]
            initial = f"{base}_edited.png"
        path = filedialog.asksaveasfilename(
            title="保存图片", defaultextension=".png", initialfile=initial,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")])
        if not path:
            return
        d = os.path.dirname(path)
        if d and not os.path.isdir(d):
            try:
                os.makedirs(d, exist_ok=True)
            except Exception as e:
                messagebox.showerror("错误", f"无法创建目录：{e}")
                return
        if not io_utils.imwrite_unicode(path, self.working_bgr):
            messagebox.showerror("错误", "保存失败，请更换保存路径或格式后重试。")
            return
        self.status_var.set(f"已保存：{path}")
        messagebox.showinfo("完成", f"已导出成品图片：\n{path}")

    # ---------------------------------------------------------------- 帮助
    def show_help(self):
        msg = (
            "操作方式：\n"
            "· 滚轮              放大 / 缩小图片（以光标为中心）\n"
            "· 右键 / 中键拖动   平移画面（放大后查看细节）\n"
            "· 左键拖拽          框选要替换的文字区域\n"
            "· 框内拖拽          移动选框（确认替换前可用；确认后位置锁死）\n"
            "· 复制选框 / Ctrl+D 复制当前选框，连同编辑文字与字体特征一起复制\n"
            "· Delete 删除选框    Ctrl+Z 撤销    Ctrl+Y / Ctrl+Shift+Z 恢复（重做）\n"
            "· 适应窗口 / 放大 / 缩小   工具栏按钮同样可控制视图\n\n"
            "使用流程：\n"
            "1. 上传图片，滚轮放大到能看清文字。\n"
            "2. 左键拖拽框选要修改的文字。\n"
            "3. 程序自动擦除旧文字并提取颜色/字号/间距/清晰度/粗细/倾斜。\n"
            "4. 输入替换文字，从候选字体小样中挑选最像的。\n"
            "5. 微调参数，右侧与独立预览窗口实时显示效果。\n"
            "6. 点「应用替换」写入，可继续框选下一处。\n"
            "7. 点「保存图片」导出成品（未应用的预览会自动应用）。\n\n"
            + config.LIMITATIONS_TEXT
        )
        messagebox.showinfo("使用说明", msg)


def run():
    root = tk.Tk()
    App(root)
    root.mainloop()
