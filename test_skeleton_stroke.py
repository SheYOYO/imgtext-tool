"""骨架细化笔画厚度 + 自动模糊质感融合 自测。

覆盖用户本轮硬性要求：
1. 用「骨架细化算法」（Zhang-Suen）计算笔画厚度，结果合理且与距离变换口径自洽。
2. 字体匹配阶段纳入骨架口径，字重判定稳定（仍精确选回生成原图的字体）。
3. 自动模糊：render_text 必须真正把 blur 参数施加到新文字上，
   且不受 sharpness 门控（此前自动模式下 blur_level 永远到不了渲染层，是 BUG）。
4. 全自动参数映射：extract_features 的 blur_level / noise_level 会进入滑块，
   从而驱动渲染阶段的模糊/噪点质感融合。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app import config
from app import feature_extract as fe
from app import render as render_mod
from app import font_match as fm

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "_test_output")
os.makedirs(OUT, exist_ok=True)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))


def _cn_font_path():
    pool = config.resolve_font_pool()
    cn = [f for f in pool if f.get("cn")]
    return cn[0]["path"] if cn else None


def _render_layer(text, font_path, size, bg=245):
    """渲染文字到纯色背景，返回 RGBA（4 通道）numpy，便于按 alpha 裁切。"""
    font = ImageFont.truetype(font_path, size)
    tmp = Image.new("RGBA", (1, 1), 0)
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + 20
    h = bbox[3] - bbox[1] + 20
    img = Image.new("RGBA", (max(1, w), max(1, h)), 0)
    d = ImageDraw.Draw(img)
    d.text((-bbox[0] + 10, -bbox[1] + 10), text, font=font, fill=(20, 20, 20, 255))
    return np.array(img)


def _region_from_layer(rgba):
    """把整张 RGBA（含 pad 背景）合成到白底，再裁出「文字 + 白底边」的 region。

    【口径统一】旧实现直接按 alpha>127 裁切、取 RGB 通道——透明 pad 的 RGB 是
    (0,0,0)，导致 region 几乎只有文字像素、无背景边。笔画测量（距离变换）在
    「贴字无背景边」上会失真（dt stroke 虚高，与真实框选口径不一致，进而使字体
    自匹配选错）。这里先整图合成白底（同 test_stroke_weight 口径），再按文字
    bbox 外扩保留背景边，与真实「框选文字 + 背景」一致。
    """
    al = rgba[:, :, 3] / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)
    bg = np.full_like(rgb, 245.0)
    comp = rgb * al[:, :, None] + bg * (1 - al[:, :, None])
    comp = comp.astype(np.uint8)
    a = rgba[:, :, 3]
    ys, xs = np.where(a > 127)
    if ys.size == 0 or xs.size == 0:
        return comp
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    # 外扩保留背景边（贴近真实框选）
    py = max(3, (y1 - y0) // 10)
    px = max(3, (x1 - x0) // 10)
    yy0, yy1 = max(0, y0 - py), min(comp.shape[0], y1 + py + 1)
    xx0, xx1 = max(0, x0 - px), min(comp.shape[1], x1 + px + 1)
    return comp[yy0:yy1, xx0:xx1]


def _edge_sharpness(arr):
    """用 alpha 通道拉普拉斯方差衡量边缘锐利度（越小越糊）。"""
    g = cv2.Laplacian(arr[:, :, 3].astype(np.float32), cv2.CV_32F)
    return float(g.var())


def main():
    fp = _cn_font_path()
    if not fp:
        print("  [SKIP] 无可用中文字体，跳过得骨架/模糊自测")
        return

    print("== 1. 骨架细化笔画厚度 ==")
    orig = _render_layer("粗细测试", fp, 48)
    region = _region_from_layer(orig)
    skel = fe.measure_stroke_px_skeleton(region)
    dt = fe.measure_stroke_px(region)
    print(f"  骨架口径={skel:.2f}px  距离变换口径={dt:.2f}px")
    check("骨架笔画厚度 > 0", skel > 0)
    check("骨架厚度在合理量程 [0.5,40]", 0.5 <= skel <= 40.0)
    check("骨架与距离变换口径自洽（偏差<=4px）",
          abs(skel - dt) <= 4.0, f"差值={abs(skel-dt):.2f}")

    feat = fe.extract_features(region)
    check("extract_features 含 stroke_px_skeleton 字段",
          "stroke_px_skeleton" in feat)
    check("extract_features 的 stroke_px_skeleton 与直接测量一致",
          abs(feat["stroke_px_skeleton"] - skel) < 1e-6)

    print("\n== 2. 字体匹配纳入骨架口径 ==")
    cands = fm.match_fonts("替换文字", region,
                           target_stroke=feat["stroke_weight"],
                           target_size=feat["font_size"],
                           target_stroke_px=feat["stroke_px"],
                           reference_text="粗细测试")
    check("返回候选字体", len(cands) > 0)
    check("骨架口径下精确选回生成字体", cands[0]["path"] == fp,
          f"实际={os.path.basename(cands[0]['path'])}")

    print("\n== 3. 自动模糊：blur 真正施加（不受 sharpness 门控）==")
    base_params = dict(
        color_rgb=(20, 20, 20), font_size=48, letter_spacing=0,
        line_spacing=0, skew_angle=0.0, sharpness=0.7,  # 自动模式 sharpness≈0.7
        edge_soften=0.0, noise=0.0, bold=False,
        target_stroke_px=skel, opacity=1.0,
    )
    p_blur = dict(base_params, blur=1.2)
    p_none = dict(base_params, blur=0.0)
    arr_blur = render_mod.render_text("模糊融合", p_blur, font_path=fp)
    arr_none = render_mod.render_text("模糊融合", p_none, font_path=fp)
    s_blur = _edge_sharpness(arr_blur)
    s_none = _edge_sharpness(arr_none)
    print(f"  无模糊锐度={s_none:.1f}  施加 blur=1.2 后锐度={s_blur:.1f}")
    check("施加模糊后新文字明显变糊", s_blur < s_none * 0.8,
          f"{s_blur:.1f} < {s_none*0.8:.1f}")
    # 关键回归：自动模式下 sharpness 高也应让 blur 生效
    check("自动模式(sharpness=0.7)下 blur 仍生效", s_blur < s_none * 0.9)

    print("\n== 4. 全自动参数映射（blur_level/noise_level → 渲染参数，精简面板后无滑块）==")
    import tkinter as tk
    root = tk.Tk()
    from app import gui
    app = gui.App(root)
    app.features = {
        "color_rgb": (10, 10, 10), "font_size": 40, "letter_spacing": 0,
        "line_spacing": 0, "skew_angle": 0.0, "sharpness": 0.7,
        "stroke_weight": 0.5,
        "stroke_px": 4.0, "stroke_px_skeleton": 4.0,
        "blur_level": 0.6, "antialias": 0.5, "noise_level": 0.4,
    }
    # 自动字号套用到唯一保留的「字号」滑杆
    app._auto_apply_features()
    fs_slider = float(app.vars["font_size"].get())
    check("自动字号已套用到字号滑杆",
          abs(fs_slider - 40) < 1e-6, f"slider={fs_slider}")
    # 其余自动参数无滑块，由 _collect_params 直接从 features 读入渲染参数
    params = app._collect_params()
    root.destroy()
    blur_p = float(params.get("blur", 0.0))
    noise_p = float(params.get("noise", 0.0))
    print(f"  blur_level=0.6 → 渲染参数 blur={blur_p}（期望=0.6）；"
          f"noise_level=0.4 → 渲染参数 noise={noise_p}（期望=0.4）")
    check("blur_level 自动进入渲染参数（无滑块直通）",
          abs(blur_p - 0.6) < 1e-6)
    check("noise_level 自动进入渲染参数（无滑块直通）",
          abs(noise_p - 0.4) < 1e-6)
    check("模糊自动生效（渲染参数>0）", blur_p > 0.0)
    check("噪点自动生效（渲染参数>0）", noise_p > 0.0)

    print(f"\n骨架/模糊自测：通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：" + "；".join(FAIL))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    print("骨架细化笔画厚度 + 自动模糊质感融合 自测通过 ✓")
