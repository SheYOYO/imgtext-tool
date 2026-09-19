"""字号精准识别 + 渲染墨迹对齐专项自测。

验证第六轮的两项核心修复：
1. estimate_font_size 基于「字符级墨迹高度」精准估算，不再粗略。
2. render_text 接收墨迹像素高度后，通过反推 em 字号，使渲染出的文字
   墨迹高度 ≈ 原图测量值（解决「生成文字大小和原图对不上」）。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from app import config
from app import feature_extract as fe
from app import render as render_mod


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_output")
os.makedirs(OUT, exist_ok=True)


def _cn_fonts():
    pool = config.resolve_font_pool()
    return [f for f in pool if f.get("cn")]


def _render_text_layer(text, font_path, size, color=(0, 0, 0, 255)):
    font = ImageFont.truetype(font_path, size)
    tmp = Image.new("RGBA", (1, 1), 0)
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + 20
    h = bbox[3] - bbox[1] + 20
    img = Image.new("RGBA", (max(1, w), max(1, h)), 0)
    d = ImageDraw.Draw(img)
    d.text((-bbox[0] + 10, -bbox[1] + 10), text, font=font, fill=color)
    return np.array(img)


def _to_bgr(rgba, bg=245):
    a = rgba[:, :, 3] / 255.0
    bg3 = np.full(rgba.shape[:2] + (3,), bg, np.uint8).astype(np.float32)
    rgb = rgba[:, :, :3].astype(np.float32)
    out = np.empty_like(bg3)
    for c in range(3):
        out[:, :, c] = rgb[:, :, c] * a + bg3[:, :, c] * (1 - a)
    return out.astype(np.uint8)


def _ink_height(rgba):
    """整块文字的墨迹竖直跨度（像素）。"""
    a = rgba[:, :, 3]
    mask = a > 40
    if not mask.any():
        return 0.0
    rows = np.where(mask.any(axis=1))[0]
    return float(rows.max() - rows.min() + 1)


def test_estimate_font_size_precision():
    """estimate_font_size 应精准逼近原图的真实墨迹像素高度。"""
    fonts = _cn_fonts()
    if not fonts:
        print("  [SKIP] 无可用中文字体，跳过字号精准度测试")
        return
    fp = fonts[0]["path"]

    # 用真实 size=48 渲染原图；原图真实墨迹高度即为参考值
    orig = _render_text_layer("国测字号", fp, 48)
    bgr = _to_bgr(orig)
    ys, xs = np.where(orig[:, :, 3] > 127)
    region = bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 原图真实墨迹高度（用生成时同款字体测，作为基准真值）
    ref_font = ImageFont.truetype(fp, 48)
    sample = "国" if any('\u4e00' <= c <= '\u9fff' for c in "国测字号") else "M"
    rb = ref_font.getbbox(sample)
    true_ink = rb[3] - rb[1]

    feat = fe.extract_features(region)
    est = feat["font_size"]
    print(f"  原图真实墨迹高度≈{true_ink:.1f}px｜estimate_font_size={est}")
    # 估算值应直接等于「真实墨迹像素高度」（渲染侧会据其反推 em 字号），
    # 与 true_ink 误差应很小（去掉了原先重复的 0.96 压缩，避免生成文字偏小）
    assert abs(est - true_ink) <= 4, \
        f"字号估算偏差过大：估算 {est} vs 期望≈{true_ink:.1f}"
    print("  [PASS] 字号估算精准（字符级墨迹高度，已对齐真实像素尺寸）")


def test_render_ink_alignment():
    """把估算的字号（墨迹高度）传给 render_text，渲染出的文字大小应≈原图。"""
    fonts = _cn_fonts()
    if not fonts:
        print("  [SKIP] 无可用中文字体，跳过墨迹对齐测试")
        return
    fp = fonts[0]["path"]

    # 原图
    orig = _render_text_layer("国测字号", fp, 48)
    bgr = _to_bgr(orig)
    ys, xs = np.where(orig[:, :, 3] > 127)
    region = bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    feat = fe.extract_features(region)
    target = feat["font_size"]   # 墨迹像素高度（目标）

    # 以原图同款字体渲染，验证墨迹对齐
    params = {
        "color_rgb": (20, 20, 20), "font_size": target,
        "letter_spacing": 0, "line_spacing": 0, "skew_angle": 0.0,
        "sharpness": 0.7, "blur": 0.0, "edge_soften": 0.0, "noise": 0.0,
        "bold": False, "target_stroke_px": feat.get("stroke_px", 0.0), "opacity": 1.0,
    }
    new = render_mod.render_text("国测字号", params, font_path=fp)
    new_ink = _ink_height(new)
    print(f"  目标墨迹={target}px｜渲染后墨迹={new_ink:.1f}px")
    assert abs(new_ink - target) <= 3, \
        f"渲染墨迹未对齐原图：渲染 {new_ink:.1f} ≠ 目标 {target}"
    print("  [PASS] 渲染墨迹高度对齐原图（不偏大、不偏小）")


if __name__ == "__main__":
    test_estimate_font_size_precision()
    test_render_ink_alignment()
    print("\n字号精准识别 + 渲染墨迹对齐自测通过 ✓")
