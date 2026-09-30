"""核心逻辑自测脚本：验证特征提取、擦除、渲染、字体匹配能正确运行。

生成一张带文字的测试图，模拟「框选-提取-擦除-渲染-合成」完整链路。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from app import feature_extract as fe
from app import erase as erase_mod
from app import render as render_mod
from app import font_match as fm
from app import config
from app import io_utils

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_output")
os.makedirs(OUT, exist_ok=True)


def make_test_image():
    """生成一张白底黑字的测试图。"""
    W, H = 600, 200
    img = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(img)
    # 找一个可用字体
    font_path = None
    pool = config.resolve_font_pool()
    for f in pool:
        if f["cn"]:
            font_path = f["path"]
            break
    if not font_path and pool:
        font_path = pool[0]["path"]

    try:
        font = ImageFont.truetype(font_path, 48) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    text = "替换测试"
    d.text((60, 70), text, font=font, fill=(20, 20, 20))
    img.save(os.path.join(OUT, "source.png"))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), font_path


def test_feature_extract(bgr):
    # 模拟框选文字区域
    box = (40, 40, 320, 150)
    region = bgr[box[1]:box[3], box[0]:box[2]]
    feat = fe.extract_features(region)
    print("== 特征提取 ==")
    for k, v in feat.items():
        print(f"  {k}: {v}")
    assert feat["font_size"] > 10, "字号估算异常"
    assert all(0 <= c <= 255 for c in feat["color_rgb"]), "颜色异常"
    print("  [PASS] 特征提取\n")
    return feat


def test_erase(bgr):
    box = (40, 40, 320, 150)
    erased = erase_mod.erase_text(bgr, box, strength=0.5)
    region = erased[box[1]:box[3], box[0]:box[2]]
    # 擦除后文字区域应接近背景（变亮/变均匀）
    diff = float(np.abs(region.astype(int) - 245).mean())
    print("== 擦除 ==")
    print(f"  擦除后区域与背景平均差: {diff:.1f}（越小越接近背景）")
    assert diff < 40, "擦除效果不佳"
    print("  [PASS] 擦除\n")
    return erased


def test_render(font_path):
    params = {
        "color_rgb": (20, 20, 20),
        "font_size": 48,
        "letter_spacing": 0,
        "line_spacing": 0,
        "skew_angle": 0.0,
        "sharpness": 0.7,
        "blur": 0.0,
        "edge_soften": 0.0,
        "noise": 0.0,
        "bold": False,
        "opacity": 1.0,
    }
    rgba = render_mod.render_text("新文字", params, font_path=font_path)
    print("== 渲染 ==")
    print(f"  渲染文字块尺寸: {rgba.shape}")
    assert rgba.shape[2] == 4, "应输出 RGBA"
    assert rgba[:, :, 3].max() > 0, "文字应为非空"
    print("  [PASS] 渲染\n")
    return rgba


def test_font_match(bgr):
    box = (40, 40, 320, 150)
    region = bgr[box[1]:box[3], box[0]:box[2]]
    cands = fm.match_fonts("替换测试", region, target_stroke=0.5)
    print("== 字体匹配 ==")
    if not cands:
        print("  [WARN] 无可用字体")
        return
    for c in cands:
        print(f"  {c['name']}: 相似度 {c['similarity']}")
    assert 1 <= len(cands) <= 6, "候选数量异常"
    print("  [PASS] 字体匹配\n")


def test_full_pipeline(bgr, font_path):
    box = (40, 40, 320, 150)
    region = bgr[box[1]:box[3], box[0]:box[2]]
    feat = fe.extract_features(region)
    params = {
        "color_rgb": feat["color_rgb"],
        "font_size": feat["font_size"],
        "letter_spacing": feat["letter_spacing"],
        "line_spacing": 0,
        "skew_angle": feat["skew_angle"],
        "sharpness": feat["sharpness"],
        "blur": 0.0,
        "edge_soften": 0.0,
        "noise": 0.0,
        # 不再用「阈值强行加粗」，而是按原图实测笔画厚度对齐字重
        "target_stroke_px": feat.get("stroke_px", 0.0),
        "opacity": 1.0,
    }
    erased = erase_mod.erase_text(bgr, box, strength=0.5)
    rgba = render_mod.render_text("全新内容", params, font_path=font_path)
    x0, y0, x1, y1 = box
    th, tw = rgba.shape[:2]
    px = x0 + max(0, ((x1 - x0) - tw) // 2)
    py = y0 + max(0, ((y1 - y0) - th) // 2)
    result = render_mod.composite_text(erased, rgba, (px, py))
    ok = io_utils.imwrite_unicode(os.path.join(OUT, "result.png"), result)
    print("== 完整链路 ==")
    print(f"  结果已保存 result.png 成功={ok}")
    print("  [PASS] 完整链路\n")


if __name__ == "__main__":
    print("生成测试图...")
    bgr, font_path = make_test_image()
    feat = test_feature_extract(bgr)
    erased = test_erase(bgr)
    test_render(font_path)
    test_font_match(bgr)
    test_full_pipeline(bgr, font_path)
    print("全部自测通过 ✓")
