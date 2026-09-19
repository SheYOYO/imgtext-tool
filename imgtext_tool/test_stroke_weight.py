"""字重对齐专项自测：验证「渲染文字不强行加粗、也不强行加细，按原图实测笔画
厚度控制字重」。

核心断言：
1. 原图笔画厚度可被准确测量（stroke_px > 0）。
2. 把原图厚度传给 match_fonts 后，选出的字体其「天然」笔画厚度最接近原图
   ——即字重由「字体选择」决定，而不是事后描边。
3. 用该字体渲染（bold=False）时，渲染出的笔画厚度与字体的天然厚度一致，
   证明渲染阶段没有施加任何描边（既不强行加粗、也不强行加细）。
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
from app import font_match as fm


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_output")
os.makedirs(OUT, exist_ok=True)


def _cn_fonts():
    pool = config.resolve_font_pool()
    return [f for f in pool if f.get("cn")]


def _render_text_layer(text, font_path, size, color=(0, 0, 0, 255), stroke=0):
    font = ImageFont.truetype(font_path, size)
    tmp = Image.new("RGBA", (1, 1), 0)
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + 20
    h = bbox[3] - bbox[1] + 20
    img = Image.new("RGBA", (max(1, w), max(1, h)), 0)
    d = ImageDraw.Draw(img)
    d.text((-bbox[0] + 10, -bbox[1] + 10), text, font=font, fill=color,
            stroke_width=stroke, stroke_fill=color)
    return np.array(img)


def _to_bgr(rgba: np.ndarray, bg=245) -> np.ndarray:
    a = rgba[:, :, 3] / 255.0
    bg3 = np.full(rgba.shape[:2] + (3,), bg, np.uint8).astype(np.float32)
    rgb = rgba[:, :, :3].astype(np.float32)
    out = np.empty_like(bg3)
    for c in range(3):
        out[:, :, c] = rgb[:, :, c] * a + bg3[:, :, c] * (1 - a)
    return out.astype(np.uint8)


def _measure_rgba_stroke(rgba: np.ndarray) -> float:
    """测量渲染图的笔画厚度。

    【口径统一修复】旧实现用固定阈值 `a > 40` + 距离变换量 stroke，而
    `fm.measure_font_stroke_px` 用 **OTSU + 前景为白** 的 `_binarize_like_original`
    口径。渲染图有抗锯齿羽化（alpha 渐变），固定阈值会把边缘羽化区一并算入，
    导致 stroke 系统性虚高（实测 4.39 → 5.60）。现改为：把 RGBA 合成到白底 BGR
    后走 `fe.measure_stroke_px`（OTSU 口径），与原图、与字体天然厚度**同一把尺**，
    才能真正验证「渲染不施加描边」。
    """
    a = rgba[:, :, 3]
    if not (a > 40).any():
        return 0.0
    bgr = _to_bgr(rgba, bg=255)
    return fe.measure_stroke_px(bgr)


def _all_cn_natural_strokes(text, size):
    out = {}
    for f in _cn_fonts():
        s = fm.measure_font_stroke_px(text, f["path"], size)
        if s > 0:
            out[f["path"]] = s
    return out


def test_thin_original_matches_back_no_stroke_added():
    """细体原图：应精确选回生成它的字体，且渲染不施加任何描边。"""
    fonts = _cn_fonts()
    if not fonts:
        print("  [SKIP] 无可用中文字体，跳过得字重对齐测试")
        return
    fp = fonts[0]["path"]
    print(f"  使用基准字体：{os.path.basename(fp)}")

    # 合成一张细体原图（与基准字体同款、无描边）
    orig = _render_text_layer("细字测试", fp, 48)
    bgr = _to_bgr(orig)
    ys, xs = np.where(orig[:, :, 3] > 127)
    region = bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    feat = fe.extract_features(region)
    s_orig = feat["stroke_px"]
    print(f"  原图实测：stroke_px={s_orig:.2f}  font_size={feat['font_size']}")
    assert s_orig > 0, "原图笔画厚度应 > 0"

    # 全自动选字：传入实测笔画厚度
    # reference_text = 原图原文：候选与原图测量同一批字符，口径完全对齐
    # （不传时会退化为固定样本，引入约 8% 的样本偏差导致选错字体）
    cands = fm.match_fonts("替换细字", region,
                           target_stroke=feat["stroke_weight"],
                           target_size=feat["font_size"],
                           target_stroke_px=s_orig,
                           reference_text="细字测试")
    assert cands, "应返回候选字体"
    best = cands[0]
    print(f"  自动选字：{best['name']}（相似度 {best['similarity']:.3f}）")

    # 关键断言①：按字重+宽窄，应精确选回生成原图的字体（天然厚度完全一致）
    assert best["path"] == fp, \
        f"字重/宽窄匹配失效：应选回 {os.path.basename(fp)}，实际 {best['name']}"

    # 关键断言②：渲染不施加描边——渲染厚度 == 字体天然厚度，且贴合原图
    params = {
        "color_rgb": (20, 20, 20), "font_size": feat["font_size"],
        "letter_spacing": 0, "line_spacing": 0, "skew_angle": 0.0,
        "sharpness": 0.7, "blur": 0.0, "edge_soften": 0.0, "noise": 0.0,
        "bold": False, "target_stroke_px": s_orig, "opacity": 1.0,
    }
    new = render_mod.render_text("替换细字", params, font_path=best["path"])
    new_stroke = _measure_rgba_stroke(new)
    font_natural = fm.measure_font_stroke_px("替换细字", best["path"], feat["font_size"])
    print(f"  新文字实测 stroke_px={new_stroke:.2f}｜字体天然 stroke_px={font_natural:.2f}")

    # 没有人为描边：渲染厚度与字体天然厚度一致
    assert abs(new_stroke - font_natural) <= 1.0, \
        f"渲染施加了描边（疑似强行加粗/加细）：渲染 {new_stroke:.2f} ≠ 天然 {font_natural:.2f}"
    # 且贴合原图（选回同款字体，误差应在容差内）
    assert abs(new_stroke - s_orig) <= 1.0, \
        f"新文字与原图字重偏差过大：新 {new_stroke:.2f} vs 原 {s_orig:.2f}"
    print("  [PASS] 细体原图 → 选回同款字体，渲染零描边、字重精确对齐")


def test_bold_original_picks_closest_weight_no_overthicken():
    """较粗原图：应挑选天然字重最接近的字体（厚度在字体池可达范围内），
    渲染既不强行加粗、也不强行加细。

    说明：本工具禁止用描边把字「描粗」去强行对齐远超字体天然厚度的原图
    （那正是「强行加粗」）。因此本用例的原图厚度必须落在字体池可达范围内——
    做法是用「池内一款天然更粗的字体」生成原图，验证匹配按字重选回它。
    """
    fonts = _cn_fonts()
    if not fonts:
        print("  [SKIP] 无可用中文字体，跳过")
        return

    # 找一款天然笔画比首款更粗的字体（通常在 weight=='bold' 的系统黑体等）
    base = fonts[0]["path"]
    base_nat = fm.measure_font_stroke_px("替换粗字", base, 48)
    heavy = None
    for f in fonts:
        n = fm.measure_font_stroke_px("替换粗字", f["path"], 48)
        if n > base_nat + 0.3:
            heavy = f
            break
    # 若池内没有更粗的字体（如只有一款中文字体），退化为用 base 自身验证
    src_font = heavy["path"] if heavy else base
    src_name = heavy["name"] if heavy else fonts[0]["name"]
    print(f"  较粗基准字体：{src_name}（天然 stroke≈{fm.measure_font_stroke_px('替换粗字', src_font, 48):.2f}）")

    # 用该字体（无额外描边）生成原图，厚度落在池内可达范围
    orig = _render_text_layer("粗体字", src_font, 48)
    bgr = _to_bgr(orig)
    ys, xs = np.where(orig[:, :, 3] > 127)
    region = bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    feat = fe.extract_features(region)
    s_orig = feat["stroke_px"]
    print(f"  原图实测 stroke_px={s_orig:.2f}")

    cands = fm.match_fonts("替换粗字", region,
                           target_stroke=feat["stroke_weight"],
                           target_size=feat["font_size"],
                           target_stroke_px=s_orig)
    assert cands, "应返回候选字体"
    best = cands[0]
    print(f"  自动选字：{best['name']}（相似度 {best['similarity']:.3f}）")

    # 关键断言①：所选字体的天然字重应最接近原图（在候选池能力范围内）
    naturals = _all_cn_natural_strokes("替换粗字", feat["font_size"])
    if naturals:
        best_nat = naturals.get(best["path"], 0.0)
        min_diff = min(abs(v - s_orig) for v in naturals.values())
        my_diff = abs(best_nat - s_orig)
        print(f"  所选天然 {best_nat:.2f}｜池内最小偏差 {min_diff:.2f}")
        assert my_diff <= min_diff + 0.6, \
            f"字重未选最接近者：所选偏差 {my_diff:.2f} > 池内最小 {min_diff:.2f}"

        # 若池内存在更粗字体且原图就是它生成的，应被精确选回
        if heavy and os.path.normcase(os.path.abspath(best["path"])) \
                == os.path.normcase(os.path.abspath(src_font)):
            print("  [PASS] 较粗原图 → 精确选回同款更粗字体")
        else:
            print("  [PASS] 较粗原图 → 选回字重最接近的字体")
    else:
        print("  [WARN] 无法测量候选天然厚度，跳过最接近性断言")

    # 关键断言②：渲染不施加描边（既不强行加粗、也不强行加细）
    params = {
        "color_rgb": (20, 20, 20), "font_size": feat["font_size"],
        "letter_spacing": 0, "line_spacing": 0, "skew_angle": 0.0,
        "sharpness": 0.7, "blur": 0.0, "edge_soften": 0.0, "noise": 0.0,
        "bold": False, "target_stroke_px": s_orig, "opacity": 1.0,
    }
    new = render_mod.render_text("替换粗字", params, font_path=best["path"])
    new_stroke = _measure_rgba_stroke(new)
    font_natural = fm.measure_font_stroke_px("替换粗字", best["path"], feat["font_size"])
    print(f"  新文字实测 stroke_px={new_stroke:.2f}｜字体天然 {font_natural:.2f}")

    # 没有人为描边：渲染厚度 == 字体天然厚度
    assert abs(new_stroke - font_natural) <= 1.0, \
        f"渲染施加了描边（疑似强行加粗/加细）：渲染 {new_stroke:.2f} ≠ 天然 {font_natural:.2f}"
    # 厚度封顶：不超出原图（不在字体天然厚度之外再加粗）
    assert new_stroke <= s_orig + 1.0, \
        f"字重越界：新 {new_stroke:.2f} 超出 原 {s_orig:.2f}"
    print("  [PASS] 较粗原图 → 渲染零描边、未失控变粗")


if __name__ == "__main__":
    test_thin_original_matches_back_no_stroke_added()
    test_bold_original_picks_closest_weight_no_overthicken()
    print("\n字重对齐（不强行加粗/加细）自测通过 ✓")
