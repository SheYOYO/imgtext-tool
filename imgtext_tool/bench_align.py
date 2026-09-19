"""细节对齐调试工作台（分层归因）。

用途：把「看着不对」翻译成可测量的数字，并区分两类完全不同的故障——
  · 特征层故障：原图测出来的字号/字距/笔画/基线/倾斜 本身就是错的（测错了）
  · 渲染层故障：特征读数是对的，但渲染出来的文字没达到目标值（渲错了）

两种模式：
  1) 合成基准（有真值）：用已知字体/字号/字距画一张图，看测量层读数偏离真值多少。
     python bench_align.py --synth
  2) 真实图（无真值）：读原图选区特征 → 渲染新文字 → 回量渲染结果，
     看渲染层是否忠实达成目标值。
     python bench_align.py --image a.png --box 40,70,200,115 --text "替换文字"

输出：分层读数 + 偏差对比表 + 并排对比图（_bench_out/）。
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from app import config, feature_extract as fe, font_match as fm, render as rd  # noqa: E402
from app import io_utils  # noqa: E402  （中文路径必须用 imencode+tofile 落盘）

OUT_DIR = os.path.join(_HERE, "_bench_out")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ------------------------------------------------------------------ 基础工具
def rgba_to_bgr_on_white(rgba: np.ndarray) -> np.ndarray:
    """把渲染出来的 RGBA（透明底）压到纯白底，便于用同一套测量口径回量。"""
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32) * a + 255.0 * (1.0 - a)
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)


def crop_to_ink(bgr: np.ndarray) -> np.ndarray:
    """裁到墨迹包围盒，去透明 padding，保证测量口径与原图一致。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(bw > 0)
    if ys.size == 0:
        return bgr
    return bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def measure_side(bgr: np.ndarray) -> dict:
    """对一侧（原图 或 渲染结果）跑同一套测量，口径完全一致。"""
    bgr = crop_to_ink(bgr)
    em, reliable = fe.estimate_em_size(bgr)
    return {
        "em": int(em),
        "em可靠": "是" if reliable else "否",
        "墨迹高": int(fe.estimate_ink_height(bgr)),
        "字间距": int(fe.estimate_letter_spacing(bgr)),
        "笔画(距离变换)": round(float(fe.measure_stroke_px(bgr)), 2),
        "笔画(骨架)": round(float(fe.measure_stroke_px_skeleton(bgr)), 2),
        "基线y": fe.estimate_baseline_y(bgr, em),
        "倾斜": round(float(fe.estimate_skew_angle(bgr)), 2),
        "颜色RGB": tuple(int(c) for c in fe.extract_text_color(bgr)),
    }


# ------------------------------------------------------------------ 合成基准
def make_synth(text, font_path, em, letter_spacing, color=(20, 20, 20),
               bg=240, pad=30) -> tuple:
    """按已知真值画一张图：字号=em、字距=letter_spacing、颜色已知。"""
    try:
        font = ImageFont.truetype(font_path, int(em)) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # 逐字推进，字距就是「相邻字墨迹之间的空白像素」
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    widths, total = [], 0
    for ch in text:
        w = tmp.textbbox((0, 0), ch, font=font)[2] - tmp.textbbox((0, 0), ch, font=font)[0]
        widths.append(w)
        total += w
    total += max(0, len(text) - 1) * int(letter_spacing)

    W, H = total + pad * 2, int(em) * 2 + pad * 2
    img = Image.new("RGB", (W, H), (bg, bg, bg))
    d = ImageDraw.Draw(img)
    x, y = pad, pad
    for ch, w in zip(text, widths):
        d.text((x, y), ch, font=font, fill=color)
        x += w + int(letter_spacing)

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return bgr, (pad, pad, total, int(em) * 2)


# ------------------------------------------------------------------ 渲染路径
def render_like_app(features: dict, text: str, font_path: str, box_h: int):
    """用与 App 一致的口径渲染：em 字号直接给渲染层，不做二次缩放。"""
    params = {
        "color_rgb": features.get("color_rgb", (0, 0, 0)),
        "font_size": int(features.get("font_size", 16)),
        "letter_spacing": int(features.get("letter_spacing", 0)),
        "line_spacing": int(features.get("line_spacing", 0)),
        "skew_angle": float(features.get("skew_angle", 0.0)),
        "sharpness": float(features.get("sharpness", 0.7)),
        "blur": float(features.get("blur_level", 0.0)),
        "bold": False,
        "noise": float(features.get("noise_level", 0.0)),
        "opacity": 1.0,
        "target_stroke_px": float(features.get("stroke_px_skeleton", 0.0)),
    }
    rgba, metrics = rd.render_text_with_metrics(text, params, font_path)
    return rgba, metrics


# ------------------------------------------------------------------ 报告
def print_table(title, rows):
    print("\n" + title)
    print("-" * 68)
    width = max(len(r[0]) for r in rows) + 2
    for r in rows:
        name = r[0].ljust(width)
        if len(r) == 3:
            a, b, c = r[1], r[2], ""
        else:
            a, b, c = r[1], r[2], r[3]
        print(f"  {name} {str(a):>14} {str(b):>14} {str(c):>16}")
    print("-" * 68)


def save_compare(orig_bgr, rend_bgr, tag):
    os.makedirs(OUT_DIR, exist_ok=True)
    o = crop_to_ink(orig_bgr)
    r = crop_to_ink(rend_bgr)
    h = max(o.shape[0], r.shape[0])
    def _pad(im):
        if im.shape[0] >= h:
            return im
        canvas = np.full((h, im.shape[1], 3), 255, dtype=np.uint8)
        canvas[:im.shape[0], :] = im
        return canvas
    both = np.hstack([_pad(o), np.full((h, 12, 3), 200, dtype=np.uint8), _pad(r)])
    p = os.path.join(OUT_DIR, f"compare_{tag}.png")
    if not io_utils.imwrite_unicode(p, both):
        return f"{p}  (保存失败：路径含非 ASCII 且写入被拒)"
    return p


# ------------------------------------------------------------------ 模式一
def run_synth():
    pool = config.resolve_font_pool()
    cn = [f for f in pool if f.get("cn")]
    if not cn:
        print("字体池为空，无法跑合成基准")
        return
    f0 = cn[0]
    print(f"合成基准字体：{f0.get('name')}  {f0.get('path')}")

    cases = [
        ("测试文字", 48, 0),
        ("测试文字", 48, 8),
        ("测试文字", 32, 4),
        ("验收报告", 64, 6),
        ("2026年第3期", 40, 3),
    ]
    print("\n【模式一】合成基准：真值 vs 特征层读数（检出「测错了」）")
    print_table(
        "  用例                真值em  读数em   真值字距 读数字距",
        [["(表头)", "真值em", "读数em", "真值字距/读数字距"]],
    )
    for text, em, ls in cases:
        bgr, _ = make_synth(text, f0.get("path"), em, ls)
        m = measure_side(bgr)
        flag = "" if abs(m["em"] - em) <= 1 else "  <-- em 偏差"
        flag2 = "" if abs(m["字间距"] - ls) <= 1 else "  <-- 字距偏差"
        print(f"  {text:<14} em {em:>3} -> {m['em']:>3}{flag:<16} "
              f"字距 {ls:>2} -> {m['字间距']:>2}{flag2}")

    # 渲染忠实度：合成图原图侧 vs 渲染侧
    print("\n【模式一·续】渲染忠实度：原图读数 vs 渲染实测（检出「渲错了」）")
    for text, em, ls in cases:
        bgr, _ = make_synth(text, f0.get("path"), em, ls)
        feats = fe.extract_features(bgr)
        rgba, metrics = render_like_app(feats, text, f0.get("path"), bgr.shape[0])
        rend = rgba_to_bgr_on_white(rgba)
        a = measure_side(bgr)
        b = measure_side(rend)
        print(f"\n  用例 「{text}」 em={em} 字距={ls}")
        for k in ("em", "墨迹高", "字间距", "笔画(骨架)", "倾斜"):
            d = round(float(b[k]) - float(a[k]), 2) if b[k] is not None and a[k] is not None else "-"
            print(f"     {k:<12} 原图 {str(a[k]):>8}   渲染 {str(b[k]):>8}   偏差 {d:>7}")
        p = save_compare(bgr, rend, f"synth_{em}_{ls}")
        print(f"     对比图：{p}")


# ------------------------------------------------------------------ 模式二
def run_real(image_path, box, text):
    img = io_utils.imread_unicode(image_path)
    if img is None:
        print("读图失败：", image_path)
        return
    x, y, w, h = box
    x, y = max(0, x), max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    region = img[y:y + h, x:x + w]

    src = measure_side(region)
    feats = fe.extract_features(region)

    cands = fm.match_fonts(text, bgr_region=region,
                           target_stroke=feats.get("stroke_weight", 0.5),
                           target_size=int(feats.get("font_size", 16)),
                           target_stroke_px=float(feats.get("stroke_px_skeleton", 0.0)),
                           top_k=3, reference_text="")
    if not cands:
        print("字体池为空")
        return
    fp = cands[0].get("path")
    fname = cands[0].get("name")

    print(f"\n【模式二】真实图：{os.path.basename(image_path)}  框 ({x},{y},{w},{h})")
    print(f"  替换文字：「{text}」   选中字体：{fname}")
    print(f"  字体候选：{[c.get('name') for c in cands]}")

    rgba, metrics = render_like_app(feats, text, fp, h)
    rend = rgba_to_bgr_on_white(rgba)
    dst = measure_side(rend)

    print_table(
        "  项目            原图实测        渲染实测        偏差",
        [[k, src[k], dst[k],
          (round(float(dst[k]) - float(src[k]), 2)
           if isinstance(src[k], (int, float)) and isinstance(dst[k], (int, float)) else "-")]
         for k in src],
    )
    print(f"  渲染 metrics：baseline_y={metrics.get('baseline_y')} "
          f"em_size={metrics.get('em_size')}")
    print(f"  原图 text_bbox={feats.get('text_bbox')}  "
          f"baseline_y={feats.get('baseline_y')}  em_reliable={feats.get('em_reliable')}")

    p = save_compare(region, rend, "real")
    print(f"\n  并排对比图（左=原图选区 右=渲染结果）：{p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="真实图片路径")
    ap.add_argument("--box", help="选区 x,y,w,h")
    ap.add_argument("--text", default="替换文字", help="要渲染的替换文字")
    ap.add_argument("--synth", action="store_true", help="跑合成基准（有真值）")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.image and args.box:
        box = tuple(int(v) for v in args.box.split(","))
        run_real(args.image, box, args.text)
    else:
        run_synth()
    print("\n完成。工作台目录：", OUT_DIR)


if __name__ == "__main__":
    main()
