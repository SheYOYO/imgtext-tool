"""字体匹配 / 笔画口径 二次诊断（量化钉死真实病灶）。

聚焦四个可证伪的问题：
  1. 笔画口径分辨率：距离变换 vs 骨架，谁的读数能真正区分字重？
  2. 自匹配率：拿字体 X 渲染原图，匹配能否把 X 选回第一？（字体匹配的金标准）
  3. 字数变化对字形匹配（aspect/coverage）的破坏
  4. 字号估算在字符粘连 / 不同字符下的稳定性
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import feature_extract as fe     # noqa: E402
from app import font_match as fm          # noqa: E402
from app import render as render_mod      # noqa: E402
from app import config                    # noqa: E402

LINE = "=" * 78


def _render_gray(text, font_path, size, pad=6):
    """渲染文字到灰度图（白底黑字），返回 numpy 灰度。"""
    font = ImageFont.truetype(font_path, size)
    tmp = Image.new("RGB", (1, 1), (255, 255, 255))
    d0 = ImageDraw.Draw(tmp)
    bb = d0.textbbox((0, 0), text, font=font)
    img = Image.new("RGB", (bb[2] - bb[0] + pad * 2, bb[3] - bb[1] + pad * 2),
                    (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((pad - bb[0], pad - bb[1]), text, font=font, fill=(0, 0, 0))
    return np.array(img)


# ------------------------------------------------------- 1 口径分辨率
def diag_resolution(pool):
    print(LINE)
    print("1. 笔画口径分辨率：能否区分不同字重？")
    print(LINE)
    size = 48
    text = "中国文字"
    dts, sks = [], []
    rows = []
    for f in pool[:18]:
        dt = fm.measure_font_stroke_px(text, f["path"], size)
        sk = fm.measure_font_stroke_px_skeleton(text, f["path"], size)
        if dt > 0:
            dts.append(dt)
        if sk > 0:
            sks.append(sk)
        rows.append((os.path.basename(f["path"]), dt, sk))

    print(f"  {'字体':<26}{'距离变换':>12}{'骨架':>10}")
    for name, dt, sk in rows:
        print(f"  {name:<26}{dt:>12.2f}{sk:>10.2f}")

    def _uniq(v):
        return len(set(round(x, 3) for x in v))

    print(f"\n  {len(rows)} 款字体在 {size}px 下的读数：")
    print(f"    距离变换口径：不同取值 {_uniq(dts):>3} 个  "
          f"范围 [{min(dts):.2f}, {max(dts):.2f}]  跨度 {max(dts) - min(dts):.2f}px")
    print(f"    骨架口径    ：不同取值 {_uniq(sks):>3} 个  "
          f"范围 [{min(sks):.2f}, {max(sks):.2f}]  跨度 {max(sks) - min(sks):.2f}px")
    print(f"\n  >> 取值个数越少 = 分辨率越低 = 越无法区分字重。")
    verdict = _uniq(dts) < len(dts) * 0.6
    print(f"  >> 判定：{'❌ 距离变换口径分辨率不足（大量字体读数撞车）' if verdict else '✅ 距离变换分辨率尚可'}")
    return _uniq(dts), len(dts), _uniq(sks), len(sks)


# ------------------------------------------------------- 2 自匹配率
def diag_self_match(pool):
    """字体匹配的金标准：拿字体 X 渲染原图，匹配结果第一名应当是 X。"""
    print()
    print(LINE)
    print("2. 自匹配率（金标准）：用字体 X 渲染原图 → 匹配结果第一名是否为 X？")
    print(LINE)
    text = "中国文字替换测试"
    hit = 0
    total = 0
    miss = []
    for f in pool:
        font_path = f["path"]
        name = os.path.basename(font_path)
        try:
            gray = _render_gray(text, font_path, 56)
            bgr = cv2.cvtColor(gray, cv2.COLOR_RGB2BGR)
        except Exception:
            continue
        fs = fe.extract_features(bgr)
        # reference_text=原图原文，模拟 OCR 正确识别原文的常规场景
        cands = fm.match_fonts(text, bgr_region=bgr,
                               target_stroke=fs["stroke_weight"],
                               target_size=fs["font_size"],
                               target_stroke_px=fs["stroke_px"], top_k=3,
                               reference_text=text)
        if not cands:
            continue
        total += 1
        top = os.path.basename(cands[0]["path"])
        ok = (top == name)
        if ok:
            hit += 1
        else:
            miss.append((name, top, cands[0]["similarity"]))
        print(f"  {'✅' if ok else '❌'} 原字体 {name:<24} → 首选 {top:<24} "
              f"(评分 {cands[0]['similarity']:.3f})")

    rate = hit / total if total else 0
    print(f"\n  >> 自匹配命中率：{hit}/{total} = {rate * 100:.1f}%")
    print(f"  >> 判定：{'❌ 字体匹配逻辑失效（多数选不回原字体）' if rate < 0.6 else '✅ 可接受'}")
    return rate, miss


# ------------------------------------------------------- 3 字数变化
def diag_length_change(font_path):
    """字数变化时，字形匹配（aspect/coverage）是否被破坏。"""
    print()
    print(LINE)
    print("3. 字数变化对字形匹配的破坏")
    print(LINE)
    text = "中国文字"
    gray = _render_gray(text, font_path, 56)
    bgr = cv2.cvtColor(gray, cv2.COLOR_RGB2BGR)
    _, bw = cv2.threshold(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                          0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw > 127).mean() > 0.5:
        bw = 255 - bw
    tf = fm._glyph_features(bw)
    print(f"  原图「{text}」（4 字）：aspect={tf['aspect']:.3f} "
          f"coverage={tf['coverage']:.3f} stroke={tf['stroke']:.4f}")

    print(f"\n  {'候选新文字':<22}{'字数':>6}{'aspect':>10}{'coverage':>10}"
          f"{'Δaspect':>10}{'宽窄得分':>10}")
    bad = False
    for cand, n in (("中国文字", 4), ("中国文字替换", 6),
                    ("中国文字替换测试内容", 10), ("中", 1), ("一二三四五六七八", 8)):
        s = fm._render_sample(cand, font_path, size=56)
        if s is None or s.max() == 0:
            continue
        sf = fm._glyph_features(s)
        d_aspect = abs(sf["aspect"] - tf["aspect"])
        # 复刻 match_fonts 的宽窄得分公式
        w_aspect = 1.0 - min(1.0, d_aspect / 0.6)
        print(f"  {cand:<22}{n:>6}{sf['aspect']:>10.3f}{sf['coverage']:>10.3f}"
              f"{d_aspect:>10.3f}{w_aspect:>10.3f}")
        if w_aspect < 0.6:
            bad = True
    print(f"\n  >> 宽窄得分越接近 1.0 越好；字数一变，得分崩塌说明 aspect 指标依赖字数。")
    print(f"  >> 判定：{'❌ 缺陷成立（aspect 与字数强耦合，字数一变匹配失效）' if bad else '✅ 可接受'}")
    return bad


# ------------------------------------------------------- 4 字号估算
def diag_font_size(pool):
    """字号估算稳定性：不同字符 / 粘连情况下的偏差。"""
    print()
    print(LINE)
    print("4. 字号（墨迹高）估算稳定性")
    print(LINE)
    font_path = pool[0]["path"]
    font = ImageFont.truetype(font_path, 64)
    ascent, descent = font.getmetrics()
    # 「真实」参考：整串渲染的墨迹高
    print(f"  {'样本文字':<24}{'估算字号':>10}{'真实墨迹高':>12}{'偏差':>10}")
    bad = False
    for text in ("中国文字", "国", "一", "中国文字替换测试",
                 "ABCabc", "中国ABC文字", "中 国 文 字"):
        gray = _render_gray(text, font_path, 64)
        bgr = cv2.cvtColor(gray, cv2.COLOR_RGB2BGR)
        est = fe.estimate_font_size(bgr)
        # 真实：单字墨迹高（用整串 bbox 高度，单行即为行墨迹高）
        ys = np.where((cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) < 127).any(axis=1))[0]
        real = int(ys[-1] - ys[0] + 1) if ys.size else 0
        if real <= 0:
            continue
        err = est / real
        flag = "❌" if abs(err - 1.0) > 0.15 else "  "
        if abs(err - 1.0) > 0.15:
            bad = True
        print(f"  {text:<24}{est:>10}{real:>12}{err:>9.2f}x {flag}")
    print(f"\n  >> 判定：{'❌ 缺陷成立（字号估算随内容剧烈波动）' if bad else '✅ 可接受'}")
    return bad


def main():
    pool = config.resolve_font_pool()
    cn_pool = [f for f in pool if f.get("cn")] or pool
    print(f"字体池：共 {len(pool)} 款（中文 {len([f for f in pool if f.get('cn')])} 款）")
    print()
    diag_resolution(cn_pool)
    rate, miss = diag_self_match(cn_pool)
    diag_length_change(cn_pool[0]["path"])
    diag_font_size(cn_pool)

    print()
    print(LINE)
    print("二次诊断结论")
    print(LINE)
    print(f"  2 字体自匹配率：{rate * 100:.1f}%  "
          f"{'❌ 失效' if rate < 0.6 else '✅ 可接受'}")
    if miss:
        print(f"     选错的样本：{', '.join(m[0] + '→' + m[1] for m in miss[:6])}")
    print(LINE)


if __name__ == "__main__":
    main()
