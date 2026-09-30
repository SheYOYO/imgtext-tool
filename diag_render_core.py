"""底层渲染缺陷诊断脚本（不修改业务代码，只量化证明缺陷）。

目标：用实测数据证明三个底层算法错误：
  A. 基线对齐 —— 当前用「墨迹包围盒中心」对齐，而非真实 baseline
  B. 笔画测量口径错位 —— 候选字体在「墨迹高度」下测量，却在更大的 em 字号下渲染
  C. 字体匹配样本不一致 —— 原图测「原文字」笔画，候选测「新文字」笔画
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import feature_extract as fe          # noqa: E402
from app import font_match as fm               # noqa: E402
from app import render as render_mod           # noqa: E402
from app import config                         # noqa: E402

P312 = sys.executable
LINE = "=" * 74


def _pick_font():
    pool = config.resolve_font_pool()
    cn = [f for f in pool if f.get("cn")]
    return (cn or pool)[0]["path"]


def _ink_bbox_alpha(arr):
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 40)
    if ys.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


# ---------------------------------------------------------------- A 基线
def diag_baseline(font_path):
    """证明：墨迹 bbox 中心对齐 ≠ 基线对齐，且误差随字符剧烈变化。

    做法：在同一 em 字号下，渲染若干字符，测量每个字符墨迹 bbox 的
    「底边相对 baseline 的距离 / 中心相对 baseline 的距离」。
    若中心对齐正确，则所有字符的 (baseline - ink_center_y) 应当相同。
    """
    print(LINE)
    print("A. 基线对齐缺陷诊断")
    print(LINE)
    em = 64
    font = ImageFont.truetype(font_path, em)
    ascent, descent = font.getmetrics()

    probe_chars = ["国", "中", "文", "一", "。", "字", "丁", "人", "口"]
    rows = []
    for ch in probe_chars:
        img = Image.new("RGBA", (em * 3, em * 3), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        # anchor 默认 "la"：y 是 ascender 顶；baseline = y + ascent
        y_top = em
        d.text((em, y_top), ch, font=font, fill=(0, 0, 0, 255))
        baseline_y = y_top + ascent

        arr = np.array(img)
        ink = _ink_bbox_alpha(arr)
        if ink is None:
            continue
        ink_cy = (ink[1] + ink[3]) / 2.0
        ink_bottom = ink[3]
        # 中心对齐法隐含的 baseline 偏移：把 ink 中心对齐到目标中心后，
        # baseline 会落在 ink_cy + (baseline_y - ink_cy)
        off_center = baseline_y - ink_cy        # 基线相对墨迹中心的偏移
        off_bottom = baseline_y - ink_bottom    # 基线相对墨迹底边的偏移
        rows.append((ch, off_center, off_bottom, ink[3] - ink[1]))

    if not rows:
        print("  [SKIP] 无可用字形")
        return

    print(f"  em={em}  ascent={ascent} descent={descent}")
    print(f"  {'字符':<6}{'墨迹高':>8}{'基线-墨迹中心':>16}{'基线-墨迹底边':>16}")
    for ch, oc, ob, ih in rows:
        print(f"  {ch:<6}{ih:>8}{oc:>16.1f}{ob:>16.1f}")

    offs = [r[1] for r in rows]
    span = max(offs) - min(offs)
    print(f"\n  >> 中心对齐法下，各字符「基线相对墨迹中心」的偏移跨度 = {span:.1f} px")
    print(f"     （若中心对齐正确，该跨度应为 0；跨度越大，字符高低错位越严重）")
    bad = span > em * 0.10
    print(f"  >> 判定：{'❌ 缺陷成立（偏移随字符剧烈变化）' if bad else '✅ 可接受'}")
    return span


# ---------------------------------------------------------------- B 笔画口径
def diag_stroke_scale(font_path):
    """证明：候选字体测量与最终渲染不在同一字号，导致字重系统性偏差。"""
    print()
    print(LINE)
    print("B. 笔画测量口径错位诊断")
    print(LINE)
    text = "中国文字替换测试"
    print(f"  {'目标墨迹高':>12}{'候选测量size':>14}{'实际渲染em':>12}"
          f"{'测量笔画':>10}{'实际笔画':>10}{'偏差':>10}")
    bad = False
    for target_ink in (24, 32, 44, 56, 72):
        # 字体匹配阶段：render_size = target_size = 原图实测墨迹高
        measured = fm.measure_font_stroke_px(text, font_path, target_ink)
        measured_skel = fm.measure_font_stroke_px_skeleton(text, font_path, target_ink)

        # 渲染阶段：反推 em 字号
        font = render_mod._resolve_font_for_ink(font_path, target_ink, text)
        em_size = font.size

        # 实际渲染出来的真实笔画（在该 em 下渲染后实测）
        real = fm.measure_font_stroke_px(text, font_path, em_size)
        real_skel = fm.measure_font_stroke_px_skeleton(text, font_path, em_size)

        if measured <= 0 or real <= 0:
            continue
        err = real / measured
        err_s = (real_skel / measured_skel) if measured_skel > 0 else float("nan")
        flag = "❌" if abs(err - 1.0) > 0.08 else "  "
        if abs(err - 1.0) > 0.08:
            bad = True
        print(f"  {target_ink:>12}{target_ink:>14}{em_size:>12}"
              f"{measured:>10.2f}{real:>10.2f}{err:>9.2f}x {flag}")
        print(f"  {'':>38}骨架口径 {measured_skel:.2f} -> {real_skel:.2f}  ({err_s:.2f}x)")

    print(f"\n  >> 若偏差 ≈ 1.00x 则口径一致；实测远离 1.00x 说明")
    print(f"     「按 target_ink 挑选的字体」渲染出来的笔画与挑选依据不符。")
    print(f"  >> 判定：{'❌ 缺陷成立（测量与渲染字号不一致）' if bad else '✅ 可接受'}")
    return bad


# ---------------------------------------------------------------- C 样本不一致
def diag_sample_mismatch(font_path):
    """证明：同一个字体、同一字号，测「原文字」与测「新文字」笔画差异巨大。

    字体匹配时：原图厚度来自被替换的原文，候选厚度来自用户输入的新文字。
    两者字符不同 → 距离变换 95% 分位口径本身就不可比。
    """
    print()
    print(LINE)
    print("C. 字体匹配样本口径不一致诊断")
    print(LINE)
    size = 48
    samples = ["国", "一", "中国文字", "替换后的文字内容", "ABCabc123", "。", "丁"]
    print(f"  同一字体 {os.path.basename(font_path)} @ {size}px：")
    print(f"  {'样本':<26}{'距离变换口径':>14}{'骨架口径':>12}")
    vals, vals_s = [], []
    for s in samples:
        a = fm.measure_font_stroke_px(s, font_path, size)
        b = fm.measure_font_stroke_px_skeleton(s, font_path, size)
        vals.append(a)
        vals_s.append(b)
        print(f"  {s:<26}{a:>14.2f}{b:>12.2f}")
    if vals:
        print(f"\n  距离变换口径：min={min(vals):.2f} max={max(vals):.2f} "
              f"极差={max(vals) - min(vals):.2f}  相对极差={(max(vals) - min(vals)) / max(1e-6, min(vals)) * 100:.0f}%")
    if vals_s:
        print(f"  骨架口径    ：min={min(vals_s):.2f} max={max(vals_s):.2f} "
              f"极差={max(vals_s) - min(vals_s):.2f}  相对极差={(max(vals_s) - min(vals_s)) / max(1e-6, min(vals_s)) * 100:.0f}%")
    ratio = (max(vals) - min(vals)) / max(1e-6, min(vals)) if vals else 0
    bad = ratio > 0.5
    print(f"\n  >> 同一字体同一字号，仅因「测什么字」不同，厚度读数就相差 {ratio * 100:.0f}%。")
    print(f"  >> 判定：{'❌ 缺陷成立（样本不可比，字重匹配失真）' if bad else '✅ 可接受'}")
    return bad


# ---------------------------------------------------------------- D 端到端
def diag_end_to_end(font_path):
    """端到端：生成一张原图 → 走完整「测量→匹配→渲染」链路，看输出偏差。"""
    print()
    print(LINE)
    print("D. 端到端链路偏差（合成原图 → 测量 → 匹配 → 渲染）")
    print(LINE)

    # 用「加粗黑体」造一张原图，字号 56
    src_size = 56
    src_text = "原始标题文字"
    font_src = ImageFont.truetype(font_path, src_size)
    tmp = Image.new("RGB", (1, 1), (255, 255, 255))
    d0 = ImageDraw.Draw(tmp)
    bb = d0.textbbox((0, 0), src_text, font=font_src)
    img = Image.new("RGB", (bb[2] - bb[0] + 20, bb[3] - bb[1] + 20), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10 - bb[0], 10 - bb[1]), src_text, font=font_src, fill=(0, 0, 0))
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    fs = fe.extract_features(bgr)
    stroke_dt = fs["stroke_px"]
    stroke_sk = fs["stroke_px_skeleton"]
    print(f"  原图：字号(墨迹高)={fs['font_size']}  笔画(DT)={stroke_dt:.2f}  "
          f"笔画(骨架)={stroke_sk:.2f}")

    # 字体匹配：模拟新文字不同
    pool = config.resolve_font_pool()
    for new_text in ("原始标题文字", "替换后的内容", "一二三"):
        cands = fm.match_fonts(new_text, bgr_region=bgr,
                               target_stroke=fs["stroke_weight"],
                               target_size=fs["font_size"],
                               target_stroke_px=stroke_dt, top_k=1)
        if not cands:
            continue
        best = cands[0]
        # 实际渲染后测真实笔画与真实墨迹高
        font = render_mod._resolve_font_for_ink(best["path"], fs["font_size"], new_text)
        real_stroke = fm.measure_font_stroke_px(new_text, best["path"], font.size)
        real_stroke_sk = fm.measure_font_stroke_px_skeleton(new_text, best["path"], font.size)
        ink_h = fm.measure_font_stroke_px  # 仅占位
        # 真实墨迹高
        tb = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
            (0, 0), new_text.split("\n")[0], font=font)
        rendered_ink_h = tb[3] - tb[1]

        print(f"\n  新文字「{new_text}」→ 选中 {os.path.basename(best['path'])} "
              f"(评分 {best['similarity']})")
        print(f"    目标笔画 DT={stroke_dt:.2f}  实际渲染 DT={real_stroke:.2f}  "
              f"偏差 {real_stroke / max(1e-6, stroke_dt):.2f}x")
        print(f"    目标笔画 骨架={stroke_sk:.2f}  实际渲染 骨架={real_stroke_sk:.2f}  "
              f"偏差 {real_stroke_sk / max(1e-6, stroke_sk):.2f}x")
        print(f"    目标墨迹高={fs['font_size']}  实际渲染墨迹高={rendered_ink_h}  "
              f"偏差 {rendered_ink_h / fs['font_size']:.2f}x")
    print()


def main():
    font_path = _pick_font()
    print(f"诊断字体：{font_path}")
    print(f"Python：{P312}")
    print()
    a = diag_baseline(font_path)
    b = diag_stroke_scale(font_path)
    c = diag_sample_mismatch(font_path)
    diag_end_to_end(font_path)

    print(LINE)
    print("诊断结论汇总")
    print(LINE)
    print(f"  A 基线对齐      ：{'缺陷成立' if (a and a > 6) else '通过'}"
          f"（字符间基线偏移跨度 {a:.1f}px）" if a is not None else "  A 基线对齐：跳过")
    print(f"  B 笔画测量口径  ：{'缺陷成立' if b else '通过'}")
    print(f"  C 匹配样本口径  ：{'缺陷成立' if c else '通过'}")
    print(LINE)


if __name__ == "__main__":
    main()
