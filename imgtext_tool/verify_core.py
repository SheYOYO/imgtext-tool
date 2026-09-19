"""底层重构验证：em 字号推定 + 基线推定 的精度。

这是「底层算法正确性的第一道闸门」——只有这两项准了，
字体匹配、字重对齐、基线对齐才有意义。
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import config                 # noqa: E402
from app import feature_extract as fe  # noqa: E402

LINE = "=" * 78


def _render(text, font, pad=8):
    """渲染到白底 RGB，返回 BGR。"""
    tmp = Image.new("RGB", (1, 1), (255, 255, 255))
    d0 = ImageDraw.Draw(tmp)
    bb = d0.multiline_textbbox((0, 0), text, font=font)
    img = Image.new("RGB", (bb[2] - bb[0] + pad * 2, bb[3] - bb[1] + pad * 2),
                    (255, 255, 255))
    ImageDraw.Draw(img).multiline_text((pad - bb[0], pad - bb[1]), text,
                                       font=font, fill=(0, 0, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def verify_em(font_path):
    print(LINE)
    print("1. em 字号推定精度（标定后）")
    print(LINE)
    cases = [
        ("中国文字", 64, "常规汉字", True),
        ("中国文字替换测试内容", 64, "长串汉字", True),
        ("中 国 文 字", 64, "带空格", True),
        ("中国ABC文字abc", 64, "中英混排", True),
        ("中国文字\n第二行文字", 64, "多行", True),
        ("中国文字", 24, "小字号", True),
        ("中国文字", 96, "大字号", True),
        ("中国文字一二三", 64, "混合含扁平字", True),
        # 以下为「不可靠」场景：算法需正确标记 reliable=False 交由周边参照兜底
        ("一", 64, "★单字扁平", False),
        ("一一一一", 64, "★全扁平字符", False),
        ("。，、", 64, "★纯标点", False),
    ]
    print(f"  {'样本':<24}{'真实em':>8}{'估算em':>8}{'偏差':>9}{'可靠':>7}  说明")
    print("  " + "-" * 74)
    ok_em, ok_rel, total = 0, 0, 0
    for text, em_true, note, want_reliable in cases:
        font = ImageFont.truetype(font_path, em_true)
        bgr = _render(text, font)
        est, rel = fe.estimate_em_size(bgr)
        err = est / em_true
        total += 1
        if abs(err - 1.0) <= 0.15:
            ok_em += 1
        if rel == want_reliable:
            ok_rel += 1
        mark = "" if abs(err - 1.0) <= 0.15 else "  ❌"
        rel_mark = "✅" if rel == want_reliable else "⚠"
        print(f"  {text!r:<24}{em_true:>8}{est:>8}{err:>8.2f}x{str(rel):>7}{mark}"
              f" {rel_mark} {note}")
    print("  " + "-" * 74)
    print(f"\n  可靠场景 em 精度合格：{ok_em}/{total}（±15%）")
    print(f"  可靠性标记正确      ：{ok_rel}/{total}")
    return ok_em, ok_rel, total


def verify_baseline(font_path):
    print()
    print(LINE)
    print("2. 基线推定精度（与「墨迹中心对齐」旧法对比）")
    print(LINE)
    em = 64
    font = ImageFont.truetype(font_path, em)
    ascent, descent = font.getmetrics()

    # 构造：把若干字符画在同一基线上，检查推定基线是否稳定
    cases = ["国文字中", "一", "。", "中国", "一了丁", "ABCgpy", "中国gpy"]
    print(f"  em={em}  ascent={ascent}")
    print(f"  {'样本':<16}{'真实基线':>10}{'推定基线':>10}{'误差':>8}"
          f"{'旧法(中心)误差':>16}")
    print("  " + "-" * 62)
    errs_new, errs_old = [], []
    for text in cases:
        img = Image.new("RGB", (em * 12, em * 3), (255, 255, 255))
        d = ImageDraw.Draw(img)
        y_top = em
        d.text((10, y_top), text, font=font, fill=(0, 0, 0))
        true_baseline = y_top + ascent
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        est = fe.estimate_baseline_y(bgr, em)
        if est is None:
            print(f"  {text:<16}{true_baseline:>10}{'None':>10}")
            continue

        # 旧法：墨迹 bbox 中心 → 反推的基线（用中心到基线的距离，以「国」为参照）
        tb = fe.foreground_bbox(bgr)
        old_center = (tb[1] + tb[3]) / 2.0
        # 旧法把新文字的墨迹中心对齐到原文字墨迹中心，等价于认为
        # 「基线 = 墨迹中心 + 固定偏移」；这里用 em 字符的固定偏移做基准比较
        tmp = Image.new("RGB", (em * 3, em * 3), (255, 255, 255))
        ImageDraw.Draw(tmp).text((em, em), "国", font=font, fill=(0, 0, 0))
        ref_bgr = cv2.cvtColor(np.array(tmp), cv2.COLOR_RGB2BGR)
        ref_tb = fe.foreground_bbox(ref_bgr)
        ref_center = (ref_tb[1] + ref_tb[3]) / 2.0
        ref_baseline = em + ascent
        center_to_baseline = ref_baseline - ref_center  # 「国」的中心→基线距离
        old_est = old_center + center_to_baseline

        e_new = abs(est - true_baseline)
        e_old = abs(old_est - true_baseline)
        errs_new.append(e_new)
        errs_old.append(e_old)
        flag_new = "❌" if e_new > 3 else "✅"
        flag_old = "❌" if e_old > 3 else "  "
        print(f"  {text:<16}{true_baseline:>10}{est:>10}{e_new:>7.1f}px {flag_new}"
              f"{e_old:>14.1f}px {flag_old}")

    print("  " + "-" * 62)
    if errs_new:
        print(f"\n  新法（基线推定）平均误差：{np.mean(errs_new):.1f}px  "
              f"最大 {max(errs_new):.1f}px")
        print(f"  旧法（墨迹中心）平均误差：{np.mean(errs_old):.1f}px  "
              f"最大 {max(errs_old):.1f}px")
        improve = np.mean(errs_old) - np.mean(errs_new)
        print(f"  >> 新法较旧法平均改善 {improve:.1f}px")
        return np.mean(errs_new), np.mean(errs_old)
    return None, None


def verify_stroke_normalized():
    print()
    print(LINE)
    print("3. 归一化笔画厚度：跨字号不变量检验")
    print(LINE)
    print("  同一字体在不同字号下，归一化厚度应当基本一致（不变量）。")
    print(f"\n  {'字体':<22}" + "".join(f"{s:>9}px" for s in (24, 48, 72, 96))
          + f"{'极差':>10}")
    print("  " + "-" * 62)
    all_ok = True
    for f in [x for x in config.resolve_font_pool() if x.get("cn")]:
        vals = []
        row = f"  {os.path.basename(f['path']):<22}"
        for size in (24, 48, 72, 96):
            font = ImageFont.truetype(f["path"], size)
            bgr = _render("中国文字", font)
            v = fe.measure_stroke_normalized(bgr, size)
            vals.append(v)
            row += f"{v:>11.4f}"
        rng = max(vals) - min(vals)
        row += f"{rng:>10.4f}"
        ok = rng < 0.02
        if not ok:
            all_ok = False
        print(row + ("  ✅" if ok else "  ❌ 跨字号漂移过大"))
    print()
    print(f"  >> 判定：{'✅ 归一化厚度是不变量，可用于字重匹配' if all_ok else '❌ 仍随字号漂移'}")

    # 区分度：不同字体的归一化厚度应有明显差异
    print()
    print("  字重区分度（48px，归一化厚度）：")
    vals = {}
    for f in [x for x in config.resolve_font_pool() if x.get("cn")]:
        font = ImageFont.truetype(f["path"], 48)
        bgr = _render("中国文字", font)
        vals[os.path.basename(f["path"])] = fe.measure_stroke_normalized(bgr, 48)
    for k, v in sorted(vals.items(), key=lambda x: x[1]):
        print(f"    {k:<22} {v:.4f}")
    span = max(vals.values()) - min(vals.values())
    uniq = len(set(round(v, 4) for v in vals.values()))
    print(f"\n    跨度={span:.4f}  不同取值={uniq}/{len(vals)}")
    print(f"  >> 判定：{'✅ 能区分全部字体' if uniq == len(vals) else '⚠ 存在撞车'}")


def main():
    pool = config.resolve_font_pool()
    cn = [f for f in pool if f.get("cn")] or pool
    font_path = cn[0]["path"]
    print(f"验证字体：{os.path.basename(font_path)}")
    print()
    verify_em(font_path)
    verify_baseline(font_path)
    verify_stroke_normalized()
    print()
    print(LINE)
    print("底层验证结束")
    print(LINE)


if __name__ == "__main__":
    main()
