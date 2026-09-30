"""全局配置：字体池、限制说明、路径等。

本工具的核心限制（必须让用户知悉）：
1. 本工具只能在「内置常用免费字体库」中进行相似度匹配，不做全网字体识别。
2. 不能精准识别小众字体、付费商用字体、以及被拉伸/变形/艺术化处理的字体。
3. 如果原图字体不在内置字体库中，只能选择最接近的替代字体。
4. 复杂背景、低清晰度截图、强压缩图片，无法做到 100% 无痕。
5. 最终效果依赖用户预览和微调。
"""
import os
import sys
from typing import List, Dict, Tuple

# ---------- 路径 ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS_DIR = os.path.join(BASE_DIR, "fonts")

# 常用系统字体目录（Windows / Linux / macOS 兜底查找）
_SYSTEM_FONT_DIRS = [
    os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts"),
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"),
    "/System/Library/Fonts",
    "/Library/Fonts",
]

# ---------- 内置字体池 ----------
# 每个字体条目：名称、字体文件名（在 fonts/ 目录或系统目录中查找）、
#             是否中文字体、是否粗体、默认权重
# 思源黑体/思源宋体/微软雅黑 等因版权与体积原因通常不直接内嵌在发行包里，
# 这里做「系统已安装则自动发现 + fonts/ 目录可手动放入」的混合策略。
FONT_POOL: List[Dict] = [
    # 中文字体
    {"name": "思源黑体", "files": ["SourceHanSansSC-Regular.otf", "SourceHanSansSC-Regular.ttf",
                                   "NotoSansSC-Regular.otf", "NotoSansCJKsc-Regular.otf",
                                   "NotoSansCJK-Regular.ttc"], "cn": True, "weight": "regular"},
    {"name": "思源宋体", "files": ["SourceHanSerifSC-Regular.otf", "SourceHanSerifSC-Regular.ttf",
                                   "NotoSerifSC-Regular.otf", "NotoSerifCJKsc-Regular.otf"],
     "cn": True, "weight": "regular"},
    {"name": "微软雅黑", "files": ["msyh.ttc", "msyh.ttf", "Microsoft YaHei.ttf"],
     "cn": True, "weight": "regular"},
    {"name": "系统宋体", "files": ["simsun.ttc", "simsun.ttf", "SimSun.ttf"],
     "cn": True, "weight": "regular", "serif": True},
    {"name": "系统黑体", "files": ["simhei.ttf", "SimHei.ttf"], "cn": True, "weight": "bold"},
    {"name": "等线", "files": ["Deng.ttf", "Dengb.ttf", "DengXian.ttf"],
     "cn": True, "weight": "regular"},
    # 英文字体
    {"name": "Arial", "files": ["arial.ttf", "Arial.ttf"], "cn": False, "weight": "regular"},
    {"name": "Arial Bold", "files": ["arialbd.ttf", "Arial Bold.ttf"], "cn": False, "weight": "bold"},
    {"name": "Calibri", "files": ["calibri.ttf", "Calibri.ttf"], "cn": False, "weight": "regular"},
    {"name": "Calibri Bold", "files": ["calibrib.ttf", "Calibri Bold.ttf"], "cn": False, "weight": "bold"},
    {"name": "Verdana", "files": ["verdana.ttf", "Verdana.ttf"], "cn": False, "weight": "regular"},
    {"name": "Times New Roman", "files": ["times.ttf", "Times New Roman.ttf"], "cn": False,
     "weight": "regular", "serif": True},
    {"name": "Times New Roman Bold", "files": ["timesbd.ttf", "Times New Roman Bold.ttf"],
     "cn": False, "weight": "bold", "serif": True},
]


def find_font_file(candidate_files: List[str]) -> str | None:
    """在 fonts/ 目录与系统字体目录中查找字体文件，返回绝对路径或 None。"""
    # 1. 优先项目内置 fonts/ 目录
    for fname in candidate_files:
        p = os.path.join(FONTS_DIR, fname)
        if os.path.exists(p):
            return p
    # 2. 系统字体目录兜底
    for d in _SYSTEM_FONT_DIRS:
        if not os.path.isdir(d):
            continue
        for fname in candidate_files:
            p = os.path.join(d, fname)
            if os.path.exists(p):
                return p
            # 忽略大小写再试一次
            if os.path.isdir(d):
                for existing in os.listdir(d):
                    if existing.lower() == fname.lower():
                        return os.path.join(d, existing)
    return None


# ---------- 扫描 fonts/ 目录 ----------
_FONT_EXTENSIONS = (".ttf", ".otf", ".ttc")
# 扫描上限，避免 fonts/ 目录字体过多时卡顿
MAX_SCAN_FONTS = 80


def _guess_weight_from_name(fname: str) -> str:
    """根据文件名粗判字重（Pillow 无法直接读取字重元数据）。"""
    low = fname.lower()
    for kw in ("black", "heavy", "extrabold", "bold", "semibold", "demibold", "bd"):
        if kw in low:
            return "bold"
    return "regular"


def _font_has_cjk(path: str) -> bool:
    """探测字体是否包含中文字形。

    渲染「中」与私用区字符（通常落到 .notdef）对比：
    若中文字形缺失，两者渲染结果一致，则判定不支持中文。
    """
    try:
        from PIL import ImageFont
        font = ImageFont.truetype(path, 48)
        m_cjk = font.getmask("中")
        bbox = m_cjk.getbbox()
        if bbox is None or bbox[2] <= 0 or bbox[3] <= 0:
            return False
        try:
            m_nul = font.getmask("\ue000")
            if bytes(m_cjk) == bytes(m_nul):
                return False
        except Exception:
            pass
        return True
    except Exception:
        return False


def scan_fonts_dir() -> List[Dict]:
    """扫描 fonts/ 目录下的所有字体文件，全部纳入候选池。"""
    found: List[Dict] = []
    if not os.path.isdir(FONTS_DIR):
        return found
    for fname in sorted(os.listdir(FONTS_DIR)):
        if not fname.lower().endswith(_FONT_EXTENSIONS):
            continue
        p = os.path.join(FONTS_DIR, fname)
        if not os.path.isfile(p):
            continue
        found.append({
            "name": os.path.splitext(fname)[0],
            "files": [fname],
            "path": p,
            "cn": _font_has_cjk(p),
            "weight": _guess_weight_from_name(fname),
            "source": "fonts",
        })
        if len(found) >= MAX_SCAN_FONTS:
            break
    return found


def resolve_font_pool() -> List[Dict]:
    """返回实际可用的字体列表（含解析出的绝对路径）。

    顺序：优先 fonts/ 目录内字体（用户自行放入的优先），
    其次系统字体（FONT_POOL 中能在系统找到的）。
    """
    resolved: List[Dict] = []
    seen = set()

    def _add(item):
        path = item.get("path")
        if not path:
            return
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            return
        seen.add(key)
        resolved.append(item)

    for f in scan_fonts_dir():
        _add(f)
    for f in FONT_POOL:
        path = find_font_file(f["files"])
        if not path:
            continue
        item = dict(f)
        item["path"] = path
        item["source"] = "system"
        _add(item)
    return resolved


def get_default_font() -> Tuple[str, int] | None:
    """返回一个默认可用字体，优先中文字体。"""
    pool = resolve_font_pool()
    for f in pool:
        if f["cn"]:
            return f["path"], 0
    if pool:
        return pool[0]["path"], 0
    return None


# ---------- 限制说明（写入注释与使用说明） ----------
LIMITATIONS_TEXT = (
    "本工具限制说明：\n"
    "1. 仅在内置常用免费字体库中进行相似度匹配，不做全网字体识别。\n"
    "2. 无法精准识别小众字体、付费商用字体、被拉伸/变形/艺术化的字体。\n"
    "3. 原图字体若不在内置字体库中，只能选择最接近的替代字体。\n"
    "4. 复杂背景、低清晰度截图、强压缩图片，无法做到 100% 无痕。\n"
    "5. 最终效果依赖用户预览与手动微调。"
)
