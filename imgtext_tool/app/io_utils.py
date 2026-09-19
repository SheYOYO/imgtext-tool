"""图片读写工具：兼容含中文/非 ASCII 的路径。

OpenCV 的 imread/imwrite 在 Windows 上无法处理非 ASCII 路径
（例如含中文、全角括号的用户名目录），这里用 numpy 读取 + imdecode/imencode 绕过。
"""
import numpy as np
import cv2


def imread_unicode(path: str) -> np.ndarray | None:
    """读取图片（兼容中文路径），返回 BGR 或 None。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def imwrite_unicode(path: str, bgr: np.ndarray) -> bool:
    """保存图片（兼容中文路径），返回是否成功。"""
    try:
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else "png"
        if ext not in ("png", "jpg", "jpeg", "bmp", "webp"):
            ext = "png"
        # 扩展名决定编码格式
        ext_map = {
            "png": ".png", "jpg": ".jpg", "jpeg": ".jpg",
            "bmp": ".bmp", "webp": ".webp",
        }
        ok, buf = cv2.imencode(ext_map[ext], bgr)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False
