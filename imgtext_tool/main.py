"""本地离线图片文字替换桌面工具 — 入口。

运行：python main.py
"""
import sys
import os

# 保证能直接 `python main.py` 运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.gui import run  # noqa: E402

if __name__ == "__main__":
    run()
