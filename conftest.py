"""pytest 全局配置：把 packaging/ 与 models/ 纳入 import 搜索路径。"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _dir in ("packaging", "models"):
    _p = str(_ROOT / _dir)
    if _p not in sys.path:
        sys.path.insert(0, _p)
