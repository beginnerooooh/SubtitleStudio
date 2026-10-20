"""speechbrain 在 Windows 上的递归崩溃修复。

问题
────
speechbrain 1.x 用 `LazyModule` 做延迟导入。它的 `ensure_module` 里有
一段防护：如果当前的调用栈来自 `inspect.py`，就主动抛 `AttributeError`
切断递归。判断写的是：

    importer_frame.filename.endswith("/inspect.py")

**这在 Windows 上永远为 False**——Windows 的路径分隔符是 `\\`，
`C:\\...\\inspect.py` 不以 `/inspect.py` 结尾。于是防护完全失效：

    LazyModule.__getattr__
      → ensure_module
        → inspect.getframeinfo
          → inspect.getmodule
            → 访问对象的 __module__ 属性
              → LazyModule.__getattr__  ← 递归回到起点

结果就是 `RecursionError: maximum recursion depth exceeded`。
触发点通常不是直接调用，而是「先做了什么、再做声纹」的特定顺序
（实测：载入弹幕后走声纹过滤时必崩）。

修法
────
不改第三方库文件（升级会被覆盖），而是在导入 speechbrain 之前给
`LazyModule.ensure_module` 打一个运行时补丁：把判断改成同时接受
`/inspect.py` 与 `\\inspect.py`（用 `os.path.normpath` 统一）。

这是**幂等**的：重复调用只打一次补丁。
"""
from __future__ import annotations

import os
import sys

_PATCHED = False


def apply_patch() -> bool:
    """给 speechbrain 的 LazyModule 打补丁；返回是否成功打过/已打过。

    失败时返回 False（例如未安装 speechbrain），调用方可以忽略——
    没有 speechbrain 时不会有这个递归问题。
    """
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from speechbrain.utils import importutils as iu
    except Exception:
        return False

    lazy_cls = getattr(iu, "LazyModule", None)
    if lazy_cls is None or not hasattr(lazy_cls, "ensure_module"):
        return False

    original = lazy_cls.ensure_module

    def ensure_module(self, stacklevel: int = 1):
        # 复刻原逻辑，只把「是否来自 inspect.py」的判断换成跨平台写法。
        # 原始实现见 speechbrain/utils/importutils.py::LazyModule。
        import inspect
        import importlib
        import warnings

        importer_frame = None
        try:
            importer_frame = inspect.getframeinfo(
                sys._getframe(stacklevel + 1))
        except AttributeError:
            warnings.warn(
                "Failed to inspect frame to check if we should ignore "
                "importing a module lazily.")
        except RecursionError:
            # 已经在递归里了：直接当作「来自 inspect」，切断递归
            raise AttributeError()

        if importer_frame is not None:
            filename = importer_frame.filename or ""
            # 关键修复：同时匹配 / 与 \ 两种分隔符
            if os.path.basename(filename) == "inspect.py":
                raise AttributeError()

        if self.lazy_module is None:
            try:
                if self.package is None:
                    self.lazy_module = importlib.import_module(self.target)
                else:
                    self.lazy_module = importlib.import_module(
                        f".{self.target}", self.package)
            except Exception as e:
                raise ImportError(f"Lazy import of {repr(self)} failed") from e
        return self.lazy_module

    lazy_cls.ensure_module = ensure_module
    _PATCHED = True
    return True


def patched() -> bool:
    """补丁是否已生效（供日志/诊断显示）。"""
    return _PATCHED


__all__ = ["apply_patch", "patched"]
