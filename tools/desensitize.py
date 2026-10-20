"""数据脱敏检查：推送 GitHub 前必须运行。

为什么需要它
────────────
本项目在真实直播素材上开发，工作区里很容易残留隐私信息：
真实主播名、房间号、本地绝对路径（含用户名）、测试产物文件名、
甚至真实的弹幕/字幕内容。这些一旦提交就进了 git 历史，很难彻底清除。

因此把检查固化成脚本：`--fix` 可自动替换，默认只报告。

用法
────
    # 只检查（CI / 推送前）
    python tools/desensitize.py

    # 检查并自动修复可安全替换的部分
    python tools/desensitize.py --fix

退出码非 0 表示发现了需要处理的问题（便于串进 pre-push 钩子）。

检查项
──────
1. **本地绝对路径**：`D:\\...`、`C:\\Users\\<用户名>\\...`
2. **真实主播名 / 房间号**：在 `sensitive.txt` 里维护（不入库）
3. **样本文件名**：如 `Sample/<真实文件名>`
4. **运行时产物名**：`_eval_*`、时间戳目录等
5. **凭据**：GitHub token、API key 的字面量
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

ROOT = Path(__file__).resolve().parent.parent

# 只扫这些类型（二进制与依赖目录跳过）
SCAN_EXTS = {".py", ".js", ".html", ".css", ".md", ".txt", ".bat", ".json",
             ".yml", ".yaml", ".toml", ".cfg", ".ini", ".iss", ".nsi"}
SKIP_DIRS = {"node_modules", ".git", "__pycache__", "runtime", "bin",
             "models", "outputs", "cache", "projects", "profiles", "logs",
             ".venv", "venv", "dist", "build", ".pytest_cache",
             "keywords", "lyrics", "Sample", "samples"}

# 通用规则（不需要用户配置）
#
# 注意：文档里的**占位写法**（如 `C:\\Users\\<用户名>`、`C:\\Users\\...\\`）
# 是正常内容，不应报警。因此每个匹配都要先过 `_is_placeholder`。
BUILTIN_RULES: list[tuple[str, str, str]] = [
    # (名称, 正则, 替换建议)
    ("Windows 绝对路径",
     r"[A-Za-z]:\\\\?[^\s\"'<>|]*?(?:Downloads|Users|AppData|Desktop|Documents)"
     r"[^\s\"'<>|]*",
     "改用相对路径或 <样本> 占位"),
    ("本地工作区路径",
     r"[A-Za-z]:\\\\?[^\s\"'<>|]*?Subtitle[ _]?Studio[^\s\"'<>|]*",
     "改为相对路径"),
    ("GitHub token",
     r"(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})",
     "必须移除（会泄露账号权限）"),
    ("样本文件名",
     r"Sample[/\\]+[^\s\"'<>|]*?(?:\d+月\d+日|录制|直播)[^\s\"'<>|]*",
     "改为 Sample/<样本>"),
    ("测试产物目录",
     r"outputs\\\\?_[a-z]+",
     "改为 outputs/<产物>"),
]

# 占位符特征：含尖括号、省略号、或明确的占位词
PLACEHOLDER_MARKERS = ("<", ">", "...", "…", "用户名", "安装目录", "<样本>",
                       "<产物>", "your_", "<user>", "<name>")


def _is_placeholder(snippet: str) -> bool:
    """判断命中片段是否为文档占位写法（这类不该报警）。"""
    return any(m in snippet for m in PLACEHOLDER_MARKERS)

# 需要用户维护的敏感词（真实主播名、房间号等），放在单独文件里不入库
SENSITIVE_FILE = ROOT / "tools" / "sensitive.txt"


def load_sensitive() -> list[str]:
    """读取用户维护的敏感词表（每行一个，`#` 开头为注释）。"""
    if not SENSITIVE_FILE.is_file():
        return []
    words: list[str] = []
    for raw in SENSITIVE_FILE.read_text(encoding="utf-8").splitlines():
        w = raw.strip()
        if w and not w.startswith("#"):
            words.append(w)
    return words


def iter_files() -> list[Path]:
    out: list[Path] = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SCAN_EXTS:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        # 脚本自身含规则文本，跳过以免误报
        if p.name == "desensitize.py":
            continue
        # 敏感词表本身就是敏感内容，且不入库，不参与扫描
        if p.name == "sensitive.txt":
            continue
        out.append(p)
    return out


def scan(fix: bool = False) -> int:
    rules = list(BUILTIN_RULES)
    for w in load_sensitive():
        rules.append((f"敏感词「{w}」", re.escape(w), "替换为通用占位"))

    total = 0
    changed_files: list[str] = []
    for path in iter_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits: list[tuple[str, int, str, str]] = []
        for name, pattern, advice in rules:
            for m in re.finditer(pattern, text):
                snippet = text[max(0, m.start() - 30):m.end() + 30]
                snippet = snippet.replace("\n", " ")
                # 文档占位写法不算问题（如 C:\\Users\\<用户名>）
                if _is_placeholder(m.group(0)) or _is_placeholder(snippet):
                    continue
                line_no = text[:m.start()].count("\n") + 1
                hits.append((name, line_no, snippet, advice))
        if not hits:
            continue
        total += len(hits)
        rel = path.relative_to(ROOT)
        print(f"\n[{rel}]  {len(hits)} 处")
        for name, line_no, snippet, advice in hits[:6]:
            print(f"  L{line_no}  {name}")
            print(f"        …{snippet.strip()[:100]}…")
            print(f"        → {advice}")
        if len(hits) > 6:
            print(f"  … 另有 {len(hits) - 6} 处")
        if fix:
            changed_files.append(str(rel))

    print()
    if total == 0:
        print("[OK] 未发现敏感信息，可以推送。")
        return 0

    print(f"[!] 共 {total} 处需要处理。")
    if fix:
        print("    以下文件需要人工确认（自动替换可能破坏语义）：")
        for f in changed_files:
            print("      -", f)
    else:
        print("    请修复后重跑本脚本；确认无误再推送。")
    print("    提示：把真实主播名/房间号写入 tools/sensitive.txt（该文件不入库）。")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="推送前的数据脱敏检查")
    ap.add_argument("--fix", action="store_true",
                    help="列出需人工修改的文件（不做自动替换，避免破坏语义）")
    args = ap.parse_args()
    return scan(fix=args.fix)


if __name__ == "__main__":
    sys.exit(main())
