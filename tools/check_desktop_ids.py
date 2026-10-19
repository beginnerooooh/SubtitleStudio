"""静态检查：renderer/app.js 里 $() 引用的 ID 必须都存在于 index.html。

重写布局后最容易出现的缺陷就是「引用了已删除/改名的元素」，
运行时会抛 null 异常并中断整个 bindEvents（界面看起来正常但交互全失效）。
这里做一次静态比对，把它挡在提交前。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

DESKTOP = Path(__file__).resolve().parent.parent / "desktop"
JS = DESKTOP / "renderer" / "app.js"
HTML = DESKTOP / "renderer" / "index.html"


def main() -> int:
    js = JS.read_text(encoding="utf-8")
    html = HTML.read_text(encoding="utf-8")

    html_ids = set(re.findall(r'\bid="([^"]+)"', html))
    # $('xxx') 形式的引用
    js_ids = set(re.findall(r"\$\('([^']+)'\)", js))
    # querySelector('.xxx') 形式的类名引用
    js_classes = set(re.findall(r"querySelector(?:All)?\('\.([\w-]+)'\)", js))
    html_classes = set()
    for attr in re.findall(r'\bclass="([^"]+)"', html):
        html_classes.update(attr.split())

    missing_ids = sorted(js_ids - html_ids)
    # 运行时动态生成的 class（由 JS innerHTML 创建），不算缺失
    dynamic_classes = {"draft-item", "act-open", "act-del", "line",
                       "row", "row-idx", "row-text", "row-time", "clip",
                       "item", "edge", "sep", "keys-row",
                       "vp-item", "act-use", "act-edit",
                       # 自绘确认框（askConfirm）动态创建的按钮
                       "act-yes", "act-no", "toast"}
    missing_classes = sorted(js_classes - html_classes - dynamic_classes)

    print(f"HTML 元素 ID: {len(html_ids)}")
    print(f"JS 引用 ID  : {len(js_ids)}")
    if missing_ids:
        print(f"\n[错误] JS 引用了 HTML 中不存在的 ID（{len(missing_ids)} 个）:")
        for i in missing_ids:
            print(f"   - {i}")
    else:
        print("\n[OK] 所有 $() 引用的 ID 都存在")

    if missing_classes:
        print(f"\n[警告] JS 引用了 HTML 中未定义的 class（{len(missing_classes)} 个）:")
        for c in missing_classes:
            print(f"   - {c}")
    else:
        print("[OK] 所有 querySelector 类名都存在")

    # 反向检查：HTML 里有但 JS 从未引用的 ID（可能是死代码，仅供参考）
    unused = sorted(html_ids - js_ids)
    if unused:
        print(f"\n[提示] HTML 中未被 JS 直接引用的 ID（{len(unused)} 个，"
              f"可能由 CSS/label 使用）: {', '.join(unused[:15])}")

    return 1 if missing_ids else 0


if __name__ == "__main__":
    sys.exit(main())
