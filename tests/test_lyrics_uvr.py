"""歌词局部匹配、UVR5 集成、时间轴拖动回归测试。

这些测试针对**真实数据**上暴露的问题，每个都写清了失败现象，
避免以后重构时把它们改回去。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from core.lyrics_library import (  # noqa: E402
    apply_report, match_document, parse_lyric_text)
from core.models import SubtitleLine, SubtitleWord  # noqa: E402
from core.song import norm_text, score_similarity  # noqa: E402


def _line(text: str, start: float, end: float) -> SubtitleLine:
    return SubtitleLine(words=[SubtitleWord(text=text, start=start, end=end)])


# ---------------- 歌词匹配：多行合并 ----------------

def test_lyric_matches_when_one_line_split_across_subtitles():
    """回归：一句歌词被切成两行字幕时也必须匹配上。

    真实数据实测：歌词「放开了拳头 反而更自由」被 ASR 切成字幕的
    「放开了拳头」「反而更自由」两行。按「一行对一行」比较时每行只
    覆盖歌词的一半，相似度上不去 → 整首歌匹配失败（早期得分 0.000）。
    """
    subs = [_line("放开了拳头", 0.0, 2.0),
            _line("反而更自由", 2.0, 4.0)]
    doc = parse_lyric_text("放开了拳头 反而更自由", title="x")
    rep = match_document(subs, doc, threshold=0.55)
    assert rep.score > 0.5, f"应当匹配上，实际 {rep.score}"
    assert len(rep.matches) >= 1


def test_lyric_merge_covers_multiple_subtitle_lines():
    """回归：合并窗口应记录覆盖到的最后一行（merged_to）。"""
    subs = [_line("在你的心里", 0.0, 2.0),
            _line("我早已落榜", 2.0, 4.0),
            _line("劝自己", 4.0, 6.0)]
    doc = parse_lyric_text("在你的心里我早已落榜\n劝自己别再去假装坚强", title="x")
    rep = match_document(subs, doc, threshold=0.55)
    assert rep.matches, "应有匹配"
    # 至少有一条覆盖了多行
    assert any(m.merged_to > m.line_index for m in rep.matches), \
        "跨行的匹配应记录 merged_to"


# ---------------- 歌词替换：不写错行 ----------------

def test_apply_report_does_not_shift_lines_when_merging():
    """回归：多行合并不许让后续替换写错行。

    真实数据实测：边删边改会让后面（行号更小）的匹配偏移，
    把歌词写到相邻错行上（「让爱的人都沉默」被写成下一句）。
    修法是「先算计划再一次性重建」。
    """
    subs = [_line("潺潺的溜走", 0.0, 2.0),
            _line("命运好幽默", 2.0, 4.0),
            _line("让爱的人都沉默", 4.0, 6.0),
            _line("一整个宇宙", 6.0, 8.0)]
    doc = parse_lyric_text(
        "闪过的念头 潺潺的溜走\n命运好幽默 让爱的人都沉默\n一整个宇宙 换一颗红豆",
        title="x")
    rep = match_document(subs, doc, threshold=0.5)
    out, n = apply_report(subs, rep, min_score=0.7)
    # 每个替换后的文本必须与它原来的内容相似（不能张冠李戴）
    for i in range(min(len(out), len(subs))):
        if out[i].text != subs[i].text:
            sim = score_similarity(norm_text(subs[i].text),
                                   norm_text(out[i].text))
            assert sim >= 0.4, (
                f"第 {i} 行被写成了无关内容：{subs[i].text!r} → {out[i].text!r}")


def test_apply_report_keeps_timeline_sane():
    """回归：替换后时间轴必须单调且区间有效。"""
    subs = [_line("测试一", 0.0, 2.0), _line("测试二", 2.0, 4.0),
            _line("测试三", 4.0, 6.0)]
    doc = parse_lyric_text("测试一\n测试二\n测试三", title="x")
    rep = match_document(subs, doc, threshold=0.5)
    out, _ = apply_report(subs, rep, min_score=0.7)
    prev_end = -1.0
    for ln in out:
        assert ln.end > ln.start, f"区间无效：{ln.start}-{ln.end}"
        assert ln.start >= prev_end - 1e-6, "时间轴出现重叠/倒退"
        prev_end = ln.end


def test_duplicate_lyric_text_not_applied_twice():
    """回归：同一句歌词不应被写到多个位置。

    实测：「在你的心里」和「我早已落宝」两行都被替换成
    「在你的心里我早已落榜」，等于丢了一行。
    """
    subs = [_line("在你的心里", 0.0, 2.0), _line("我早已落宝", 2.0, 4.0)]
    doc = parse_lyric_text("在你的心里我早已落榜", title="x")
    rep = match_document(subs, doc, threshold=0.5)
    out, n = apply_report(subs, rep, min_score=0.7)
    texts = [o.text for o in out]
    # 替换条数不超过不同歌词行数
    assert n <= 1


# ---------------- 歌词匹配：副歌重复 ----------------

def test_chorus_repetition_allowed():
    """回归：副歌重复出现时应能匹配到不同位置。

    真实歌曲副歌重复 2~3 次，歌词文件里是不同行。早期「每句只留 1 个
    候选」会让第二次副歌全部被挡掉（实测情歌只命中 8 处 / 33 处可替换）。
    """
    subs = [_line("命运好幽默", 0.0, 2.0),
            _line("让爱的人都沉默", 2.0, 4.0),
            _line("命运好幽默", 10.0, 12.0),
            _line("让爱的人都沉默", 12.0, 14.0)]
    doc = parse_lyric_text("命运好幽默 让爱的人都沉默\n命运好幽默 让爱的人都沉默",
                           title="x")
    rep = match_document(subs, doc, threshold=0.5)
    # 至少能匹配到两处不同位置
    assert len(rep.matches) >= 2, f"副歌重复应能多处匹配，实际 {len(rep.matches)}"


# ---------------- UVR5 ----------------

def test_uvr_detects_installed_root():
    """UVR5 探测：找不到时应返回 None 而不是抛错。"""
    from core import uvr5

    root = uvr5.detect_uvr_root()
    if root:
        assert Path(root, "models").is_dir()
    else:
        assert root is None


def test_uvr_status_shape():
    """状态对象字段齐全（UI 依赖这些字段）。"""
    from core import uvr5

    st = uvr5.status()
    d = st.to_dict()
    for key in ("available", "root", "models", "preferred", "reason"):
        assert key in d
    if d["available"]:
        assert d["preferred"], "可用时必须给出首选模型"
        assert d["models"], "可用时必须列出模型"


def test_uvr_model_shape_is_respected():
    """回归：MDX 模型的输入维度必须按实际形状读取。

    实测踩坑：按 n_fft=2048 实现会得到 1025 个频点，而模型要求 3072，
    报 `Got invalid dimensions ... Expected: 3072`。
    """
    from core import uvr5

    root = uvr5.detect_uvr_root()
    if not root:
        pytest.skip("未安装 UVR5")
    model = uvr5.pick_vocal_model(root)
    if model is None:
        pytest.skip("UVR5 目录里没有模型")
    src = (ROOT / "app" / "core" / "uvr5.py").read_text(encoding="utf-8")
    # 必须从模型实际形状推导，而不是写死
    assert "get_inputs()" in src
    assert "n_freq" in src and "n_fft" in src
    # 必须记录踩坑经验
    assert "3072" in src


def test_uvr_separation_accumulates_overlap():
    """回归：分块重叠区必须取平均，不能直接赋值。

    实测踩坑：直接赋值会让未覆盖的帧保持全零，重建后「只有首尾有声音」；
    而复相加会在重叠区因相位不一致相互抵消，同样导致大片静音。
    正确做法是「幅度累加取平均 + 沿用原相位」。
    """
    src = (ROOT / "app" / "core" / "uvr5.py").read_text(encoding="utf-8")
    assert "mag_acc" in src, "应按幅度累加"
    assert "wsum" in src, "应统计权重以便取平均"
    assert "np.angle" in src, "应沿用原谱相位"


def test_uvr_output_is_voiceprint_compatible():
    """回归：分离结果必须能直接喂给声纹引擎（16kHz 单声道）。"""
    src = (ROOT / "app" / "core" / "uvr5.py").read_text(encoding="utf-8")
    assert "separate_segment_for_voiceprint" in src
    assert "16000" in src, "必须重采样到 16kHz"


# ---------------- 时间轴拖动 ----------------

def _desktop(name: str) -> str:
    return (ROOT / "desktop" / "renderer" / name).read_text(encoding="utf-8")


def test_timeline_supports_drag_panning():
    """回归：时间轴必须支持左键拖动平移（用户反馈「不能拖动」）。"""
    js = _desktop("app.js")
    assert "panning" in js, "缺少拖动平移状态"
    assert "m.startView" in js, "平移应基于起始视图位置"
    # 必须区分「字幕轨起手=框选」与「其它区域=平移」
    assert "inSubLane" in js


def test_pan_uses_real_track_width():
    """回归：平移不能解构 viewSpan().width。

    viewSpan() 返回 {start, span, dur}，没有 width。早期写成
    `const {width, span} = viewSpan()` 会得到 undefined，
    使 `width <= 0` 恒真 → 拖了也不动（实测 panned=0）。
    """
    js = _desktop("app.js")
    start = js.index("if (!m.inSubLane)")
    seg = js[start: start + 900]        # 取该分支的完整代码块
    assert "const { width, span } = viewSpan()" not in seg, \
        "viewSpan 没有 width 字段，不能这样解构"
    assert "subRect.width" in seg or "rect.width" in seg, \
        "应使用轨道的实际宽度"


def test_pan_clamps_when_proxy_shorter_than_span():
    """回归：代理片段比跨度还短时，上界必须用 max(0, dur-span)。

    否则 hi 为负，Math.min(hi, ...) 得到负数，viewStart 变成非法值。
    """
    js = _desktop("app.js")
    start = js.index("if (!m.inSubLane)")
    seg = js[start: start + 900]
    assert "Math.max(0, dur - span)" in seg


# ---------------- 前端脚本必须能编译 ----------------

def _run_structure_check(target: Path) -> tuple[int, str]:
    """调用 tools/check_js_structure.js 检查指定 JS 文件。"""
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("未安装 node")
    script = ROOT / "tools" / "check_js_structure.js"
    assert script.is_file(), "缺少 tools/check_js_structure.js"
    proc = subprocess.run([node, str(script), str(target)],
                          capture_output=True, encoding="utf-8",
                          errors="replace", timeout=120)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def test_renderer_app_js_compiles():
    """回归：app.js 必须能真正编译通过。

    关键：`node --check` **漏报**了 `await` 位于非 async 函数中的错误
    ——实测 `node --check` 返回 0，但浏览器加载时抛 SyntaxError，
    整个界面变成白板（所有按钮失效）。必须用 `new vm.Script()` 编译。
    """
    rc, out = _run_structure_check(ROOT / "desktop" / "renderer" / "app.js")
    assert rc == 0, "app.js 结构检查失败（浏览器会白屏）：\n" + out[:1500]


def test_structure_checker_catches_bad_await():
    """自检：检查工具本身必须能发现 await 位置错误。

    否则这个守卫是假的——测试会一直「通过」却抓不到问题。
    """
    import tempfile

    bad = "function ok() {\n  return 1;\n}\n  await foo();\n"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as f:
        f.write(bad)
        tmp = Path(f.name)
    try:
        rc, out = _run_structure_check(tmp)
        assert rc != 0, "检查器没能发现 await 位置错误，守卫失效"
        assert "await" in out.lower()
    finally:
        tmp.unlink(missing_ok=True)
