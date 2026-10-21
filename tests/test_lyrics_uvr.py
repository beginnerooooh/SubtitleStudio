"""歌词局部匹配、声纹注册、时间轴拖动回归测试。

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


# ---------------- speechbrain Windows 递归修复 ----------------

def test_sb_compat_patches_lazy_module():
    """回归：speechbrain 的 LazyModule 必须被打上跨平台补丁。

    根因：speechbrain 的 `ensure_module` 用
      `importer_frame.filename.endswith("/inspect.py")`
    判断调用栈是否来自 inspect，但 Windows 路径是反斜杠，
    该判断**永远为 False** → 防护失效 → 无限递归：
      __getattr__ → ensure_module → inspect.getmodule
      → __getattr__ → ...
    """
    from core import sb_compat

    ok = sb_compat.apply_patch()
    assert ok, "补丁应成功应用（未安装 speechbrain 时才会失败）"
    assert sb_compat.patched()

    import speechbrain.utils.importutils as iu

    assert "ensure_module" in iu.LazyModule.__dict__, \
        "LazyModule.ensure_module 应被替换（不能只改实例）"


def test_sb_compat_is_idempotent():
    """补丁可重复调用，不会层层包装。"""
    from core import sb_compat

    assert sb_compat.apply_patch() is True
    import speechbrain.utils.importutils as iu

    first = iu.LazyModule.ensure_module
    sb_compat.apply_patch()
    assert iu.LazyModule.ensure_module is first, "重复调用不应再次包装"


def test_sb_compat_handles_windows_paths():
    """回归：判断「来自 inspect.py」必须跨平台。

    直接验证 os.path.basename 判断在反斜杠路径下也成立——
    这正是官方实现漏掉的地方。
    """
    import os

    win_path = r"C:\Python311\Lib\inspect.py"
    posix = "/usr/lib/python3.11/inspect.py"
    assert os.path.basename(win_path) == "inspect.py"
    assert os.path.basename(posix) == "inspect.py"
    # 官方写法在 Windows 路径上失效（这就是 bug 本身）
    assert not win_path.endswith("/inspect.py")


def test_voiceprint_load_applies_patch():
    """回归：声纹模型加载前必须先打补丁。"""
    src = (ROOT / "app" / "core" / "voiceprint.py").read_text(encoding="utf-8")
    assert "sb_compat" in src, "voiceprint 加载前应调用 sb_compat.apply_patch()"
    assert src.index("sb_compat.apply_patch") < src.index(
        "from speechbrain.inference.speaker import"), \
        "补丁必须在导入 speechbrain 之前应用"


def test_streamer_load_applies_patch():
    """回归：streamer 的 ECAPA 加载同样要打补丁。"""
    src = (ROOT / "app" / "core" / "streamer.py").read_text(encoding="utf-8")
    assert "sb_compat" in src
    assert src.index("sb_compat.apply_patch") < src.index(
        "from speechbrain.inference.speaker import")


# ---------------- 声纹注册：列表刷新 ----------------

def test_enroll_uses_real_duration_not_huge_number():
    """回归：enroll() 的时间参数必须用实际时长，不能传 1e9。

    `enroll(wav, start, end)` 的参数是**秒**，内部 `_cap` 会按文件时长
    裁剪。传 1e9 会被算成越界区间并抛
    「无效音频区间：[499999985.0, 500000015.0)」（实测踩坑）。
    """
    src = (ROOT / "app" / "desktop_server.py").read_text(encoding="utf-8")
    assert "engine.enroll(speak, 0.0, 1e9)" not in src, \
        "/api/profiles 仍在使用 1e9 作为结束时间"
    assert "_wav_seconds" in src, "应使用 _wav_seconds 取实际时长"


def test_frontend_refreshes_both_voiceprint_lists():
    """回归：注册成功后必须同时刷新「弹窗列表」和「工具栏下拉框」。

    实测事故：声纹注册成功后只调了 refreshVoiceList()（刷新弹窗的
    vp-list），没调 loadProfiles()（刷新 #profile 下拉框），
    用户看到的现象是「注册了但列表里没有新声纹」。
    """
    js = _desktop("app.js")
    for anchor in ("async function enrollVoiceprintFromSpan",
                   "async function importDryVoice"):
        start = js.index(anchor)
        nxt = js.find("\nasync function ", start + 10)
        nxt2 = js.find("\nfunction ", start + 10)
        end = min(x for x in (nxt, nxt2, len(js)) if x > 0)
        body = js[start:end]
        assert "refreshVoiceList()" in body, f"{anchor} 缺少 refreshVoiceList()"
        assert "loadProfiles()" in body, (
            f"{anchor} 缺少 loadProfiles()——下拉框不会出现新声纹")


# ---------------- CLI ----------------

def _run_cli(*argv, timeout: int = 300):
    """运行 CLI，返回 (退出码, 标准输出+错误)。"""
    import subprocess

    # 必须用**当前解释器**（runtime/python.exe），不能用 shutil.which("python")：
    # 实测该函数会返回 MSYS2 的 python，它没装本项目依赖，
    # 导致 CLI 在 import 阶段就崩，测试结果完全失真。
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "studio_cli.py"), *argv],
        capture_output=True, encoding="utf-8", errors="replace",
        timeout=timeout, cwd=str(ROOT))
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def test_cli_help_works():
    """CLI 应能打印帮助并以 0 退出（无输入时打印帮助）。"""
    rc, out = _run_cli()
    assert rc == 1, "无输入参数应返回 1"
    assert "usage" in out.lower()


def test_cli_info_works():
    """--info 应显示环境信息。"""
    rc, out = _run_cli("--info")
    assert rc == 0
    assert "FFmpeg" in out
    assert "PyTorch" in out or "未安装" in out


def test_cli_init_config_generates_template(tmp_path):
    """--init-config 应生成带注释的配置模板。"""
    dst = tmp_path / "my.toml"
    rc, out = _run_cli("--init-config", str(dst))
    assert rc == 0, out[:500]
    assert dst.is_file()
    text = dst.read_text(encoding="utf-8")
    # 必须有注释说明与关键分节
    assert "#" in text
    for section in ("[general]", "[asr]", "[segment]", "[voiceprint]"):
        assert section in text, f"模板缺少 {section}"


def test_cli_init_config_refuses_overwrite(tmp_path):
    """已存在的配置默认不覆盖，需 --force。"""
    dst = tmp_path / "my.toml"
    dst.write_text("x = 1\n", encoding="utf-8")
    rc, out = _run_cli("--init-config", str(dst))
    assert rc == 1, "已存在时应拒绝"
    assert "已存在" in out or "\u5df2\u5b58\u5728" in out
    # 加 --force 才覆盖
    rc2, _ = _run_cli("--init-config", str(dst), "--force")
    assert rc2 == 0


def test_cli_print_config_reflects_file(tmp_path):
    """--print-config 应把配置文件的值合并进最终结果。"""
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        "[asr]\nchunk_seconds = 60\nbeam_size = 3\n\n"
        "[segment]\nmax_chars = 12\n",
        encoding="utf-8")
    rc, out = _run_cli("--print-config", "--config", str(cfg))
    assert rc == 0, out[:600]
    assert "asr.chunk_seconds = 60.0" in out
    assert "asr.beam_size = 3" in out
    assert "segment.max_chars = 12" in out


def test_cli_command_line_overrides_config(tmp_path):
    """命令行参数优先级最高，应覆盖配置文件。"""
    cfg = tmp_path / "c.toml"
    cfg.write_text("[asr]\nchunk_seconds = 60\n", encoding="utf-8")
    rc, out = _run_cli("--print-config", "--config", str(cfg),
                       "--chunk", "200")
    assert rc == 0
    assert "asr.chunk_seconds = 200.0" in out


def test_cli_rejects_missing_input():
    """输入文件不存在应返回 2。"""
    rc, out = _run_cli("no_such_file_xyz.mp4")
    assert rc == 2, out[:400]
    assert "不存在" in out or "not exist" in out.lower()


def test_cli_rejects_bad_format():
    """不支持的格式应返回 1。"""
    # 造一个假文件路径（存在性检查在前，所以先用真实文件）
    sample = ROOT / "outputs" / "_eval" / "burn_src.mp4"
    if not sample.is_file():
        pytest.skip("缺少测试视频")
    rc, out = _run_cli(str(sample), "--format", "docx")
    assert rc == 1, out[:400]
    assert "不支持" in out or "unsupported" in out.lower()


def test_cli_config_missing_file_errors():
    """配置文件不存在应明确报错。"""
    rc, out = _run_cli("--print-config", "--config", "no_such.toml")
    assert rc != 0
    assert "不存在" in out


def test_cli_has_all_documented_flags():
    """回归：docs/CLI.md 里提到的每个参数都必须真实存在。

    避免文档与实现脱节（用户照着文档敲却报 unknown argument）。
    """
    import re

    rc, out = _run_cli("--help")
    assert rc == 0 or rc == 1
    doc = (ROOT / "docs" / "CLI.md").read_text(encoding="utf-8")
    # 从文档表格里抽出 `--xxx` 形式的参数
    flags = set(re.findall(r"`(--[a-z][a-z0-9-]+)", doc))
    assert flags, "文档里没找到参数"
    missing = [f for f in flags if f not in out]
    assert not missing, f"文档提到但 CLI 没有的参数: {missing}"


def test_cli_documents_config_sections():
    """配置模板里的每个分节都应在文档里有说明。"""
    tmpl = (ROOT / "config.example.toml").read_text(encoding="utf-8")
    doc = (ROOT / "docs" / "CLI.md").read_text(encoding="utf-8")
    import re

    sections = set(re.findall(r"^\[([a-z]+)\]", tmpl, re.M))
    assert sections, "模板里没有分节"
    # 文档至少要提到这些分节名
    for s in sections:
        assert s in doc.lower(), f"文档未说明配置分节 [{s}]"
