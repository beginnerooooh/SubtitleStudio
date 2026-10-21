#!/usr/bin/env python
"""Subtitle Studio CLI —— 快速出字幕，导出到其它剪辑软件。

设计目标
────────
1. **一条命令出字幕**：`subtitle-studio video.mp4` 即可，不需要开界面；
2. **配置文件驱动**：把常用参数写进 `config.toml`，命令行只覆盖差异；
3. **导出到别的软件**：SRT / ASS / VTT / LRC / JSON 任选，
   SRT 与 ASS 可直接拖进剪映、Premiere、DaVinci、Final Cut。

典型用法
────────
    # 最快：用默认参数出 SRT
    subtitle-studio 录制.flv

    # 指定输出目录与格式
    subtitle-studio 录制.flv -o ./subs --format srt,ass

    # 用配置文件（推荐：把常用参数写进去）
    subtitle-studio 录制.flv --config my.toml

    # 有弹幕文件时：用弹幕热词提升专名识别
    subtitle-studio 录制.flv --danmaku 录制.xml

    # 只处理前 10 分钟做快速验证
    subtitle-studio 录制.flv --start 0 --duration 600

    # 查看当前生效的全部参数（排查问题用）
    subtitle-studio --print-config --config my.toml

    # 生成一份带注释的配置文件模板
    subtitle-studio --init-config my.toml

退出码
──────
    0  成功
    1  参数错误
    2  输入文件不存在
    3  处理失败（识别/导出）
    130 用户中断（Ctrl+C）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 便携版 Embedded Python 不会把当前目录放进 sys.path，
# 因此显式按文件位置补上项目根与 app/。
ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EXIT_OK = 0
EXIT_ARGS = 1
EXIT_NO_INPUT = 2
EXIT_FAILED = 3
EXIT_INTERRUPT = 130


def _force_utf8() -> None:
    """强制 stdout/stderr 用 UTF-8。

    为什么必须做：Windows 控制台默认代码页是 GBK（cp936），
    CLI 里大量中文输出会被替换成 `?` 或乱码；被父进程以管道捕获时
    更严重（父进程按 UTF-8 解码会得到 `\\ufffd`）。
    这里在启动时重配编码，保证管道与重定向都能拿到正确中文。
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


_force_utf8()


# ---------------- 配置文件 ----------------

def _toml_load(path: Path) -> dict:
    """读 TOML。Python 3.11+ 用 tomllib；更早版本用简易解析。"""
    try:
        import tomllib                                    # Python 3.11+
        with open(path, "rb") as f:
            return tomllib.load(f)
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        raise SystemExit(f"配置文件解析失败：{path}\n  {exc}")
    # 回退：极简解析（只支持 [section] 与 key = value）
    out: dict[str, dict] = {}
    cur = out.setdefault("", {})
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            cur = out.setdefault(line[1:-1].strip(), {})
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if v.lower() in ("true", "false"):
            val: object = v.lower() == "true"
        elif v.startswith('"') and v.endswith('"'):
            val = v[1:-1]
        elif v.startswith("'") and v.endswith("'"):
            val = v[1:-1]
        else:
            try:
                val = int(v)
            except ValueError:
                try:
                    val = float(v)
                except ValueError:
                    val = v
        cur[k.strip()] = val
    out.pop("", None)
    return out


def _build_config(args) -> "AppConfig":
    """把「配置文件 + 命令行参数」合成 AppConfig。"""
    from core.config import AppConfig

    cfg = AppConfig.from_env()

    # ---- 1) 配置文件 ----
    if args.config:
        p = Path(args.config)
        if not p.is_file():
            raise SystemExit(f"配置文件不存在：{p}")
        data = _toml_load(p)
        g = data.get("general", {})
        a = data.get("asr", {})
        s = data.get("segment", {})
        t = data.get("timing", {})
        so = data.get("song", {})
        v = data.get("voiceprint", {})
        kw = data.get("keywords", {})

        if g.get("output_dir"):
            cfg.work_dir = str(g["output_dir"])
        if g.get("device"):
            cfg.device = str(g["device"])
        if g.get("compute_type"):
            cfg.compute_type = str(g["compute_type"])
        if kw.get("dir"):
            cfg.keywords_dir = str(kw["dir"])
        if a.get("engine"):
            cfg.asr.engine = str(a["engine"])
        if a.get("model"):
            cfg.asr.model = str(a["model"])
        for key in ("chunk_seconds", "chunk_overlap", "beam_size", "temperature"):
            if key in a:
                setattr(cfg.asr, key, float(a[key])
                        if key != "beam_size" else int(a[key]))
        if "language" in a:
            cfg.asr.language = str(a["language"])
        if "hotwords" in a:
            cfg.asr.hotwords = str(a["hotwords"])
        if "vad_filter" in a:
            cfg.asr.vad_filter = bool(a["vad_filter"])
        for key in ("target_chars", "max_chars", "max_duration"):
            if key in s:
                setattr(cfg.segment, key, float(s[key])
                        if key == "target_chars" else type(
                            getattr(cfg.segment, key))(s[key]))
        if "use_punctuation" in s:
            cfg.segment.use_punctuation = bool(s["use_punctuation"])
        if "inner_space" in s:
            cfg.segment.inner_space = bool(s["inner_space"])
        if "refine" in t:
            cfg.timing.refine = bool(t["refine"])
        if "enabled" in so:
            cfg.song.enabled = bool(so["enabled"])
        if "manual_titles" in so:
            cfg.song.manual_titles = _split_list(str(so["manual_titles"]))
        for key in ("lyric_correct", "segment_by_lyrics"):
            if key in so:
                setattr(cfg.song, key, bool(so[key]))
        if "lyric_library_dir" in so:
            cfg.song.lyric_library_dir = str(so["lyric_library_dir"])
        for key in ("lyric_doc_threshold", "lyric_line_threshold"):
            if key in so:
                setattr(cfg.song, key, float(so[key]))
        if "enabled" in v:
            cfg.voiceprint.enabled = bool(v["enabled"])
        if "profile_name" in v:
            cfg.voiceprint.profile_name = str(v["profile_name"])
        for key in ("threshold", "bgm_threshold"):
            if key in v:
                setattr(cfg.voiceprint, key, float(v[key]))
        if "bgm_policy" in v:
            cfg.voiceprint.bgm_policy = str(v["bgm_policy"])

    # ---- 2) 命令行覆盖（优先级最高）----
    if args.output_dir:
        cfg.work_dir = args.output_dir
    if args.device:
        cfg.device = args.device
    if args.engine:
        cfg.asr.engine = args.engine
    if args.model:
        cfg.asr.model = args.model
    if args.chunk is not None:
        cfg.asr.chunk_seconds = args.chunk
    if args.beam is not None:
        cfg.asr.beam_size = args.beam
    if args.language:
        cfg.asr.language = args.language
    if args.hotwords:
        cfg.asr.hotwords = args.hotwords
    if args.max_chars is not None:
        cfg.segment.max_chars = args.max_chars
    if args.target_chars is not None:
        cfg.segment.target_chars = args.target_chars
    if args.no_refine:
        cfg.timing.refine = False
    if args.no_punct:
        cfg.segment.use_punctuation = False
    if args.songs:
        cfg.song.enabled = True
    if args.manual_titles:
        cfg.song.manual_titles = _split_list(args.manual_titles)
    if args.lyric_correct:
        cfg.song.lyric_correct = True
    if args.voiceprint:
        cfg.voiceprint.enabled = True
    if args.profile:
        cfg.voiceprint.profile_name = args.profile
        cfg.voiceprint.enabled = True
    if args.bgm_policy:
        cfg.voiceprint.bgm_policy = args.bgm_policy

    return cfg


def _split_list(text: str) -> tuple:
    """把「逗号/顿号/空白分隔」的字符串拆成元组。"""
    out = []
    for part in str(text or "").replace("、", ",").replace("，", ",").split(","):
        for piece in part.split():
            if piece.strip():
                out.append(piece.strip())
    return tuple(out)


# ---------------- 子命令：生成 ----------------

def _clip_input(src: Path, start: float, duration: float) -> Path | None:
    """用 ffmpeg 切出 [start, start+duration) 的临时文件。

    为什么要切文件而不是在 Pipeline 里加区间参数：
    区间会牵动识别分块、时间轴校正、导出时间戳三处逻辑，改动面大；
    CLI 的「快速验证」只需要看到一段的字幕，切文件最简单也最可靠。
    注意：切出来的时间戳从 0 开始，导出后如需对齐原片请自行加偏移。
    """
    from core.env import detect_env

    ff = detect_env().ffmpeg_path or "ffmpeg"
    tmp = ROOT / "cache" / "cli"
    tmp.mkdir(parents=True, exist_ok=True)
    out = tmp / f"clip_{int(time.time())}{src.suffix}"
    cmd = [ff, "-y", "-v", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    if duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(src), "-c", "copy", str(out)]
    try:
        import subprocess

        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0 or not out.is_file():
            # -c copy 对某些容器会失败，退回重编码
            cmd2 = [ff, "-y", "-v", "error"]
            if start > 0:
                cmd2 += ["-ss", f"{start:.3f}"]
            if duration > 0:
                cmd2 += ["-t", f"{duration:.3f}"]
            cmd2 += ["-i", str(src), "-c:v", "libx264", "-preset",
                     "veryfast", "-c:a", "aac", str(out)]
            subprocess.run(cmd2, check=True, capture_output=True)
        return out if out.is_file() else None
    except Exception:
        return None


def cmd_generate(args) -> int:
    """识别视频 → 导出字幕。"""
    from core.pipeline2 import Pipeline, PipelineRequest

    src = Path(args.input)
    if not src.is_file():
        print(f"错误：输入文件不存在：{src}", file=sys.stderr)
        return EXIT_NO_INPUT

    cfg = _build_config(args)
    formats = [f.strip().lower() for f in (args.format or "srt").split(",")
               if f.strip()]
    valid = {"srt", "ass", "vtt", "lrc", "json"}
    bad = [f for f in formats if f not in valid]
    if bad:
        print(f"错误：不支持的格式 {bad}；可选 {sorted(valid)}", file=sys.stderr)
        return EXIT_ARGS

    req = PipelineRequest(
        input_path=str(src),
        formats=tuple(formats),
        live_xml=str(args.danmaku or ""),
        reference_srt=str(args.reference or ""),
        title=args.title or src.stem,
        enable_song_detect=cfg.song.enabled,
        enable_voiceprint=cfg.voiceprint.enabled,
        profile_name=cfg.voiceprint.profile_name,
    )
    # 时长裁剪：只处理一段（快速验证用）。
    # 做法是先用 ffmpeg 切出临时文件——PipelineRequest 没有区间字段，
    # 而在请求上加区间会牵动识别/校正/导出多处，风险更大。
    clipped: Path | None = None
    if args.start or args.duration:
        clipped = _clip_input(src, float(args.start or 0.0),
                              float(args.duration or 0.0))
        if clipped is None:
            print("错误：裁剪输入失败（检查 --start/--duration）",
                  file=sys.stderr)
            return EXIT_ARGS
        req.input_path = str(clipped)

    t0 = time.time()
    last = [""]

    def on_progress(ratio: float, message: str) -> None:
        if message and message != last[0]:
            last[0] = message
            pct = int(max(0.0, min(1.0, ratio)) * 100)
            print(f"\r[{pct:3d}%] {message[:60]:<60}", end="", flush=True)

    def on_log(message: str) -> None:
        if args.verbose:
            print(f"\n  · {message}", end="", flush=True)

    try:
        res = Pipeline(cfg, req, on_progress=on_progress, on_log=on_log).run()
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return EXIT_INTERRUPT
    except Exception as exc:                                  # noqa: BLE001
        print(f"\n处理失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return EXIT_FAILED
    finally:
        print()
        if clipped is not None:
            clipped.unlink(missing_ok=True)

    elapsed = time.time() - t0
    print(f"完成：{len(res.lines)} 行字幕，耗时 {elapsed:.1f}s")
    if res.out_dir:
        print(f"输出目录：{res.out_dir}")
    # res.files 是 {格式: 路径} 的字典
    for fmt, fpath in (res.files or {}).items():
        print(f"  {fmt:<6} {fpath}")
    if res.warnings:
        print(f"\n注意：{len(res.warnings)} 条警告")
        for w in res.warnings[:8]:
            print(f"  - {w}")

    # 评测（可选）
    if args.reference and res.evaluation is not None:
        ev = res.evaluation
        print("\n评测（对比参考字幕）")
        print(f"  完全匹配  {ev.exact_match_rate:.1%}")
        print(f"  文本匹配  {ev.text_match_rate:.1%}")
        print(f"  时间匹配  {ev.time_match_rate:.1%}")
        print(f"  字错误率  {ev.cer:.3f}")
        print(f"  字符召回  {ev.char_recall:.1%}")

    if args.json:
        payload = {
            "ok": True, "lines": len(res.lines), "seconds": round(elapsed, 1),
            "out_dir": str(res.out_dir),
            "files": {k: str(v) for k, v in (res.files or {}).items()},
            "warnings": res.warnings,
            "timings": res.timings,
        }
        if res.evaluation is not None:
            payload["evaluation"] = res.evaluation.to_dict()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return EXIT_OK


# ---------------- 子命令：声纹 ----------------

def cmd_voiceprint(args) -> int:
    """从干声文件注册声纹（配合 UVR5 等工具分离出的纯人声）。"""
    from core.env import detect_env
    from core.streamer import VoiceprintEngine, StreamerProfile, save_profile

    src = Path(args.input)
    if not src.is_file():
        print(f"错误：文件不存在：{src}", file=sys.stderr)
        return EXIT_NO_INPUT
    name = args.name or src.stem
    tmp = ROOT / "cache" / "voiceprint"
    tmp.mkdir(parents=True, exist_ok=True)
    wav = tmp / f"vp_cli_{int(time.time())}.wav"
    try:
        import subprocess

        ff = detect_env().ffmpeg_path or "ffmpeg"
        subprocess.run([ff, "-y", "-v", "error", "-i", str(src),
                        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                        str(wav)], check=True, capture_output=True)
        import wave

        with wave.open(str(wav), "rb") as w:
            dur = w.getnframes() / float(w.getframerate() or 1)
        eng = VoiceprintEngine(device=args.device or "cpu")
        vec = eng.enroll(str(wav), 0.0, dur)
        saved = save_profile(StreamerProfile(name=name, speak=vec),
                             profiles_dir=str(ROOT / "profiles"))
        print(f"已注册声纹「{name}」（{dur:.1f}s，{len(vec)} 维）")
        print(f"  {saved}")
        return EXIT_OK
    except Exception as exc:                                  # noqa: BLE001
        print(f"注册失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED
    finally:
        wav.unlink(missing_ok=True)


# ---------------- 子命令：信息 ----------------

def cmd_info(args) -> int:
    """显示环境与可用资源。"""
    from core.env import detect_env

    env = detect_env()
    print("Subtitle Studio CLI")
    print(f"  项目目录    {ROOT}")
    print(f"  Python      {sys.version.split()[0]}")
    print(f"  FFmpeg      {env.ffmpeg_path or '未找到'}")
    print(f"  FFprobe     {env.ffprobe_path or '未找到'}")
    try:
        import torch

        print(f"  PyTorch     {torch.__version__}")
        print(f"  CUDA 可用   {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU         {torch.cuda.get_device_name(0)}")
    except Exception:
        print("  PyTorch     未安装")
    profiles = sorted(p.stem for p in (ROOT / "profiles").glob("*.npy")) \
        if (ROOT / "profiles").is_dir() else []
    print(f"  已注册声纹  {len(profiles)} 个"
          + (f"（{', '.join(profiles[:5])}）" if profiles else ""))
    lyrics = list((ROOT / "lyrics").glob("*.lrc")) if (ROOT / "lyrics").is_dir() else []
    print(f"  歌词库      {len(lyrics)} 个文件")
    return EXIT_OK


# ---------------- 配置文件模板 ----------------

def cmd_init_config(args) -> int:
    """把带注释的配置模板写到指定路径。"""
    src = ROOT / "config.example.toml"
    if not src.is_file():
        print("错误：找不到 config.example.toml", file=sys.stderr)
        return EXIT_FAILED
    dst = Path(args.init_config)
    if dst.exists() and not args.force:
        print(f"错误：{dst} 已存在（加 --force 覆盖）", file=sys.stderr)
        return EXIT_ARGS
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"已生成配置模板：{dst}")
    print("按需修改后，用 --config 指定：")
    print(f"  subtitle-studio video.mp4 --config {dst}")
    return EXIT_OK


def cmd_print_config(args) -> int:
    """打印当前生效的全部参数（排查问题用）。"""
    import dataclasses

    cfg = _build_config(args)
    print(f"# 生效配置（来自默认值 + 配置文件 {args.config or '无'} + 命令行）")

    def dump(obj, prefix: str) -> None:
        for f in dataclasses.fields(obj):
            val = getattr(obj, f.name)
            if dataclasses.is_dataclass(val):
                dump(val, f"{prefix}{f.name}.")
            else:
                print(f"{prefix}{f.name} = {val!r}")

    dump(cfg, "")
    return EXIT_OK


# ---------------- 参数解析 ----------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="subtitle-studio",
        description="Subtitle Studio CLI：快速生成字幕并导出到剪辑软件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  subtitle-studio video.flv\n"
            "  subtitle-studio video.flv -o ./subs --format srt,ass\n"
            "  subtitle-studio video.flv --config my.toml\n"
            "  subtitle-studio --init-config my.toml\n"
            "  subtitle-studio --print-config --config my.toml\n"
            "\n参数详解见 docs/CLI.md"
        ))
    p.add_argument("input", nargs="?", help="输入视频/音频文件")
    p.add_argument("-o", "--output-dir", help="输出目录（默认 outputs）")
    p.add_argument("-c", "--config", help="配置文件（TOML）")
    p.add_argument("-f", "--format", default="",
                   help="导出格式，逗号分隔：srt,ass,vtt,lrc,json（默认 srt）")
    p.add_argument("-n", "--name", help="输出文件名（不含扩展名）")
    p.add_argument("-t", "--title", help="标题（写入元数据）")
    p.add_argument("--reference", help="参考字幕，用于评测准确率")
    p.add_argument("--danmaku", help="弹幕 XML（提供热词，提升专名识别）")

    g = p.add_argument_group("识别")
    g.add_argument("--engine", choices=["whisper", "funasr"], help="识别引擎")
    g.add_argument("--model", help="模型：large-v3 / medium / small")
    g.add_argument("--chunk", type=float, help="分块长度（秒），默认 120")
    g.add_argument("--beam", type=int, help="束搜索宽度，默认 1（贪心）")
    g.add_argument("--language", help="语言：auto / zh / en …")
    g.add_argument("--hotwords", help="热词，空格分隔")
    g.add_argument("--device", help="设备：auto / cuda / cpu")

    g = p.add_argument_group("断句")
    g.add_argument("--target-chars", type=float, help="断句目标字数，默认 11")
    g.add_argument("--max-chars", type=int, help="单行最大字数，默认 17")
    g.add_argument("--no-punct", action="store_true", help="关闭标点辅助断句")
    g.add_argument("--no-refine", action="store_true", help="关闭时间轴校正")

    g = p.add_argument_group("歌曲与声纹")
    g.add_argument("--songs", action="store_true", help="启用歌曲识别")
    g.add_argument("--manual-titles", help="本场演唱曲目，逗号分隔")
    g.add_argument("--lyric-correct", action="store_true",
                   help="用歌词库校正文本")
    g.add_argument("--voiceprint", action="store_true", help="启用声纹过滤")
    g.add_argument("--profile", help="声纹名称（自动启用声纹过滤）")
    g.add_argument("--bgm-policy", choices=["mark", "filter", "off"],
                   help="BGM 策略，默认 mark")

    g = p.add_argument_group("范围")
    g.add_argument("--start", type=float, help="只处理从该秒开始的片段")
    g.add_argument("--duration", type=float, help="只处理这么长（秒）")

    g = p.add_argument_group("其它")
    g.add_argument("--json", action="store_true", help="末尾输出 JSON 摘要")
    g.add_argument("-v", "--verbose", action="store_true", help="显示详细日志")
    g.add_argument("--print-config", action="store_true",
                   help="打印生效配置后退出")
    g.add_argument("--init-config", metavar="PATH",
                   help="生成带注释的配置模板后退出")
    g.add_argument("--force", action="store_true", help="配合 --init-config 覆盖")
    g.add_argument("--voiceprint-enroll", metavar="AUDIO",
                   help="从干声文件注册声纹后退出")
    g.add_argument("--voiceprint-name", help="配合 --voiceprint-enroll 指定名称")
    g.add_argument("--info", action="store_true", help="显示环境信息后退出")
    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    if args.init_config:
        return cmd_init_config(args)
    if args.print_config:
        return cmd_print_config(args)
    if args.info:
        return cmd_info(args)
    if args.voiceprint_enroll:
        args.input = args.voiceprint_enroll
        args.name = args.voiceprint_name or Path(args.voiceprint_enroll).stem
        return cmd_voiceprint(args)

    if not args.input:
        # 无输入：打印帮助。退出码用 1（参数错误），
        # 便于 `cli` 与 `cli --help` 区分（后者由 argparse 返回 0）。
        p.print_help()
        return EXIT_ARGS
    return cmd_generate(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(EXIT_INTERRUPT)
