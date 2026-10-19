"""Subtitle Studio 命令行：生成字幕、评测对比、批量处理。

用法示例
────────
    python -m tools.cli generate input.mp4 --ref reference.srt --eval
    python -m tools.cli evaluate --ref reference.srt --hyp out.srt
    python -m tools.cli batch ./videos --out ./outputs
    python -m tools.cli info

设计：与 UI 共用 core.pipeline2 / core.evaluate，不重复业务逻辑；
输出人类可读摘要 + 可机读 JSON（--json）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 便携版使用 Embedded Python：它不会把当前目录（也不含 tools/ 的父目录）
# 放进 sys.path，因此这里显式按文件位置补上项目根与 app/。
ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.config import AppConfig                       # noqa: E402
from core.evaluate import evaluate, load_srt            # noqa: E402
from core.pipeline2 import Pipeline, PipelineRequest     # noqa: E402


def _build_config(args) -> AppConfig:
    overrides = {
        "asr": {
            "engine": args.engine,
            "model": args.model,
            "chunk_seconds": args.chunk,
            "chunk_overlap": args.overlap,
        },
        "timing": {"refine": not args.no_refine},
        "segment": {"use_punctuation": not args.no_punct},
        "song": {"enabled": not args.no_songs},
        "voiceprint": {"enabled": bool(args.profile),
                       "profile_name": args.profile or ""},
    }
    return AppConfig.from_env(**overrides)


def cmd_generate(args) -> int:
    cfg = _build_config(args)
    req = PipelineRequest(
        input_path=args.input,
        reference_srt=args.ref or "",
        live_xml=args.live_xml or "",
        enable_separation=args.separate,
        enable_voiceprint=bool(args.profile),
        enable_song_detect=not args.no_songs,
        formats=tuple(args.formats),
    )
    t0 = time.time()
    res = Pipeline(cfg, req, on_log=print).run()
    print(f"\n完成：{len(res.lines)} 行，用时 {time.time() - t0:.1f}s")
    print(f"输出目录：{res.out_dir}")
    for key, path in res.files.items():
        print(f"  {key}: {path}")
    if res.evaluation is not None:
        report = res.evaluation
        print("\n" + report.summary_md("评测：生成结果 vs 人工校对样本"))
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_evaluate(args) -> int:
    ref = load_srt(args.ref)
    hyp = load_srt(args.hyp)
    report = evaluate(ref, hyp)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(report.summary_md(args.title),
                                       encoding="utf-8")
        (out / "report.json").write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.summary_md(args.title))
    return 0 if report.cer < args.max_cer else 2


def cmd_batch(args) -> int:
    src = Path(args.batch_dir)
    files = [p for p in src.iterdir()
             if p.suffix.lower() in {".mp4", ".mkv", ".mov", ".flv", ".mp3",
                                     ".wav", ".flac", ".m4a", ".aac", ".webm"}]
    if not files:
        print(f"目录中没有可处理的音视频：{src}")
        return 1
    print(f"共 {len(files)} 个文件")
    failed = 0
    for i, f in enumerate(sorted(files), 1):
        print(f"\n[{i}/{len(files)}] {f.name}")
        cfg = _build_config(args)
        req = PipelineRequest(input_path=str(f),
                              formats=tuple(args.formats),
                              enable_song_detect=not args.no_songs)
        try:
            res = Pipeline(cfg, req, on_log=print).run()
            print(f"   -> {len(res.lines)} 行")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"   失败：{exc}")
    print(f"\n完成 {len(files) - failed}/{len(files)}")
    return 0 if failed == 0 else 1


def cmd_info(args) -> int:
    from core.env import detect_env
    from core.providers import available_providers

    env = detect_env()
    print("运行环境")
    print(f"  设备: {env.device}  量化: {env.compute_type}  "
          f"显存: {env.vram_gb or 0:.1f}GB")
    print(f"  FFmpeg: {env.ffmpeg_path or '未找到'}")
    print(f"  识别后端: {', '.join(available_providers())}")
    cfg = AppConfig.from_env()
    print("\n默认配置")
    print(f"  引擎/模型: {cfg.asr.engine}/{cfg.asr.model}")
    print(f"  分块: {cfg.asr.chunk_seconds}s（重叠 {cfg.asr.chunk_overlap}s）")
    print(f"  VAD 过滤: {cfg.asr.vad_filter}")
    print(f"  时间轴校正: {cfg.timing.refine}")
    print(f"  标点辅助: {cfg.segment.use_punctuation}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="subtitle-studio", description="Subtitle Studio 命令行")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--engine", default="whisper")
        sp.add_argument("--model", default="large-v3")
        sp.add_argument("--chunk", type=float, default=120.0)
        sp.add_argument("--overlap", type=float, default=10.0)
        sp.add_argument("--no-refine", action="store_true")
        sp.add_argument("--no-punct", action="store_true")
        sp.add_argument("--no-songs", action="store_true")
        sp.add_argument("--formats", nargs="+",
                        default=["srt", "lrc", "ass", "vtt", "json"])

    g = sub.add_parser("generate", help="生成字幕")
    g.add_argument("input")
    g.add_argument("--ref", help="人工校对样本（生成后自动评测）")
    g.add_argument("--live-xml", help="直播弹幕 XML")
    g.add_argument("--profile", help="主播声纹名（启用单目标声纹过滤）")
    g.add_argument("--separate", action="store_true", help="人声分离")
    g.add_argument("--json", action="store_true", help="同时输出 JSON 报告")
    add_common(g)
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("evaluate", help="评测两份字幕")
    e.add_argument("--ref", required=True)
    e.add_argument("--hyp", required=True)
    e.add_argument("--out", help="报告输出目录")
    e.add_argument("--title", default="评测报告")
    e.add_argument("--json", action="store_true")
    e.add_argument("--max-cer", type=float, default=1.0,
                   help="CER 超过该值时返回非零退出码（CI 用）")
    e.set_defaults(func=cmd_evaluate)

    b = sub.add_parser("batch", help="批量处理目录")
    b.add_argument("batch_dir")
    b.add_argument("--out", default="outputs")
    add_common(b)
    b.set_defaults(func=cmd_batch)

    i = sub.add_parser("info", help="显示环境与默认配置")
    i.set_defaults(func=cmd_info)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
