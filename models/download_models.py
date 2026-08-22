"""离线模型预下载：制作"纯离线完整版"安装包。

用法一（开发机打包前，由 build_portable.py --with-models 自动调用）：
    python models/download_models.py --preset full

用法二（用户手动补模型，便携目录内双击或命令行）：
    runtime\\python.exe models\\download_models.py --preset basic
    runtime\\python.exe models\\download_models.py --whisper small --aligner zh --demucs

下载位置由环境变量控制（launcher.py 已预热为安装目录 models/）：
    HF_HOME          → faster-whisper / ECAPA / wav2vec2（huggingface_hub）
    TORCH_HOME       → Demucs htdemucs（torch hub checkpoints）

中国大陆网络建议：set HF_ENDPOINT=https://hf-mirror.com
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

WHISPER_MODELS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}
VOICEPRINT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"     # 声纹 ECAPA-TDNN
ALIGNER_MODELS = {
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
}

# 预设：basic = 默认盲识别 + 中文对齐 + 声纹；full = 全模型（含英文对齐与 Demucs）
PRESETS = {
    "basic": {"whisper": ["small"], "voiceprint": True,
              "aligner": ["zh"], "demucs": False},
    "full": {"whisper": ["base", "small"], "voiceprint": True,
             "aligner": ["zh", "en"], "demucs": True},
}


def resolve_plan(preset: str | None = None,
                 whisper: list[str] | None = None,
                 aligner: list[str] | None = None,
                 voiceprint: bool | None = None,
                 demucs: bool | None = None) -> dict:
    """合并预设与显式参数（显式参数优先于预设），返回下载计划。

    返回 {"hf": [repo_id...], "demucs": bool}；hf 顺序稳定、去重。
    """
    plan = PRESETS.get(preset or "basic", PRESETS["basic"]).copy()
    if whisper:
        plan["whisper"] = whisper
    if aligner:
        plan["aligner"] = aligner
    if voiceprint is not None:
        plan["voiceprint"] = voiceprint
    if demucs is not None:
        plan["demucs"] = demucs

    repos: list[str] = []
    for size in plan["whisper"]:
        if size not in WHISPER_MODELS:
            raise SystemExit(f"未知的 whisper 规格：{size}（可选 {'/'.join(WHISPER_MODELS)}）")
        repos.append(WHISPER_MODELS[size])
    if plan["voiceprint"]:
        repos.append(VOICEPRINT_MODEL)
    for lang in plan["aligner"]:
        if lang not in ALIGNER_MODELS:
            raise SystemExit(f"未知的对齐语言：{lang}（可选 {'/'.join(ALIGNER_MODELS)}）")
        repos.append(ALIGNER_MODELS[lang])
    # 去重保序
    seen: set[str] = set()
    ordered = [r for r in repos if not (r in seen or seen.add(r))]
    return {"hf": ordered, "demucs": bool(plan["demucs"])}


def _ensure_cache_env(root: Path) -> None:
    """未由 launcher 预热时（独立运行），把缓存指到脚本同级目录。"""
    os.environ.setdefault("HF_HOME", str(root / "hf"))
    os.environ.setdefault("TORCH_HOME", str(root / "torch"))
    os.environ.setdefault("MODELSCOPE_CACHE", str(root / "modelscope"))
    for d in ("hf", "torch", "modelscope"):
        (root / d).mkdir(parents=True, exist_ok=True)


def download_hf_model(repo_id: str) -> str:
    """huggingface_hub 快照下载（受 HF_HOME / HF_ENDPOINT 控制）。"""
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id)
    return str(path)


def download_demucs_model() -> None:
    """Demucs htdemucs：实例化一次 Separator 触发下载（TORCH_HOME）。"""
    from demucs.api import Separator

    Separator(model="htdemucs", progress=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="download_models", description="Subtitle Studio 离线模型预下载")
    p.add_argument("--preset", choices=list(PRESETS), default=None,
                   help="模型预设（basic=盲识别+中文对齐+声纹；full=全量）")
    p.add_argument("--whisper", nargs="*", choices=list(WHISPER_MODELS),
                   help="faster-whisper 规格（覆盖预设）")
    p.add_argument("--aligner", nargs="*", choices=list(ALIGNER_MODELS),
                   help="wav2vec2 对齐模型语言（覆盖预设）")
    p.add_argument("--voiceprint", choices=["1", "0"], default=None,
                   help="是否下载 ECAPA 声纹模型（1/0，覆盖预设）")
    p.add_argument("--demucs", choices=["1", "0"], default=None,
                   help="是否下载 Demucs 人声分离模型（1/0，覆盖预设）")
    p.add_argument("--list", action="store_true", help="仅显示下载计划，不下载")
    args = p.parse_args(argv)

    plan = resolve_plan(
        preset=args.preset,
        whisper=args.whisper,
        aligner=args.aligner,
        voiceprint=None if args.voiceprint is None else args.voiceprint == "1",
        demucs=None if args.demucs is None else args.demucs == "1",
    )

    print("下载计划：")
    for repo in plan["hf"]:
        print(f"  [HF]  {repo}")
    print(f"  [Torch] demucs htdemucs: {'是' if plan['demucs'] else '否'}")
    if args.list:
        return 0

    root = Path(__file__).resolve().parent
    _ensure_cache_env(root)
    print(f"HF_HOME = {os.environ['HF_HOME']}")
    print(f"TORCH_HOME = {os.environ['TORCH_HOME']}")

    failures: list[str] = []
    for repo in plan["hf"]:
        print(f"\n>>> 下载 {repo} …")
        try:
            path = download_hf_model(repo)
            print(f"    完成：{path}")
        except Exception as exc:
            print(f"    失败：{exc}", file=sys.stderr)
            failures.append(repo)

    if plan["demucs"]:
        print("\n>>> 下载 Demucs htdemucs …")
        try:
            download_demucs_model()
            print("    完成")
        except Exception as exc:
            print(f"    失败：{exc}", file=sys.stderr)
            failures.append("demucs/htdemucs")

    if failures:
        print(f"\n{len(failures)} 个模型下载失败：{', '.join(failures)}", file=sys.stderr)
        print("建议：检查网络；国内设置 HF_ENDPOINT=https://hf-mirror.com 后重试", file=sys.stderr)
        return 1
    print("\n全部模型就绪，本目录现已支持纯离线运行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
