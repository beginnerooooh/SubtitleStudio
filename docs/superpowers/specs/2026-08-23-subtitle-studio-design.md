# Subtitle Studio 设计文档

日期：2026-08-23
状态：已确认（用户逐节审批通过）

## 1. 目标

一个本地运行的 Gradio WebUI 工具，实现"剪映"式字幕生成：

- **盲识别模式**：无参考文本，faster-whisper VAD 分段听写，输出词级时间戳
- **强制对齐模式**：用户粘贴歌词/台词，wav2vec2 CTC 逐字对齐，毫秒级时间戳
- **输出**：SRT（句级）、LRC（行级）、ASS（`\k` 逐字卡拉OK）
- **长音频**：支持最长 5 小时直播回放（流式分块、可取消、实时进度）

## 2. 已确认的关键决策

| 决策点 | 结论 | 理由 |
|---|---|---|
| GUI 框架 | Gradio | 上传/表格/进度/下载全是现成组件；用户推荐 |
| 盲识别引擎 | faster-whisper（CTranslate2） | 4x 速度、低显存、VAD 内置、MIT |
| 强制对齐 | 自研模块：wav2vec2 中文模型 + `torchaudio.functional.forced_align` | 依赖干净无版本冲突；中文词表即汉字，逐字精度最高；核心对齐算法是 torchaudio 官方 API |
| 人声分离 | Demucs（htdemucs，two_stems=vocals），可选开关 | 歌声识别前置增强 |
| 目标硬件 | CUDA/CPU 自动检测，自动选择量化策略 | 用户确认 |
| 内容语言 | 中文为主，英文文本自动切换对齐模型 | 用户确认 |
| 不用 WhisperX 整库 | 依赖版本锁死敏感，与 faster-whisper 共存易冲突 | 方案对比结论 |
| 不基于现有开源项目改造 | AutoKaraoke（最接近）是 PyQt6 beta 无分离；Whisper-WebUI 无对齐无卡拉OK | 开源调研结论（2026-08-23） |

## 3. 项目结构

```
（仓库根 = 项目根）
├── app.py                  # Gradio 入口：UI 布局 + 事件绑定
├── requirements.txt        # 依赖 + CUDA/FFmpeg/HF镜像 安装注释
├── core/
│   ├── __init__.py
│   ├── env.py              # 环境检测：CUDA、FFmpeg、显存 → 量化策略
│   ├── audio.py            # FFmpeg 抽取/标准化 16kHz 16bit mono WAV
│   ├── separator.py        # Demucs 人声分离（可选）
│   ├── transcriber.py      # faster-whisper 盲识别
│   ├── aligner.py          # wav2vec2 CTC 强制对齐（核心新代码）
│   ├── pipeline.py         # 流水线编排 + 进度回调 + 取消
│   └── subtitle.py         # SRT / LRC / ASS 生成器（纯函数）
└── tests/
    ├── test_subtitle.py    # 字幕格式生成（无模型依赖）
    ├── test_text.py        # 文本标准化/分字分词（无模型依赖）
    └── test_pipeline_smoke.py  # 集成冒烟（标记 slow，需真实模型）
```

模块边界：`pipeline.py` 只做编排不碰算法；core 各模块可独立使用；模型全部懒加载 + 进程内缓存（换文件不重载模型）。

## 4. 数据模型（统一中间表示）

```python
@dataclass
class SubtitleWord:
    text: str      # 单个汉字或英文单词
    start: float   # 秒
    end: float

@dataclass
class SubtitleLine:
    words: list[SubtitleWord]
    @property start / end   # 首尾字时间推导
```

三种导出格式全部从 `SubtitleLine[]` 生成，不重复计算时间。

## 5. 流水线设计

### 5.1 全局进度状态机

```
抽取音频(5%) → 人声分离(25%, 可选) → 识别/对齐(60%) → 聚合导出(10%)
```

- 各模块通过 `on_progress(stage, ratio, message)` 回调上报
- `pipeline.py` 汇总为全局进度推送给 UI
- 每个阶段/分块处理前检查 `threading.Event` 取消标志 → 安全退出

### 5.2 env.py — 环境探测

- FFmpeg：`shutil.which("ffmpeg")` + Windows 常见路径（`C:\ffmpeg\bin` 等）回退；启动时检测缓存；缺失 → UI 红条提示 + 禁用执行按钮
- 量化策略表（自动，UI 可覆盖）：
  - GPU 显存 ≥ 8GB → `float16`
  - GPU 显存 < 8GB → `int8_float16`
  - CPU → `int8`
- 警告规则：CPU + 时长 > 30 分钟 + 开启人声分离 → UI 提示预计耗时极长

### 5.3 audio.py — FFmpeg 抽取

- `ffprobe` 探测时长/编码
- `ffmpeg -i <in> -ac 1 -ar 16000 -sample_fmt s16 <out.wav>`；subprocess 超时保护；非零退出抛 `AudioProcessError`（附 stderr 尾部 500 字符）
- 输出到 `outputs/<文件名>/<时间戳>/`，任务结束不自动删除
- **长音频约束：任何模块不得整段加载音频做变换**；WAV 落盘后用 `soundfile.blocks()` 流式分块读取（5h ≈ 1.1GB float32）

### 5.4 separator.py — Demucs 分离

- `demucs.api.Separator`，模型 `htdemucs`，`two_stems="vocals"`（只出人声干声，比四轨省一半时间）
- 设备跟随全局策略（GPU/CPU）
- 失败（OOM/依赖缺失）→ 警告日志 + 回退原始音频继续（不中断任务）

### 5.5 transcriber.py — 盲识别

- `WhisperModel(model_size, device, compute_type)` 懒加载 + 缓存
- `transcribe(vad_filter=True, word_timestamps=True, beam_size=5)`
- segments 生成器逐段消费，按「已处理音频时长 / 总时长」实时上报进度
- OOM 自动降档重试链：`float16 → int8_float16 → int8 → CPU int8`，每次降档记日志
- 语言：`language="zh"` 默认（UI 可改 auto/zh/en）

### 5.6 aligner.py — 强制对齐（核心）

文本标准化：
- 全半角统一（全角→半角）、剥离空行与内嵌 `[00:12.34]` 时间戳、剥离常见歌词标记行（[ti:] [ar:] 等）
- 切分：中文按**字**、英文/数字按**词**（连续拉丁字母串为一个 token），保留原文映射（导出时还原空格与原文）

分块对齐算法：
1. Silero VAD 找语音区间（去静音/前奏/间奏）
2. 歌词行按字数比例分配到各语音区间（比例假设：语速均匀）
3. 每个区间内：wav2vec2 前向得 log_probs → `torchaudio.functional.forced_align(log_probs, targets)` → 字级帧索引 → ×0.02s（wav2vec2 stride 320 @16kHz）→ 秒级时间戳（毫秒精度，与 `SubtitleWord` 单位一致）
4. 每区间输出对齐置信度（空白/重复 token 占比），低置信 → 日志标记「建议人工复核」

声学模型：
- 中文（CJK 占比 > 30%）：`jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`（词表即汉字）
- 英文：`jonatasgrosman/wav2vec2-large-xlsr-53-english`（按文本语言启发式自动切换）

### 5.7 subtitle.py — 导出（纯函数）

- SRT：序号 + `HH:MM:SS,mmm --> HH:MM:SS,mmm`；聚合规则：标点句末 / 最大时长 5s / 最大 25 字符
- LRC：`[mm:ss.xx]` + `[ti:][re:]` 元标签
- ASS：完整 header（PlayResX/Y、Default 样式 + Karaoke 高亮样式）；`{\k<n>}` 逐字（n = 厘秒）；末字自动延音至行尾；行文本 = words 用原文连接还原

## 6. GUI 设计（Gradio）

```
┌─────────────────────────────────────────────┐
│ 标题 + 环境状态徽章（设备·量化·FFmpeg 状态）     │
├─────────────────────────────────────────────┤
│ 文件上传（mp4/mkv/mov/mp3/wav/flac/m4a/aac/ogg/webm）│
│ 歌词文本框 + [清空] [填入示例]                 │
│ （留空=盲识别，有内容=强制对齐，动态提示模式）   │
├─────────────────────────────────────────────┤
│ 设备下拉(自动/CUDA/CPU) · 模型大小(tiny~large-v3) │
│ 伴奏分离开关 · 语言(自动/中/英) · 导出格式多选    │
├─────────────────────────────────────────────┤
│ [开始生成] [取消] 进度条 + 阶段文字             │
│ 实时日志面板（滚动 append）                    │
│ 预览表格（行号/起止/文本，前 500 行 + 提示）     │
│ 下载区（每个勾选格式一个文件按钮）               │
└─────────────────────────────────────────────┘
```

## 7. 线程模型

- 「开始生成」→ `threading.Thread(daemon=False)` 启动 pipeline
- Gradio 事件函数为 generator：每 0.5s 从 `queue.Queue` 拉取最新进度/日志/状态 `yield` 刷回 UI
- 取消按钮只置位 `threading.Event`，pipeline 在分块边界检查并安全退出（模型缓存保留）
- 单任务串行；运行中禁用开始按钮（防并发重入）

## 8. 异常处理清单

| 异常 | 处理 |
|---|---|
| FFmpeg 缺失/超时/非零退出 | 启动预检 + `AudioProcessError` 带 stderr 摘要 → UI 红字 |
| CUDA OOM | 量化降档重试链（5.5 节），全程记日志 |
| Demucs 失败/依赖缺失 | 警告 + 回退原始音频继续 |
| 歌词全空/仅标点 | 前端 + 后端双重校验，自动转盲识别并提示 |
| 对齐低置信区间 | 日志标记「建议人工复核」，不中断 |
| 模型下载失败 | 报错提示检查网络 / HF_ENDPOINT 镜像 |
| 不支持格式 | 白名单校验 |
| 未知异常 | 兜底捕获 → 日志完整 traceback → UI 友好摘要 |

## 9. 测试策略

- `test_subtitle.py`：SRT/LRC/ASS 纯逻辑（时间格式化、ms→厘秒换算、延音、聚合规则、原文还原）
- `test_text.py`：标准化/分字分词/内嵌时间戳剥离/中英混合切分
- `test_pipeline_smoke.py`：`@pytest.mark.slow`，需真实模型 + 样例音频，本地手动
- 手动验收：短视频（盲识别）+ 一首歌（对齐）各跑通全流程，含 LRC/ASS 输出在播放器验证

## 10. 依赖与环境

requirements.txt（注释含安装指引）：

```
gradio
faster-whisper
transformers
torch / torchaudio（CUDA 版经 --index-url 安装，注释说明）
demucs
soundfile
numpy
```

环境要求：Python ≥ 3.10；FFmpeg 系统安装（Windows 配 PATH）；NVIDIA GPU 可选（自动检测）；HF 镜像可选（HF_ENDPOINT）。

## 11. 范围外（未来可做）

- 说话人分离（diarization）
- 歌词自动抓取（LRCLIB/Genius）
- 字幕内嵌烧录（视频合成）
- 任务队列/多文件批处理
- 对齐后的人工校准波形编辑器
