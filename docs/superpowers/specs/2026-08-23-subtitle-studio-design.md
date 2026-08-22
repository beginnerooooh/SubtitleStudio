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
| 对齐区间分配 | 短音频（≤15min）全局 CTC 对齐；长音频 VAD 分块 + ±2 行重叠滑窗 | 评审反馈：副歌长音下「语速均匀」线性分配会错位 |
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
│   ├── models.py           # SubtitleWord / SubtitleLine 数据模型
│   ├── text.py             # 歌词标准化/分字分词/标点剥离回填（纯逻辑，无重依赖）
│   ├── env.py              # 环境检测：CUDA、FFmpeg、显存 → 量化策略
│   ├── audio.py            # FFmpeg 抽取/标准化 16kHz 16bit mono WAV
│   ├── separator.py        # Demucs 人声分离（可选）
│   ├── transcriber.py      # faster-whisper 盲识别
│   ├── aligner.py          # wav2vec2 CTC 强制对齐（全局 + 滑窗分块策略）
│   ├── voiceprint.py       # 主声线声纹：ECAPA 提取/双音色 Profile/分段比对过滤
│   ├── song_recognizer.py  # 听歌识曲：shazamio 封装 + 演唱块合并 + 歌单时间戳导出
│   ├── pipeline.py         # 流水线编排 + 进度回调 + 取消
│   └── subtitle.py         # SRT / LRC / ASS 生成器（纯函数）
├── profiles/               # 主播声纹库（<主播名>.npy，双音色 embedding）
├── packaging/              # Windows 商用打包（详见 docs/PACKAGING.md）
│   ├── build_portable.py   # 一键构建便携目录（Embedded Python/依赖/FFmpeg/源码过滤）
│   ├── build.bat           # Windows 构建入口（参数透传）
│   ├── launcher.py         # 启动器：环境变量预热/端口健康检查/自动开浏览器/优雅退出
│   ├── run.bat / run-debug.bat / stop.bat   # 用户双击入口（静默/调试/停止）
│   ├── installer.iss       # Inno Setup 6 安装包脚本（中文向导/VC++检测/卸载保留数据）
│   ├── version.txt         # 版本号唯一来源
│   ├── app.ico             # 应用图标（多尺寸）
│   └── LICENSE.txt         # EULA 模板（发布前替换）
├── models/
│   └── download_models.py  # 离线模型预下载（basic/full 预设，制作纯离线完整版）
├── docs/
│   └── PACKAGING.md        # 打包实操指南（从克隆到 Setup.exe 的完整命令流程）
└── tests/
    ├── test_text.py        # 标准化/分字分词/标点回填（纯逻辑）
    ├── test_subtitle.py    # 字幕格式生成（纯逻辑）
    ├── test_env.py         # 环境探测（mock）
    ├── test_audio.py       # FFmpeg 转码（mock + ffmpeg 静音夹具）
    ├── test_transcriber.py # 降档链/进度回调/clip_timestamps（mock）
    ├── test_separator.py   # 失败回退/显存释放（mock）
    ├── test_aligner.py     # 行分配/重叠去歧义/置信度（mock 前向）
    ├── test_voiceprint.py  # Profile 存取/相似度比对/区域合并（mock 模型接缝）
    ├── test_song_recognizer.py # 块合并/时间戳格式化/识曲控制流（mock shazamio）
    ├── test_pipeline.py    # 编排状态机（mock 各阶段）
    ├── test_launcher.py    # 启动器：环境预热/端口探测/健康检查/停止信号
    ├── test_build_portable.py  # 构建脚本：源码过滤/._pth/FFmpeg 提取/瘦身/下载
    ├── test_download_models.py # 模型预下载：预设与显式参数合并
    └── test_pipeline_smoke.py  # 集成冒烟（slow，需真实模型）
```

模块边界：`pipeline.py` 只做编排不碰算法；core 各模块可独立使用；模型全部懒加载 + 进程内缓存（换文件不重载模型）。

**打包设计要点（零环境依赖）**：最终用户机器无需预装 Python/Git/CUDA Toolkit/FFmpeg。
Embedded Python + 全依赖内置于 `runtime/`；FFmpeg 静态二进制内置于 `bin/`，
由 launcher 进程内注入 `PATH`（`shutil.which` 直接命中，核心代码零改动）；
`HF_HOME`/`TORCH_HOME`/`MODELSCOPE_CACHE` 全部指向安装目录 `models/`，
不写 C 盘用户缓存；卸载时可选保留 outputs/profiles/models 用户数据。

## 4. 数据模型（统一中间表示）

```python
@dataclass
class SubtitleWord:
    text: str      # 显示形式：汉字/英文词 + 尾随标点（对齐用剥离形式）
    start: float   # 秒
    end: float

@dataclass
class SubtitleLine:
    words: list[SubtitleWord]
    @property start / end   # 首尾字时间推导
```

三种导出格式全部从 `SubtitleLine[]` 生成，不重复计算时间。

直播场景扩展数据模型（`core/voiceprint.py` / `core/song_recognizer.py`）：

```python
@dataclass
class VoiceProfile:          # 主播声纹（双音色注册集）
    name: str                # 主播名（即 profiles/<name>.npy 文件名）
    speak: np.ndarray|None   # 说话声线 embedding（L2 归一化，192 维）
    sing:  np.ndarray|None   # 唱歌声线 embedding（可选注册）

@dataclass
class SegmentVerdict:        # 单个 VAD 语音段的声纹判定
    start: float; end: float
    speak_sim: float|None    # 与说话声纹的余弦相似度
    sing_sim:  float|None    # 与唱歌声纹的余弦相似度
    best_sim   = max(两者)   # 双音色取最大：解决唱歌音色漂移误过滤
    is_singing = sing_sim > speak_sim  # 唱歌状态判定

@dataclass
class SongEntry:             # 歌单时间戳条目
    start: float; end: float
    title: str; artist: str
    confidence: float|None   # Shazam 返回时展示，缺省省略
```

## 5. 流水线设计

### 5.1 全局进度状态机

```
抽取音频(5%) → 人声分离(25%, 可选) → 识别/对齐(60%) → 聚合导出(10%)
```

直播场景（两项开关均关闭时与上图完全一致）：

```
抽取音频 → 人声分离(可选) → [语音分析：VAD分段 → 声纹过滤 → 唱歌段识曲] → 语音转写(仅保留区域) → 导出(字幕 + 歌单时间戳)
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
- 分离完成后显式释放：`del` 模型引用 → `gc.collect()` → `torch.cuda.empty_cache()`（Demucs 峰值 2~4GB，防止与后续 faster-whisper / wav2vec2 显存争抢）

### 5.5 transcriber.py — 盲识别

- `WhisperModel(model_size, device, compute_type)` 懒加载 + 缓存
- `transcribe(vad_filter=True, word_timestamps=True, beam_size=5)`
- segments 生成器逐段消费，按「已处理音频时长 / 总时长」实时上报进度
- OOM 自动降档重试链：`float16 → int8_float16 → int8 → CPU int8`，每次降档记日志
- 语言：`language="zh"` 默认（UI 可改 auto/zh/en）

### 5.6 aligner.py — 强制对齐（核心）

文本预处理（`core/text.py`，纯逻辑可单测）：
- 全半角统一（全角→半角）、剥离空行与内嵌 `[00:12.34]` 时间戳、剥离常见歌词标记行（[ti:] [ar:] 等）
- 切分：中文按**字**、英文/数字按**词**（连续拉丁字母串为一个 token）
- 标点：对齐 token 不含标点（声学模型词表无标点）；每 token 记录**显示形式**（词本体 + 尾随标点），`SubtitleWord.text` 存显示形式，标点随词回填，导出 SRT/ASS 时自然还原

对齐策略（按音频时长自动选择，阈值 15 分钟）：

**短音频（≤ 15 分钟）— 全局对齐**（默认路径，覆盖绝大多数歌曲）：
1. 完整人声分块过 wav2vec2：**32s 窗 / 30s 步进**，拼接时仅保留每窗中央 30s 帧（每帧 ≥1s 双侧上下文，消除分块边界伪影），log_probs 在全局时间轴拼接
2. 完整 token 序列与全局 log_probs 执行**一次** `torchaudio.functional.forced_align(log_probs, targets)`：CTC DP 全局最优、天然单调，无累计漂移；前奏/间奏/静音由 blank 吸收
3. token 全局帧索引 ×0.02s（stride 320 @16kHz）→ 秒级时间戳（毫秒精度）；按歌词原始行结构（`\n`）还原 `SubtitleLine`
4. 内存特征：log_probs 落 CPU float32，15 分钟峰值约 900MB（本地工具可接受）

**长音频（> 15 分钟）— VAD 分块 + 重叠滑窗行分配**（直播回放类）：
1. Silero VAD 找语音区间，相邻区间合并为对齐块（目标 ~10 分钟语音/块，只在 VAD 区间边界切割）
2. 歌词行按「字数 ↔ 语音时长」比例初步分配到块（仅作锚点，精度不依赖语速均匀假设）
3. 每块行范围向前后各扩 **±2 行重叠缓冲**，块内走上述全局对齐流程
4. 缓冲行在相邻块被对齐两次 → **保留距块边缘更远的结果**（滑窗去歧义），丢弃另一个；逐块处理即释放内存
5. token 挤压块边缘或置信度异常 → 日志警告「该边界区域建议人工复核」

置信度：全局/每块输出 blank 与重复 token 占比，低置信 → 日志标记「建议人工复核」。

声学模型：
- 中文（CJK 占比 > 30%）：`jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`（词表即汉字）
- 英文：`jonatasgrosman/wav2vec2-large-xlsr-53-english`（按文本语言启发式自动切换）

### 5.7 subtitle.py — 导出（纯函数）

- SRT：序号 + `HH:MM:SS,mmm --> HH:MM:SS,mmm`；聚合规则：标点句末 / 最大时长 5s / 最大 25 字符
- LRC：`[mm:ss.xx]` + `[ti:][re:]` 元标签
- ASS：完整 header（PlayResX/Y、Default 样式 + Karaoke 高亮样式）；`{\k<n>}` 逐字（n = 厘秒）；末字自动延音至行尾；行文本 = words 用原文连接还原

### 5.8 voiceprint.py — 主声线声纹过滤（直播场景）

**声学模型**：`speechbrain/spkrec-ecapa-voxceleb`（ECAPA-TDNN，192 维 embedding，
~80MB，CPU 提取 ~0.1x 实时；懒加载 + 进程内缓存）。备选 `3D-Speaker/CAM++`
（modelscope），当前版本未集成。

**双音色注册集（Enrollment Profiles）**：
- UI 录入两个参考音频（各 10~30s 清晰干声）：A=日常说话声线（必填）、B=唱歌/高音
  音色（可选）。上传样本先经 FFmpeg 转为 16kHz mono WAV（复用 `audio.extract_wav`），
  再过 ECAPA 提取 embedding，L2 归一化后持久化为 `profiles/<主播名>.npy`
  （`{"speak": ..., "sing": ...}` dict 容器，np.save 对象数组）
- 主播名校验：非空、无路径分隔符/非法字符、≤64 字符；样本近静音（峰值 < 1e-4）拒绝

**分段比对逻辑**（仅盲识别模式；对齐模式歌词即目标内容，不过滤）：
1. Silero VAD（复用 faster-whisper 内置 `get_speech_timestamps`）切出语音段
2. 每段音频**居中截取 ≤30s** 送 ECAPA（模型训练分布为短语句；唱歌长段截中段），
   按累计 ≤60s 分批前向控制内存
3. `Sim = max(Sim_speak, Sim_sing)` —— 唱歌音色整体漂移由唱歌声纹兜底，
   避免副歌高音区被误过滤
4. `Sim < threshold`（UI 可调 0.3~0.8，默认 0.55）→ 判为背景音/BGM 人声/其他人声，
   **转写前剔除**（不浪费 whisper 算力）
5. 保留段合并（间隔 ≤1s 合并、边界 ±0.25s padding、<0.5s 碎片丢弃）后作为
   faster-whisper `clip_timestamps` 定向转写，时间戳保持全局时间轴

**边界**：全部段落被过滤 → 空字幕 + 警告（不报错）；声纹过滤对
「音色相近的连麦嘉宾」区分度有限（ECAPA 等误识率场景），阈值建议从宽松
（0.45）起调。

### 5.9 song_recognizer.py — 唱歌检测与听歌识曲（直播场景）

**触发条件**（`enable_song_detect` 开启时）：
- 候选段 = 双音色 Profile 判定为唱歌（`is_singing`）的语音段；无唱歌声纹时
  退化为「持续 > 30s 的人声段」
- 相邻候选段间隔 ≤10s 合并为**演唱块**（VAD 间奏切分不碎歌）；块总时长 <30s 丢弃
- 每块从**原始输入文件**（含伴奏，非分离人声轨）FFmpeg 截取前 12s 高保真片段
  （44.1kHz stereo），经 `shazamio`（异步库，`asyncio.run` + 30s 超时包装为同步）
  识别 `title / subtitle(artist) / confidence`
- 识别成功的演唱块**从转写区域剔除**（歌声 whisper 转写只会产生幻觉歌词）；
  识别失败的块保留转写（尽力兜底）；相邻同名歌曲条目（间隔 ≤30s）合并为一条

**歌单导出**（与字幕同目录 `outputs/<任务名>/`）：
- `songs_timeline.md`：`- [HH:MM:SS - HH:MM:SS] 《title》- artist (置信度: xx%)`
- `songs_timeline.csv`：`start,end,title,artist,confidence`（csv 模块转义）
- 开关开启即生成（无歌曲时输出「未检测到歌曲」提示头）

**隐私与可用性**：识曲需联网访问 Shazam 服务（上传音频指纹）；网络不可达/无匹配
→ 日志提示并跳过，任务不中断。

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
│ ▸ 折叠区：主播声纹与歌单（直播场景，可选）        │
│   [x] 启用主声线声纹过滤  [x] 自动识曲生成歌单   │
│   主播 Profile 下拉(历史) + [刷新]              │
│   声纹匹配严格度滑块(0.3~0.8, 默认0.55)         │
│   新增主播：名称 + 说话样本 + 唱歌样本(可选)      │
│           + [提取并保存声纹] + 状态提示          │
├─────────────────────────────────────────────┤
│ [开始生成] [取消] 进度条 + 阶段文字             │
│ 实时日志面板（滚动 append）                    │
│ 预览表格（行号/起止/文本，前 500 行 + 提示）     │
│ 歌单时间戳面板（md 文本 + 一键复制 + 下载）      │
│ 下载区（每个勾选格式一个文件按钮 + 歌单两件）     │
└─────────────────────────────────────────────┘
```

## 7. 线程模型

- 「开始生成」→ `threading.Thread(daemon=False)` 启动 pipeline
- Gradio 事件函数为 generator：每 0.5s 从 `queue.Queue` 拉取最新进度/日志/状态 `yield` 刷回 UI
- 取消按钮只置位 `threading.Event`，pipeline 在分块边界检查并安全退出（模型缓存保留）
- 进度组件遵循 Gradio 5.x 规范：`gr.Progress`（阶段进度，支持 `track_tqdm`）+ generator `yield` 增量刷新日志/预览/下载组件
- 并发防护双保险：运行中禁用开始按钮 + `threading.Lock` 串行锁（获取失败立即返回「任务进行中」提示）

## 8. 异常处理清单

| 异常 | 处理 |
|---|---|
| FFmpeg 缺失/超时/非零退出 | 启动预检 + `AudioProcessError` 带 stderr 摘要 → UI 红字 |
| CUDA OOM | 量化降档重试链（5.5 节），全程记日志 |
| Demucs 失败/依赖缺失 | 警告 + 回退原始音频继续 |
| 歌词全空/仅标点 | 前端 + 后端双重校验，自动转盲识别并提示 |
| 对齐低置信区间 | 日志标记「建议人工复核」，不中断 |
| 声纹 Profile 缺失/损坏 | `VoiceprintError` → UI 明确提示先创建/重选 Profile |
| 全部语音段被声纹过滤 | 空字幕 + 警告，任务正常结束 |
| Shazam 网络不可达/超时/无匹配 | 日志提示 + 该段保留转写，任务不中断 |
| ECAPA 模型加载失败（speechbrain 未装） | `VoiceprintError` 明确指引安装 |
| 模型下载失败 | 报错提示检查网络 / HF_ENDPOINT 镜像 |
| 不支持格式 | 白名单校验 |
| 未知异常 | 兜底捕获 → 日志完整 traceback → UI 友好摘要 |

## 9. 测试策略

- `test_subtitle.py`：SRT/LRC/ASS 纯逻辑（时间格式化、ms→厘秒换算、延音、聚合规则、原文还原）
- `test_text.py`：标准化/分字分词/内嵌时间戳剥离/中英混合切分/**标点剥离与回填 round-trip**
- `test_aligner.py`：长音频行分配 + ±2 行重叠去歧义 + 置信度（纯逻辑部分，模型前向 monkeypatch）
- `test_env.py` / `test_audio.py` / `test_transcriber.py` / `test_separator.py` / `test_pipeline.py`：mock 依赖测控制流（降档链、回退、进度汇总、取消、声纹过滤区域传递、识曲剔除）
- `test_voiceprint.py`：Profile 存取 round-trip、余弦相似度/双音色取最大/唱歌判定、区域合并（padding/碎片丢弃）、名称与静音校验（模型接缝 monkeypatch）
- `test_song_recognizer.py`：演唱块合并（间奏不碎歌）、同名条目合并、md/csv 精确格式、识曲成功/失败/超时控制流（mock shazamio 与 FFmpeg 截取）
- `test_pipeline_smoke.py`：`@pytest.mark.slow`，需真实模型 + 样例音频，本地手动
- 手动验收：短视频（盲识别）+ 一首歌（对齐）+ 长音频（≥30min 进度/取消/显存）各跑通全流程，含 LRC/ASS 输出在播放器验证

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
speechbrain          # 声纹提取（ECAPA-TDNN，直播场景；懒加载，未装时明确报错）
shazamio             # 听歌识曲（直播场景；需联网访问 Shazam 服务）
```

环境要求：Python ≥ 3.10；FFmpeg 系统安装（Windows 配 PATH）；NVIDIA GPU 可选（自动检测）；HF 镜像可选（HF_ENDPOINT）。

## 11. 范围外（未来可做）

- 说话人分离（diarization）
- 歌词自动抓取（LRCLIB/Genius）
- 字幕内嵌烧录（视频合成）
- 任务队列/多文件批处理
- 对齐后的人工校准波形编辑器
