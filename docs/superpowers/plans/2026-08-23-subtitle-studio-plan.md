# Subtitle Studio 实现计划

日期：2026-08-23
依据：[设计规格](../specs/2026-08-23-subtitle-studio-design.md)（含评审反馈修订版）
状态：待用户确认后执行

## 总体策略

- **顺序**：纯逻辑（TDD，CI 无 GPU 可跑）→ 环境/音频层 → 模型层（识别/分离/对齐）→ 编排 → UI → 集成验收
- **每阶段有独立验证手段，通过后才进入下一阶段**
- 模型相关逻辑的控制流（降档链、回退、进度、取消）全部用 monkeypatch 单测；真实模型只出现在 slow 冒烟与手动验收

## Phase 0 — 脚手架

| 任务 | 产出 |
|---|---|
| 目录与包初始化、gitignore | `core/__init__.py`、`.gitignore` |
| 依赖清单（注释含 CUDA `--index-url`、FFmpeg 安装、HF_ENDPOINT 镜像指引） | `requirements.txt` |

验证：`pip install -r requirements.txt` 成功；`python -c "import core"` 无报错。

## Phase 1 — 纯逻辑核心（TDD：先写测试再实现）

| 任务 | 文件 | 要点 |
|---|---|---|
| 数据模型 | `core/models.py` | `SubtitleWord`（text=显示形式含尾随标点, start, end）、`SubtitleLine`（words + start/end 属性） |
| 文本处理 | `core/text.py` | 全半角统一；内嵌 `[00:12.34]` 时间戳与 `[ti:]` 标记行剥离；中字/英词 token 化；标点剥离与回填 round-trip；CJK 占比判定 |
| 字幕生成 | `core/subtitle.py` | SRT（标点/5s/25 字聚合）、LRC、ASS（`\k` 厘秒 + 末字延音） |
| 单测 | `tests/test_text.py`、`tests/test_subtitle.py` | 标点 round-trip、中英混排、时间格式化、ms→cs、聚合边界、延音 |

验证：`pytest tests/test_text.py tests/test_subtitle.py` 全绿。

## Phase 2 — 环境与音频层

| 任务 | 文件 | 要点 |
|---|---|---|
| 环境探测 | `core/env.py` | `torch.cuda` + 显存查询；FFmpeg `shutil.which` + Windows 路径回退；量化策略表（≥8G→float16 / <8G→int8_float16 / CPU→int8）；结果缓存；CPU+长音频+分离警告 |
| 音频抽取 | `core/audio.py` | ffprobe 时长；ffmpeg → 16kHz/16bit/mono WAV；subprocess 超时；`AudioProcessError`（stderr 尾部 500 字符）；格式白名单 |
| 单测 | `tests/test_env.py`、`tests/test_audio.py` | monkeypatch `which`/`subprocess`；ffmpeg 生成静音 WAV 夹具跑真实转码 |

验证：`pytest tests/`；本地真实视频转码一次。

## Phase 3 — 盲识别

| 任务 | 文件 | 要点 |
|---|---|---|
| faster-whisper 封装 | `core/transcriber.py` | 懒加载 + 模型缓存；`vad_filter=True, word_timestamps=True, beam_size=5`；segments 逐段消费按已处理时长报进度；取消 Event 检查；OOM 降档链 float16→int8_float16→int8→CPU int8 |
| 单测 | `tests/test_transcriber.py` | monkeypatch `WhisperModel` 验证降档链触发顺序与进度回调 |

验证：单测全绿 + 本地 small 模型真实听写人工检查。

## Phase 4 — 人声分离

| 任务 | 文件 | 要点 |
|---|---|---|
| Demucs 封装 | `core/separator.py` | `htdemucs` + `two_stems="vocals"`；设备跟随全局；失败→警告+回退原始音频；完成后 `del` + `gc.collect()` + `torch.cuda.empty_cache()` |
| 单测 | `tests/test_separator.py` | mock `Separator` 验证回退路径与资源释放调用 |

验证：单测全绿 + 本地一首歌分离人声听感检查。

## Phase 5 — 强制对齐（核心）

| 任务 | 文件 | 要点 |
|---|---|---|
| 全局对齐（≤15min） | `core/aligner.py` | 32s 窗/30s 步进分块前向（保留中央帧去边界伪影）→ log_probs 全局拼接 → 一次 `forced_align` → 帧索引 ×0.02s → 按 `\n` 还原行 |
| 滑窗分块（>15min） | `core/aligner.py` | VAD 区间合并 ~10min 块（仅 VAD 边界切割）→ 行按字数↔语音时长比例分配 → ±2 行重叠缓冲 → 块内全局对齐 → 重复行取距边更远者 |
| 置信度 | `core/aligner.py` | blank/重复占比；低置信→「建议人工复核」日志 |
| 资源控制 | `core/aligner.py` | log_probs 落 CPU float32（15min 峰值 ~900MB）；长音频逐块处理即释放 |
| 单测 | `tests/test_aligner.py` | 行分配/重叠去歧义/置信度纯函数；前向与 forced_align monkeypatch |

验证：单测全绿 + 本地「一首歌 + 粘贴歌词」端到端，ASS 拖入播放器验证逐字高亮。

## Phase 6 — 流水线编排

| 任务 | 文件 | 要点 |
|---|---|---|
| 状态机 | `core/pipeline.py` | 抽取 5% → 分离 25%（可选）→ 识别/对齐 60% → 聚合导出 10%；on_progress 汇总；取消 Event 贯穿；异常映射为用户可读消息 |
| 单测 | `tests/test_pipeline.py` | mock 各阶段：进度汇总、中途取消、Demucs 回退继续、歌词空自动转盲识别 |

验证：`pytest tests/ -m "not slow"` 全绿。

## Phase 7 — Gradio UI

| 任务 | 要点 |
|---|---|
| `app.py` 布局 | 三段式：上传+歌词（清空/示例）；参数（设备/模型/分离/语言/格式多选）；执行+进度+日志+预览（前 500 行）+下载 |
| 线程与进度 | `threading.Thread` 跑 pipeline；generator 每 0.5s 从 `queue.Queue` yield 刷新；`gr.Progress` 阶段进度 |
| 并发防护 | 运行中禁用按钮 + `threading.Lock`（获取失败返回「任务进行中」） |
| 动态提示 | 歌词框空/非空 → 盲识别/强制对齐模式提示；环境徽章（设备·量化·FFmpeg） |

验证：`python app.py` 启动，UI 全组件手动走查。

## Phase 8 — 集成验收

| 验收路径 | 内容 |
|---|---|
| A 盲识别 | 短视频 → 听写 → SRT/LRC/ASS 三格式下载正常 |
| B 强制对齐 | 歌曲 + 粘贴歌词 → 逐字 ASS 在播放器卡拉OK高亮正确 |
| C 长音频 | ≥30min 音频 → 进度实时、中途取消生效、显存稳定 |
| D 回归 | `pytest tests/ -m "not slow"` 全绿；slow 冒烟本地通过 |

产出：`tests/test_pipeline_smoke.py`（`@pytest.mark.slow`）。

验收结果（2026-08-23）：
- A/B/D 通过：真实模型（faster-whisper tiny + wav2vec2-base-960h）端到端冒烟 3/3 绿；
  对齐词级时间戳与语音精确吻合，ASS `{\k}` 时长正确；回归 172 单测全绿。
- C 通过（单测级）：长音频分块/去重、取消传播、多格式导出均有覆盖；真实 ≥30min
  音频的手动验收留给用户在本机 GPU 环境执行。
- 冒烟过程发现并修复两处真实缺陷：wav2vec2 英文词表仅大写导致小写全部映射
  unk（`_vocab_id` 大写回退）；盲识别英文词间空格丢失（whisper 前导空格转移
  到前词末尾，CJK 侧不补空格）。

## 关键风险与对策

| 风险 | 对策 |
|---|---|
| wav2vec2 分块前向边界伪影 | 32s/30s 重叠窗保留中央帧；Phase 5 端到端验收为关卡 |
| demucs ↔ torch 版本冲突 | requirements 固定已知兼容组合；分离失败自动回退不影响主流程 |
| 5h 音频 CPU 场景过慢 | env.py 启动警告 + VAD 跳静音；文档注明预期 |
| forced_align 长序列内存 | 15min 阈值切分全局/分块路径；log_probs 落 CPU |
