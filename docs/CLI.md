# Subtitle Studio CLI 使用手册

命令行版的目标很明确：**一条命令出字幕，直接拖进剪辑软件**。

不需要打开界面，适合批处理、脚本化和服务器环境。

---

## 快速开始

```bash
# 最简单：出 SRT
python tools/studio_cli.py 录制.flv

# 指定输出目录和格式
python tools/studio_cli.py 录制.flv -o ./subs --format srt,ass

# 只处理前 10 分钟（先验证效果再跑全片）
python tools/studio_cli.py 录制.flv --start 0 --duration 600

# 用配置文件（推荐）
python tools/studio_cli.py 录制.flv --config my.toml
```

输出形如：

```
[100%] 全部完成
完成：28 行字幕，耗时 114.9s
输出目录：outputs/录制/20261021-124706
  srt    outputs/录制/20261021-124706/录制.srt
  ass    outputs/录制/20261021-124706/录制.ass
```

生成的 `.srt` / `.ass` 可直接拖进：

| 软件 | 用法 |
|---|---|
| 剪映 / CapCut | 时间轴 → 导入字幕 |
| Premiere Pro | 文件 → 导入 → 选 SRT |
| DaVinci Resolve | 文件 → 导入 → 字幕 |
| Final Cut Pro | 导入 → 字幕文件 |
| Aegisub | 直接打开 ASS（可继续精修）|

---

## 配置文件

把常用参数写进 `config.toml`，命令行只覆盖差异。

### 生成模板

```bash
python tools/studio_cli.py --init-config my.toml
```

模板里每个参数都有注释说明（见 `config.example.toml`）。

### 优先级

```
内置默认值  <  配置文件  <  命令行参数
```

### 配置分节一览

配置文件分为 8 个分节，与命令行参数一一对应：

| 分节 | 作用 | 关键项 |
|---|---|---|
| `[general]` | 输出目录、设备、精度 | `output_dir`、`device`、`compute_type` |
| `[asr]` | 识别引擎与解码参数 | `engine`、`model`、`chunk_seconds`、`beam_size` |
| `[segment]` | 断句规则 | `target_chars`、`max_chars`、`use_punctuation` |
| `[timing]` | 时间轴校正 | `refine` |
| `[song]` | 歌曲识别与歌词校正 | `enabled`、`manual_titles`、`lyric_correct` |
| `[voiceprint]` | 声纹过滤与 BGM 策略 | `enabled`、`profile_name`、`bgm_policy` |
| `[keywords]` | 关键词字库目录 | `dir` |
| `[output]` | 导出格式 | `formats` |

各分节内的每个参数在模板 `config.example.toml` 里都有逐行注释。

### 查看当前生效的全部参数

```bash
python tools/studio_cli.py --print-config --config my.toml
```

排查「为什么效果和预期不一样」时，这条命令最有用——它会把
默认值、配置文件、命令行三者合并后的**最终值**全部打印出来。

---

## 参数详解

### 基本参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `input` | — | 输入视频/音频文件。支持 mp4/mkv/flv/mov/ts/avi/wmv/mp3/wav 等 |
| `-o, --output-dir` | `outputs` | 输出目录。每跑一次会在其下建「文件名/时间戳」子目录，不会互相覆盖 |
| `-c, --config` | — | 配置文件路径（TOML） |
| `-f, --format` | `srt` | 导出格式，逗号分隔：`srt,ass,vtt,lrc,json` |
| `-n, --name` | 输入文件名 | 输出文件名（不含扩展名） |
| `-t, --title` | 输入文件名 | 标题，写入元数据 |
| `--reference` | — | 参考字幕，用于评测准确率 |
| `--danmaku` | — | 弹幕 XML（B 站录播）。提供热词，提升专名识别 |

**格式怎么选**：

- `srt` —— 通用性最好，几乎所有软件都认。**优先选这个**
- `ass` —— 带样式（字号/颜色/位置），可逐字卡拉OK。适合直接压制
- `vtt` —— 网页播放器用
- `lrc` —— 音乐播放器用（带时间轴）
- `json` —— 程序处理用（含词级时间戳）

可以同时要多个：`--format srt,ass`

### 识别参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--engine` | `whisper` | `whisper`（通用，多语言）/ `funasr`（中文优化） |
| `--model` | `large-v3` | `large-v3` 最准（需 4GB+ 显存）/ `medium` / `small` |
| `--chunk` | `120` | 分块长度（秒）。**不要超过 300** |
| `--beam` | `1` | 束搜索宽度。1 = 贪心解码 |
| `--language` | `auto` | 语言：`auto` / `zh` / `en` / `ja` … |
| `--hotwords` | — | 热词，空格分隔。如 `--hotwords "脆脆鲨 舰长"` |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |

**关于 `--chunk`**：whisper 在过长的块上会**静默跳过内容**（不报错，
但字幕会缺段）。实测同一素材：

| 分块 | 字错误率 |
|---|---|
| 1200 秒 | 0.72（严重漏识别）|
| 120 秒 | 0.34 |

显存紧张可降到 60，但不要超过 300。

**关于 `--beam`**：实测在长音频直播上 `beam=1` **又快又准**——
所有指标都优于 `beam=5`。原因是大 beam 更容易陷入重复循环与幻听。
一般不要改。

### 断句参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--target-chars` | `11` | 断句目标字数。11 字接近真人字幕的呼吸节奏 |
| `--max-chars` | `17` | 单行最大字数，超过强制断开 |
| `--no-punct` | 关 | 关闭标点辅助断句（会少加载一个模型，快一点） |
| `--no-refine` | 关 | 关闭时间轴校正 |

**关于 `--no-refine`**：时间轴校正用互相关估计整体偏移。关掉能省
1~2 秒，但长时间录音容易出现整体偏早/偏晚（几秒级）。除非时间紧，
建议保持开启。

### 歌曲与声纹

| 参数 | 默认 | 说明 |
|---|---|---|
| `--songs` | 关 | 启用歌曲识别（识别演唱段、对齐歌词） |
| `--manual-titles` | — | 本场演唱曲目，逗号分隔。**强烈建议填** |
| `--lyric-correct` | 关 | 用歌词库校正识别文本 |
| `--voiceprint` | 关 | 启用声纹过滤（需先注册声纹） |
| `--profile` | — | 声纹名称（会自动启用声纹过滤） |
| `--bgm-policy` | `mark` | `mark` 只标注 / `filter` 剔除 / `off` 关闭 |

**关于 `--manual-titles`**：直播里 BGM 与主播跟唱极难区分。手工录入
本场唱了哪些歌，能显著减少误判：

```bash
--manual-titles "情歌,落榜"
```

**关于 `--bgm-policy`**：默认 `mark`（只标注不删除）是**安全**选择。
`filter` 会直接删掉判定为 BGM 的行，但实测准确率只有 0.76，可能误删
主播自己唱的内容。除非你确认素材很干净，否则不要用 `filter`。

### 范围参数（快速验证）

| 参数 | 说明 |
|---|---|
| `--start` | 只处理从该秒开始的片段 |
| `--duration` | 只处理这么长（秒） |

```bash
# 只看第 10~11 分钟的效果
python tools/studio_cli.py 录制.flv --start 600 --duration 60
```

**注意**：这两个参数会用 ffmpeg 先切出临时文件，因此**导出的时间戳
从 0 开始**，不是原片的时间。要拿到原片时间轴上的字幕，请去掉这两个
参数跑全片。

### 其它

| 参数 | 说明 |
|---|---|
| `--json` | 末尾输出 JSON 摘要（便于脚本解析） |
| `-v, --verbose` | 显示详细日志（排查问题时用） |
| `--info` | 显示环境信息（FFmpeg、GPU、声纹、歌词库）后退出 |
| `--print-config` | 打印生效配置后退出 |
| `--init-config PATH` | 生成带注释的配置模板后退出 |
| `--force` | 配合 `--init-config`，允许覆盖已存在文件 |

---

## 声纹注册

声纹用于把主播人声与 BGM/其他人分开。**需要干净的人声样本**。

推荐流程：先用 [UVR5](https://github.com/Anjok07/ultimatevocalremovergui)
（Ultimate Vocal Remover）把素材分离成干声，再用 CLI 注册：

```bash
python tools/studio_cli.py --voiceprint-enroll 干声.wav --voiceprint-name 主播
```

然后识别时启用：

```bash
python tools/studio_cli.py 录制.flv --profile 主播
```

> **为什么需要先分离**：内置的人声分离已移除（不同机器上模型/依赖
> 差异导致调用不稳定）。UVR5 的分离质量更好，而且你能亲眼确认分离
> 结果是否干净。

注册后的声纹存于 `profiles/*.npy`，可在 `--info` 里查看。

---

## 常见问题

### 字幕缺失大段内容

检查 `--chunk` 是否过大。whisper 在长块上会**静默跳过**内容而不报错。
用默认 120 秒。

### 人名/术语总是识别错

两个办法一起用：

1. `--hotwords "脆脆鲨 舰长"` —— 引导模型输出正确写法
2. 关键词字库（`keywords/keywords.json`）—— 生成后再纠正一次

### 时间轴整体偏移几秒

确认没有加 `--no-refine`。时间轴校正会估计系统性偏移并补正。

### 输出目录找不到

每跑一次会在 `--output-dir` 下建「文件名/时间戳」子目录。加
`--json` 后可从 JSON 里的 `out_dir` 精确拿到路径。

### 显存不足 / OOM

按顺序尝试：

1. `--chunk 60`（减小分块）
2. `--model medium`（换小模型）
3. `--device cpu`（用 CPU，慢但稳）

---

## 退出码

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 1 | 参数错误 |
| 2 | 输入文件不存在 |
| 3 | 处理失败（识别/导出） |
| 130 | 用户中断（Ctrl+C） |

便于脚本判断：

```bash
python tools/studio_cli.py 录制.flv -o ./subs
if [ $? -ne 0 ]; then echo "失败"; fi
```

---

## 批量处理

CLI 本身不含批量命令，用 shell 循环即可（每次都独立进程，显存会释放）：

```bash
# Windows PowerShell
Get-ChildItem *.flv | ForEach-Object {
    python tools/studio_cli.py $_.FullName -o ./subs --format srt
}

# bash
for f in *.flv; do
    python tools/studio_cli.py "$f" -o ./subs --format srt
done
```

这样跑的好处是每个文件独立进程，不会因为显存碎片累积而 OOM。
