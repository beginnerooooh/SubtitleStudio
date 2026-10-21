# Subtitle Studio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Electron](https://img.shields.io/badge/Desktop-Electron-9feaf9.svg)](https://www.electronjs.org/)

[中文](README.md) · **English**

A local-first subtitle generator and editor for livestreams and video. It turns recordings
into **ready-to-use** subtitles, then lets you proofread, split, and re-time them on a
timeline in a lightweight desktop app modelled on professional editing software.

**Highlights**

- **Fully local** — recognition, speaker verification, and lyric alignment run on your machine. No audio or video ever leaves it.
- **Built for livestreams** — handles the audio problems that livestreams actually have: singing, vocals inside BGM, and multi-person chatter.
- **Measurable** — compares output against a human-proofread reference line by line, and reports accuracy you can interpret.
- **Editable** — dual-track timeline plus video preview, with on-canvas subtitle positioning.

---

## Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Interface](#interface)
- [How It Works](#how-it-works)
- [Command Line](#command-line)
- [Backend API](#backend-api)
- [Configuration](#configuration)
- [Project Layout](#project-layout)
- [Development](#development)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Features

### Recognition

- **Multiple engines** — faster-whisper (large-v3 and others) / FunASR (paraformer-zh) / external HTTP service
- **Chunked decoding** — long audio is decoded in overlapping chunks so the model cannot silently skip whole passages
- **VAD disabled by default** — during livestreams, singing and soft onsets tend to be swallowed by VAD
- **Timeline correction** — cross-correlates the subtitle coverage curve with audio activity across the full recording
- **Smart line breaking** — dynamic programming over length, punctuation, and pause costs produces natural reading breaks
- **Hotword boosting** — frequent nicknames and terms extracted from chat XML are fed to the engine to reduce name errors

### Speaker Verification & BGM

- **Single-target verification** — answers only "is this the current streamer?", using both speech and singing voiceprints
- **Multi-factor BGM detection** — voiceprint similarity + instrumental presence + stereo side signal + loudness variance
- **Marks by default, never deletes** — window-level discrimination is limited, and aggressive deletion removes the
  streamer's own speech. So suspected original-vocal segments are listed with their evidence for you to confirm.

### Songs

Four-stage chain, increasing in cost, stopping at the first hit:

```
song requests in chat  →  local lyric library  →  online sources (NetEase / Kugou / LRCLIB)  →  audio fingerprinting
```

Online sources support **automatic fallback**: if the best match from one source has the wrong
artist, the next source is tried. A self-hosted lyric service can also be plugged in.

### Lyric Library & Text Correction

Recognition accuracy on **singing** is noticeably lower than on speech (sustained vowels,
instrumental interference). But song text is known, so there is a correction path independent
of recognition:

```
Lyric library (import .lrc / .txt / .srt, or fetch online by title + artist)
   ↓  order-aware similarity matching against the subtitles (limited skips allowed)
   ↓  high-confidence matches → replace recognised text with the canonical lyric
```

- **Lyrics inform line breaking too** — subtitles break at the lyric's own phrase boundaries,
  so line timing and length follow the song rather than being patched afterwards
- **Only replaces lines where the lyric is clearly more reliable** — the score must pass a
  threshold *and* the lyric must not be shorter than the recognised text, so extra recognised
  content is never discarded
- **Inner breaks use spaces** — when a line must be split internally, a space is inserted
  rather than introducing punctuation

### Chat-Based Correction

In livestreams, chat/gifts/SC frequently **repeat what the streamer just said or named**
(chat spam, song requests, thank-yous). These are time-aligned external text sources:

- **Time-weighted** — chat close to the subtitle carries more weight, since audience reactions
  are near-synchronous
- **Homophone fixes** — when recognition mishears a proper noun (e.g. 翠翠沙 for 脆脆鲨), the
  spelling that recurs in chat replaces it
- **Text only, never the timeline** — with a safety valve that abandons the whole pass if more
  than half the lines would change
- Parses **user IDs, nicknames, gift names, chat text, and SC messages**

### Manual Song List

You can enter every song performed in this stream:

- Drives lyric-library lookup directly instead of relying on fingerprinting
- Narrows the BGM decision space, **substantially reducing false "original vocal" flags**
- Serves as the lyric source for line breaking, so subtitles split at the right phrases

### Waiting Feedback

Every operation that takes time shows a **centred progress dialog**:

- Full-screen backdrop: clicks during the wait cannot trigger a second run
  (more reliable than disabling buttons one by one, which always misses an entry point)
- Shows the operation name, percentage, and elapsed time
- Appears only after 120 ms, so fast requests do not flicker
- One task at a time — a repeat trigger reports "the previous operation is still running"

Covers: recognition, export, lyric fetch/import, voiceprint enrolment, cache cleanup,
chat parsing, video transcoding, and project save/recover.

### Video Format Support

Chromium can only play its own container set (MP4 / WebM / Ogg). For **FLV, MKV, TS, AVI,
WMV, RMVB** and similar formats, the backend transcodes to a `faststart` MP4 and caches it:

- Prefers `-c copy` remux (roughly 0.1–2.4 s for a 20-second clip, lossless)
- Falls back to re-encoding when codecs are incompatible
- Cache is keyed by path + mtime + size, so reopening is instant

### Editing

- **Dual-track timeline** — video track (filmstrip thumbnails) + subtitle track
- **Drag to retime** — drag a clip to move it, drag either edge to change start or end (green = start, orange = end)
- **Marquee selection** — drag on empty space with the left button to select many subtitles, then act on them from the context menu
- **Timecode** — uniformly `HH:MM:SS:FF`, with the frame rate read from the video
- **Subtitle styling** — size / colour / outline / position (presets or free dragging)
- **Convenience** — autosave, undo/redo, drafts, crash recovery, cache cleanup

### Lyric Library & Keyword Correction

**Lyric library** (lyrics/): import .lrc / .txt / .srt, or fetch online by title + artist.

Matching scores the **best contiguous run** rather than the whole-file average —
this is the key design decision:

| Scenario | Whole-file average (old) | Local run (now) |
|---|---|---|
| 3 lyric lines alone | 0.836 OK | 0.836 OK |
| 3 lyric lines among 200 chat lines | **0.006 rejected** | **0.836 matched** |

A song typically occupies only a few minutes of a recording. Averaging across the
whole file dilutes the score below any sensible threshold and rejects the song
entirely — the root cause behind "lyric matching does not work". The matcher now
slides over the subtitles looking for a **contiguous run of hits** and scores only
that region.

**Three fixes found on real data** (1014 actual subtitle lines + two `.lrc` files):

| Symptom | Root cause | Fix |
|---|---|---|
| both songs scored 0.000 | one lyric line is usually split across 2–3 subtitle lines, so a line-for-line comparison only covers half of it | merge up to 3 subtitle lines before comparing (`_windows`) |
| only 8 hits for a song with 33 replaceable spots | the chorus repeats 2–3 times but each lyric line kept a single candidate, already consumed by the first pass | keep multiple candidates per line; lines whose text repeats are exempt from the ordering check |
| lyrics written onto the neighbouring line | merging deleted lines while iterating, shifting earlier (lower-index) matches | compute all replacements first, then rebuild the list in one pass |

An **actual-similarity guard** was added on top: a window may include
neighbouring subtitles, so if the old text and the new lyric are genuinely
dissimilar (< 0.45) the replacement is skipped. On the real data this produced
**zero suspicious replacements**.

**Keyword store** (keywords/): streamer nicknames, jargon, catchphrases.

- Used as recognition hints (initial_prompt) so the model gets names right from the start
- A post-pass then corrects output: alias replacement via correct|wrong1,wrong2,
  plus fuzzy whole-line correction
- Text only — the timeline is never touched

### Subtitle Styling & Video Burn-in

- **Style scope** — global by default; a single line can override it, or use
  "apply to all subtitles" to bake the current style into every line
- **Burn-in** — render subtitles into the picture (hard subs) with styling identical
  to the preview (size / colour / outline / position), via an ASS filter, preferring
  hardware encoding

### Evaluation

Compare generated output against a human-proofread reference:

- Line-level exact match, text match, and timing match rates
- Character error rate (CER) and character recall
- **Ceiling analysis** — quantifies why 100% is unattainable
- A list of boundary cases that still need human review

---

## Getting Started

### Requirements

| Dependency | Notes |
|---|---|
| Python 3.10+ | Verified on 3.12 |
| FFmpeg | Audio/video extraction and thumbnail generation |
| Node.js 18+ | **Desktop app only** (build time); not needed at runtime |
| NVIDIA GPU (optional) | 4 GB VRAM runs large-v3 at int8_float16; falls back to CPU without one |

### Installation

```bash
git clone https://github.com/beginnerooooh/SubtitleStudio.git
cd SubtitleStudio

# Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Verify the environment (reports GPU / FFmpeg / available engines)
python tools/cli.py info
```

> Models are downloaded automatically on first use. For a fully offline setup, place them
> in `models/` beforehand — see [Configuration](#configuration).

### Generate Your First Subtitle

```bash
# Generate subtitles from a video
python tools/cli.py generate path/to/video.mp4

# Generate and evaluate against a reference
python tools/cli.py generate path/to/video.mp4 --ref path/to/reference.srt

# Results land in outputs/<video name>/<timestamp>/ (srt / lrc / ass / vtt / json)
```

### Launch a GUI

**Desktop (recommended)**

```bash
cd desktop
npm install      # installs Electron on first run
npm start
```

The desktop app starts its own local backend — no need to install Python separately or
remember a port.

**Web (Gradio)**

```bash
python launcher.py          # opens http://127.0.0.1:7860 in your browser
python launcher.py --debug  # keeps the console and DEBUG logs
```

---

## Interface

### Desktop

The layout follows mainstream editing software:

```
┌─ Toolbar: open video · chat · subtitles · recognize · undo/redo · QC · export ──────┐
├──────────┬──────────────────────────────────────┬──────────────────────────────────┤
│ Left     │  Video preview                       │ Subtitle list                    │
│ ▸ Media  │  (subtitles overlaid live,           │  click to seek                   │
│ ▸ Library│   draggable in free mode)            │  double-click to edit text       │
│ ▸ ASR    │                                      │  context menu                    │
│ ▸ Style  │  ▶ 00:00:00:12 / 00:51:33:01 ══●═══  │                                  │
│ ▸ Output ├──────────────────────────────────────┤  150 lines                       │
│          │ split merge delete new│copy paste│fit │                                  │
│          │ ┌──────────────────────────────────┐ │                                  │
│          │ │video   ▓▓▒▒▓▓▒▒ (filmstrip)      │ │                                  │
│          │ │subtitle ███ ████ ██ ██████ ███    │ │                                  │
│          │ └──────────────────────────────────┘ │                                  │
├──────────┴──────────────────────────────────────┴──────────────────────────────────┤
│ Log drawer (collapsed by default): log / songs / BGM review                         │
└─ Status bar ────────────────────────────────────────────────────────────────────────┘
```

| Area | Capability |
|---|---|
| **Video preview** | Subtitles overlaid live; style changes apply instantly. In free mode you can drag the subtitle directly on the frame |
| **Dual-track timeline** | Video track (filmstrip, cached after first build) + subtitle track (each clip shows its text) |
| **Drag to retime** | Drag a clip to move it, drag either edge to change start or end; green = start, orange = end |
| **Marquee select** | Drag on empty space with the left button to select many subtitles; then right-click to act on all of them |
| **Timecode** | Uniform `HH:MM:SS:FF` throughout (frame rate read from the video) |
| **Visible range** | Constrained to 5 seconds – 10 minutes, with a logarithmic slider. "Fit subtitles" jumps to a readable density |
| **Resizable layout** | All four panes are draggable; stretching the timeline compresses the log first, so **preview height is preserved** |
| **State memory** | Pane visibility, section collapse state, and pane sizes are all restored automatically |
| **Export** | Choose formats and destination |

Default shortcuts: `Space` play · `Ctrl+Z`/`Ctrl+Shift+Z` undo/redo · `Ctrl+B` split ·
`Ctrl+M` merge · `Ctrl+C`/`Ctrl+V` copy/paste · `Ctrl+D` duplicate · `Delete` remove ·
`Ctrl+S` export · `←`/`→` seek 0.5 s. All are customisable in the Shortcuts dialog.

### Web (Gradio)

Five tabs: Generate / Proofread / Evaluate / Voiceprint & Lyrics / Settings. Useful when
Electron is unavailable or for remote access.

---

## How It Works

Processing pipeline (`app/core/pipeline2.py`):

```
audio/video input
  → audio extraction (16 kHz mono WAV)
  → vocal separation (optional)
  → speaker verification / BGM analysis (optional)
  → recognition (chunked decoding + exclusive-range dedup)
  → song detection and lyric alignment (optional)
  → line breaking (DP) + punctuation restoration
  → timeline correction
  → dedup / overlap repair / monotonicity
  → export + evaluation
```

**Key design decisions** — each backed by measurements; see [docs/EVALUATION.md](docs/EVALUATION.md):

- **VAD disabled** — Silero VAD swallows singing and soft onsets, losing whole passages
- **Chunking with overlap** — whisper silently skips content in long audio (26 s skipped, measured)
- **BGM detected, not deleted** — window-level accuracy is around 0.76, too low to delete automatically
- **Punctuation is a hint only** — it guides line breaking; final text comes from recognition

---

## Advanced CLI (`tools/cli.py`)

A lower-level CLI for batch jobs and evaluation pipelines:

```bash
# Generate subtitles
python tools/cli.py generate <video> [--ref <srt>] [--chunk 120] [--engine whisper]

# Evaluate two subtitle files (--max-cer sets a CI threshold; exits non-zero if exceeded)
python tools/cli.py evaluate --ref <srt> --hyp <srt> --json

# Batch-process a directory
python tools/cli.py batch <dir>

# Environment and defaults
python tools/cli.py info
```

For everyday "just give me subtitles" use, see **Command Line** below
(`tools/studio_cli.py`) — it has a config file and annotated parameters.

---

## Backend API

`app/desktop_server.py` exposes a local HTTP API built on the standard library, with no
extra dependencies. The desktop and web UIs share the same logic.

| Endpoint | Description |
|---|---|
| `GET /api/health` | Environment, available engines, and lyric sources |
| `GET/POST/DELETE /api/projects[...]` | Project CRUD, save, autosave, undo/redo, export |
| `POST /api/edit/<op>` | Stateless edits: split / merge / delete / insert / update / retime_edge / copy / paste / duplicate / qc |
| `POST /api/generate` · `GET /api/jobs/<id>` | Async generation jobs and progress polling |
| `POST /api/evaluate` | Evaluation |
| `GET/POST /api/lyrics/*` | Lyric source list, search, self-hosted endpoint |
| `GET /api/lyrics/library` | Lyric library listing |
| `POST /api/lyrics/library/{add,fetch,remove}` | Import / fetch / delete lyrics |
| `GET /api/lyrics/match` | Preview matching of subtitles against a lyric |
| `GET /api/media?path=` | Media stream (Range supported) |
| `GET /api/videoinfo?path=` | Frame rate and duration (for timecode) |
| `GET /api/thumbs?path=` | Timeline filmstrip thumbnails |
| `GET /api/danmaku?path=` | Chat summary (streamer / gifts / song requests / hotwords) |
| `GET/POST /api/cache/*` · `/api/drafts/clean` | Cache statistics and cleanup, draft cleanup |
| `GET/POST /api/profiles[...]` | Voiceprint list, enrolment, deletion |

---

## Configuration

Defaults live in `app/core/config.py` and can all be overridden by environment variables.
Common ones:

| Variable | Default | Description |
|---|---|---|
| `SS_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `SS_MODELS_DIR` | `models` | Model directory |
| `SS_CACHE_DIR` | `cache` | Regenerable intermediate files |
| `SS_OUTPUT_DIR` | `outputs` | Generated results |
| `SS_ASR_ENGINE` | `whisper` | Default engine |
| `SS_LYRIC_PROVIDERS` | `netease,kugou,lrclib` | Lyric source order |

Model layout:

```
models/
  ct2/<model>/            faster-whisper models (CTranslate2 format)
  hf/                     HuggingFace cache (ECAPA voiceprint, vocal separation)
  modelscope/             ModelScope cache (FunASR, timestamp prediction)
```

---

## Project Layout

```
app/
  app.py                 Web UI (Gradio)
  desktop_server.py      Desktop local API (stdlib HTTP)
  core/
    pipeline2.py         Pipeline orchestration
    providers/           Pluggable recognition backends (whisper / funasr / external)
    segmenter.py         Line-breaking dynamic programming
    punctuate.py         Punctuation restoration (line-breaking hint only)
    timing.py            Timeline correction
    bgm.py               Multi-factor BGM detection
    song.py              Song detection / multi-source lyric search / forced alignment
    streamer.py          Single-target voiceprint (speech + singing)
    lyrics_library.py    Lyric library (import/fetch) + subtitle similarity correction
    danmaku_correct.py   Chat-based correction (proper nouns / homophones)
    livexml.py           Chat / gift / SC / guard XML parsing
    store.py             Projects / drafts / undo / autosave / cache cleanup
    evaluate.py          Evaluation and boundary-case analysis
    editor.py            Edit operations (pure functions)
    subtitle.py          Multi-format export
    separator.py         Vocal separation (Demucs)
    audio.py             Audio extraction and probing
    models.py            Data models (SubtitleLine / Word)
    text.py              Text normalisation
    env.py               Environment detection (device / FFmpeg)
    config.py            Central configuration
desktop/                 Electron desktop app
  main.js                Main process (starts backend, window, menus)
  preload.js             Security bridge
  renderer/              UI (HTML / CSS / JS)
docs/EVALUATION.md       Evaluation report (root-cause analysis and reproduction)
tools/cli.py             Command line
tools/check_desktop_ids.py  Desktop DOM element ID consistency check
tests/test_core_v2.py    Unit tests (93 cases, no GPU required)
```

---

## Development

```bash
# Run the tests (no GPU required)
python -m pytest tests/ -q

# Check desktop element ID consistency (catches JS referencing removed nodes)
python tools/check_desktop_ids.py

# Start the backend for debugging the UI in a browser
python app/desktop_server.py
```

Conventions:

- Modules under `core/` favour **pure functions**, which keeps them testable and undoable
- Edit operations never mutate their input; they return a new list
- The desktop app handles UI and interaction only — business logic always goes through the backend API
- Before committing, make sure both `pytest` and `check_desktop_ids.py` pass

---

## Voiceprint Enrollment

Voiceprints separate the streamer's voice from BGM and other speakers.
Enrollment needs a **clean vocal sample**.

Recommended workflow: first separate the source with
[UVR5](https://github.com/Anjok07/ultimatevocalremovergui) (or similar),
then import the dry vocal here:

- **Import dry vocal (recommended)** — wav / mp3 / flac / m4a / aac / ogg,
  converted to 16 kHz mono automatically
- **Cut a range** — take a segment from the video (only suitable when the
  source already has clean vocals)

From the CLI:

`ash
python tools/studio_cli.py --voiceprint-enroll vocal.wav --voiceprint-name host
`

> **Why no built-in separation**: an earlier version embedded UVR5's MDX
> model, but it proved unreliable across machines (model/dependency
> differences caused inference failures) and the maintenance cost outweighed
> the benefit. UVR5 separates better and lets you confirm the result is
> clean. The tool now only does "dry vocal → voiceprint", which is narrower
> and more dependable.

## Command Line

For scripted or headless use:

`ash
python tools/studio_cli.py 录制.flv                      # simplest
python tools/studio_cli.py 录制.flv -o ./subs --format srt,ass
python tools/studio_cli.py 录制.flv --config my.toml
python tools/studio_cli.py 录制.flv --start 0 --duration 600
`

| File | Purpose |
|---|---|
| 	ools/studio_cli.py | CLI entry point |
| config.example.toml | Config template (every option annotated) |
| docs/CLI.md | **Full parameter reference** (defaults and tradeoffs) |

Helper commands:

`ash
python tools/studio_cli.py --init-config my.toml   # write a template
python tools/studio_cli.py --print-config --config my.toml  # effective values
python tools/studio_cli.py --info                  # environment info
`

Design stance: **the defaults are the best practice**. Parameters validated
by measurement (chunk=120, eam=1, …) are already the defaults, so you
get good results without tuning.
## Performance

Measured on a 72-minute 1080p60 livestream recording (RTX 3050 Laptop, 4 GB):

| Stage | Time | Notes |
|---|---|---|
| Audio extraction | ~5 s | FFmpeg demux, independent of video codec |
| **Recognition (ASR)** | **~6 min** | Over 85% of total; the only stage worth optimising |
| Timeline correction + export | ~2 s | Pure CPU, negligible |

**Optimisations in effect**

- **eam_size=1 (greedy decoding)** — measurably **both faster and more accurate**:
  ASR dropped from 445 s to 247 s while EXACT / TEXT / TIME / CER / recall **all improved**.
  On long livestream audio, beam search tends to fall into repetition loops and
  hallucinations (a larger beam prefers "plausible sentences" over faithful transcription).
- **chunk_seconds=120** — whisper silently skips content in long chunks. On the same
  material, CER was 0.721 with 1200 s chunks and 0.34 with 120 s chunks.
- **540p30 preview proxy** — transcoding 1080p60 takes 13 minutes and yields 910 MB for the
  full recording; 540p30 takes 7.4 minutes and 224 MB. Proofreading only needs to see the
  picture and mouth shapes.
- **Hardware acceleration** — prefers h264_nvenc encoding with -hwaccel cuda decoding,
  falling back to CPU encoding (libx264 veryfast) when no NVIDIA GPU is present.
- **Cross-process transcode cache** — cache keys use a content digest (md5), **never
  hash()**. Python's hash() is salted per process, so every restart produced a different
  key (three independent runs gave three different keys) and re-transcoded from scratch —
  the root cause of "it takes half an hour every time".
- **Head segment released first** — for long recordings the first 90 seconds become playable
  in about 12 seconds while the full version transcodes in the background.
- **Timeline driven by source duration** — the proxy segment is only 90 seconds; rendering the
  timeline from it would misalign every subtitle in a full-length recording.

**Tried and rejected**

- **Parallel chunk decoding** — ThreadPoolExecutor(max_workers=2) was *slower*
  (72 s → 164 s). CTranslate2's GPU inference already saturates the device; concurrency
  only adds contention.
- **BatchedInferencePipeline** — requires VAD segments (conflicting with the core decision
  to keep VAD off) and returns timestamps relative to each clip, needing extra compensation
  logic. Uncertain gain against a concrete misalignment risk.

## Known Limitations

- **100% exact match is unattainable** — local ASR reaches roughly 73% character accuracy on
  real livestream audio. Even with perfect line breaking, the ceiling for fully matching lines
  is around 36%. Closing the gap further requires external text sources (lyric libraries, chat,
  or user-supplied transcripts).
- **BGM detection marks rather than deletes** — window-level voiceprint separation is limited,
  and automatic deletion removes the streamer's own content.
- **Song recognition needs network access** — online lyric sources require a connection; offline
  use is limited to the local lyric library.
- **Chat XML currently covers BililiveRecorder 2.x** — other platforms need additional parsers.
- **Slow without a GPU** — use the `medium` or `small` model instead.

---

## Acknowledgements

Built on the following open-source work:

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) / [CTranslate2](https://github.com/OpenNMT/CTranslate2)
- [FunASR](https://github.com/modelscope/FunASR) / [ModelScope](https://modelscope.cn/)
- [SpeechBrain](https://speechbrain.github.io/) (ECAPA-TDNN voiceprints)
- [Demucs](https://github.com/facebookresearch/demucs) (vocal separation)
- [Gradio](https://gradio.app/) · [Electron](https://www.electronjs.org/) · [FFmpeg](https://ffmpeg.org/)

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

---

## License

[MIT License](LICENSE)
