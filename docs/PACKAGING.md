# Subtitle Studio Windows 打包实操指南

从源码到最终 `SubtitleStudio_Setup_v1.0.0.exe` 与免安装便携版的完整流程。

## 0. 产物形态

| 形态 | 路径 | 说明 |
|---|---|---|
| 便携目录 | `dist/SubtitleStudio/` | 免安装、拷贝即用，可作为绿色版直接分发（zip） |
| 安装包 | `dist/SubtitleStudio_Setup_v1.0.0.exe` | Inno Setup 向导式安装（开始菜单/桌面快捷方式/卸载程序） |

最终用户机器**无需预装** Python、Git、CUDA Toolkit、FFmpeg——全部内置于安装目录：

```
SubtitleStudio/                 # 安装根目录
├── run.bat                     # 双击启动（黑窗自动隐藏）
├── run-debug.bat               # 调试启动（保留控制台 + 详细日志）
├── stop.bat                    # 优雅停止（写 stop.flag，释放 GPU 显存）
├── launcher.py                 # 启动器：环境变量预热/端口健康检查/拉起浏览器
├── version.txt                 # 版本与构建信息
├── app.ico                     # 应用图标
├── runtime/                    # Embedded Python 3.12 + 全部依赖（约 1.5~4GB）
├── app/                        # 项目源码（app.py + core/）
├── bin/                        # ffmpeg.exe / ffprobe.exe（静态构建）
├── models/                     # 模型缓存根（hf/ torch/ modelscope/ + download_models.py）
├── profiles/                   # 主播声纹库（用户数据）
├── outputs/                    # 字幕与歌单输出（用户数据）
└── logs/app.log                # 运行日志（10MB 轮转 x3）
```

## 1. 开发机准备（一次性）

在 **Windows 10/11 x64** 开发机上：

1. **Python 3.10+**（跑构建脚本本身）：`winget install Python.Python.3.12`
2. **Inno Setup 6.x**（编译安装包）：`winget install JRSoftware.InnoSetup`
3. 获取源码并进入目录：
   ```bat
   git clone <repo> && cd subtitle-studio
   ```

> 中国大陆网络建议同时设置：`HF_ENDPOINT=https://hf-mirror.com`（模型下载），
> 并使用下文的 `--python-mirror` / `--pip-mirror` 参数。

## 2. 版本号与图标

- **版本号**：编辑 `packaging/version.txt`（如 `1.0.0`）。构建脚本与安装包文件名
  均以它为唯一来源；用 `--version` 参数可临时覆盖。
- **图标**：`packaging/app.ico`（已内置多尺寸 256/128/64/48/32/16）。替换为自有
  品牌图标时保持同名即可；`installer.iss` 会自动检测其存在。
- **许可协议**：`packaging/LICENSE.txt` 为 EULA 模板，**商用发布前必须替换为
  经法务审核的正式协议**。

## 3. 一键构建便携目录

### 3.1 CPU 标准版（约 1.5GB）

```bat
packaging\build.bat --clean
```

等价于 `python packaging\build_portable.py --clean`。脚本自动完成：

1. 创建 `dist/SubtitleStudio/` 目录骨架
2. 下载 Python 3.12.8 **Embedded（embeddable zip）** → `runtime/`，改写
   `python312._pth` 启用 site-packages，get-pip 引导安装 pip
3. `pip install -r requirements.txt`（`--no-cache-dir`）
4. 下载 **BtbN 静态 FFmpeg**（ffmpeg.exe + ffprobe.exe）→ `bin/`
5. 白名单复制源码（`app.py`、`core/`、`requirements.txt`）→ `app/`，
   自动过滤 `__pycache__` / `.git` / `tests` / `docs` / `*.pyc` / `*.log` / `.env`
6. 写入 `version.txt` 构建清单

### 3.2 GPU 版（CUDA，约 4GB）

```bat
packaging\build.bat --clean --torch-index cu124 --slim
```

- `--torch-index cu124`：从 PyTorch 官方源安装 CUDA 12.4 轮子（另有 `cu121`、
  `cu118` 可选）。**不需要用户安装 CUDA Toolkit**——运行库 DLL 随 wheel 内置，
  仅要求用户装有 NVIDIA 显卡驱动。
- `--slim`：删除 pip/setuptools/Scripts 与全部 `__pycache__`，体积约减 200MB
  （代价：该便携目录无法再执行 pip install）。

### 3.3 常用变体

```bat
rem 使用本地已有 FFmpeg（避免重复下载）
packaging\build.bat --ffmpeg-dir D:\tools\ffmpeg\bin

rem 国内镜像加速
packaging\build.bat ^
  --python-mirror https://mirrors.huaweicloud.com/python ^
  --pip-mirror https://pypi.tuna.tsinghua.edu.cn/simple

rem 只更新源码（runtime/bin 已就绪，秒级增量）
packaging\build.bat --skip-deps
```

### 3.4 验证便携版

双击 `dist\SubtitleStudio\run.bat`：

- 黑色控制台闪现 <1 秒后自动隐藏
- 约 10~60 秒后系统默认浏览器自动打开 `http://127.0.0.1:7860`
- 退出：双击 `stop.bat`（或浏览器标签页保持不管，服务后台常驻）

排障：`run-debug.bat` 保留控制台，日志同步写入 `logs\app.log`。

## 4. 离线完整版（可选：预置模型）

默认构建不带模型（首次运行联网下载）。制作**纯离线完整版**：

```bat
packaging\build.bat --clean --with-models --preset full
```

或对已有便携目录手动补模型：

```bat
cd dist\SubtitleStudio
runtime\python.exe models\download_models.py --preset full
```

| 预设 | 内容 | 体积 |
|---|---|---|
| `basic` | whisper-small、ECAPA 声纹、wav2vec2 中文对齐 | ~2GB |
| `full` | whisper-base+small、ECAPA、wav2vec2 中英、Demucs | ~4.5GB |

模型落在 `models/hf/`（HF_HOME）与 `models/torch/`（TORCH_HOME），随安装目录
整体分发；用户机器无需任何网络。

## 5. 编译安装包（Setup.exe）

> **需要 Inno Setup 7+**（脚本使用 `SetupArchitecture=x64` 生成原生 64 位安装器，
> 产物为 PE32+ x64 可执行文件；Inno Setup 6 无法编译本脚本）。

```bat
packaging\build.bat --clean --installer
```

`--installer` 会在便携目录构建完成后自动调用 Inno Setup（自动探测
`ISCC.exe`；找不到时打印手动命令）。也可以单独编译：

```bat
"C:\Program Files\Inno Setup 7\ISCC.exe" /DMyAppVersion=1.0.0 packaging\installer.iss
```

产物：`dist\SubtitleStudio_Setup_v1.0.0.exe`（原生 x64，lzma2/ultra64 固实压缩）。

### 5.1 Linux 交叉构建（推荐：NSIS，无需 Wine）

`packaging/installer.nsi` 与 `installer.iss` 功能对等（中文向导、许可页、
Program Files 默认目录、开始菜单/桌面快捷方式、VC++ 2015-2022 x64 运行库
检测、控制面板卸载注册、卸载时询问是否保留用户数据）。区别仅在技术栈：
NSIS 的 `makensis` 是**原生 Linux 程序**，不依赖 wine——在 CI 沙箱
（常禁止执行 32 位 ELF，导致 wine 无法运行 Inno 的 ISCC）也能编译：

```bash
apt install -y nsis                       # Ubuntu 24.04 = NSIS 3.09
makensis -DMyAppVersion=1.2.0 packaging/installer.nsi
```

产物：`dist/SubtitleStudio_Setup_v1.2.0.exe`（LZMA 固实压缩，约 5~10 分钟）。
安装器向导流程与 Inno 版一致；对最终用户完全透明（7-Zip / Notepad++ 同款
安装器技术）。`!insertmacro MUI_PAGE_STARTMENU` 支持勾选「不创建开始菜单
文件夹」；桌面快捷方式为可选组件（默认不勾选），与 Inno 版 tasks 行为一致。

> 升级安装：NSIS 版写入 `HKLM\Software\Subtitle Studio\InstallLocation`，
> 重复安装时自动沿用旧目录（与 Inno AppId 的升级识别目的相同）。

### 5.2 Linux 交叉构建（备选：Wine + Inno Setup）

`build_portable.py` 内置 `--wine` 模式：在 Linux 上用 Wine 执行 Embedded Python
的 `pip install`（保证依赖的平台标记为 `win_amd64`），并用 Wine 中的 ISCC 编译
Setup.exe。已在 Wine 11.15（WoW64 构建）+ Inno Setup 7.0.2 x64 上验证全流程
（含静默安装与卸载冒烟测试）。

一次性准备（root 或有 sudo 的 Linux）：

```bash
# 1) 安装 Wine（推荐 11.x 的 wow64 共享目录构建，32/64 位程序都能跑）
#    本例解包到 /opt/wine-11.15-amd64-wow64；发行版自带的 winehq-stable 亦可
apt install -y xvfb unzip          # 虚拟显示器 + 解包工具

# 2) 创建专用 prefix（WOW64 模式：同时支持 x86 与 x64 PE）
export WINEPREFIX=/root/.wine1115 WINEARCH=wow64 DISPLAY=:99
Xvfb :99 &                          # Inno Setup 安装器需要 GUI
wineboot -u

# 3) 安装 Inno Setup 7 x64（官网 jrsoftware.org 下载 innosetup-7.x.x.exe）
#    安装到 C:\InnoSetup7（find_iscc() 自动探测；也支持 C:\Program Files\*）
#    注意：官方安装器本身是原生 x64 程序，可直接在 wine 下静默安装；
#    但 Inno6 的 32 位安装器/及 32 位 setup 引导器在部分 wow64 wine 下会段错误，
#    因此交叉构建必须用 Inno 7 的 x64 安装器 + SetupArchitecture=x64
wine innosetup-7.0.2.exe /DIR=C:\\InnoSetup7 /VERYSILENT /SUPPRESSMSGBOXES
```

构建命令（在仓库根目录）：

```bash
export PATH=/opt/wine-11.15-amd64-wow64/bin:$PATH
export WINEPREFIX=/root/.wine1115 WINEARCH=wow64 DISPLAY=:99 WINEDEBUG=-all

# 完整构建 + 安装包（首次约需 30~60 分钟，视网络与压缩耗时）
python3 packaging/build_portable.py --wine \
    --wine-bin  /opt/wine-11.15-amd64-wow64/bin/wine \
    --wine-prefix /root/.wine1115 \
    --python-version 3.12.10 \
    --python-mirror https://mirrors.huaweicloud.com/python \
    --pip-mirror  https://pypi.tuna.tsinghua.edu.cn/simple \
    --slim --installer

# runtime 已就绪时只重打安装包（秒级，跳过依赖安装）
python3 packaging/build_portable.py --wine \
    --wine-bin /opt/wine-11.15-amd64-wow64/bin/wine \
    --wine-prefix /root/.wine1115 \
    --python-version 3.12.10 --skip-deps --ffmpeg btbn --installer
```

产物同样是 `dist/SubtitleStudio_Setup_v1.0.0.exe`，与 Windows 原生构建完全等价。

**Wine 下的已知差异（仅影响交叉构建机，不影响最终用户）**：

| 现象 | 说明 |
|---|---|
| 32 位 ELF `exec format error`（内核禁用 ia32） | 多数 CI 沙箱禁执行 32 位 ELF，wine 无法加载 32 位 PE（`syswow64\ntdll.dll` c0000135）。解决：改用 5.1 的 NSIS 路线，或自建完整 wow64 wine 环境 |
| Inno 7 安装器提示「不支持当前 Windows 版本」 | wine 9 对 Win11 新 API 支持不全，即便注册表伪装 Build 26100 仍被拒。解决：升级 winehq-stable 11（注意 dl.winehq.org 走代理仅 ~20KB/s，122MB 需约 100 分钟），或改用 NSIS |
| `OMP: Error #179 GetNumaNodeProcessorMaskEx() failed` | Wine 未实现该 syscall，libiomp 初始化即崩；冒烟测试时 `export KMP_AFFINITY=disabled` 绕过。真实 Windows 10/11 无此问题 |
| `pip download --platform win_amd64` 平台标记错乱 | 这正是必须通过 Wine 里的 Windows Python 执行 `pip install` 的原因 |

**交叉构建冒烟测试**（可选，验证便携目录可用性）：

```bash
cd dist/SubtitleStudio
export KMP_AFFINITY=disabled HF_ENDPOINT=https://hf-mirror.com
wine runtime\\python.exe launcher.py --debug    # 等待 "WebUI 就绪"
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:7860/   # 期望 200
touch stop.flag                                   # 优雅停止，日志见 logs/app.log
```

### 安装包行为要点

- **向导流程**：欢迎页 → 许可协议（LICENSE.txt）→ 路径选择（默认
  `C:\Program Files\SubtitleStudio`）→ 开始菜单/桌面快捷方式勾选 → 安装 →
  「立即启动」勾选
- **权限**：`PrivilegesRequired=admin`（Program Files 标准位置）
- **VC++ 运行库检测**：安装收尾自动检查 VC++ 2015-2022 x64 Redistributable，
  缺失时弹窗给出微软官方下载地址；如需完全离线部署，将 `vc_redist.x64.exe`
  放入 `packaging\redist\` 并按 `installer.iss` 内注释取消两段注释
- **卸载**：标准 `unins000.exe`；卸载前自动优雅停止运行中的服务；
  卸载完成时**询问是否删除用户数据**（outputs / profiles / models），
  选「否」则保留，重装后直接续用

### Smoke 测试清单（发布前必做）

在一台**纯净无 Python** 的 Windows 机器上：

1. 安装 Setup.exe → 勾选桌面快捷方式 → 完成
2. 双击桌面图标 → 浏览器自动打开 WebUI
3. 上传一段 1 分钟 mp4 → 盲识别 → 下载 SRT
4. 开启声纹过滤 + 听歌识曲跑一段直播回放 → 检查 `outputs/` 歌单生成
5. 双击「开始菜单 → 停止 Subtitle Studio」→ 任务管理器确认 `python.exe`
   退出、显存释放（GPU 版）
6. 控制面板卸载 → 验证「保留数据」选项行为

## 6. 用户使用说明（可随安装包分发）

- **启动**：双击桌面「Subtitle Studio」图标。首次启动需加载模型（10~60 秒），
  就绪后浏览器自动打开；也可手动访问 `http://127.0.0.1:7860`
- **停止**：开始菜单 →「停止 Subtitle Studio」，或删除安装目录下 `stop.flag`
  以外的方式均不推荐（直接杀进程可能残留显存占用数十秒）
- **端口被占**：7860 被占用时自动改用 7861~7879，实际地址见 `logs\app.log`
- **数据位置**：输出字幕/歌单在 `outputs\`，主播声纹在 `profiles\`，
  模型缓存在 `models\`——全部在安装目录内，卸载勾选保留即可迁移

## 7. 常见问题

| 现象 | 原因与处理 |
|---|---|
| 启动后浏览器没打开 | 看 `logs\app.log`；端口就绪但 webbrowser 失败时手动访问日志中的 URL |
| 报「未找到 ffmpeg」 | `bin\` 目录缺失（被杀毒软件误删？）重新安装或从便携包复制 |
| GPU 版报 `Torch not compiled with CUDA` | 构建时未加 `--torch-index cu124`；或用户显卡驱动过旧（升级驱动） |
| torch DLL 加载失败 | 缺 VC++ 2015-2022 x64 运行库：https://aka.ms/vs/17/release/vc_redist.x64.exe |
| 模型下载慢/失败 | 设 `HF_ENDPOINT=https://hf-mirror.com` 后重跑 `models\download_models.py` |
| 杀毒软件误报 | Embedded Python + PyInstaller 类方案的常见误报；提交白名单或代码签名（商用建议购买 Authenticode 证书并在 Inno `SignTool` 配置签名） |

## 8. 文件清单

| 文件 | 职责 |
|---|---|
| `packaging/build_portable.py` | 一键构建：Embedded Python / 依赖 / FFmpeg / 源码过滤 / 模型预置 |
| `packaging/build.bat` | Windows 构建入口（参数透传） |
| `packaging/launcher.py` | 启动器：环境变量预热、端口健康检查、自动开浏览器、优雅退出 |
| `packaging/run.bat` | 用户双击入口（静默模式） |
| `packaging/run-debug.bat` | 调试入口（控制台 + DEBUG 日志） |
| `packaging/stop.bat` | 优雅停止（stop.flag 信号） |
| `packaging/installer.iss` | Inno Setup 7 安装包脚本（中文向导/VC++ 检测/卸载保留数据，Windows/wine 构建） |
| `packaging/installer.nsi` | NSIS 等价脚本（Linux 原生 makensis 交叉构建，CI 沙箱友好） |
| `packaging/version.txt` | 版本号唯一来源 |
| `packaging/app.ico` | 应用图标（多尺寸） |
| `packaging/LICENSE.txt` | EULA 模板（发布前替换） |
| `models/download_models.py` | 离线模型预下载（basic/full 预设） |
