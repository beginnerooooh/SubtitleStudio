/**
 * Subtitle Studio 桌面版主进程。
 *
 * 设计要点（对应反馈「构建成类似剪映的精简版，只保留字幕与简单剪辑」）：
 * - **零配置启动**：自动拉起内置 Python 后端（app/desktop_server.py），
 *   用户不需要装 Python、不需要开浏览器、不需要记端口；
 * - **生命周期绑定**：窗口关闭 → 后端进程一并退出（不会留孤儿进程）；
 * - **本地文件直通**：用系统对话框选视频/弹幕/字幕，走 /api/media 流式播放；
 * - **无网络依赖**：界面资源全部本地（不引 CDN），离线可用。
 *
 * 进程结构：
 *   Electron 主进程 ──spawn──> runtime/python.exe app/desktop_server.py
 *        │                              │
 *        └── BrowserWindow ──HTTP──> 127.0.0.1:<port>/api/...
 */
'use strict';

const { app, BrowserWindow, dialog, ipcMain, shell, Menu } = require('electron');
const { spawn } = require('child_process');
const http = require('http');
const path = require('path');
const fs = require('fs');
const net = require('net');

const ROOT = path.resolve(__dirname, '..');           // 便携目录根
const IS_DEV = process.argv.includes('--dev');

// 兼容性开关（便携/受限环境常见问题）：
// - 部分机器缺少 GPU 依赖 DLL，Chromium 的 GPU 子进程会以
//   STATUS_DLL_NOT_FOUND(0xC0000135) 反复崩溃并直接终止整个应用；
//   字幕编辑不需要 GPU 加速，因此默认走软件渲染。
// - 受限环境（企业策略/沙箱）下 GPU 沙箱也会导致同样问题，一并关闭。
// - 用户可用 SUBTITLE_STUDIO_GPU=1 强制启用硬件加速。
const USE_GPU = process.env.SUBTITLE_STUDIO_GPU === '1';
if (!USE_GPU) {
  app.commandLine.appendSwitch('disable-gpu');
  app.commandLine.appendSwitch('disable-gpu-compositing');
  app.commandLine.appendSwitch('disable-gpu-sandbox');
  app.commandLine.appendSwitch('disable-software-rasterizer');
  app.commandLine.appendSwitch('use-gl', 'swiftshader');
  app.disableHardwareAcceleration();
}
// 便携环境常无管理员权限，关闭 Chromium 自身的沙箱以避免启动失败
app.commandLine.appendSwitch('no-sandbox');
app.commandLine.appendSwitch('disable-setuid-sandbox');

const PY_CANDIDATES = [
  path.join(ROOT, 'runtime', 'python.exe'),
  path.join(ROOT, 'runtime', 'bin', 'python3'),
  'python3',
  'python',
];

let backend = null;          // Python 后端进程
let backendPort = 0;
let mainWindow = null;

// ---------------- 后端进程管理 ----------------

function findPython() {
  for (const p of PY_CANDIDATES) {
    if (p.includes(path.sep) && !fs.existsSync(p)) continue;
    return p;
  }
  return null;
}

function findFreePort(start = 8756, tries = 20) {
  return new Promise((resolve, reject) => {
    let port = start;
    const attempt = () => {
      const srv = net.createServer();
      srv.once('error', () => {
        port += 1;
        if (port > start + tries) reject(new Error('找不到可用端口'));
        else attempt();
      });
      srv.once('listening', () => srv.close(() => resolve(port)));
      srv.listen(port, '127.0.0.1');
    };
    attempt();
  });
}

function waitForBackend(port, timeoutMs = 60000) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const poll = () => {
      const req = http.get(
        { host: '127.0.0.1', port, path: '/api/health', timeout: 3000 },
        (res) => {
          res.resume();
          if (res.statusCode === 200) return resolve(port);
          retry();
        });
      req.on('error', retry);
      req.on('timeout', () => { req.destroy(); retry(); });
    };
    const retry = () => {
      if (Date.now() > deadline) {
        reject(new Error('后端启动超时（60s）。请检查 runtime/ 与依赖是否完整。'));
      } else {
        setTimeout(poll, 400);
      }
    };
    poll();
  });
}

async function startBackend() {
  const py = findPython();
  if (!py) {
    throw new Error('未找到 Python 运行时。请确认 runtime/python.exe 存在。');
  }
  backendPort = await findFreePort(8756);
  const args = [path.join(ROOT, 'app', 'desktop_server.py'),
                '--port', String(backendPort), '--quiet'];
  backend = spawn(py, args, {
    cwd: ROOT,
    env: { ...process.env, PYTHONIOENCODING: 'utf-8',
           PYTHONDONTWRITEBYTECODE: '1' },
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });
  backend.stdout.on('data', (d) => {
    if (IS_DEV) process.stdout.write('[backend] ' + d.toString());
  });
  backend.stderr.on('data', (d) => {
    if (IS_DEV) process.stderr.write('[backend] ' + d.toString());
  });
  backend.on('exit', (code, signal) => {
    if (IS_DEV) console.log(`[backend] exited code=${code} signal=${signal}`);
    backend = null;
    // 后端异常退出时提示用户（而不是白屏）
    if (mainWindow && !mainWindow.isDestroyed() && code !== 0 && code !== null) {
      dialog.showErrorBox('后端已停止',
        `识别服务异常退出（code=${code}）。请查看日志后重试。`);
    }
  });
  return waitForBackend(backendPort);
}

function stopBackend() {
  if (!backend) return;
  try {
    if (process.platform === 'win32') {
      spawn('taskkill', ['/pid', String(backend.pid), '/f', '/t'],
            { windowsHide: true });
    } else {
      backend.kill('SIGTERM');
    }
  } catch (e) { /* 忽略退出时的清理失败 */ }
  backend = null;
}

// ---------------- 窗口 ----------------

function createWindow() {
  // 自适应屏幕：按**工作区**（排除任务栏）计算，并留 4% 余量。
  // 注意高 DPI 屏：Electron 的 size 是 DIP，若系统缩放 125%，
  // 1440 DIP 实际会占 1800 物理像素，超出屏幕导致右侧面板被裁。
  // 因此这里直接按工作区的 92%/90% 取尺寸，创建后再钳制一次。
  const { screen } = require('electron');
  const disp = screen.getPrimaryDisplay();
  const work = disp.workAreaSize;
  const width = Math.max(960, Math.min(1440, Math.round(work.width * 0.92)));
  const height = Math.max(600, Math.min(900, Math.round(work.height * 0.90)));

  mainWindow = new BrowserWindow({
    width,
    height,
    x: Math.max(0, Math.round((work.width - width) / 2)),
    y: Math.max(0, Math.round((work.height - height) / 2)),
    minWidth: 900,
    minHeight: 560,
    backgroundColor: '#1b1c1f',
    title: 'Subtitle Studio',
    show: false,
    autoHideMenuBar: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      spellcheck: false,
    },
  });
  // 兜底钳制：窗口不得超出工作区（防止系统缩放把窗口顶出屏幕）
  const b = mainWindow.getBounds();
  const clamped = {
    x: Math.min(Math.max(0, b.x), Math.max(0, work.width - b.width)),
    y: Math.min(Math.max(0, b.y), Math.max(0, work.height - b.height)),
    width: Math.min(b.width, work.width),
    height: Math.min(b.height, work.height),
  };
  mainWindow.setBounds(clamped);

  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'),
                      { query: { port: String(backendPort),
                                 diag: IS_DEV ? '1' : '0',
                                 open: process.env.SUBTITLE_STUDIO_OPEN || '' } });
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    if (IS_DEV) mainWindow.webContents.openDevTools({ mode: 'detach' });
  });

  // 自检/截图模式：SUBTITLE_STUDIO_SHOT=<输出 png 路径>
  // 用于无人值守地验证「窗口内实际渲染成什么样」以及真实 CSS 视口尺寸
  // （高 DPI 缩放下物理像素与 CSS 像素差异很大，必须实测）。
  if (process.env.SUBTITLE_STUDIO_SHOT) {
    mainWindow.webContents.on('console-message', (_e, level, message) => {
      console.log(`[renderer:${level}] ${message}`);
    });
    mainWindow.webContents.on('render-process-gone', (_e, details) => {
      console.error('RENDERER_GONE ' + JSON.stringify(details));
    });
    // 等待时长可配（胶片条抽帧需要时间，默认 3.5s 可能不够）
    const waitMs = Number(process.env.SUBTITLE_STUDIO_SHOT_WAIT || 3500);
    // 可选：截图前执行一段 JS（用于验证交互行为，如「无字幕处不应显示字幕」）
    const probe = process.env.SUBTITLE_STUDIO_PROBE || '';
    mainWindow.webContents.once('did-finish-load', () => {
      setTimeout(async () => {
        let info = '{}';
        try {
          info = await mainWindow.webContents.executeJavaScript(
            'JSON.stringify({w: innerWidth, h: innerHeight,' +
            ' scrollW: document.documentElement.scrollWidth,' +
            ' dpr: devicePixelRatio,' +
            ' booted: typeof S !== "undefined",' +
            ' film: (() => { const f = document.getElementById("filmstrip");' +
            '   if (!f) return "missing";' +
            '   return {complete: f.complete, nw: f.naturalWidth,' +
            '     nh: f.naturalHeight, w: Math.round(f.getBoundingClientRect().width),' +
            '     h: Math.round(f.getBoundingClientRect().height),' +
            '     src: (f.src || "").slice(-40)}; })()' +
            '})');
          console.log('VIEWPORT ' + info);
        } catch (err) {
          console.error('VIEWPORT_FAILED ' + err.message);
        }
        if (probe) {
          try {
            const r = await mainWindow.webContents.executeJavaScript(probe);
            console.log('PROBE ' + r);
          } catch (err) {
            console.error('PROBE_FAILED ' + err.message);
          }
        }
        try {
          const img = await mainWindow.webContents.capturePage();
          fs.writeFileSync(process.env.SUBTITLE_STUDIO_SHOT, img.toPNG());
          console.log('SHOT ' + process.env.SUBTITLE_STUDIO_SHOT);
        } catch (err) {
          console.error('SHOT_FAILED ' + err.message);
        }
        app.quit();
      }, waitMs);
    });
  }
  mainWindow.on('closed', () => { mainWindow = null; });
  // 外链走系统浏览器，不在应用内打开
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

// ---------------- IPC（文件对话框等） ----------------

ipcMain.handle('dialog:openMedia', async () => {
  const r = await dialog.showOpenDialog(mainWindow, {
    title: '选择视频或音频',
    properties: ['openFile'],
    filters: [{ name: '媒体', extensions: ['mp4', 'mkv', 'mov', 'flv', 'webm',
                                           'mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg'] }],
  });
  return r.canceled ? '' : r.filePaths[0];
});

ipcMain.handle('dialog:openAny', async (_e, kind) => {
  const filters = {
    danmaku: [{ name: '弹幕', extensions: ['xml', 'jsonl'] }],
    subtitle: [{ name: '字幕', extensions: ['srt', 'ass', 'vtt', 'lrc', 'json'] }],
    audio: [{ name: '音频', extensions: ['wav', 'mp3', 'flac', 'm4a'] }],
    lyrics: [{ name: '歌词', extensions: ['lrc', 'txt', 'srt', 'json'] }],
    media: [{ name: '音视频',
              extensions: ['mp4', 'mkv', 'flv', 'mov', 'webm', 'ts', 'avi',
                           'wmv', 'rmvb', 'mp3', 'wav', 'm4a', 'flac'] }],
  };
  const r = await dialog.showOpenDialog(mainWindow, {
    title: '选择文件',
    properties: ['openFile'],
    filters: filters[kind] || [{ name: '全部', extensions: ['*'] }],
  });
  return r.canceled ? '' : r.filePaths[0];
});

ipcMain.handle('dialog:chooseDir', async () => {
  const r = await dialog.showOpenDialog(mainWindow, {
    title: '选择导出目录',
    properties: ['openDirectory', 'createDirectory'],
  });
  return r.canceled ? '' : r.filePaths[0];
});

ipcMain.handle('dialog:saveAs', async (_e, defaultName) => {
  const r = await dialog.showSaveDialog(mainWindow, {
    title: '另存为',
    defaultPath: defaultName || 'subtitle.srt',
    filters: [
      { name: '字幕', extensions: ['srt', 'ass', 'vtt', 'lrc'] },
      { name: '全部', extensions: ['*'] },
    ],
  });
  return r.canceled ? '' : r.filePath;
});

ipcMain.handle('app:info', () => ({
  root: ROOT, port: backendPort, version: app.getVersion(),
  platform: process.platform, dev: IS_DEV,
}));

ipcMain.handle('shell:reveal', (_e, target) => {
  if (target && fs.existsSync(target)) shell.showItemInFolder(target);
});

// ---------------- 生命周期 ----------------

// 单实例：第二次启动时聚焦已有窗口（避免开多个后端）
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(async () => {
    Menu.setApplicationMenu(buildMenu());
    try {
      await startBackend();
    } catch (err) {
      dialog.showErrorBox('启动失败', String(err.message || err));
      app.quit();
      return;
    }
    createWindow();
  });

  app.on('window-all-closed', () => {
    stopBackend();
    if (process.platform !== 'darwin') app.quit();
  });
  app.on('before-quit', stopBackend);
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0 && backendPort) createWindow();
  });
}

function buildMenu() {
  const isMac = process.platform === 'darwin';
  return Menu.buildFromTemplate([
    {
      label: '文件',
      submenu: [
        { label: '打开视频…', accelerator: 'CmdOrCtrl+O',
          click: () => mainWindow?.webContents.send('menu', 'open-video') },
        { label: '导入字幕…', accelerator: 'CmdOrCtrl+Shift+O',
          click: () => mainWindow?.webContents.send('menu', 'import-subs') },
        { type: 'separator' },
        { label: '导出字幕…', accelerator: 'CmdOrCtrl+S',
          click: () => mainWindow?.webContents.send('menu', 'export') },
        { type: 'separator' },
        isMac ? { role: 'close' } : { role: 'quit', label: '退出' },
      ],
    },
    {
      label: '编辑',
      submenu: [
        { label: '撤销', accelerator: 'CmdOrCtrl+Z',
          click: () => mainWindow?.webContents.send('menu', 'undo') },
        { label: '重做', accelerator: 'CmdOrCtrl+Shift+Z',
          click: () => mainWindow?.webContents.send('menu', 'redo') },
        { type: 'separator' },
        { label: '复制', accelerator: 'CmdOrCtrl+C',
          click: () => mainWindow?.webContents.send('menu', 'copy') },
        { label: '粘贴', accelerator: 'CmdOrCtrl+V',
          click: () => mainWindow?.webContents.send('menu', 'paste') },
        { label: '原地复制', accelerator: 'CmdOrCtrl+D',
          click: () => mainWindow?.webContents.send('menu', 'duplicate') },
        { type: 'separator' },
        { label: '在播放头切分', accelerator: 'CmdOrCtrl+B',
          click: () => mainWindow?.webContents.send('menu', 'split') },
        { label: '合并选中行', accelerator: 'CmdOrCtrl+M',
          click: () => mainWindow?.webContents.send('menu', 'merge') },
        { label: '删除选中行', accelerator: 'Delete',
          click: () => mainWindow?.webContents.send('menu', 'delete') },
        { type: 'separator' },
        { label: '自动质检', accelerator: 'CmdOrCtrl+Q',
          click: () => mainWindow?.webContents.send('menu', 'qc') },
      ],
    },
    {
      label: '视图',
      submenu: [
        { role: 'reload', label: '重新加载' },
        { role: 'toggleDevTools', label: '开发者工具' },
        { type: 'separator' },
        { role: 'zoomIn', label: '放大' },
        { role: 'zoomOut', label: '缩小' },
        { role: 'resetZoom', label: '重置缩放' },
        { type: 'separator' },
        { role: 'togglefullscreen', label: '全屏' },
      ],
    },
    {
      label: '帮助',
      submenu: [
        { label: '打开项目目录',
          click: () => shell.openPath(ROOT) },
        { label: '查看评测文档',
          click: () => shell.openPath(path.join(ROOT, 'docs', 'EVALUATION.md')) },
      ],
    },
  ]);
}
