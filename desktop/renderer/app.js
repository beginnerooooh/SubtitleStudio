/**
 * Subtitle Studio 桌面版渲染层。
 *
 * 交互参考剪映等主流剪辑软件：
 * - 多轨时间轴：视频轨（胶片缩略）+ 字幕轨（可拖动边缘调时长）
 * - 字幕片段：拖动移动、左右边缘拖拽改起止、右键菜单（复制/粘贴/删除等）
 * - 预览窗口：字幕实时叠加，自由模式下可直接拖动定位
 * - 导出：自选格式 + 目录
 * - 快捷键可自定义并持久化
 *
 * 职责边界：只做 UI 与交互；所有字幕运算走后端 API（app/desktop_server.py）。
 */
'use strict';

// ---------------- 状态 ----------------

const S = {
  port: 0,
  project: null,
  lines: [],
  selection: new Set(),
  current: -1,
  job: '',
  dirty: false,
  saving: false,
  danmaku: null,
  refSrt: '',
  bgmFlags: [],
  songs: [],
  clipboard: [],
  style: {
    size: 72, color: '#ffffff', outline: 3, anchor: 'bottom',
    x: null, y: null, bg: false, preview: true,
  },
  zoom: 1,            // 由 span 派生的缩放倍数（仅用于显示与交互）
  span: 15 * 60,      // 当前可见区间（秒），受 [15s, 15min] 限制
  viewStart: 0,       // 可见区起点（秒）
  keys: {},
  autosave: true,
  _tl: null,
  _drag: null,
};

const DEFAULT_KEYS = {
  play: 'Space', undo: 'Ctrl+Z', redo: 'Ctrl+Shift+Z',
  split: 'Ctrl+B', merge: 'Ctrl+M', delete: 'Delete',
  qc: 'Ctrl+Q', export: 'Ctrl+S', open: 'Ctrl+O',
  copy: 'Ctrl+C', paste: 'Ctrl+V', duplicate: 'Ctrl+D',
  prev: 'ArrowLeft', next: 'ArrowRight',
};

const KEY_LABELS = {
  play: '播放/暂停', undo: '撤销', redo: '重做', split: '切分',
  merge: '合并', delete: '删除', qc: '质检', export: '导出',
  open: '打开视频', copy: '复制', paste: '粘贴', duplicate: '原地复制',
  prev: '后退 0.5s', next: '前进 0.5s',
};

const $ = (id) => document.getElementById(id);
const API_BASE = () => `http://127.0.0.1:${S.port}`;
const api = (path, opts) => fetch(API_BASE() + path, opts);
const mediaUrl = (p) => `${API_BASE()}/api/media?path=${encodeURIComponent(p)}`;

/**
 * 把 API 路径翻译成用户能看懂的操作名（用于阻塞层文案）。
 *
 * 放在 apiJson 层统一处理的原因：调用点有 30 多处，逐个包装既啰嗦
 * 又容易漏；而「哪些请求会让用户等」本质上由接口决定。写操作的
 * 默认耗时明显高于读操作，只有它们才值得弹窗。
 */
function apiLabel(path) {
  const p = path.split('?')[0];
  const table = [
    [/^\/api\/generate/, '正在识别字幕'],
    [/^\/api\/evaluate/, '正在评测'],
    [/^\/api\/lyrics\/library\/fetch/, '正在拉取歌词'],
    [/^\/api\/lyrics\/library\/add/, '正在导入歌词'],
    [/^\/api\/lyrics\/library\/remove/, '正在删除歌词'],
    [/^\/api\/lyrics\/search/, '正在检索歌词'],
    [/^\/api\/lyrics\/match/, '正在比对歌词'],
    [/^\/api\/profiles\/delete/, '正在删除声纹'],
    [/^\/api\/profiles/, '正在提取声纹'],
    [/^\/api\/cache\/clean/, '正在清理缓存'],
    [/^\/api\/drafts\/clean/, '正在清理草稿'],
    [/^\/api\/danmaku/, '正在解析弹幕'],
    [/^\/api\/projects\/[^/]+\/export/, '正在导出字幕'],
    [/^\/api\/projects\/[^/]+\/undo/, '正在撤销'],
    [/^\/api\/projects\/[^/]+\/redo/, '正在重做'],
    [/^\/api\/projects\/[^/]+\/save/, '正在保存'],
    [/^\/api\/projects\/[^/]+\/autosave/, ''],      // 自动保存不打扰用户
    [/^\/api\/projects\/[^/]+\/recover/, '正在恢复草稿'],
    [/^\/api\/projects\/[^/]+\/history/, ''],
    [/^\/api\/projects\/[^/]+\/rename/, '正在重命名'],
    [/^\/api\/projects\/[^/]+\/export/, '正在导出'],
    [/^\/api\/projects$/, ''],
    [/^\/api\/projects\/[^/]+$/, '正在打开项目'],
    [/^\/api\/edit\//, '正在应用修改'],
    [/^\/api\/thumbs/, '正在生成缩略图'],
    [/^\/api\/videoinfo/, ''],
    [/^\/api\/media/, ''],
    [/^\/api\/health/, ''],
  ];
  for (const [re, label] of table) {
    if (re.test(p)) return label;
  }
  return '';
}

// 这些读接口不值得弹窗（很快，且频繁）
const SILENT_GET = /^\/api\/(health|videoinfo|projects(\/[^/]+(\/history)?)?|profiles|lyrics\/(providers|library)|cache\/stats|media)/;

/**
 * 发请求。写操作（或耗时 GET）自动套上阻塞层，防止等待期间重复点击。
 * `__silent` 选项可让调用方显式跳过弹窗（例如自动保存）。
 */
async function apiJson(path, method = 'GET', body, opts = {}) {
  const p = path.split('?')[0];
  const label = opts.label ?? apiLabel(path);
  const silent = opts.silent === true
    || (method === 'GET' && SILENT_GET.test(p))
    || !label;
  const run = async () => {
    const res = await api(path, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const data = await res.json().catch(
      () => ({ ok: false, error: '响应解析失败' }));
    if (!data.ok && data.error) throw new Error(data.error);
    return data;
  };
  if (silent) {
    // 静默请求不弹窗，但仍走同一套解析与错误处理
    try {
      return await run();
    } catch (err) {
      err.__handled = true;
      throw err;
    }
  }
  return withBusy(label, run);
}

// ---------------- 全局阻塞进度层 ----------------
//
// 设计：所有「需要用户等待」的操作都被包一层 `withBusy()`。
// 它做三件事：
//   1. 显示居中进度窗口（遮罩吃掉所有点击 → 天然防重复点击）
//   2. 显示操作名 / 进度 / 已用时间
//   3. 同一时刻只允许一个任务：重复触发时直接忽略并提示
//
// 为什么要用遮罩而不是逐个 disable 按钮：入口太多（工具栏、右键菜单、
// 快捷键、表格双击…），逐个 disable 一定会漏。

const Busy = {
  depth: 0,          // 允许嵌套（内层不重复显示）
  timer: 0,
  startedAt: 0,
  label: '',
  // 操作超过这个时长才显示弹窗，避免快速请求「闪一下」
  delay: 120,
};

function busyShow(label, sub = '') {
  Busy.label = label;
  Busy.startedAt = Date.now();
  $('busy-text').textContent = label;
  $('busy-sub').textContent = sub;
  $('busy-time').textContent = '';
  $('busy-fill').style.width = '0%';
  $('busy-fill').classList.add('indeterminate');
  $('busy').classList.remove('hidden');
  clearInterval(Busy.timer);
  Busy.timer = setInterval(() => {
    const s = ((Date.now() - Busy.startedAt) / 1000).toFixed(1);
    $('busy-time').textContent = `已用 ${s}s`;
  }, 200);
}

function busyHide() {
  clearInterval(Busy.timer);
  Busy.timer = 0;
  $('busy').classList.add('hidden');
  $('busy-fill').classList.remove('indeterminate');
}

/** 更新进度（ratio 为 null 时显示不确定态）。 */
function busyProgress(ratio, sub) {
  const fill = $('busy-fill');
  if (sub != null) $('busy-sub').textContent = sub;
  if (ratio == null || !Number.isFinite(ratio)) {
    fill.classList.add('indeterminate');
  } else {
    fill.classList.remove('indeterminate');
    fill.style.width = `${Math.max(0, Math.min(100, ratio * 100))}%`;
  }
}

/**
 * 包裹一次异步操作：显示阻塞层、防重入、出错时用弹窗提示。
 *
 * 重入时抛错而不是返回 undefined——静默返回会让调用方拿到假数据，
 * 进而写入项目状态（很难排查）。
 *
 * @param {string} label   给用户看的操作名
 * @param {Function} fn    实际工作（可以是 async）
 */
async function withBusy(label, fn) {
  if (Busy.depth > 0) {
    const e = new Error('上一操作尚未完成，请稍候');
    e.__handled = true;
    setSaveState('正在处理，请稍候', 'warn');
    throw e;
  }
  Busy.depth += 1;
  const showTimer = setTimeout(() => busyShow(label), Busy.delay);
  try {
    return await fn();
  } catch (err) {
    // 统一在这里报错，调用方不必到处写 alert
    if (!err || !err.__handled) {
      alert(`${label}失败：${err && err.message ? err.message : err}`);
    }
    throw err;
  } finally {
    clearTimeout(showTimer);
    Busy.depth -= 1;
    if (Busy.depth === 0) busyHide();
  }
}

// ---------------- 工具 ----------------

// 帧率：时间码「时:分:秒:帧」需要它；启动后由 /api/videoinfo 覆盖。
// 必须声明在使用之前（let 有暂时性死区，放后面会 ReferenceError）。
let FPS = 24;
const FPS_FALLBACK = 24;

/**
 * 统一的时长格式。全站时间都用「时:分:秒」或「时:分:秒:帧」，
 * 不再混用 mm:ss.mmm 与 hh:mm:ss（用户要求统一）。
 * ms=true 时保留帧号（依赖 FPS）。
 */
function fmt(t, withFrames = true) {
  return fmtTC(t, withFrames);
}

function setSaveState(text, cls = '') {
  const el = $('save-state');
  el.textContent = text;
  el.style.color = cls === 'warn' ? 'var(--warn)'
    : cls === 'ok' ? 'var(--ok)' : 'var(--muted)';
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// 组合键 → 归一化字符串（用于比较）
function keyName(e) {
  const parts = [];
  if (e.ctrlKey || e.metaKey) parts.push('Ctrl');
  if (e.shiftKey) parts.push('Shift');
  if (e.altKey) parts.push('Alt');
  let k = e.key;
  if (k === ' ') k = 'Space';
  else if (k.length === 1) k = k.toUpperCase();
  if (!['Control', 'Shift', 'Alt', 'Meta'].includes(k)) parts.push(k);
  return parts.join('+');
}

function loadKeys() {
  try {
    const saved = JSON.parse(localStorage.getItem('ss.keys') || '{}');
    S.keys = { ...DEFAULT_KEYS, ...saved };
  } catch (e) { S.keys = { ...DEFAULT_KEYS }; }
}

function saveKeys() {
  localStorage.setItem('ss.keys', JSON.stringify(S.keys));
}

function loadStyle() {
  try {
    const saved = JSON.parse(localStorage.getItem('ss.style') || '{}');
    Object.assign(S.style, saved);
  } catch (e) { /* 用默认 */ }
}

function saveStyle() {
  localStorage.setItem('ss.style', JSON.stringify(S.style));
}

// ---------------- 启动 ----------------

async function boot() {
  const params = new URLSearchParams(location.search);
  if (params.get('port')) S.port = Number(params.get('port'));
  else if (window.desktop) S.port = (await window.desktop.info()).port;
  else S.port = 8756;

  if (params.get('diag') === '1') {
    const setDiag = () => {
      document.title = `W${window.innerWidth}x${window.innerHeight}`;
    };
    setDiag();
    window.addEventListener('resize', setDiag);
  }

  loadKeys();
  loadStyle();
  try {
    const health = await apiJson('/api/health');
    $('status-backend').textContent =
      `后端就绪 · ${health.device}/${health.compute_type}`;
    $('status-backend').className = 'ok';
    fillSelect($('engine'), health.engines);
  } catch (err) {
    $('status-backend').textContent = '后端未连接：' + err.message;
    $('status-backend').className = 'err';
  }
  await loadProfiles();
  bindEvents();
  bindSplitters();
  bindDetailsMemory();
  loadLayout();
  updateZoomUI();
  syncStyleControls();
  updateOverlay();
  drawTimeline();
  setInterval(autosaveTick, 2000);

  if (params.get('open')) {
    try {
      await openProject(params.get('open'));
      if (S.lines.length) {
        $('video').currentTime = S.lines[0].start + 0.2;
        selectRow(0);
      }
    } catch (e) {
      console.error('open failed: ' + e.message);
    }
  }
}

function fillSelect(sel, values) {
  const cur = sel.value;
  sel.innerHTML = '';
  for (const v of values) {
    const o = document.createElement('option');
    o.value = o.textContent = v;
    sel.appendChild(o);
  }
  if (values.includes(cur)) sel.value = cur;
}

async function loadProfiles() {
  try {
    const d = await apiJson('/api/profiles');
    const sel = $('profile');
    sel.innerHTML = '<option value="">（未选择）</option>';
    for (const name of d.profiles) {
      const o = document.createElement('option');
      o.value = o.textContent = name;
      sel.appendChild(o);
    }
  } catch (e) { /* 无声纹 */ }
}

// ---------------- 项目 / 视频 ----------------

async function openVideo() {
  let path = '';
  if (window.desktop) path = await window.desktop.openMedia();
  else path = prompt('视频路径：') || '';
  if (!path) return;
  const name = path.split(/[\\/]/).pop().replace(/\.[^.]+$/, '');
  const d = await apiJson('/api/projects', 'POST', { video: path, name });
  S.project = { id: d.id, meta: d.meta, settings: {} };
  setVideo(path);
  S.lines = []; S.bgmFlags = []; S.songs = []; S.selection.clear();
  renderList(); renderClips(); renderSongs([]); renderBgm([]);
  $('status-project').textContent = `项目：${d.meta.name}`;
  setSaveState('已新建项目', 'ok');
  loadFilmstrip(path);
}

function setVideo(path) {
  const v = $('video');
  const badge = path.split(/[\\/]/).pop();
  $('video-info').textContent = badge;
  $('video-info').title = path;
  $('preview-badge').textContent = badge;
  // 先确保格式可播（必要时转码并显示进度），再设置 src
  preparePlayable(path).then(() => {
    // 关键：必须用绝对 URL。渲染层是 file:// 页面，相对路径 /api/media
    // 会被解析成 file:///api/media 而无法加载（实测视频黑屏的根因）。
    v.src = mediaUrl(path);
  }).catch(() => {
    v.src = mediaUrl(path);
  });
  loadVideoInfo(path);
  v.onloadedmetadata = () => {
    $('status-dur').textContent = fmt(v.duration);
    $('seek').max = String(Math.max(1, Math.round(v.duration * 1000)));
    drawTimeline();
    updateOverlay();
  };
  v.onerror = () => {
    $('preview-badge').textContent = '视频加载失败（格式或路径问题）';
  };
}

/** 取视频帧率，供「时:分:秒:帧」时间码使用。 */
async function loadVideoInfo(path) {
  try {
    const d = await apiJson(
      '/api/videoinfo?path=' + encodeURIComponent(path));
    if (d.fps > 0) {
      FPS = d.fps;
      if (d.estimated) {
        $('status-backend').textContent += '（帧率估计）';
      }
    }
  } catch (e) {
    FPS = FPS_FALLBACK;
  }
  // 帧率变化后重画所有时间显示
  renderList();
  renderClips();
  drawTimeline();
  updateOverlay();
  if ($('status-dur').textContent) {
    $('status-dur').textContent = fmt($('video').duration || 0);
  }
}

/**
 * 处理「需要转码才能预览」的格式。
 *
 * Chromium 播不了 FLV/MKV/HEVC 等，后端要转码。长视频重编码可能十几分钟，
 * 因此这里的策略是**前段就放行**：
 *   1. 弹居中进度窗（同时挡住重复点击）
 *   2. 轮询状态，一旦 `ready`（后端起出前段）立刻返回，让视频可播
 *   3. 后台继续转完，用户下一次打开就是完整时长
 *
 * 不能等「全量转完」再放行——4GB/72 分钟 HEVC 要 ~20 分钟，
 * 那样用户会以为程序卡死（实测超时 7 分钟仍未返回）。
 */
async function preparePlayable(path) {
  let st;
  try {
    st = await apiJson('/api/transcode/status?path=' + encodeURIComponent(path),
                       'GET', undefined, { silent: true });
  } catch (e) {
    return;                       // 查询失败就按原样尝试播放
  }
  if (!st.needed || st.ready) return;

  const label = st.copyable ? '正在转换封装格式' : '正在转码视频（HEVC 需重编码）';
  Busy.depth += 1;
  busyShow(label, st.copyable ? '通常几秒完成' : '正在生成可预览片段…');
  try {
    // 触发后端开始转码（媒体请求本身会启动任务）
    const probe = new Image();
    probe.src = `/api/media?path=${encodeURIComponent(path)}&_t=${Date.now()}`;
    // 轮询到「有可用片段」即放行（不是等转完）
    const t0 = Date.now();
    const LIMIT_MS = st.copyable ? 120000 : 180000;   // copy 2 分钟，转码 3 分钟
    let lastP = 0;
    while (Date.now() - t0 < LIMIT_MS) {
      await new Promise((r) => setTimeout(r, 1200));
      let s2;
      try {
        s2 = await apiJson('/api/transcode/status?path=' + encodeURIComponent(path),
                           'GET', undefined, { silent: true });
      } catch (e) { continue; }
      if (s2.ready) {
        busyProgress(1, '已可预览，后台继续转码');
        await new Promise((r) => setTimeout(r, 350));
        return;
      }
      if (Number.isFinite(s2.progress) && s2.progress > lastP) lastP = s2.progress;
      busyProgress(lastP || null,
        lastP ? `已转码 ${Math.round(lastP * 100)}%` : '正在生成可预览片段…');
    }
    // 超时也放行：至少让用户尝试播放，而不是一直转圈
    busyProgress(null, '转码较慢，先尝试播放');
    await new Promise((r) => setTimeout(r, 400));
  } finally {
    Busy.depth -= 1;
    if (Busy.depth === 0) busyHide();
  }
}

async function loadFilmstrip(path) {
  const img = $('filmstrip');
  const hint = $('filmstrip-hint');
  hint.textContent = '正在生成画面缩略…';
  hint.classList.remove('hidden');
  const dur = $('video').duration || 0;
  const count = Math.max(12, Math.min(80, Math.round((dur || 600) / 10)));
  img.src = `${API_BASE()}/api/thumbs?path=${encodeURIComponent(path)}`
    + `&count=${count}&height=${Math.round(40 * (window.devicePixelRatio || 1))}`;
  img.onload = () => { hint.classList.add('hidden'); };
  img.onerror = () => { hint.textContent = '无法生成缩略图（不影响编辑）'; };
}

async function ensureProject() {
  if (S.project) return S.project;
  const d = await apiJson('/api/projects', 'POST', { name: '未命名' });
  S.project = { id: d.id, meta: d.meta, settings: {} };
  $('status-project').textContent = `项目：${d.meta.name}`;
  return S.project;
}

function settingsPayload() {
  return {
    engine: $('engine').value,
    model: $('model').value,
    chunk_seconds: Number($('chunk').value) || 120,
    refine: $('opt-refine').checked,
    use_punctuation: $('opt-punc').checked,
    voiceprint: $('opt-voiceprint').checked,
    profile_name: $('profile').value,
    bgm_policy: $('bgm-policy').value,
    songs: $('opt-songs').checked,
    style: { ...S.style },
  };
}

async function markDirty() {
  S.dirty = true;
  setSaveState('未保存…', 'warn');
}

async function autosaveTick() {
  if (!S.autosave || !S.project || !S.dirty || S.saving) return;
  S.saving = true;
  try {
    await apiJson(`/api/projects/${S.project.id}/autosave`, 'POST',
      { lines: S.lines });
    S.dirty = false;
    setSaveState('已自动保存', 'ok');
  } catch (e) {
    setSaveState('保存失败', 'warn');
  } finally {
    S.saving = false;
  }
}

async function saveNow(note = 'edit') {
  const p = await ensureProject();
  const d = await apiJson(`/api/projects/${p.id}/save`, 'POST',
    { lines: S.lines, settings: settingsPayload(), songs: S.songs,
      bgm_flags: S.bgmFlags, note });
  S.dirty = false;
  setSaveState('已保存', 'ok');
  updateHistory(d);
}

function updateHistory(d) {
  const u = d.undo ?? 0, r = d.redo ?? 0;
  $('history-info').textContent = `撤销 ${u} · 重做 ${r}`;
  $('btn-undo').disabled = u === 0;
  $('btn-redo').disabled = r === 0;
}

// ---------------- 生成 ----------------

async function startGenerate() {
  if (!S.project || !S.project.meta.video) {
    alert('请先打开视频文件。');
    return;
  }
  const st = settingsPayload();
  const d = await apiJson('/api/generate', 'POST', {
    config: {
      asr: { engine: st.engine, model: st.model,
             chunk_seconds: st.chunk_seconds,
             chunk_overlap: Math.max(4, st.chunk_seconds * 0.08) },
      timing: { refine: st.refine },
      segment: { use_punctuation: st.use_punctuation },
      song: {
        enabled: st.songs,
        // 手动录入的本场曲目：逗号/顿号/空白分隔
        manual_titles: ($('manual-titles').value || '')
          .split(/[,，、\n]/).map((x) => x.trim()).filter(Boolean),
        lyric_correct: $('opt-lyric-correct').checked,
        danmaku_correct: $('opt-danmaku-correct').checked,
        segment_by_lyrics: true,
      },
      voiceprint: { enabled: st.voiceprint, profile_name: st.profile_name,
                    bgm_policy: st.bgm_policy },
    },
    request: {
      input_path: S.project.meta.video,
      enable_separation: $('opt-separate').checked,
      enable_voiceprint: $('opt-voiceprint').checked,
      profile_name: $('profile').value,
      enable_song_detect: $('opt-songs').checked,
      live_xml: S.danmaku ? S.danmaku.path : '',
      reference_srt: S.refSrt,
      formats: ['srt', 'lrc', 'ass', 'vtt', 'json'],
    },
  });
  S.job = d.job;
  $('btn-generate').disabled = true;
  $('btn-cancel').disabled = false;
  $('log').textContent = '';
  setDrawer(true);
  // 生成是最长的操作：全程保持阻塞层，并实时反映进度。
  // pollJob 的请求必须 silent，否则会与这层阻塞冲突（重入被拒）。
  pollJobWhileBusy();
}

/** 轮询生成任务，同时把进度写进居中弹窗。 */
async function pollJobWhileBusy() {
  Busy.depth += 1;
  busyShow('正在识别字幕', '准备中…');
  try {
    await pollJob();
  } finally {
    Busy.depth -= 1;
    if (Busy.depth === 0) busyHide();
  }
}

async function pollJob() {
  if (!S.job) return;
  let done = false;
  try {
    const j = await apiJson(`/api/jobs/${S.job}`, 'GET', undefined,
                            { silent: true });
    $('progress-bar').style.width = `${Math.min(100, (j.ratio || 0) * 100)}%`;
    $('progress-text').textContent = j.message || '';
    busyProgress(Number.isFinite(j.ratio) ? j.ratio : null, j.message || '');
    if (j.log && j.log.length) {
      const atBottom = $('log').scrollTop + $('log').clientHeight
        >= $('log').scrollHeight - 30;
      $('log').textContent = j.log.join('\n');
      if (atBottom) $('log').scrollTop = $('log').scrollHeight;
    }
    if (j.status === 'done') {
      done = true;
      busyProgress(1, '正在写入结果…');
      await applyResult(j.result);
    } else if (j.status === 'error') {
      done = true;
      $('progress-text').textContent = '失败：' + j.error;
      busyProgress(null, '失败：' + j.error);
      alert('识别失败：' + j.error);
    }
  } catch (e) {
    $('progress-text').textContent = '轮询失败：' + e.message;
    busyProgress(null, '轮询失败：' + e.message);
  }
  if (done) {
    S.job = '';
    $('btn-generate').disabled = false;
    $('btn-cancel').disabled = true;
    return;
  }
  await new Promise((r) => setTimeout(r, 700));
  return pollJob();
}

async function applyResult(result) {
  if (!result) return;
  S.lines = result.lines || [];
  S.songs = result.songs || [];
  S.bgmFlags = result.bgm_flags || [];
  renderList(); renderClips(); renderSongs(S.songs); renderBgm(S.bgmFlags);
  if (result.live && result.live.streamer) {
    S.danmaku = S.danmaku || {};
    $('danmaku-info').textContent =
      `${result.live.danmaku} 条 / 礼物 ${result.live.gifts}`;
  }
  drawTimeline();
  await ensureProject();
  await saveNow('generate');
  const bgmCount = S.bgmFlags.length;
  $('progress-text').textContent = `完成：${S.lines.length} 行`
    + (bgmCount ? ` · ${bgmCount} 处疑似 BGM 待复核` : '');
}

// ---------------- 预览：字幕叠加 + 自由拖动 ----------------

function previewScale() {
  const v = $('video');
  const shown = videoRect().w || 0;
  // 以成片 1920 宽为基准换算（所见即所得）
  return shown > 0 ? shown / 1920 : 0.33;
}

/** 视频在页面上的实际显示矩形（考虑 object-fit 的 letterbox）。
 *
 * 注意：`object-fit: contain` 下元素尺寸与画面尺寸不同——元素可能比
 * 画面「更高/更宽」，两侧或上下留黑边。字幕必须按**画面**定位，
 * 否则会偏到黑边上（实测字幕跑到画面外）。
 */
function videoRect() {
  const v = $('video');
  const frame = $('video-frame');
  const fr = frame.getBoundingClientRect();
  const er = v.getBoundingClientRect();
  const vw = v.videoWidth || 16;
  const vh = v.videoHeight || 9;

  // 元素在页面上的位置（相对 frame）与可用尺寸
  const availW = er.width || fr.width || 1;
  const availH = er.height || fr.height || 1;
  const k = Math.min(availW / vw, availH / vh);
  const w = Math.max(1, vw * k);
  const h = Math.max(1, vh * k);
  return {
    w, h,
    left: (er.left - fr.left) + (availW - w) / 2,
    top: (er.top - fr.top) + (availH - h) / 2,
  };
}

function applyStyleToOverlay() {
  const line = $('ov-line');
  if (!line) return;
  const k = previewScale();
  line.style.fontSize = Math.max(9, Math.round(S.style.size * k)) + 'px';
  line.style.color = S.style.color;
  const o = Math.max(0, S.style.outline * k);
  line.style.textShadow = o > 0.2
    ? `0 0 ${o}px #000, ${o}px 0 0 #000, -${o}px 0 0 #000, 0 ${o}px 0 #000, `
      + `0 -${o}px 0 #000, 0 2px 4px rgba(0,0,0,.9)`
    : '0 2px 4px rgba(0,0,0,.9)';
  line.classList.toggle('bgbar', S.style.bg);

  // 坐标以「成片 1920x1080」为基准；先算出视频实际画面矩形，
  // 再把画面内坐标加上黑边偏移，得到相对 video-frame 的位置。
  const r = videoRect();
  const kx = r.w / 1920;
  const ky = r.h / 1080;
  line.style.transform = 'translate(-50%, -50%)';
  if (S.style.anchor === 'free' && S.style.x != null && S.style.y != null) {
    line.style.left = (r.left + S.style.x * kx) + 'px';
    line.style.top = (r.top + S.style.y * ky) + 'px';
  } else if (S.style.anchor === 'top') {
    line.style.left = (r.left + r.w / 2) + 'px';
    line.style.top = (r.top + r.h * 0.12) + 'px';
  } else if (S.style.anchor === 'middle') {
    line.style.left = (r.left + r.w / 2) + 'px';
    line.style.top = (r.top + r.h / 2) + 'px';
  } else {
    line.style.left = (r.left + r.w / 2) + 'px';
    line.style.top = (r.top + r.h * 0.88) + 'px';
  }
  const draggable = S.style.anchor === 'free';
  line.classList.toggle('draggable', draggable);
  // 不再渲染辅助点：拖动是直接在字幕上按下并移动，无需可见手柄。
}

function updateOverlay() {
  const v = $('video');
  const t = v.currentTime || 0;
  let ln = S.lines.find((l) => t > l.start && t < l.end);
  // 「选中即预览」只在暂停状态生效：刚选中一行时想立刻看到样式效果。
  // 播放中/跳转后必须严格按时间匹配，否则会一直显示选中的那一行
  // （实测 bug：播放到无字幕段落时仍显示选中字幕）。
  if (!ln && S._previewSelection && v.paused) {
    ln = S.lines[[...S.selection].sort((a, b) => a - b)[0]];
  }
  const text = (S.style.preview && ln) ? ln.text : '';
  $('ov-line').textContent = text;
  applyStyleToOverlay();
}

function beginOverlayDrag(e) {
  if (S.style.anchor !== 'free') {
    // 非自由模式：拖动即自动切到自由定位，并以当前位置为起点
    const r = videoRect();
    if (S.style.x == null || S.style.y == null) {
      S.style.x = Math.round(1920 / 2);
      S.style.y = Math.round(S.style.anchor === 'top' ? 1080 * 0.12
        : S.style.anchor === 'middle' ? 1080 / 2 : 1080 * 0.88);
    }
    S.style.anchor = 'free';
    $('st-anchor').value = 'free';
    $('st-x').value = S.style.x;
    $('st-y').value = S.style.y;
    applyStyleToOverlay();
    void r;
  }
  S._drag = { kind: 'overlay' };
  e.preventDefault();
  window.addEventListener('mousemove', onOverlayDrag);
  window.addEventListener('mouseup', endDrag);
}

function onOverlayDrag(e) {
  const d = S._drag;
  if (!d || d.kind !== 'overlay') return;
  const frame = $('video-frame');
  const fr = frame.getBoundingClientRect();
  const r = videoRect();
  if (r.w <= 1 || r.h <= 1) return;
  const kx = 1920 / r.w;
  const ky = 1080 / r.h;
  const x = (e.clientX - fr.left - r.left) * kx;
  const y = (e.clientY - fr.top - r.top) * ky;
  S.style.x = Math.round(Math.max(0, Math.min(1920, x)));
  S.style.y = Math.round(Math.max(0, Math.min(1080, y)));
  $('st-x').value = S.style.x;
  $('st-y').value = S.style.y;
  applyStyleToOverlay();
}

// ---------------- 列表 ----------------

function visibleLines() {
  const q = ($('filter').value || '').trim();
  if (!q) return S.lines.map((ln, i) => [ln, i]);
  return S.lines.map((ln, i) => [ln, i]).filter(([ln]) => ln.text.includes(q));
}

function renderList() {
  const wrap = $('list-wrap');
  const rows = visibleLines();
  wrap.innerHTML = '';
  const frag = document.createDocumentFragment();
  for (const [ln, i] of rows) {
    const div = document.createElement('div');
    div.className = 'row'
      + (S.selection.has(i) ? ' selected' : '')
      + (i === S.current ? ' playing' : '')
      + (ln.low_confidence ? ' flagged' : '');
    div.dataset.index = String(i);
    div.innerHTML =
      `<span class="row-idx">${i + 1}</span>` +
      `<span class="row-text">${escapeHtml(ln.text)}</span>` +
      `<span class="row-time">${fmt(ln.start)}</span>`;
    frag.appendChild(div);
  }
  wrap.appendChild(frag);
  $('list-count').textContent = `${rows.length}/${S.lines.length}`;
  $('stats').textContent = `${S.lines.length} 行`
    + (S.selection.size ? ` · 选中 ${S.selection.size}` : '');
  $('status-sel').textContent = S.selection.size
    ? `已选中 ${S.selection.size} 行` : '未选中';
}

function rowIndex(el) {
  const row = el.closest('.row');
  return row ? Number(row.dataset.index) : -1;
}

function selectRow(index, e) {
  if (index < 0 || index >= S.lines.length) return;
  if (e && (e.ctrlKey || e.metaKey)) {
    S.selection.has(index) ? S.selection.delete(index) : S.selection.add(index);
  } else if (e && e.shiftKey && S.selection.size) {
    const anchor = Math.max(...S.selection);
    S.selection.clear();
    for (let k = Math.min(anchor, index); k <= Math.max(anchor, index); k++) {
      S.selection.add(k);
    }
  } else {
    S.selection.clear();
    S.selection.add(index);
  }
  // 选中即预览（仅暂停时生效，见 updateOverlay）
  S._previewSelection = true;
  renderList();
  renderClips();
  updateOverlay();
}

function jumpToLine(i) {
  const ln = S.lines[i];
  if (!ln) return;
  const v = $('video');
  S._previewSelection = false;   // 跳转后按真实时间显示
  v.currentTime = Math.max(0, ln.start);
  v.play().catch(() => {});
  S.current = i;
  renderList();
  renderClips();
  updateOverlay();
}

// ---------------- 编辑操作 ----------------

async function editOp(op, body) {
  const d = await apiJson(`/api/edit/${op}`, 'POST',
    { lines: S.lines, ...body });
  if (d.lines) S.lines = d.lines;
  renderList();
  renderClips();
  updateOverlay();
  if (d.clipboard) S.clipboard = d.clipboard;
  await markDirty();
  $('progress-text').textContent = d.message || '';
  return d;
}

async function doSplit() {
  const at = $('video').currentTime || 0;
  const idx = S.lines.findIndex((l) => at > l.start && at < l.end);
  if (idx < 0) { alert('播放头不在任何字幕行内。'); return; }
  const ln = S.lines[idx];
  const ratio = (at - ln.start) / Math.max(0.01, ln.end - ln.start);
  const pos = Math.max(1, Math.min(ln.text.length - 1,
                                   Math.round(ln.text.length * ratio)));
  await editOp('split', { index: idx, char_pos: pos });
  await saveNow('split');
}

async function doMerge() {
  const idx = [...S.selection].sort((a, b) => a - b);
  if (idx.length < 2) { alert('请用 Ctrl/Shift 选中至少两行。'); return; }
  await editOp('merge', { indices: idx });
  S.selection.clear();
  await saveNow('merge');
}

async function doDelete() {
  const idx = [...S.selection];
  if (!idx.length) { alert('请先选中要删除的行。'); return; }
  await editOp('delete', { indices: idx });
  S.selection.clear();
  await saveNow('delete');
}

async function doInsert() {
  const at = $('video').currentTime || 0;
  const idx = [...S.selection].sort((a, b) => a - b)[0] ?? S.lines.length;
  await editOp('insert', { index: idx, text: '新字幕行',
                           start: at, end: at + 2 });
  await saveNow('insert');
}

async function doQc() {
  const d = await editOp('qc', { max_chars: 18 });
  await saveNow('qc');
  $('progress-text').textContent = d.message;
}

async function doCopy() {
  const idx = [...S.selection];
  if (!idx.length) { alert('请先选中要复制的行。'); return; }
  const d = await editOp('copy', { indices: idx });
  $('progress-text').textContent = d.message;
}

async function doPaste() {
  if (!S.clipboard.length) { alert('剪贴板为空，请先复制。'); return; }
  const at = $('video').currentTime || 0;
  await editOp('paste', { clipboard: S.clipboard, at });
  await saveNow('paste');
}

async function doDuplicate() {
  const idx = [...S.selection];
  if (!idx.length) { alert('请先选中要复制的行。'); return; }
  await editOp('duplicate', { indices: idx });
  await saveNow('duplicate');
}

async function doUndo() {
  if (!S.project) return;
  const d = await apiJson(`/api/projects/${S.project.id}/undo`, 'POST', {});
  if (d.lines) { S.lines = d.lines; renderList(); renderClips(); updateOverlay(); }
  updateHistory(d);
  setSaveState('已撤销', 'ok');
}

async function doRedo() {
  if (!S.project) return;
  const d = await apiJson(`/api/projects/${S.project.id}/redo`, 'POST', {});
  if (d.lines) { S.lines = d.lines; renderList(); renderClips(); updateOverlay(); }
  updateHistory(d);
  setSaveState('已重做', 'ok');
}

// ---------------- 时间轴（多轨 + 缩放 + 拖动） ----------------

function timelineDuration() {
  return ($('video').duration || 0)
    || S.lines.reduce((m, l) => Math.max(m, l.end), 0) || 600;
}

/** 时间轴可见区间的边界（秒）。 */
const SPAN_MIN = 5;                        // 最短：5 秒
const SPAN_MAX = 10 * 60;                  // 最长：10 分钟
const SPAN_DEFAULT = 10 * 60;              // 默认显示 10 分钟

/**
 * 把「希望显示的时长」夹到允许区间，并同时受视频总长约束。
 *
 * 限制的意义（用户需求）：区间太大会让每行字幕窄到看不清，
 * 太小则失去上下文。因此统一夹在 [15s, 15min]，且不超过片长。
 */
function clampSpan(span) {
  const dur = timelineDuration();
  const hi = Math.min(SPAN_MAX, Math.max(SPAN_MIN, dur));
  return Math.max(SPAN_MIN, Math.min(hi, span));
}

function viewSpan() {
  const dur = timelineDuration();
  const span = clampSpan(S.span || SPAN_DEFAULT);
  const start = Math.max(0, Math.min(Math.max(0, dur - span), S.viewStart));
  return { start, span, dur };
}

/** 把任意缩放倍数换算成区间（供滑块/滚轮复用）。 */
function zoomToSpan(zoom) {
  return clampSpan(timelineDuration() / Math.max(0.01, zoom));
}

/**
 * 缩放到「字幕可读」的密度。
 *
 * 实测：842px 宽的轨道上，2 秒中位字幕要显示文字需要约 11px/秒。
 * 这里按中位时长反推区间，并交给 clampSpan 保证不小于 15 秒。
 */
function fitReadable() {
  if (!S.lines.length) { S.viewStart = 0; setSpan(SPAN_MAX); return; }
  const durs = S.lines.map((l) => l.end - l.start).sort((a, b) => a - b);
  const median = durs[Math.floor(durs.length / 2)] || 2;
  const lanes = $('tl-lanes').clientWidth || 800;
  // 目标：中位字幕约占 36px（能读约 3~4 个汉字）
  const targetPx = 36;
  const needSpan = lanes * median / targetPx;
  S.viewStart = S.lines[0]?.start || 0;
  setSpan(needSpan);
}

function timeToX(t) {
  const { start, span } = viewSpan();
  const w = $('tl-lanes').clientWidth || 800;
  return (t - start) / span * w;
}

function xToTime(x) {
  const { start, span } = viewSpan();
  const w = $('tl-lanes').clientWidth || 800;
  return start + (x / w) * span;
}

function drawRuler() {
  const r = $('ruler');
  const lanes = $('tl-lanes');
  const w = lanes.clientWidth || 800;
  const dpr = window.devicePixelRatio || 1;
  r.width = w * dpr;
  r.height = 18 * dpr;
  r.style.width = w + 'px';
  r.style.height = '18px';
  const ctx = r.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, 18);
  ctx.fillStyle = '#26282c';
  ctx.fillRect(0, 0, w, 18);

  const { start, span } = viewSpan();
  const scale = w / span;
  const cands = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800];
  let step = cands[cands.length - 1];
  for (const v of cands) { if (v * scale >= 62) { step = v; break; } }
  ctx.strokeStyle = '#3d4046';
  ctx.fillStyle = '#8d939b';
  ctx.font = '10px Consolas, monospace';
  const first = Math.ceil(start / step) * step;
  for (let t = first; t <= start + span; t += step) {
    const x = (t - start) * scale;
    ctx.beginPath(); ctx.moveTo(x, 12); ctx.lineTo(x, 18); ctx.stroke();
    ctx.fillText(fmt(t, false), x + 3, 10);
  }
}

function renderClips() {
  const box = $('clips');
  const { start, span } = viewSpan();
  const w = $('tl-lanes').clientWidth || 800;
  const scale = w / span;
  box.innerHTML = '';
  const isBgm = (ln) => S.bgmFlags.some((f) => ln.start < f.end && f.start < ln.end);
  const frag = document.createDocumentFragment();
  let lastEnd = -1, lane = 0;
  for (let i = 0; i < S.lines.length; i++) {
    const ln = S.lines[i];
    if (ln.end < start || ln.start > start + span) { continue; }
    lane = ln.start < lastEnd - 0.001 ? (lane === 0 ? 1 : 0) : 0;
    lastEnd = Math.max(lastEnd, ln.end);
    const px = Math.max(4, (ln.end - ln.start) * scale);
    const div = document.createElement('div');
    div.className = 'clip'
      + (S.selection.has(i) ? ' selected' : '')
      + (ln.low_confidence || isBgm(ln) ? ' flagged' : '')
      + (lane === 1 ? ' lane2' : '')
      + (px < 18 ? ' tiny' : '');
    div.dataset.index = String(i);
    div.style.left = ((ln.start - start) * scale) + 'px';
    div.style.width = px + 'px';
    div.title = `${fmt(ln.start)} – ${fmt(ln.end)}\n${ln.text}`;
    // 片段上直接显示字幕内容：宽度足够时显示文本，否则留空（避免挤出乱码）。
    // 阈值取 18px（约 2 个中文字），再小的片段只能显示色块。
    let label = '';
    if (px >= 18) {
      const perChar = 9.5;                 // 10px 字号下中英文混合的近似字宽
      const fit = Math.max(1, Math.floor((px - 8) / perChar));
      label = ln.text.length > fit ? ln.text.slice(0, fit) : ln.text;
    }
    div.innerHTML = `<span class="clip-text">${escapeHtml(label)}</span>`
      + `<span class="edge l" data-edge="l"></span>`
      + `<span class="edge r" data-edge="r"></span>`;
    frag.appendChild(div);
  }
  box.appendChild(frag);
}

function drawTimeline() {
  drawRuler();
  renderClips();
  updatePlayhead();
  updateZoomUI();
}

function updatePlayhead() {
  const t = $('video').currentTime || 0;
  const { start, span } = viewSpan();
  const w = $('tl-lanes').clientWidth || 800;
  const x = (t - start) / span * w;
  const ph = $('playhead');
  ph.style.left = Math.max(0, Math.min(w - 2, x)) + 'px';
  // 播放头超出可见区时自动滚动
  if (x < 0 || x > w) {
    S.viewStart = Math.max(0, Math.min(timelineDuration() - span, t - span / 2));
    drawTimeline();
  }
}

// 片段拖动：移动 / 拖左右边缘改起止
function beginClipDrag(e, mode) {
  const clip = e.target.closest('.clip');
  if (!clip) return;
  const i = Number(clip.dataset.index);
  const ln = S.lines[i];
  if (!ln) return;
  const lanes = $('tl-lanes');
  const rect = lanes.getBoundingClientRect();
  S._drag = {
    kind: 'clip', mode, index: i, rect,
    startX: e.clientX, origStart: ln.start, origEnd: ln.end,
    moved: false,
  };
  e.preventDefault();
  e.stopPropagation();
  window.addEventListener('mousemove', onClipDrag);
  window.addEventListener('mouseup', endDrag);
}

function onClipDrag(e) {
  const d = S._drag;
  if (!d || d.kind !== 'clip') return;
  const { span } = viewSpan();
  const w = d.rect.width || 1;
  const dt = (e.clientX - d.startX) / w * span;
  if (Math.abs(e.clientX - d.startX) > 2) d.moved = true;
  const ln = S.lines[d.index];
  if (!ln) return;
  // 实时预览（本地改，松手才提交后端）
  let s = d.origStart, en = d.origEnd;
  if (d.mode === 'move') {
    const shift = Math.max(-d.origStart, dt);
    s = d.origStart + shift;
    en = d.origEnd + shift;
  } else if (d.mode === 'l') {
    s = Math.max(0, Math.min(d.origEnd - 0.2, d.origStart + dt));
  } else {
    en = Math.max(d.origStart + 0.2, d.origEnd + dt);
  }
  ln.start = Math.round(s * 1000) / 1000;
  ln.end = Math.round(en * 1000) / 1000;
  renderClips();
  updatePlayhead();
  $('status-sel').textContent = `${fmt(ln.start)} – ${fmt(ln.end)}`;
}

async function endDrag() {
  window.removeEventListener('mousemove', onClipDrag);
  window.removeEventListener('mousemove', onOverlayDrag);
  const d = S._drag;
  S._drag = null;
  if (!d) return;
  if (d.kind === 'overlay') {
    saveStyle();
    return;
  }
  if (d.kind === 'clip' && d.moved) {
    const ln = S.lines[d.index];
    if (!ln) return;
    const body = { index: d.index };
    if (d.mode === 'move') { body.start = ln.start; body.end = ln.end; }
    else if (d.mode === 'l') body.start = ln.start;
    else body.end = ln.end;
    await editOp('retime_edge', body);
    await saveNow('drag');
  }
  renderList();
  renderClips();
}

// ---------------- 右键菜单 ----------------

const MENU_ITEMS = [
  { id: 'split', label: '在播放头切分', key: 'split' },
  { id: 'merge', label: '合并选中行', key: 'merge' },
  { sep: true },
  { id: 'copy', label: '复制', key: 'copy' },
  { id: 'paste', label: '在播放头粘贴', key: 'paste' },
  { id: 'duplicate', label: '原地复制', key: 'duplicate' },
  { sep: true },
  { id: 'insert', label: '在播放头新增', key: null },
  { id: 'delete', label: '删除', key: 'delete', danger: true },
];

function showContextMenu(x, y, index) {
  const menu = $('ctxmenu');
  if (index != null && index >= 0) {
    if (!S.selection.has(index)) selectRow(index);
  }
  const hasSel = S.selection.size > 0;
  menu.innerHTML = MENU_ITEMS.map((it) => {
    if (it.sep) return '<div class="sep"></div>';
    const disabled = !hasSel && ['merge', 'copy', 'duplicate', 'delete'].includes(it.id);
    const key = it.key ? (S.keys[it.key] || '') : '';
    return `<div class="item${it.danger ? ' danger' : ''}"
      data-act="${it.id}" ${disabled ? 'style="opacity:.4;pointer-events:none"' : ''}>
      <span>${it.label}</span><span class="key">${key}</span></div>`;
  }).join('');
  menu.classList.remove('hidden');
  const mw = menu.offsetWidth || 180;
  const mh = menu.offsetHeight || 200;
  menu.style.left = Math.min(x, window.innerWidth - mw - 6) + 'px';
  menu.style.top = Math.min(y, window.innerHeight - mh - 6) + 'px';
  menu.querySelectorAll('.item').forEach((el) => {
    el.onclick = () => { hideContextMenu(); runAction(el.dataset.act); };
  });
}

function hideContextMenu() {
  $('ctxmenu').classList.add('hidden');
}

function runAction(id) {
  ({
    split: doSplit, merge: doMerge, delete: doDelete, insert: doInsert,
    copy: doCopy, paste: doPaste, duplicate: doDuplicate, qc: doQc,
    export: showExport, undo: doUndo, redo: doRedo,
  }[id] || (() => {}))();
}

// ---------------- 导出（自选路径） ----------------

async function showExport() {
  if (!S.project) { alert('请先打开视频或新建项目。'); return; }
  const name = S.project.meta.name || 'subtitle';
  $('exp-name').value = name;
  $('exp-dir').value = '';
  $('modal-export').classList.remove('hidden');
}

async function doExport() {
  const fmts = [...$('exp-fmts').querySelectorAll('input:checked')]
    .map((c) => c.value);
  if (!fmts.length) { alert('请至少选择一种格式。'); return; }
  await saveNow('pre-export');
  const body = { fmt: fmts, name: ($('exp-name').value || 'subtitle').trim() };
  const dir = ($('exp-dir').value || '').trim();
  if (dir) body.out_dir = dir;
  try {
    const d = await apiJson(`/api/projects/${S.project.id}/export`, 'POST', body);
    $('modal-export').classList.add('hidden');
    $('progress-text').textContent =
      `已导出 ${d.files.length} 个文件到 ${d.out_dir}`;
    if (window.desktop && d.files.length) window.desktop.reveal(d.files[0]);
  } catch (e) {
    alert('导出失败：' + e.message);
  }
}

// ---------------- 快捷键设置 ----------------

function renderKeys() {
  const box = $('keys-list');
  box.innerHTML = Object.keys(DEFAULT_KEYS).map((id) => `
    <div class="keys-row">
      <span class="name">${KEY_LABELS[id] || id}</span>
      <input data-key="${id}" readonly value="${S.keys[id] || ''}" />
    </div>`).join('');
  box.querySelectorAll('input').forEach((inp) => {
    inp.onkeydown = (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.key === 'Escape') { inp.value = S.keys[inp.dataset.key] = ''; saveKeys(); return; }
      if (['Control', 'Shift', 'Alt', 'Meta'].includes(e.key)) return;
      const name = keyName(e);
      S.keys[inp.dataset.key] = name;
      inp.value = name;
      saveKeys();
    };
  });
}

// ---------------- 草稿箱 ----------------

async function showDrafts() {
  const d = await apiJson('/api/projects');
  const list = $('draft-list');
  if (!d.projects.length) {
    list.innerHTML = '<p class="muted">暂无项目。</p>';
  } else {
    list.innerHTML = d.projects.map((p) => `
      <div class="draft-item" data-id="${p.id}">
        <div class="info">
          <div class="name">${escapeHtml(p.name)}${p.draft ? ' <span class="muted">(草稿)</span>' : ''}</div>
          <div class="sub">${p.line_count} 行 · ${escapeHtml((p.video || '').split(/[\\\\/]/).pop() || '无视频')}</div>
        </div>
        <button class="btn sm act-open">打开</button>
        <button class="btn sm act-del">删除</button>
      </div>`).join('');
  }
  $('modal-drafts').classList.remove('hidden');
  list.querySelectorAll('.draft-item').forEach((item) => {
    const id = item.dataset.id;
    item.querySelector('.act-open').onclick = () => openProject(id);
    item.querySelector('.act-del').onclick = async () => {
      if (!confirm('确认删除该项目？')) return;
      await apiJson(`/api/projects/${id}`, 'DELETE');
      item.remove();
    };
  });
}

async function openProject(id) {
  const d = await apiJson(`/api/projects/${id}`);
  S.project = { id, meta: d.meta, settings: d.settings || {} };
  S.lines = d.lines || [];
  S.songs = d.songs || [];
  S.bgmFlags = d.bgm_flags || [];
  S.selection.clear();
  if (d.settings && d.settings.style) {
    Object.assign(S.style, d.settings.style);
    saveStyle();
  }
  $('modal-drafts').classList.add('hidden');
  $('status-project').textContent = `项目：${d.meta.name}`;
  if (d.meta.video) {
    setVideo(d.meta.video);
    loadFilmstrip(d.meta.video);
  }
  try {
    const rec = await apiJson(`/api/projects/${id}/recover`, 'POST', {});
    if (rec.lines && rec.lines.length !== S.lines.length
        && confirm('检测到自动保存的版本与上次保存不一致，是否恢复？')) {
      S.lines = rec.lines;
    }
  } catch (e) { /* 无 autosave */ }
  renderList(); renderClips(); renderSongs(S.songs); renderBgm(S.bgmFlags);
  syncStyleControls();
  updateOverlay();
  const h = await apiJson(`/api/projects/${id}/history`);
  updateHistory(h);
  setSaveState('已打开', 'ok');
}

async function cleanCache() {
  const stats = await apiJson('/api/cache/stats');
  if (!confirm(`当前缓存约 ${stats.freed_mb} MB / ${stats.files} 个文件。\n`
    + '将清理 7 天前的中间产物（抽取音频、缩略图等）。\n'
    + '不会删除字幕、声纹、歌词库。继续？')) return;
  const d = await apiJson('/api/cache/clean', 'POST',
    { max_age_days: 7, max_total_mb: 2048 });
  alert(`已清理 ${d.files} 个文件，释放约 ${d.freed_mb} MB。`);
}

async function importDanmaku() {
  let path = '';
  if (window.desktop) path = await window.desktop.openFile('danmaku');
  else path = prompt('弹幕 XML 路径：') || '';
  if (!path) return;
  const d = await apiJson('/api/danmaku?path=' + encodeURIComponent(path));
  S.danmaku = { path, ...d };
  $('danmaku-info').textContent =
    `${d.danmaku} 条 / 礼物 ${d.gifts} / 点歌 ${d.requests.length}`;
  $('log').textContent = [
    `主播：${d.streamer}`,
    `直播：${d.title}（${(d.duration / 60).toFixed(1)} 分钟）`,
    `弹幕 ${d.danmaku} 条，礼物 ${d.gifts} 条`,
    `点歌请求：${d.requests.map((r) => r[2]).join('、') || '无'}`,
    `热词：${d.hotwords.slice(0, 20).join('、')}`,
  ].join('\n');
  setDrawer(true);
}

async function importSubs() {
  let path = '';
  if (window.desktop) path = await window.desktop.openFile('subtitle');
  else path = prompt('字幕路径：') || '';
  if (!path) return;
  S.refSrt = path;
  $('ref-info').textContent = path.split(/[\\/]/).pop();
}

// ---------------- 面板 / 抽屉 / 样式 ----------------

function setDrawer(open) {
  $('drawer').classList.toggle('collapsed', !open);
  $('btn-drawer').textContent = open ? '日志 ▼' : '日志 ▲';
  if (open && !$('drawer').style.height) {
    $('drawer').style.height = '200px';
  }
  if (!open) $('drawer').style.height = '';
  if (window.__syncSplitters) window.__syncSplitters();
  saveLayout();
  if (open) drawTimeline();
}

function togglePanel(side) {
  const el = $(side === 'left' ? 'panel-left' : 'panel-right');
  const btn = $(side === 'left' ? 'btn-left' : 'btn-right');
  el.classList.toggle('hidden');
  btn.classList.toggle('on', !el.classList.contains('hidden'));
  if (window.__syncSplitters) window.__syncSplitters();
  saveLayout();
  setTimeout(drawTimeline, 180);
}

function syncStyleControls() {
  $('st-size').value = S.style.size;
  $('st-size-v').textContent = S.style.size;
  $('st-color').value = S.style.color;
  $('st-outline').value = S.style.outline;
  $('st-outline-v').textContent = S.style.outline;
  $('st-anchor').value = S.style.anchor;
  $('st-x').value = S.style.x ?? '';
  $('st-y').value = S.style.y ?? '';
  $('st-bg').checked = S.style.bg;
  $('st-preview').checked = S.style.preview;
  applyStyleToOverlay();
}

const PRESETS = {
  standard: { size: 72, color: '#ffffff', outline: 3, anchor: 'bottom', bg: false },
  cinema: { size: 64, color: '#f5f5f0', outline: 2, anchor: 'bottom', bg: false },
  variety: { size: 88, color: '#ffe14d', outline: 4, anchor: 'middle', bg: true },
  bold: { size: 96, color: '#ffffff', outline: 6, anchor: 'bottom', bg: false },
};

function renderSongs(songs) {
  const box = $('songs');
  if (!songs || !songs.length) {
    box.innerHTML = '<p class="muted">未识别到歌曲。开启「歌曲识别」后会用'
      + '弹幕点歌 / 歌词库检索 / 识曲三级链路定位曲目。</p>';
    return;
  }
  box.innerHTML = songs.map((s) => `
    <div class="list-item">
      <div class="title">《${escapeHtml(s.title || '未识别')}》${escapeHtml(s.artist || '')}</div>
      <div class="meta">${fmt(s.start, false)} – ${fmt(s.end, false)}
        · 来源 ${escapeHtml(s.source || '-')}
        · ${s.aligned ? '逐行对齐' : '近似'} · ${s.line_count || 0} 行</div>
    </div>`).join('');
}

function renderBgm(flags) {
  const box = $('bgm');
  if (!flags || !flags.length) {
    box.innerHTML = '<p class="muted">没有疑似 BGM 原唱片段。启用「主播声纹过滤」后，'
      + '被判为伴奏原唱的时间段会列在这里（默认只标注、不删除）。</p>';
    return;
  }
  box.innerHTML = flags.map((f) => `
    <div class="list-item">
      <div class="title">${fmt(f.start)} – ${fmt(f.end)}
        <span class="muted">（声纹 ${(f.vocal_sim ?? 0).toFixed(2)}
        / 伴奏度 ${(f.music ?? 0).toFixed(2)}）</span></div>
      <div class="meta">${escapeHtml((f.reasons || []).join('；'))}</div>
    </div>`).join('');
}

// ---------------- 歌词库 ----------------

async function openLyrics() {
  $('modal-lyrics').classList.remove('hidden');
  await refreshLyricList();
}

async function refreshLyricList() {
  try {
    const d = await apiJson('/api/lyrics/library');
    $('ly-count').textContent = `（${d.size} 首）`;
    const box = $('ly-list');
    if (!d.docs.length) {
      box.innerHTML = '<p class="muted small">曲库为空。用上方按钮导入或拉取歌词。</p>';
      return;
    }
    box.innerHTML = d.docs.map((doc) => `
      <div class="vp-item" data-title="${escapeHtml(doc.title)}"
           data-artist="${escapeHtml(doc.artist)}">
        <span class="name">《${escapeHtml(doc.title)}》
          ${doc.artist ? escapeHtml(doc.artist) : ''}</span>
        <span class="tags">${doc.line_count} 行${doc.synced ? ' · 带时间' : ''}</span>
        <button class="btn sm act-use">加入本场</button>
        <button class="btn sm act-del">删除</button>
      </div>`).join('');
    box.querySelectorAll('.vp-item').forEach((item) => {
      const title = item.dataset.title;
      item.querySelector('.act-use').onclick = () => {
        const cur = ($('manual-titles').value || '').trim();
        const parts = cur ? cur.split(/[,，、]/).map((s) => s.trim()).filter(Boolean) : [];
        if (!parts.includes(title)) parts.push(title);
        $('manual-titles').value = parts.join('，');
        $('ly-status').textContent = `已把《${title}》加入本场曲目`;
      };
      item.querySelector('.act-del').onclick = async () => {
        if (!confirm(`从曲库删除《${title}》？`)) return;
        await apiJson('/api/lyrics/library/remove', 'POST', { title });
        await refreshLyricList();
      };
    });
  } catch (e) {
    $('ly-list').innerHTML =
      `<p class="muted small">读取失败：${escapeHtml(e.message)}</p>`;
  }
}

async function fetchLyric() {
  const title = ($('ly-title').value || '').trim();
  const artist = ($('ly-artist').value || '').trim();
  if (!title) { $('ly-status').textContent = '请先填写歌名'; return; }
  $('ly-status').textContent = '正在拉取…';
  try {
    const d = await apiJson('/api/lyrics/library/fetch', 'POST',
      { title, artist });
    if (d.added && d.added.length) {
      $('ly-status').textContent =
        `已添加 ${d.added.length} 首：${d.added.map((x) => `《${x.title}》`).join('')}`;
    } else {
      $('ly-status').textContent = d.error || '没有找到可用歌词';
    }
    if (d.log && d.log.length) {
      $('log').textContent = d.log.join('\n');
    }
    await refreshLyricList();
  } catch (e) {
    $('ly-status').textContent = '拉取失败：' + e.message;
  }
}

async function importLyricFile() {
  if (!window.desktop) { $('ly-status').textContent = '浏览器模式请用粘贴导入'; return; }
  const path = await window.desktop.openFile('lyrics');
  if (!path) return;
  $('ly-status').textContent = '正在导入…';
  try {
    const d = await apiJson('/api/lyrics/library/add', 'POST', { path });
    $('ly-status').textContent = `已导入《${d.title}》（${d.line_count} 行）`;
    await refreshLyricList();
  } catch (e) {
    $('ly-status').textContent = '导入失败：' + e.message;
  }
}

async function pasteLyric() {
  const ta = $('ly-text');
  if (ta.classList.contains('hidden')) {
    ta.classList.remove('hidden');
    ta.focus();
    $('ly-status').textContent = '粘贴歌词后再次点击「粘贴导入」';
    return;
  }
  const title = ($('ly-title').value || '').trim();
  const artist = ($('ly-artist').value || '').trim();
  const text = ta.value.trim();
  if (!title) { $('ly-status').textContent = '请先填写歌名'; return; }
  if (!text) { $('ly-status').textContent = '请先粘贴歌词内容'; return; }
  try {
    const d = await apiJson('/api/lyrics/library/add', 'POST',
      { title, artist, text });
    $('ly-status').textContent = `已导入《${d.title}》（${d.line_count} 行）`;
    ta.value = '';
    ta.classList.add('hidden');
    await refreshLyricList();
  } catch (e) {
    $('ly-status').textContent = '导入失败：' + e.message;
  }
}

// ---------------- 面板分隔条（可拉伸，但不影响视频预览） ----------------

const LAYOUT_KEY = 'ss.layout';

function loadLayout() {
  try {
    const saved = JSON.parse(localStorage.getItem(LAYOUT_KEY) || '{}');
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    if (saved.left) {
      const w = clamp(saved.left, 160, 460);
      $('panel-left').style.flexBasis = w + 'px';
      $('panel-left').style.width = w + 'px';
    }
    if (saved.right) {
      const w = clamp(saved.right, 180, 520);
      $('panel-right').style.flexBasis = w + 'px';
      $('panel-right').style.width = w + 'px';
    }
    // 时间轴高度上限按当前窗口算，避免恢复出「预览被压没」的布局
    if (saved.timeline) {
      const maxH = Math.max(120, window.innerHeight * 0.42);
      const h = clamp(saved.timeline, 120, maxH);
      $('timeline').style.height = h + 'px';
      $('timeline').style.flexBasis = h + 'px';
    }
    if (saved.drawer && !$('drawer').classList.contains('collapsed')) {
      const h = clamp(saved.drawer, 80, window.innerHeight * 0.5);
      $('drawer').style.height = h + 'px';
    }
    // 兜底：如果此时预览高度不足（窗口变小、或旧布局把空间都占了），
    // 主动收缩时间轴与日志，保证至少留出 260px 给视频。
    //
    // 注意不能用 requestAnimationFrame —— loadLayout 在 boot 早期调用，
    // 那时 flex 尚未完成首帧布局，getBoundingClientRect 读到的是 0
    // 或旧值，兜底会静默失效（实测坏布局仍然把预览压到 207px）。
    // 用 setTimeout 等布局稳定后再校正。
    setTimeout(ensurePreviewSpace, 120);
    // 左面板折叠状态
    restoreDetails(saved.details);
    // 面板显隐（用按钮的高亮状态保持一致）
    if (saved.panels) {
      const setPanel = (side, visible) => {
        const el = $(side === 'left' ? 'panel-left' : 'panel-right');
        const btn = $(side === 'left' ? 'btn-left' : 'btn-right');
        el.classList.toggle('hidden', !visible);
        btn.classList.toggle('on', visible);
      };
      if (saved.panels.left === false) setPanel('left', false);
      if (saved.panels.right === false) setPanel('right', false);
    }
    // 日志抽屉初始展开状态
    if (saved.drawerOpen) {
      $('drawer').classList.remove('collapsed');
      $('btn-drawer').textContent = '日志 ▼';
    }
  } catch (e) { /* 忽略损坏的布局设置 */ }
}

function saveLayout() {
  try {
    localStorage.setItem(LAYOUT_KEY, JSON.stringify({
      left: Math.round($('panel-left').getBoundingClientRect().width),
      right: Math.round($('panel-right').getBoundingClientRect().width),
      timeline: Math.round($('timeline').getBoundingClientRect().height),
      drawer: Math.round($('drawer').getBoundingClientRect().height),
      // 左面板各折叠区的展开状态（按标题记，增删章节也不会错位）
      details: [...document.querySelectorAll('#panel-left details')]
        .map((d) => ({ t: d.querySelector('summary')?.textContent?.trim() || '',
                       o: d.open })),
      panels: {
        left: !$('panel-left').classList.contains('hidden'),
        right: !$('panel-right').classList.contains('hidden'),
      },
      drawerOpen: !$('drawer').classList.contains('collapsed'),
    }));
  } catch (e) { /* 忽略 */ }
}

/**
 * 保证预览区有足够高度。
 *
 * 场景：用户之前把日志抽屉展开、时间轴拖得很高，布局被记忆下来；
 * 下次打开（或换到更小的窗口）时预览会被压到很小，看起来像「没画面」。
 * 这里主动回收空间，优先级：日志抽屉 → 时间轴。
 *
 * 保留 PREVIEW_MIN_H 给视频，低于此值才动手，避免干扰正常布局。
 */
const PREVIEW_MIN_H = 260;

function ensurePreviewSpace() {
  try {
    const preview = $('preview');
    if (!preview) return;
    let need = PREVIEW_MIN_H - preview.getBoundingClientRect().height;
    if (need <= 0) return;

    // 1) 先从日志抽屉扣（最小保留 30px 的折叠条）
    const drawer = $('drawer');
    if (!drawer.classList.contains('collapsed')) {
      const dh = drawer.getBoundingClientRect().height;
      const take = Math.min(need, Math.max(0, dh - 30));
      if (take > 0) {
        const nh = Math.max(30, dh - take);
        drawer.style.height = nh + 'px';
        if (nh <= 30) setDrawer(false);
        need -= take;
      }
    }
    // 2) 还不够就压时间轴（最小保留 120px）
    if (need > 0) {
      const tl = $('timeline');
      const th = tl.getBoundingClientRect().height;
      const take = Math.min(need, Math.max(0, th - 120));
      if (take > 0) {
        const nh = Math.max(120, th - take) + 'px';
        tl.style.height = nh;
        tl.style.flexBasis = nh;
      }
    }
    drawTimeline();
    updateOverlay();
  } catch (e) { /* 忽略：布局校正失败不应影响编辑 */ }
}

/** 恢复左面板折叠状态：按标题匹配，找不到就保持默认。 */
function restoreDetails(saved) {
  if (!Array.isArray(saved) || !saved.length) return;
  const nodes = [...document.querySelectorAll('#panel-left details')];
  for (const item of saved) {
    const node = nodes.find(
      (d) => (d.querySelector('summary')?.textContent?.trim() || '') === item.t);
    if (node) node.open = !!item.o;
  }
}

/** 折叠状态变化时立即写盘（不必等拖拽结束）。 */
function bindDetailsMemory() {
  document.querySelectorAll('#panel-left details').forEach((d) => {
    d.addEventListener('toggle', saveLayout);
  });
}

/**
 * 绑定分隔条拖拽。
 *
 * 关键约束（对应反馈「不能影响视频预览窗口」）：
 * 时间轴与日志的拉伸只改变它们自身高度，预览按 flex:1 吃掉剩余空间，
 * 并有 min-height 保底；因此拉伸不会把预览压没。
 */
function bindSplitters() {
  const startDrag = (el, e, onMove) => {
    el.classList.add('dragging');
    const move = (ev) => onMove(ev);
    const up = () => {
      el.classList.remove('dragging');
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
      saveLayout();
      drawTimeline();
      updateOverlay();
    };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
    e.preventDefault();
  };

  // 左面板宽度
  $('split-left').addEventListener('mousedown', (e) => {
    const startX = e.clientX;
    const startW = $('panel-left').getBoundingClientRect().width;
    startDrag($('split-left'), e, (ev) => {
      const w = Math.max(160, Math.min(460, startW + (ev.clientX - startX)));
      $('panel-left').style.flexBasis = w + 'px';
      $('panel-left').style.width = w + 'px';
      drawTimeline();
    });
  });

  // 右面板宽度（从右侧拖，方向相反）
  $('split-right').addEventListener('mousedown', (e) => {
    const startX = e.clientX;
    const startW = $('panel-right').getBoundingClientRect().width;
    startDrag($('split-right'), e, (ev) => {
      const w = Math.max(180, Math.min(520, startW - (ev.clientX - startX)));
      $('panel-right').style.flexBasis = w + 'px';
      $('panel-right').style.width = w + 'px';
      drawTimeline();
    });
  });

  // 时间轴高度（向上拖变高）。
  // 「不影响视频预览」的实现：把时间轴的增量从**日志抽屉**里扣掉。
  // 若日志已收起，则先把日志展开出来承接（相当于挤压日志而非预览）；
  // 只有日志已到最小高度时，才真正占用预览空间。
  $('split-timeline').addEventListener('mousedown', (e) => {
    const startY = e.clientY;
    const startH = $('timeline').getBoundingClientRect().height;
    const drawer = $('drawer');
    const drawerWasCollapsed = drawer.classList.contains('collapsed');
    const startDrawerH = drawerWasCollapsed ? 30
      : drawer.getBoundingClientRect().height;
    startDrag($('split-timeline'), e, (ev) => {
      const h = Math.max(120, Math.min(420, startH - (ev.clientY - startY)));
      $('timeline').style.height = h + 'px';
      $('timeline').style.flexBasis = h + 'px';
      const grow = h - startH;
      // 从日志里抵扣（可用空间 = 日志当前高度 - 最小 30px）
      const available = Math.max(0, startDrawerH - 30);
      const take = Math.min(available, Math.max(0, grow));
      if (take > 0) {
        if (drawerWasCollapsed) setDrawer(true);
        drawer.style.height = Math.max(30, startDrawerH - take) + 'px';
        if (drawer.style.height === '30px') setDrawer(false);
      }
      drawTimeline();
    });
  });

  // 日志高度（向上拖变高）
  $('split-drawer').addEventListener('mousedown', (e) => {
    if ($('drawer').classList.contains('collapsed')) return;
    const startY = e.clientY;
    const startH = $('drawer').getBoundingClientRect().height;
    startDrag($('split-drawer'), e, (ev) => {
      const h = Math.max(80, Math.min(window.innerHeight * 0.6,
                                      startH - (ev.clientY - startY)));
      $('drawer').style.height = h + 'px';
    });
  });

  // 双面板都隐藏时分隔条也隐藏（避免误拖）
  const syncSplitters = () => {
    $('split-left').classList.toggle('hidden',
      $('panel-left').classList.contains('hidden'));
    $('split-right').classList.toggle('hidden',
      $('panel-right').classList.contains('hidden'));
    $('split-drawer').classList.toggle('hidden',
      $('drawer').classList.contains('collapsed'));
  };
  window.__syncSplitters = syncSplitters;
  syncSplitters();
}

// ---------------- 缩放条 ----------------

/**
 * 滑块用「区间」而非「倍数」驱动：
 * 0 = 最长区间（15 分钟），1000 = 最短区间（15 秒），对数刻度更符合手感。
 */
// ---------------- 时间码：时:分:秒:帧 ----------------

/**
 * 秒 → 「时:分:秒:帧」。
 *
 * 帧号按 floor((t - 整秒) * fps) 取，并对浮点误差做半帧容差，
 * 避免 1.999999 被算成第 23 帧而不是第 24 帧（时间码的常见坑）。
 */
function fmtTC(t, withFrames = true) {
  t = Math.max(0, Number(t) || 0);
  const fps = FPS > 0 ? FPS : 25;
  const per = Math.max(1, Math.round(fps));
  let total = Math.floor(t * fps + 1e-6);
  const ff = total % per;
  total = Math.floor(total / per);
  const ss = total % 60;
  const mm = Math.floor(total / 60) % 60;
  const hh = Math.floor(total / 3600);
  const base = `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`
    + `:${String(ss).padStart(2, '0')}`;
  return withFrames ? `${base}:${String(ff).padStart(2, '0')}` : base;
}

/** 「时:分:秒:帧」或其它常见写法 → 秒。 */
function parseTC(text) {
  const s = String(text).trim().replace(',', '.');
  if (!s) return NaN;
  const parts = s.split(':').map((x) => x.trim());
  const fps = FPS > 0 ? FPS : 25;
  const per = Math.max(1, Math.round(fps));
  const num = (v) => (v === '' || v == null ? 0 : Number(v));
  if (parts.length >= 4) {
    const [h, m, sec, f] = parts.map(num);
    if ([h, m, sec, f].some(Number.isNaN)) return NaN;
    // 兼容把「第 per 帧」写成 per 的写法（等价于下一秒 0 帧）
    const carry = f >= per ? 1 : 0;
    const fr = f >= per ? 0 : f / fps;
    return h * 3600 + m * 60 + sec + fr + carry;
  }
  if (parts.length === 3) {
    const [h, m, sec] = parts.map(num);
    if ([h, m, sec].some(Number.isNaN)) return NaN;
    return h * 3600 + m * 60 + sec;
  }
  if (parts.length === 2) {
    const [m, sec] = parts.map(num);
    if ([m, sec].some(Number.isNaN)) return NaN;
    return m * 60 + sec;
  }
  return num(parts[0]);
}

// ---------------- 框选（按住左键拖动批量选择） ----------------

const MARQUEE_THRESHOLD = 4;   // 移动超过 4px 才算框选，否则视为点击定位

function beginMarquee(e) {
  const lanes = $('tl-lanes');
  const laneSub = $('lane-sub');
  const rect = lanes.getBoundingClientRect();
  const subRect = laneSub.getBoundingClientRect();
  S._marquee = {
    active: false, startX: e.clientX, startY: e.clientY,
    originX: e.clientX - rect.left, originY: e.clientY - rect.top,
    rect, subRect,
    base: new Set(e.ctrlKey || e.metaKey ? S.selection : []),
  };
  window.addEventListener('mousemove', onMarquee);
  window.addEventListener('mouseup', endMarquee);
  e.preventDefault();
}

function onMarquee(e) {
  const m = S._marquee;
  if (!m) return;
  const dx = e.clientX - m.startX;
  const dy = e.clientY - m.startY;
  if (!m.active) {
    if (Math.abs(dx) < MARQUEE_THRESHOLD && Math.abs(dy) < MARQUEE_THRESHOLD) {
      return;
    }
    m.active = true;
    $('marquee').classList.remove('hidden');
  }
  const curX = e.clientX - m.rect.left;
  const curY = e.clientY - m.rect.top;
  const left = Math.min(m.originX, curX);
  const top = Math.min(m.originY, curY);
  const w = Math.abs(curX - m.originX);
  const h = Math.abs(curY - m.originY);
  const box = $('marquee');
  // marquee 绝对定位在 lane-sub 内，纵向需减去字幕轨的偏移
  const subOffset = m.subRect.top - m.rect.top;
  box.style.left = left + 'px';
  box.style.top = (top - subOffset) + 'px';
  box.style.width = w + 'px';
  box.style.height = h + 'px';

  // 命中检测：横向按时间区间取交，纵向须覆盖字幕轨
  const t0 = Math.min(xToTime(m.originX), xToTime(curX));
  const t1 = Math.max(xToTime(m.originX), xToTime(curX));
  const subTop = subOffset;
  const subBottom = subTop + m.subRect.height;
  const y0 = Math.min(m.originY, curY);
  const y1 = Math.max(m.originY, curY);
  const hitLane = y1 >= subTop && y0 <= subBottom;

  S.selection = new Set(m.base);
  if (hitLane) {
    S.lines.forEach((ln, i) => {
      if (ln.end > t0 && ln.start < t1) S.selection.add(i);
    });
  }
  renderList();
  renderClips();
  updateOverlay();
}

function endMarquee(e) {
  const m = S._marquee;
  S._marquee = null;
  window.removeEventListener('mousemove', onMarquee);
  window.removeEventListener('mouseup', endMarquee);
  if (!m) return;
  if (!m.active) {
    // 没拖动 = 单击空白处 → 定位播放头（保持原有行为）
    $('video').currentTime = Math.max(
      0, Math.min(timelineDuration(), xToTime(m.startX - m.rect.left)));
    return;
  }
  $('marquee').classList.add('hidden');
  $('status-sel').textContent = S.selection.size
    ? `已选中 ${S.selection.size} 行` : '未选中';
}

// ---------------- 菜单 / 缩放 ----------------

function spanToSlider(span) {
  const lo = Math.log(SPAN_MIN), hi = Math.log(SPAN_MAX);
  const v = Math.log(clampSpan(span));
  return Math.round((hi - v) / (hi - lo) * 1000);
}

function sliderToSpan(v) {
  const lo = Math.log(SPAN_MIN), hi = Math.log(SPAN_MAX);
  return Math.exp(hi - (Number(v) / 1000) * (hi - lo));
}

function updateZoomUI() {
  const z = $('zoom');
  if (z && document.activeElement !== z) z.value = String(spanToSlider(S.span));
  const lab = $('zoom-label');
  if (lab) {
    const { span } = viewSpan();
    // 比例以「最短区间 5 秒 = 100%」为基准：区间越短比例越大。
    // 例：5s → 100%，10s → 50%，1min → 8.3%，10min → 0.8%。
    const pct = SPAN_MIN / span * 100;
    const pctText = pct >= 10 ? `${Math.round(pct)}%`
      : pct >= 1 ? `${pct.toFixed(1)}%` : `${pct.toFixed(2)}%`;
    lab.textContent = `${fmtSpan(span)} · ${pctText}`;
  }
}

/** 区间时长：统一用 h:mm:ss（不足 1 分钟时用 Ns 更直观） */
function fmtSpan(sec) {
  const s = Math.round(sec * 10) / 10;
  if (s < 60) return `${s % 1 === 0 ? s : s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const rest = Math.round(s - m * 60);
  if (m < 60) return `${m}:${String(rest).padStart(2, '0')}`;
  const h = Math.floor(m / 60);
  return `${h}:${String(m % 60).padStart(2, '0')}:${String(rest).padStart(2, '0')}`;
}

function setSpan(span, anchorTime) {
  const dur = timelineDuration();
  const prev = viewSpan();
  const at = anchorTime ?? (prev.start + prev.span / 2);
  const next = clampSpan(span);
  S.span = next;
  S.zoom = dur / next;                     // 同步派生的缩放倍数
  S.viewStart = Math.max(0, Math.min(Math.max(0, dur - next), at - next / 2));
  updateZoomUI();
  drawTimeline();
}

function setZoom(z, anchorTime) {
  // 兼容旧调用：把倍数换算成区间
  const dur = timelineDuration();
  setSpan(dur / Math.max(0.01, z), anchorTime);
}

// ---------------- 声纹管理 ----------------

const VP = { speak: '', sing: '', playing: null };

async function openVoiceprint() {
  $('modal-voice').classList.remove('hidden');
  await refreshVoiceList();
}

async function refreshVoiceList() {
  try {
    const d = await apiJson('/api/profiles');
    const box = $('vp-list');
    if (!d.profiles.length) {
      box.innerHTML = '<p class="muted small">还没有注册任何声纹。</p>';
      return;
    }
    box.innerHTML = d.profiles.map((name) => `
      <div class="vp-item" data-name="${escapeHtml(name)}">
        <span class="name">${escapeHtml(name)}</span>
        <span class="tags">已注册</span>
        <button class="btn sm act-use">用于识别</button>
        <button class="btn sm act-del">删除</button>
      </div>`).join('');
    box.querySelectorAll('.vp-item').forEach((item) => {
      const name = item.dataset.name;
      item.querySelector('.act-use').onclick = () => {
        $('profile').value = name;
        $('opt-voiceprint').checked = true;
        $('modal-voice').classList.add('hidden');
        setSaveState(`已选用声纹「${name}」`, 'ok');
      };
      item.querySelector('.act-del').onclick = async () => {
        if (!confirm(`删除声纹「${name}」？`)) return;
        try {
          await apiJson('/api/profiles/delete', 'POST', { name });
          await refreshVoiceList();
          await loadProfiles();
        } catch (e) { alert('删除失败：' + e.message); }
      };
    });
  } catch (e) {
    $('vp-list').innerHTML = `<p class="muted small">读取失败：${escapeHtml(e.message)}</p>`;
  }
}

async function pickVoiceSample(kind) {
  let path = '';
  if (window.desktop) path = await window.desktop.openFile('audio');
  else path = prompt('音频文件路径：') || '';
  if (!path) return;
  VP[kind] = path;
  const label = kind === 'speak' ? $('vp-speak-label') : $('vp-sing-label');
  const audio = kind === 'speak' ? $('vp-speak-audio') : $('vp-sing-audio');
  const drop = kind === 'speak' ? $('vp-speak-drop') : $('vp-sing-drop');
  label.textContent = path.split(/[\\/]/).pop();
  drop.classList.add('filled');
  audio.src = mediaUrl(path);
  audio.classList.remove('hidden');
}

async function saveVoiceprint() {
  const name = ($('vp-name').value || '').trim();
  if (!name) { $('vp-status').textContent = '请填写主播名'; return; }
  if (!VP.speak) { $('vp-status').textContent = '请先选择说话样本'; return; }
  $('vp-status').textContent = '正在提取声纹…';
  try {
    const d = await apiJson('/api/profiles', 'POST', {
      name, speak_wav: VP.speak, sing_wav: VP.sing || undefined,
    });
    $('vp-status').textContent = `已保存：${d.path}`;
    $('profile').value = name;
    $('opt-voiceprint').checked = true;
    await loadProfiles();
    await refreshVoiceList();
  } catch (e) {
    $('vp-status').textContent = '注册失败：' + e.message;
  }
}

// ---------------- 事件绑定 ----------------

function bindEvents() {
  $('btn-open').onclick = openVideo;
  $('btn-danmaku').onclick = importDanmaku;
  $('btn-import').onclick = importSubs;
  $('btn-generate').onclick = startGenerate;
  $('btn-cancel').onclick = async () => {
    if (S.job) await apiJson(`/api/jobs/${S.job}/cancel`, 'POST', {});
  };
  $('btn-undo').onclick = doUndo;
  $('btn-redo').onclick = doRedo;
  $('btn-qc').onclick = doQc;
  $('btn-export').onclick = showExport;
  $('btn-drafts').onclick = showDrafts;
  $('btn-drafts2').onclick = showDrafts;
  $('btn-cache').onclick = cleanCache;
  $('btn-left').onclick = () => togglePanel('left');
  $('btn-right').onclick = () => togglePanel('right');
  $('btn-drawer').onclick = () => setDrawer($('drawer').classList.contains('collapsed'));
  $('btn-shortcuts').onclick = () => {
    renderKeys();
    $('modal-keys').classList.remove('hidden');
  };
  $('keys-close').onclick = () => $('modal-keys').classList.add('hidden');
  $('keys-reset').onclick = () => {
    S.keys = { ...DEFAULT_KEYS };
    saveKeys();
    renderKeys();
  };
  $('draft-close').onclick = () => $('modal-drafts').classList.add('hidden');
  $('exp-cancel').onclick = () => $('modal-export').classList.add('hidden');
  $('exp-ok').onclick = doExport;
  $('exp-browse').onclick = async () => {
    if (!window.desktop) { alert('浏览器模式下请直接输入绝对路径。'); return; }
    const dir = await window.desktop.chooseDir();
    if (dir) $('exp-dir').value = dir;
  };

  $('btn-split').onclick = doSplit;
  $('btn-merge').onclick = doMerge;
  $('btn-delete').onclick = doDelete;
  $('btn-insert').onclick = doInsert;
  $('btn-copy').onclick = doCopy;
  $('btn-paste').onclick = doPaste;
  $('btn-dup').onclick = doDuplicate;
  $('btn-zoom-fit').onclick = () => { S.viewStart = 0; setSpan(SPAN_MAX); };
  $('btn-zoom-subs').onclick = fitReadable;
  $('zoom').oninput = () => setSpan(sliderToSpan($('zoom').value));

  // 声纹管理
  $('btn-voiceprint').onclick = openVoiceprint;
  $('vp-close').onclick = () => $('modal-voice').classList.add('hidden');
  $('vp-save').onclick = saveVoiceprint;
  $('vp-refresh').onclick = refreshVoiceList;
  $('vp-speak-drop').onclick = () => pickVoiceSample('speak');
  $('vp-sing-drop').onclick = () => pickVoiceSample('sing');

  // 歌词库
  $('btn-lyrics').onclick = openLyrics;
  $('ly-close').onclick = () => $('modal-lyrics').classList.add('hidden');
  $('ly-fetch').onclick = fetchLyric;
  $('ly-import').onclick = importLyricFile;
  $('ly-paste').onclick = pasteLyric;
  // 拖入音频文件
  for (const kind of ['speak', 'sing']) {
    const drop = $(kind === 'speak' ? 'vp-speak-drop' : 'vp-sing-drop');
    drop.addEventListener('dragover', (e) => {
      e.preventDefault();
      drop.classList.add('over');
    });
    drop.addEventListener('dragleave', () => drop.classList.remove('over'));
    drop.addEventListener('drop', (e) => {
      e.preventDefault();
      drop.classList.remove('over');
      const f = e.dataTransfer?.files?.[0];
      if (!f) return;
      const p = f.path || '';
      if (!p) { $('vp-status').textContent = '请用点击选择（浏览器模式无文件路径）'; return; }
      VP[kind] = p;
      const label = $(kind === 'speak' ? 'vp-speak-label' : 'vp-sing-label');
      const audio = $(kind === 'speak' ? 'vp-speak-audio' : 'vp-sing-audio');
      label.textContent = f.name;
      drop.classList.add('filled');
      audio.src = mediaUrl(p);
      audio.classList.remove('hidden');
    });
  }

  $('filter').oninput = renderList;

  // 播放控制
  $('btn-play').onclick = () => {
    const v = $('video');
    v.paused ? v.play().catch(() => {}) : v.pause();
  };
  $('seek').oninput = () => {
    $('video').currentTime = Number($('seek').value) / 1000;
  };

  // 列表交互
  $('list-wrap').addEventListener('click', (e) => {
    const i = rowIndex(e.target);
    if (i < 0) return;
    selectRow(i, e);
    if (!e.target.isContentEditable) jumpToLine(i);
  });
  $('list-wrap').addEventListener('dblclick', (e) => {
    const t = e.target.closest('.row-text');
    if (!t) return;
    t.contentEditable = 'true';
    t.focus();
    document.execCommand('selectAll', false, null);
  });
  $('list-wrap').addEventListener('blur', async (e) => {
    const t = e.target.closest?.('.row-text');
    if (!t) return;
    t.contentEditable = 'false';
    const i = rowIndex(t);
    const raw = t.textContent.trim();
    if (i < 0 || !S.lines[i] || raw === S.lines[i].text) return;
    await editOp('update', { index: i, text: raw });
    await saveNow('text');
  }, true);
  $('list-wrap').addEventListener('contextmenu', (e) => {
    const i = rowIndex(e.target);
    if (i < 0) return;
    e.preventDefault();
    showContextMenu(e.clientX, e.clientY, i);
  });

  // 时间轴：点击定位 / 片段拖动 / 右键
  const lanes = $('tl-lanes');
  lanes.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    const edge = e.target.closest('.edge');
    if (edge) { beginClipDrag(e, edge.dataset.edge); return; }
    const clip = e.target.closest('.clip');
    if (clip) {
      const i = Number(clip.dataset.index);
      // 已多选时按住其中一个拖动 = 整体移动（保持多选）
      const multi = S.selection.size > 1 && S.selection.has(i);
      if (!multi) selectRow(i, e);
      beginClipDrag(e, 'move');
      return;
    }
    // 空白处：开始框选（拖动超过阈值即框选，否则松手时定位播放头）
    beginMarquee(e);
  });
  lanes.addEventListener('contextmenu', (e) => {
    const clip = e.target.closest('.clip');
    e.preventDefault();
    showContextMenu(e.clientX, e.clientY,
                    clip ? Number(clip.dataset.index) : null);
  });
  // 滚轮：以光标处为锚点缩放；Shift+滚轮横向平移
  lanes.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (e.shiftKey) {
      const { span } = viewSpan();
      S.viewStart = Math.max(0, S.viewStart + (e.deltaY > 0 ? 1 : -1) * span * 0.2);
      drawTimeline();
    } else {
      const rect = lanes.getBoundingClientRect();
      const anchor = xToTime(e.clientX - rect.left);
      const factor = e.deltaY < 0 ? 1 / 1.2 : 1.2;
      setSpan(viewSpan().span * factor, anchor);
    }
  }, { passive: false });

  // 预览字幕：直接按住拖动即定位（自由模式下）
  $('ov-line').addEventListener('mousedown', beginOverlayDrag);

  // 字幕样式
  $('st-size').oninput = () => {
    S.style.size = Number($('st-size').value);
    $('st-size-v').textContent = S.style.size;
    applyStyleToOverlay(); saveStyle();
  };
  $('st-color').oninput = () => {
    S.style.color = $('st-color').value;
    applyStyleToOverlay(); saveStyle();
  };
  $('st-outline').oninput = () => {
    S.style.outline = Number($('st-outline').value);
    $('st-outline-v').textContent = S.style.outline;
    applyStyleToOverlay(); saveStyle();
  };
  $('st-anchor').onchange = () => {
    S.style.anchor = $('st-anchor').value;
    if (S.style.anchor !== 'free') { S.style.x = null; S.style.y = null; }
    syncStyleControls(); saveStyle();
  };
  $('st-x').oninput = $('st-y').oninput = () => {
    const x = Number($('st-x').value), y = Number($('st-y').value);
    if (!Number.isNaN(x)) S.style.x = Math.max(0, Math.min(1920, x));
    if (!Number.isNaN(y)) S.style.y = Math.max(0, Math.min(1080, y));
    if (S.style.anchor !== 'free') {
      S.style.anchor = 'free';
      $('st-anchor').value = 'free';
    }
    applyStyleToOverlay(); saveStyle();
  };
  $('st-bg').onchange = () => {
    S.style.bg = $('st-bg').checked;
    applyStyleToOverlay(); saveStyle();
  };
  $('st-preview').onchange = () => {
    S.style.preview = $('st-preview').checked;
    updateOverlay(); saveStyle();
  };
  $('opt-autosave').onchange = () => {
    S.autosave = $('opt-autosave').checked;
  };
  document.querySelectorAll('.preset').forEach((btn) => {
    btn.onclick = () => {
      Object.assign(S.style, PRESETS[btn.dataset.preset] || {});
      syncStyleControls(); saveStyle();
      document.querySelectorAll('.preset').forEach(
        (b) => b.classList.toggle('active', b === btn));
    };
  });

  // 抽屉页签
  document.querySelectorAll('.drawer-tabs .tab').forEach((tab) => {
    tab.onclick = () => {
      document.querySelectorAll('.drawer-tabs .tab')
        .forEach((t) => t.classList.toggle('active', t === tab));
      document.querySelectorAll('.tab-pane')
        .forEach((p) => p.classList.toggle('active',
          p.id === 'tab-' + tab.dataset.tab));
      setDrawer(true);
    };
  });

  // 视频联动
  const v = $('video');
  v.addEventListener('timeupdate', () => {
    $('status-pos').textContent = fmt(v.currentTime);
    if (!$('seek').matches(':active')) {
      $('seek').value = String(Math.round(v.currentTime * 1000));
    }
    updatePlayhead();
    // 播放中不做「选中即预览」，严格按时间轴显示
    S._previewSelection = false;
    updateOverlay();
    const i = S.lines.findIndex((l) => v.currentTime >= l.start
      && v.currentTime < l.end);
    if (i !== S.current) {
      S.current = i;
      renderList();
      renderClips();
    }
  });
  v.addEventListener('play', () => {
    S._previewSelection = false;
    $('btn-play').textContent = '❚❚';
    updateOverlay();
  });
  v.addEventListener('pause', () => {
    $('btn-play').textContent = '▶';
  });
  v.addEventListener('seeked', () => {
    S._previewSelection = false;
    updateOverlay();
  });
  window.addEventListener('resize', () => { drawTimeline(); updateOverlay(); });
  document.addEventListener('click', hideContextMenu);
  $('ctxmenu').addEventListener('click', (e) => e.stopPropagation());

  // 快捷键
  document.addEventListener('keydown', (e) => {
    if (e.target.isContentEditable
        || ['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
    const name = keyName(e);
    const hit = Object.keys(S.keys).find((k) => S.keys[k] === name);
    if (!hit) return;
    e.preventDefault();
    ({
      play: () => { const vid = $('video'); vid.paused ? vid.play() : vid.pause(); },
      undo: doUndo, redo: doRedo, split: doSplit, merge: doMerge,
      delete: doDelete, qc: doQc, export: showExport, open: openVideo,
      copy: doCopy, paste: doPaste, duplicate: doDuplicate,
      prev: () => { $('video').currentTime -= 0.5; },
      next: () => { $('video').currentTime += 0.5; },
    }[hit] || (() => {}))();
  });

  if (window.desktop) {
    window.desktop.onMenu((action) => runAction(action));
  }
}

boot();
