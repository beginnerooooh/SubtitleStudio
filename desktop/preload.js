/**
 * 预加载脚本：在隔离环境下暴露最小的桌面能力给渲染层。
 *
 * 安全边界：渲染层拿不到 Node/fs，只能通过下面这几个受控通道
 * 调用主进程（文件对话框、目录定位、应用信息、菜单事件）。
 */
'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('desktop', {
  /** 选择视频/音频文件（返回绝对路径，空串表示取消） */
  openMedia: () => ipcRenderer.invoke('dialog:openMedia'),
  /** 选择弹幕/字幕/音频文件；kind: danmaku | subtitle | audio */
  openFile: (kind) => ipcRenderer.invoke('dialog:openAny', kind),
  /** 选择导出目录（返回绝对路径，空串表示取消） */
  chooseDir: () => ipcRenderer.invoke('dialog:chooseDir'),
  /** 另存为（返回绝对路径，空串表示取消） */
  saveAs: (defaultName) => ipcRenderer.invoke('dialog:saveAs', defaultName),
  /** 应用与后端信息：{root, port, version, platform, dev} */
  info: () => ipcRenderer.invoke('app:info'),
  /** 在系统文件管理器中定位文件 */
  reveal: (p) => ipcRenderer.invoke('shell:reveal', p),
  /** 订阅菜单命令（open-video / undo / redo / split / merge / delete / qc / export ...） */
  onMenu: (handler) => {
    ipcRenderer.on('menu', (_e, action) => handler(action));
  },
  isDesktop: true,
});
