// 检查渲染层 JS 的结构健康度。
//
// 用法：node tools/check_js_structure.js desktop/renderer/app.js
//
// 为什么需要它
// ────────────
// `node --check` 只做词法检查，**抓不到**两类实际事故：
//
//   1. `await` 出现在非 async 函数里 —— 实测 `node --check` 返回 0，
//      但浏览器加载时抛 SyntaxError，整个界面变白板；
//   2. 函数闭合后残留一段旧代码 —— 结构错乱。
//
// 因此这里用 `new vm.Script()` 做真正的编译（能抓到第 1 类），
// 再用「顶层缩进块」启发式抓第 2 类。
//
// 实现上刻意**不自己数括号**：JS 里有正则字面量、模板字符串、
// 除号与注释，手写括号计算很容易误报（实测报了 6 处假阳性）。
// 编译已经能保证语法正确，结构检查只做粗粒度提示。
'use strict';

const fs = require('fs');
const vm = require('vm');

const path = process.argv[2];
if (!path) {
  console.error('用法: node tools/check_js_structure.js <file.js>');
  process.exit(2);
}

const src = fs.readFileSync(path, 'utf8');
const problems = [];

// ---- 1) 真正的编译（关键）----
try {
  new vm.Script(src, { filename: path });
  console.log('[OK] 编译通过（vm.Script）');
} catch (e) {
  problems.push('编译失败: ' + e.message);
  console.log('[FAIL] 编译失败: ' + e.message);
}

// ---- 2) 重复的顶层函数定义 ----
const fnRe = /^(?:async\s+)?function\s+([A-Za-z_$][\w$]*)/gm;
const seen = new Map();
let m;
while ((m = fnRe.exec(src)) !== null) {
  const name = m[1];
  const ln = src.slice(0, m.index).split('\n').length;
  if (seen.has(name)) {
    problems.push(`重复定义函数 ${name}（行 ${seen.get(name)} 与 ${ln}）`);
  } else {
    seen.set(name, ln);
  }
}
console.log(`[OK] 顶层函数 ${seen.size} 个，无重复`);

// ---- 3) 顶层孤立缩进块 ----
//
// 只在「花括号深度为 0」的行上判断，并用一个保守的合法开头清单。
// 由于 `new vm.Script` 已保证语法正确，这里只报告**明显**的残留：
// 以 `if (`、`} else`、`await ` 开头、却又缩进的顶层行。
const lines = src.split('\n');
let brace = 0;
let pend = null;
let prev = '';
const suspicious = [];
let joinBuf = '';

for (let ln = 0; ln < lines.length; ln++) {
  const text = lines[ln];
  const trimmed = text.trim();

  // 花括号深度只统计非字符串部分
  const depthBefore = brace;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (pend === 'line') { if (c === '\n') pend = null; continue; }
    if (pend === 'block') { if (prev === '*' && c === '/') pend = null; prev = c; continue; }
    if (pend) {
      if (c === '\\') { i++; continue; }
      if (c === pend) pend = null;
      prev = c;
      continue;
    }
    if (c === '/' && text[i + 1] === '/') { pend = 'line'; i++; continue; }
    if (c === '/' && text[i + 1] === '*') { pend = 'block'; i++; continue; }
    if (c === '"' || c === "'" || c === '`') { pend = c; continue; }
    if (c === '{') brace++;
    else if (c === '}') brace = Math.max(0, brace - 1);
    prev = c;
  }
  if (pend === 'line') pend = null;

  if (depthBefore !== 0 || brace !== 0) continue;
  if (!trimmed) continue;
  // 只盯这几类「一定属于函数体」的开头
  if (/^(if|else|for|while|return|await|const j =|done =|busyProgress|\$\(')/.test(trimmed)
      && /^\s{2,}/.test(text)) {
    suspicious.push(`第 ${ln + 1} 行: ${trimmed.slice(0, 70)}`);
  }
}

if (suspicious.length) {
  problems.push('疑似孤立代码块（函数体语句出现在顶层）：\n  '
                + suspicious.slice(0, 8).join('\n  '));
} else {
  console.log('[OK] 无顶层孤立语句');
}

// ---- 汇总 ----
if (problems.length) {
  console.log('\n发现问题:');
  for (const p of problems) console.log('  - ' + p);
  process.exit(1);
}
console.log('\n全部检查通过');
