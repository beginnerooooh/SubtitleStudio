# 贡献指南

感谢你愿意改进 Subtitle Studio。本文说明如何报告问题、提交改动。

## 报告问题

请尽量包含：

- 操作系统与 Python 版本
- `python tools/cli.py info` 的输出（会隐藏本地路径之外的信息）
- 复现步骤与期望结果
- 相关日志（`--debug` 运行可保留控制台）

**请勿在 issue 中上传**：完整直播录像、含个人信息的弹幕 XML、真实主播名。
如需提供样本，请用几秒钟的截取片段，或对敏感信息做替换。

## 提交改动

1. Fork 仓库并从 `main` 拉出分支：

   ```bash
   git checkout -b fix/your-topic
   ```

2. 改动后确保检查通过：

   ```bash
   python -m pytest tests/ -q
   python tools/check_desktop_ids.py
   python tools/desensitize.py      # 数据脱敏（推送前必跑）
   ```

   > **数据脱敏是强制步骤。** 本项目在真实直播素材上开发，工作区里很容易
   > 残留真实主播名、房间号、本地绝对路径、样本文件名。这些一旦提交就进了
   > git 历史，很难彻底清除。因此仓库带了 `pre-push` 钩子，发现敏感信息会
   > **直接拒绝推送**。
   >
   > 把你自己的敏感词写进 `tools/sensitive.txt`（每行一个，该文件已加入
   > `.gitignore`，不会入库）。

3. 提交信息建议用「类型: 简述」：

   ```
   fix: 修复时间轴片段宽度不随缩放变化
   feat: 支持框选批量选择字幕
   docs: 补充歌词多源说明
   ```

4. 发起 Pull Request，说明改动动机与验证方式。

## 代码约定

- `app/core/` 下的模块以**纯函数优先**：编辑操作不修改入参，返回新列表
- 桌面端（`desktop/renderer/`）只做 UI 与交互，业务逻辑一律走 `app/desktop_server.py`
- 新增可配置项放入 `app/core/config.py`，并支持环境变量覆盖
- 涉及算法的改动，请在 `docs/EVALUATION.md` 补充实测数据

## 不要提交的内容

`.gitignore` 已排除下列内容，请勿强制添加：

- `runtime/`、`bin/`、`models/`、`desktop/node_modules/`（体积大、平台相关）
- `Sample/`、`outputs/`、`projects/`、`profiles/`、`cache/`、`logs/`
  （用户素材与运行时产物，可能含个人信息）
- `.env` 及任何含密钥的配置

## 新增识别后端

1. 在 `app/core/providers/` 新建模块，实现统一的 `transcribe()` 接口
2. 在 `app/core/providers/__init__.py` 注册
3. 补充最小单测（可用几秒的合成音频，无需真实素材）

## 许可

提交的代码将按本项目 [MIT 许可](LICENSE) 发布。
