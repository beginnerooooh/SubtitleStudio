; ============================================================================
;  Subtitle Studio — Inno Setup 安装包脚本（Inno Setup 6.x）
;
;  编译（两种方式任选其一）：
;    1) Inno Setup Compiler 打开本文件，版本号默认取下方 #define
;    2) 命令行传入版本（build.bat --installer 自动调用）：
;       iscc /DMyAppVersion=1.0.0 packaging\installer.iss
;
;  产物：dist\SubtitleStudio_Setup_v<版本>.exe
;  输入：..\dist\SubtitleStudio\*（先运行 build_portable.py 生成）
; ============================================================================

#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif

#define MyAppName "Subtitle Studio"
#define MyAppNameZh "Subtitle Studio 字幕工坊"
#define MyAppPublisher "Subtitle Studio Team"
#define MyAppURL "https://example.com/subtitle-studio"
#define MyAppId "{{7C1F2E64-9A3B-4F5D-8E77-2B9C0D4A1E55}"

[Setup]
; 唯一 AppId（升级/卸载识别用，切勿改动）
AppId={#MyAppId}
AppName={#MyAppNameZh}
AppVersion={#MyAppVersion}
AppVerName={#MyAppNameZh} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
; 标准安装位置：Program Files（admin 权限）；便携偏好者可直接解压分发 dist 目录
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppNameZh}
AllowNoIcons=yes
; 自定义图标（缺失时自动降级为默认图标，不影响编译）
#if FileExists("app.ico")
SetupIconFile=app.ico
UninstallDisplayIcon={app}\app.ico
#endif
; 压缩：lzma2/ultra64 + 固实压缩（site-packages 为文本型 py/dll，压缩比高）
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
; 仅 64 位（PyTorch / CUDA 运行库均为 x64）
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
MinVersion=10.0
; 输出
OutputDir=..\dist
OutputBaseFilename=SubtitleStudio_Setup_v{#MyAppVersion}
; 不写系统环境变量（便携设计：PATH/HF_HOME 全部由 launcher.py 进程内注入）
ChangesEnvironment=no
DisableProgramGroupPage=yes

[Languages]
; Inno Setup 6 自带中文语言包（compiler:Languages\ChineseSimplified.isl）
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
; 中文向导附加文案（英文页回退默认值）
chinesesimplified.VCRedistMissing=未检测到 Microsoft Visual C++ 2015-2022 运行库（x64）。%n%nPyTorch / CTranslate2 等组件依赖它，无法启动时请先安装：%nhttps://aka.ms/vs/17/release/vc_redist.x64.exe
english.VCRedistMissing=Microsoft Visual C++ 2015-2022 Redistributable (x64) was not detected.%n%nPyTorch / CTranslate2 components require it. Download from:%nhttps://aka.ms/vs/17/release/vc_redist.x64.exe
chinesesimplified.UninstallKeepData=卸载完成前，是否同时删除用户数据？%n%n• 输出字幕与歌单（outputs）%n• 主播声纹库（profiles）%n• 已下载模型（models，体积较大）%n%n选择「否」将保留以上文件，重装后可直接续用。
english.UninstallKeepData=Also remove user data during uninstall?%n%n• Subtitles & tracklists (outputs)%n• Streamer voiceprint profiles (profiles)%n• Downloaded models (models, large)%n%nChoose "No" to keep them for future reinstall.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; \
    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1

[Files]
; 整个便携目录原样落盘（runtime/app/bin/models 全量）
Source: "..\dist\SubtitleStudio\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs
; 许可协议
Source: "LICENSE.txt"; DestDir: "{app}"; Flags: ignoreversion
; 可选：内置 VC++ 运行库静默安装（将 vc_redist.x64.exe 放入 packaging\redist\
; 并取消下方两段注释，即可实现无网机器的完整部署）
; Source: "redist\vc_redist.x64.exe"; DestDir: "{tmp}"; \
;     Flags: deleteafterinstall; Check: VCRedistNeedsInstall

[Icons]
#if FileExists("app.ico")
Name: "{group}\{#MyAppNameZh}"; Filename: "{app}\run.bat"; \
    WorkingDir: "{app}"; IconFilename: "{app}\app.ico"
Name: "{group}\{#MyAppNameZh}（调试模式）"; Filename: "{app}\run-debug.bat"; \
    WorkingDir: "{app}"; IconFilename: "{app}\app.ico"
Name: "{group}\停止 {#MyAppNameZh}"; Filename: "{app}\stop.bat"; \
    WorkingDir: "{app}"; IconFilename: "{app}\app.ico"
Name: "{autodesktop}\{#MyAppNameZh}"; Filename: "{app}\run.bat"; \
    WorkingDir: "{app}"; IconFilename: "{app}\app.ico"; Tasks: desktopicon
#else
Name: "{group}\{#MyAppNameZh}"; Filename: "{app}\run.bat"; WorkingDir: "{app}"
Name: "{group}\{#MyAppNameZh}（调试模式）"; Filename: "{app}\run-debug.bat"; WorkingDir: "{app}"
Name: "{group}\停止 {#MyAppNameZh}"; Filename: "{app}\stop.bat"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppNameZh}"; Filename: "{app}\run.bat"; \
    WorkingDir: "{app}"; Tasks: desktopicon
#endif

[Run]
; 安装完成页勾选「立即启动」
Filename: "{app}\run.bat"; Description: "启动 {#MyAppNameZh}"; \
    Flags: nowait postinstall skipifsilent shellexec
; 可选：静默安装 VC++ 运行库（配合上方注释的 [Files] 段）
; Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; \
;     Check: VCRedistNeedsInstall; Flags: waituntilterminated; StatusMsg: \
;     "Installing Visual C++ Redistributable..."

[UninstallRun]
; 卸载前先优雅停止运行中的服务（stop.flag → launcher 自行退出释放 GPU 显存）
Filename: "{cmd}"; Parameters: "/C type NUL > ""{app}\stop.flag"" && timeout /t 3 /nobreak >NUL"; \
    Flags: runhidden; RunOnceId: "StopService"

[UninstallDelete]
; 仅清理运行期生成的临时文件；用户数据（outputs/profiles/models）由 [Code] 询问后处理
Type: files; Name: "{app}\stop.flag"
Type: filesandordirs; Name: "{app}\logs"

[Code]
// ---------------- VC++ 2015-2022 (x64) 运行库检测 ----------------
function VCRedistNeedsInstall(): Boolean;
var
  Installed: Cardinal;
begin
  Result :=
    (not RegQueryDWordValue(HKEYLM,
        'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',
        'Installed', Installed)) or (Installed <> 1);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  // 安装收尾：缺失 VC++ 运行库时给出明确指引（PyTorch DLL 加载依赖）
  if (CurStep = ssPostInstall) and VCRedistNeedsInstall() then
    MsgBox(ExpandConstant('{cm:VCRedistMissing}'), mbInformation, MB_OK);
end;

// ---------------- 卸载时询问是否保留用户数据 ----------------
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    if MsgBox(ExpandConstant('{cm:UninstallKeepData}'),
              mbConfirmation, MB_YESNO) = IDYES then
    begin
      DelTree(ExpandConstant('{app}\outputs'), True, True, True);
      DelTree(ExpandConstant('{app}\profiles'), True, True, True);
      DelTree(ExpandConstant('{app}\models'), True, True, True);
    end;
  end;
end;
