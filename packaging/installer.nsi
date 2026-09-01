; ============================================================================
;  Subtitle Studio — NSIS 安装包脚本（Linux 交叉构建用，等价于 installer.iss）
;
;  背景：Inno Setup 的 ISCC 为 Windows PE 程序，需在 wine 中运行；部分 CI
;  沙箱禁止执行 32 位 ELF 且 wine9 无法通过 Inno 7 安装器的系统版本检查。
;  NSIS 的 makensis 为原生 Linux 程序，可在无 wine 环境直接编译，产物为
;  标准 Windows 安装器（7-Zip / Notepad++ 同款技术栈）。
;
;  编译：
;    makensis -DMyAppVersion=1.2.0 packaging/installer.nsi
;  产物：
;    dist/SubtitleStudio_Setup_v<版本>.exe
;  输入：
;    ../dist/SubtitleStudio/*（先运行 build_portable.py 生成）
; ============================================================================

Unicode true

!ifndef MyAppVersion
  !define MyAppVersion "1.3.0"
!endif

!define MyAppName "Subtitle Studio"
!define MyAppNameZh "Subtitle Studio 字幕工坊"
!define MyAppPublisher "Subtitle Studio Team"
!define MyAppURL "https://example.com/subtitle-studio"
!define MyAppUninstKey "Software\Microsoft\Windows\CurrentVersion\Uninstall\${MyAppName}"

Name "${MyAppNameZh} ${MyAppVersion}"
OutFile "..\dist\SubtitleStudio_Setup_v${MyAppVersion}.exe"
; 标准安装位置：Program Files（admin 权限）；便携偏好者可直接解压分发 dist 目录
InstallDir "$PROGRAMFILES64\${MyAppName}"
; 升级时沿用旧安装目录
InstallDirRegKey HKLM "Software\${MyAppName}" "InstallLocation"
RequestExecutionLevel admin
; 压缩：LZMA 固实压缩（site-packages 为文本型 py/dll，压缩比高）
SetCompressor /SOLID lzma
ShowInstDetails show
ShowUninstDetails show

; ---------------- MUI2 界面 ----------------
!include "MUI2.nsh"
!include "FileFunc.nsh"
!insertmacro GetSize

Var StartMenuFolder

!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_RUN "$INSTDIR\run.bat"
!define MUI_FINISHPAGE_RUN_TEXT "启动 ${MyAppNameZh}"
!define MUI_FINISHPAGE_SHOWREADME ""
!define MUI_FINISHPAGE_SHOWREADME_NOTCHECKED
!define MUI_FINISHPAGE_LINK "访问主页"
!define MUI_FINISHPAGE_LINK_LOCATION "${MyAppURL}"
!define MUI_FINISHPAGE_NOREBOOTSUPPORT

; 自定义图标（存在时生效；makensis 不支持 FileExists 判断，由构建脚本保证）
!if /FileExists "app.ico"
  !define MUI_ICON "app.ico"
  !define MUI_UNICON "app.ico"
!endif

; 安装向导页：许可 → 目录 → 开始菜单 → 安装 → 完成
!insertmacro MUI_PAGE_LICENSE "LICENSE.txt"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_STARTMENU "StartMenu" $StartMenuFolder
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

; 卸载向导页
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

; 语言（中文优先，对应 installer.iss 的 ChineseSimplified + English）
!insertmacro MUI_LANGUAGE "SimpChinese"
!insertmacro MUI_LANGUAGE "English"

; ---------------- 安装 ----------------
Section "!程序文件" SEC_CORE
  SectionIn RO
  SetOutPath "$INSTDIR"
  ; 整个便携目录原样落盘（runtime/app/bin/models 全量）
  File /r "..\dist\SubtitleStudio\*"
  ; 许可协议
  File /oname=LICENSE.txt "LICENSE.txt"

  ; 写注册表：安装位置 + 卸载信息（控制面板）
  WriteRegStr HKLM "Software\${MyAppName}" "InstallLocation" "$INSTDIR"
  WriteUninstaller "$INSTDIR\uninstall.exe"
  WriteRegStr HKLM "${MyAppUninstKey}" "DisplayName" "${MyAppNameZh}"
  WriteRegStr HKLM "${MyAppUninstKey}" "DisplayVersion" "${MyAppVersion}"
  WriteRegStr HKLM "${MyAppUninstKey}" "Publisher" "${MyAppPublisher}"
  WriteRegStr HKLM "${MyAppUninstKey}" "DisplayIcon" "$INSTDIR\app.ico"
  WriteRegStr HKLM "${MyAppUninstKey}" "URLInfoAbout" "${MyAppURL}"
  WriteRegStr HKLM "${MyAppUninstKey}" "URLUpdateInfo" "${MyAppURL}"
  WriteRegStr HKLM "${MyAppUninstKey}" "UninstallString" "$\"$INSTDIR\uninstall.exe$\""
  WriteRegDWORD HKLM "${MyAppUninstKey}" "NoModify" 1
  WriteRegDWORD HKLM "${MyAppUninstKey}" "NoRepair" 1
  ; EstimatedSize（KB）：安装器收尾估算，便于控制面板显示体积
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKLM "${MyAppUninstKey}" "EstimatedSize" "$0"
SectionEnd

; 桌面图标（默认不勾选，对应 installer.iss 的 unchecked task）
Section "桌面快捷方式" SEC_DESKTOP
  CreateShortCut "$DESKTOP\${MyAppNameZh}.lnk" "$INSTDIR\run.bat" "" "$INSTDIR\app.ico"
SectionEnd

; ---------------- 安装收尾：快捷方式 + VC++ 运行库检测 ----------------
Function .onInstSuccess
  ; 开始菜单（含调试模式与停止入口）
  !insertmacro MUI_STARTMENU_WRITE_BEGIN StartMenu
    CreateDirectory "$SMPROGRAMS\$StartMenuFolder"
    CreateShortCut "$SMPROGRAMS\$StartMenuFolder\${MyAppNameZh}.lnk" "$INSTDIR\run.bat" "" "$INSTDIR\app.ico"
    CreateShortCut "$SMPROGRAMS\$StartMenuFolder\${MyAppNameZh}（调试模式）.lnk" "$INSTDIR\run-debug.bat" "" "$INSTDIR\app.ico"
    CreateShortCut "$SMPROGRAMS\$StartMenuFolder\停止 ${MyAppNameZh}.lnk" "$INSTDIR\stop.bat" "" "$INSTDIR\app.ico"
  !insertmacro MUI_STARTMENU_WRITE_END

  ; VC++ 2015-2022 (x64) 运行库检测：缺失时给出明确指引
  ; （PyTorch / CTranslate2 的 DLL 加载依赖）
  ReadRegDWORD $0 HKLM "SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" "Installed"
  ${If} $0 != 1
    MessageBox MB_ICONINFORMATION|MB_OK "未检测到 Microsoft Visual C++ 2015-2022 运行库（x64）。$\r$\n$\r$\nPyTorch / CTranslate2 等组件依赖它，无法启动时请先安装：$\r$\nhttps://aka.ms/vs/17/release/vc_redist.x64.exe"
  ${EndIf}
FunctionEnd

; ---------------- 卸载 ----------------
Section "Uninstall"
  ; 先优雅停止运行中的服务（stop.flag → launcher 自行退出释放 GPU 显存）
  IfFileExists "$INSTDIR\run.bat" 0 +3
    FileOpen $0 "$INSTDIR\stop.flag" w
    FileClose $0
    Sleep 3000

  ; 开始菜单 + 桌面快捷方式
  !insertmacro MUI_STARTMENU_GETFOLDER StartMenu $StartMenuFolder
  Delete "$SMPROGRAMS\$StartMenuFolder\${MyAppNameZh}.lnk"
  Delete "$SMPROGRAMS\$StartMenuFolder\${MyAppNameZh}（调试模式）.lnk"
  Delete "$SMPROGRAMS\$StartMenuFolder\停止 ${MyAppNameZh}.lnk"
  RMDir "$SMPROGRAMS\$StartMenuFolder"
  Delete "$DESKTOP\${MyAppNameZh}.lnk"

  ; 仅清理程序本体与运行期临时文件；用户数据（outputs/profiles/models）先询问
  Delete "$INSTDIR\stop.flag"
  RMDir /r "$INSTDIR\logs"
  RMDir /r "$INSTDIR\runtime"
  RMDir /r "$INSTDIR\app"
  RMDir /r "$INSTDIR\bin"
  Delete "$INSTDIR\uninstall.exe"
  Delete "$INSTDIR\LICENSE.txt"
  Delete "$INSTDIR\version.txt"
  Delete "$INSTDIR\launcher.py"
  Delete "$INSTDIR\app.ico"
  Delete "$INSTDIR\run.bat"
  Delete "$INSTDIR\run-debug.bat"
  Delete "$INSTDIR\stop.bat"

  ; 询问是否同时删除用户数据（选择「否」将保留，重装后可直接续用）
  MessageBox MB_ICONQUESTION|MB_YESNO|MB_DEFBUTTON2 "卸载完成前，是否同时删除用户数据？$\r$\n$\r$\n• 输出字幕与歌单（outputs）$\r$\n• 主播声纹库（profiles）$\r$\n• 已下载模型（models，体积较大）$\r$\n$\r$\n选择「否」将保留以上文件，重装后可直接续用。" IDNO +4
    RMDir /r "$INSTDIR\outputs"
    RMDir /r "$INSTDIR\profiles"
    RMDir /r "$INSTDIR\models"

  RMDir "$INSTDIR"
  DeleteRegKey HKLM "${MyAppUninstKey}"
  DeleteRegKey HKLM "Software\${MyAppName}"
  SetAutoClose true
SectionEnd
