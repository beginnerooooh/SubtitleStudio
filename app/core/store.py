"""项目与草稿存储：自动保存、撤销/重做、草稿箱、缓存清理。

为桌面端（Electron）与 Web UI 提供统一的持久化后端：
- **项目**：一次编辑会话的完整状态（视频路径、字幕行、参数、历史），
  落在 `projects/<id>/project.json`；
- **自动保存**：编辑后 debounce 写入（避免频繁磁盘 IO），崩溃后可恢复；
- **撤销/重做**：内存中的快照栈（每步编辑压栈），并定期落盘；
- **草稿箱**：未正式命名的项目自动进入草稿列表，可恢复/删除；
- **缓存清理**：按大小/时间清理中间产物（wav、切片、模型缓存），
  只清「可再生成」的东西，绝不删用户字幕与声纹。

存储布局
────────
    projects/
      index.json               项目索引（列表页读它，避免扫目录）
      <id>/
        project.json           当前状态（含 lines 与 settings）
        history/               撤销栈快照（滚动保留最近 N 步）
        autosave.json          自动保存点（崩溃恢复用）
        exports/               导出的字幕文件
    cache/                     可清理的中间产物
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# 撤销栈最多保留的步数（超出丢弃最旧的）
MAX_HISTORY = 60
# 自动保存防抖（秒）：编辑停止后这么久才落盘
AUTOSAVE_DEBOUNCE = 1.5
# 草稿自动过期（天）：超过则提示清理，不自动删除
DRAFT_TTL_DAYS = 30


@dataclass
class ProjectMeta:
    """项目索引条目（列表页展示）。"""

    id: str
    name: str = "未命名"
    video: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    line_count: int = 0
    duration: float = 0.0
    draft: bool = True                 # 未命名 → 草稿
    thumb: str = ""                    # 缩略图路径（可选）

    @property
    def age_days(self) -> float:
        return max(0.0, (time.time() - self.updated_at) / 86400.0)


@dataclass
class ProjectState:
    """项目完整状态（可 JSON 序列化）。"""

    meta: ProjectMeta
    lines: list[dict] = field(default_factory=list)     # 词级字幕行
    settings: dict = field(default_factory=dict)        # 生成参数
    songs: list[dict] = field(default_factory=list)
    bgm_flags: list[dict] = field(default_factory=list)  # BGM 判别明细
    log: list[str] = field(default_factory=list)


class ProjectStore:
    """项目仓库（线程安全；单进程内使用）。"""

    def __init__(self, root: str | Path = "projects"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._history: dict[str, list[dict]] = {}
        self._future: dict[str, list[dict]] = {}   # 重做栈
        self._last_save: dict[str, float] = {}

    # ---------------- 索引 ----------------

    def _index_path(self) -> Path:
        return self.root / "index.json"

    def list_projects(self, include_drafts: bool = True) -> list[ProjectMeta]:
        """项目列表（按更新时间倒序）。"""
        with self._lock:
            data = self._read_index()
        out = [ProjectMeta(**row) for row in data]
        if not include_drafts:
            out = [m for m in out if not m.draft]
        return sorted(out, key=lambda m: -m.updated_at)

    def drafts(self) -> list[ProjectMeta]:
        return [m for m in self.list_projects() if m.draft]

    def _read_index(self) -> list[dict]:
        path = self._index_path()
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _write_index(self, rows: list[dict]) -> None:
        self._index_path().write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    def _upsert_index(self, meta: ProjectMeta) -> None:
        rows = self._read_index()
        for i, row in enumerate(rows):
            if row.get("id") == meta.id:
                rows[i] = asdict(meta)
                break
        else:
            rows.append(asdict(meta))
        self._write_index(rows)

    def _drop_index(self, pid: str) -> None:
        rows = [r for r in self._read_index() if r.get("id") != pid]
        self._write_index(rows)

    # ---------------- 读写 ----------------

    def project_dir(self, pid: str) -> Path:
        return self.root / pid

    def create(self, video: str = "", name: str = "未命名",
               settings: Optional[dict] = None,
               draft: Optional[bool] = None) -> ProjectState:
        """新建项目。

        draft 未显式指定时按名字判断：叫「未命名」即视为草稿，
        这样「新建后还没命名」的项目会自动进草稿箱，避免堆积在正式列表里。
        """
        pid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        now = time.time()
        is_draft = (name == "未命名") if draft is None else bool(draft)
        meta = ProjectMeta(id=pid, name=name or "未命名", video=video,
                           created_at=now, updated_at=now, draft=is_draft)
        state = ProjectState(meta=meta, settings=dict(settings or {}))
        d = self.project_dir(pid)
        (d / "history").mkdir(parents=True, exist_ok=True)
        (d / "exports").mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._write_state(state)
            self._upsert_index(meta)
            self._history[pid] = []
            self._future[pid] = []
        return state

    def load(self, pid: str) -> ProjectState:
        path = self.project_dir(pid) / "project.json"
        if not path.is_file():
            raise FileNotFoundError(f"项目不存在：{pid}")
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = ProjectMeta(**data.get("meta", {}))
        state = ProjectState(
            meta=meta, lines=data.get("lines", []),
            settings=data.get("settings", {}), songs=data.get("songs", []),
            bgm_flags=data.get("bgm_flags", []), log=data.get("log", [])[-400:])
        with self._lock:
            self._history.setdefault(pid, [])
            self._future.setdefault(pid, [])
        return state

    def save(self, state: ProjectState, *, push_history: bool = True,
             note: str = "") -> None:
        """保存项目；push_history=True 时把当前状态压入撤销栈。

        自动保存（autosave）与显式保存都走这里，区别只在 debounce。
        """
        with self._lock:
            if push_history:
                self._push_history(state.meta.id, note)
            state.meta.updated_at = time.time()
            self._write_state(state)
            self._upsert_index(state.meta)
            self._last_save[state.meta.id] = time.time()

    def autosave(self, state: ProjectState) -> bool:
        """防抖自动保存（返回是否真的写了盘）。"""
        pid = state.meta.id
        now = time.time()
        with self._lock:
            if now - self._last_save.get(pid, 0.0) < AUTOSAVE_DEBOUNCE:
                return False
        self.save(state, push_history=False, note="autosave")
        d = self.project_dir(pid)
        (d / "autosave.json").write_text(
            json.dumps({"at": time.time(), "lines": state.lines},
                       ensure_ascii=False), encoding="utf-8")
        return True

    def _write_state(self, state: ProjectState) -> None:
        d = self.project_dir(state.meta.id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "project.json").write_text(
            json.dumps(asdict(state), ensure_ascii=False), encoding="utf-8")

    def recover_autosave(self, pid: str) -> Optional[list[dict]]:
        """崩溃恢复：返回 autosave.json 里的字幕行（若无返回 None）。"""
        path = self.project_dir(pid) / "autosave.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("lines")
        except Exception:
            return None

    def rename(self, pid: str, name: str) -> ProjectMeta:
        """重命名（同时把草稿转为正式项目）。"""
        state = self.load(pid)
        state.meta.name = (name or "").strip() or "未命名"
        state.meta.draft = state.meta.name == "未命名"
        self.save(state, push_history=False)
        return state.meta

    def delete(self, pid: str) -> None:
        """删除项目及其目录。"""
        with self._lock:
            shutil.rmtree(self.project_dir(pid), ignore_errors=True)
            self._drop_index(pid)
            self._history.pop(pid, None)
            self._future.pop(pid, None)

    # ---------------- 撤销 / 重做 ----------------

    def _push_history(self, pid: str, note: str) -> None:
        """把「当前」状态快照入栈（在写入新状态之前调用）。"""
        path = self.project_dir(pid) / "project.json"
        if not path.is_file():
            return
        try:
            snapshot = path.read_text(encoding="utf-8")
        except Exception:
            return
        stack = self._history.setdefault(pid, [])
        stack.append({"at": time.time(), "note": note, "data": snapshot})
        if len(stack) > MAX_HISTORY:
            stack.pop(0)
            self._trim_history_files(pid)
        self._future[pid] = []          # 新编辑使重做栈失效
        self._write_history_file(pid, stack)

    def _write_history_file(self, pid: str, stack: list[dict]) -> None:
        d = self.project_dir(pid) / "history"
        d.mkdir(parents=True, exist_ok=True)
        (d / "stack.json").write_text(
            json.dumps([{"at": s["at"], "note": s["note"]} for s in stack],
                       ensure_ascii=False), encoding="utf-8")

    def _trim_history_files(self, pid: str) -> None:
        """历史目录只保留最近若干快照文件，避免无限增长。"""
        d = self.project_dir(pid) / "history"
        if not d.is_dir():
            return
        files = sorted(d.glob("snap-*.json"))
        for old in files[:-MAX_HISTORY]:
            old.unlink(missing_ok=True)

    def undo(self, pid: str) -> Optional[ProjectState]:
        """撤销一步（返回回退后的状态）。"""
        with self._lock:
            stack = self._history.get(pid) or []
            if not stack:
                return None
            current = (self.project_dir(pid) / "project.json")
            if current.is_file():
                self._future.setdefault(pid, []).append(
                    {"at": time.time(), "note": "redo",
                     "data": current.read_text(encoding="utf-8")})
            entry = stack.pop()
            current.write_text(entry["data"], encoding="utf-8")
        return self.load(pid)

    def redo(self, pid: str) -> Optional[ProjectState]:
        """重做一步。"""
        with self._lock:
            fut = self._future.get(pid) or []
            if not fut:
                return None
            current = self.project_dir(pid) / "project.json"
            if current.is_file():
                self._history.setdefault(pid, []).append(
                    {"at": time.time(), "note": "undo",
                     "data": current.read_text(encoding="utf-8")})
            entry = fut.pop()
            current.write_text(entry["data"], encoding="utf-8")
        return self.load(pid)

    def history_info(self, pid: str) -> dict:
        with self._lock:
            return {"undo": len(self._history.get(pid) or []),
                    "redo": len(self._future.get(pid) or [])}

    # ---------------- 缓存清理 ----------------

    def clean_cache(self, cache_dir: str | Path = "cache",
                    max_age_days: float = 7.0,
                    max_total_mb: float = 2048.0,
                    dry_run: bool = False) -> dict:
        """清理可再生成的中间产物。

        只清理 `cache/` 下的内容（wav 抽取、歌曲切片、临时对齐音频）；
        **不会**碰 `projects/`（用户字幕）、`profiles/`（声纹）、
        `lyrics/`（歌词库）。返回清理统计。
        """
        root = Path(cache_dir)
        stats = {"files": 0, "freed_mb": 0.0, "skipped": 0}
        if not root.is_dir():
            return stats
        now = time.time()
        entries: list[tuple[float, Path, int]] = []
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append((st.st_mtime, p, st.st_size))
        total_mb = sum(e[2] for e in entries) / 1024 / 1024
        # 1) 过期文件
        for mtime, path, size in entries:
            age_days = (now - mtime) / 86400.0
            if age_days > max_age_days:
                stats["files"] += 1
                stats["freed_mb"] += size / 1024 / 1024
                if not dry_run:
                    path.unlink(missing_ok=True)
        # 2) 超容量时按最旧优先继续删
        if total_mb - stats["freed_mb"] > max_total_mb:
            remaining = sorted((e for e in entries
                                if not dry_run or (now - e[0]) / 86400.0 <= max_age_days),
                               key=lambda e: e[0])
            need = total_mb - stats["freed_mb"] - max_total_mb
            for _mtime, path, size in remaining:
                if need <= 0:
                    break
                if not path.exists():
                    continue
                stats["files"] += 1
                stats["freed_mb"] += size / 1024 / 1024
                need -= size / 1024 / 1024
                if not dry_run:
                    path.unlink(missing_ok=True)
        stats["freed_mb"] = round(stats["freed_mb"], 1)
        # 清理空目录
        if not dry_run:
            for d in sorted(root.rglob("*"), reverse=True):
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
        return stats

    def clean_old_drafts(self, ttl_days: float = DRAFT_TTL_DAYS,
                         dry_run: bool = True) -> list[ProjectMeta]:
        """列出（或删除）长期未动的草稿。默认只列出，交由用户确认。"""
        stale = [m for m in self.drafts() if m.age_days > ttl_days]
        if dry_run:
            return stale
        for m in stale:
            self.delete(m.id)
        return stale


__all__ = ["ProjectMeta", "ProjectState", "ProjectStore", "MAX_HISTORY",
           "AUTOSAVE_DEBOUNCE", "DRAFT_TTL_DAYS"]
