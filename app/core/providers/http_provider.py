"""外部 HTTP 识别后端：把音频发给用户自建服务，取回词级结果。

契约（POST multipart 或 JSON 均可，服务端实现其一即可）：

    POST <endpoint>
      Authorization: Bearer <api_key>            # 有配置时
      Content-Type: application/json
      {"audio_base64": "...", "format": "wav", "language": "zh",
       "model": "...", "word_timestamps": true}

服务端应返回（结构从宽）：

    {"segments": [
        {"start": 1.23, "end": 4.56, "text": "字幕文本",
         "words": [{"text": "字", "start": 1.23, "end": 1.45, "prob": 0.9}]}
    ]}
或直接返回 OpenAI-Whisper 风格：
    {"segments": [{"start": ..., "end": ..., "text": ...}]}   # 无词级时按字均分

这样用户可以接自建 Whisper、阿里云、腾讯云、火山引擎等任何服务，
只需在外面套一层适配（甚至可以直接用它们原生的 verbose_json 响应）。
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Optional

from core.providers import ASRProvider, AsrError, AsrSegment, AsrWord, register_provider


@register_provider
class ExternalHTTPProvider(ASRProvider):
    name = "external"

    def __init__(self, endpoint: str = "", api_key: str = "",
                 timeout: float = 60.0, headers: Optional[dict] = None,
                 model: str = "", **_: object):
        if not endpoint:
            raise AsrError(
                "外部识别服务未配置：请设置 SUBTITLE_STUDIO_ASR_ENDPOINT，"
                "或在界面填写服务地址")
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.headers = dict(headers or {})
        self.model = model

    def transcribe(self, wav_path, *, language="auto", total_duration=None,
                   on_progress=None, cancel_event=None) -> list[AsrSegment]:
        if cancel_event is not None and cancel_event.is_set():
            from core.errors import TaskCancelled

            raise TaskCancelled("识别已取消")
        with open(wav_path, "rb") as fh:
            payload = {
                "audio_base64": base64.b64encode(fh.read()).decode("ascii"),
                "format": "wav",
                "language": language,
                "model": self.model,
                "word_timestamps": True,
            }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.endpoint, data=body, headers=headers,
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:400]
            except Exception:
                pass
            raise AsrError(f"外部识别服务返回 {exc.code}：{detail}") from exc
        except Exception as exc:
            raise AsrError(f"外部识别服务调用失败：{exc}") from exc
        if on_progress:
            on_progress(1.0)
        return parse_external_response(data)


def parse_external_response(data: object) -> list[AsrSegment]:
    """把外部服务的响应解析为统一的 AsrSegment 列表（容错优先）。"""
    if isinstance(data, dict):
        rows = data.get("segments") or data.get("results") or data.get("data")
    else:
        rows = data
    if not isinstance(rows, list):
        raise AsrError("外部识别服务返回结构无法识别（期望 {\"segments\": [...]}）")
    out: list[AsrSegment] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        start = _num(row.get("start", row.get("start_time", 0.0)))
        end = _num(row.get("end", row.get("end_time", start)))
        text = str(row.get("text") or row.get("sentence") or "").strip()
        words = []
        for w in row.get("words") or []:
            if not isinstance(w, dict):
                continue
            wt = str(w.get("text") or w.get("word") or "")
            if not wt:
                continue
            ws = _num(w.get("start", start))
            we = _num(w.get("end", ws))
            words.append(AsrWord(text=wt, start=ws, end=we,
                                 prob=_num(w.get("prob", w.get("probability", 1.0)))))
        if not words and text:
            words = _spread_words(text, start, end)
        if not words:
            continue
        out.append(AsrSegment(start=start, end=end, text=text or "".join(
            w.text for w in words), words=words,
            avg_logprob=_num(row.get("avg_logprob", 0.0)),
            no_speech_prob=_num(row.get("no_speech_prob", 0.0))))
    out.sort(key=lambda s: s.start)
    return out


def _spread_words(text: str, start: float, end: float) -> list[AsrWord]:
    """无词级时间时，按字符数均分（下游还会用能量精修）。"""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return []
    dur = max(0.0, end - start) / len(chars)
    return [AsrWord(text=c, start=start + dur * i, end=start + dur * (i + 1))
            for i, c in enumerate(chars)]


def _num(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


__all__ = ["ExternalHTTPProvider", "parse_external_response"]
