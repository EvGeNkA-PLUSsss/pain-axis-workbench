# SPDX-FileCopyrightText: 2026 EvGeNkA-PLUSsss
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Бэкенд Ollama: любые модели из локального каталога Ollama.

Почему отдельный бэкенд, а не transformers
------------------------------------------
Ollama хранит модели в своём каталоге (у вас ``F:\\Ollama\\Models``) в виде GGUF-блобов
и отдаёт их только через HTTP-сервер. Читать эти файлы напрямую нельзя, поэтому
работа идёт через API:

* ``GET  /api/tags``   — список установленных моделей (сервер знает свой каталог);
* ``POST /api/pull``   — скачать новую модель **в тот же каталог Ollama**, а не в
  кэш Hugging Face;
* ``POST /api/chat``   — обычный чат;
* ``POST /api/show``   — сведения о модели (семейство, число блоков, квантизация);
* ``POST /api/generate`` с ``keep_alive: 0`` — выгрузить модель из памяти.

Главное ограничение, о котором надо сказать прямо: **Ollama не отдаёт внутренние
активации**. Ни hidden states, ни логиты по слоям через API не доступны, поэтому
оси, ползунки, проба «кнопки облегчения» и профиль по слоям для этих моделей
невозможны — доступен только чат. Приложение это показывает явно, а не делает вид,
что стиринг работает.

Если сервер не запущен, список моделей всё равно читается с диска (из манифестов
``OLLAMA_MODELS`` или ``~/.ollama/models``), чтобы каталог был виден.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Sequence

from ..registry import ModelSpec
from .base import Backend, GenResult, ProgressFn, SteerSpec

OLLAMA_PREFIX = "ollama:"
DEFAULT_HOST = "http://127.0.0.1:11434"
_TIMEOUT = 4.0


# ---------------------------------------------------------------------------
# Каталог моделей
# ---------------------------------------------------------------------------
def host() -> str:
    """Адрес сервера Ollama.

    Тонкость: в ``OLLAMA_HOST`` Ollama хранит адрес **привязки** сервера, и там
    часто стоит ``0.0.0.0:11434``. Как клиентский адрес он не годится (на Windows
    подключение к 0.0.0.0 не проходит), поэтому «все интерфейсы» заменяем на
    127.0.0.1, сохраняя порт.
    """
    raw = os.environ.get("OLLAMA_HOST", "").strip()
    if not raw:
        return DEFAULT_HOST
    if "://" not in raw:
        raw = "http://" + raw
    try:
        from urllib.parse import urlparse, urlunparse

        parts = urlparse(raw)
        h = (parts.hostname or "").strip()
        if h in ("", "0.0.0.0", "::", "[::]", "*"):
            port = parts.port or 11434
            return f"http://127.0.0.1:{port}"
        return urlunparse(parts).rstrip("/")
    except Exception:
        return DEFAULT_HOST


def models_dir() -> Path:
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return Path(env)
    return Path.home() / ".ollama" / "models"


def _request(path: str, payload: dict | None = None, timeout: float = _TIMEOUT):
    url = host() + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def server_available(timeout: float = 1.5) -> bool:
    try:
        with _request("/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def list_from_disk() -> list[dict]:
    """Модели из манифестов на диске — работает и без запущенного сервера."""
    root = models_dir() / "manifests"
    out: list[dict] = []
    if not root.exists():
        return out
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if len(parts) < 3:
            continue
        # registry.ollama.ai/library/qwen2.5-coder/7b -> qwen2.5-coder:7b
        name, tag = parts[-2], parts[-1]
        try:
            size = sum(b.get("size", 0) for b in json.loads(path.read_text(encoding="utf-8")).get("layers", []))
        except Exception:
            size = 0
        out.append({"name": f"{name}:{tag}", "size": size, "source": "диск"})
    return sorted(out, key=lambda m: m["name"])


def list_models() -> tuple[list[dict], str]:
    """Список моделей и состояние сервера: (модели, 'сервер' | 'диск' | 'пусто')."""
    try:
        with _request("/api/tags", timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = []
        for m in data.get("models", []):
            det = m.get("details") or {}
            models.append({
                "name": m.get("name") or m.get("model"),
                "size": int(m.get("size") or 0),
                "family": det.get("family", ""),
                "params": det.get("parameter_size", ""),
                "quant": det.get("quantization_level", ""),
                "source": "сервер",
            })
        return sorted(models, key=lambda m: m["name"]), "сервер"
    except Exception:
        disk = list_from_disk()
        return disk, ("диск" if disk else "пусто")


def show(name: str) -> dict:
    """Сведения о модели: семейство, параметры, число блоков."""
    try:
        with _request("/api/show", {"model": name}, timeout=8.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}
    info = data.get("model_info") or {}
    blocks = next((int(v) for k, v in info.items() if k.endswith(".block_count")), 0)
    hidden = next((int(v) for k, v in info.items() if k.endswith(".embedding_length")), 0)
    return {
        "family": data.get("details", {}).get("family", ""),
        "params": data.get("details", {}).get("parameter_size", ""),
        "quant": data.get("details", {}).get("quantization_level", ""),
        "n_layers": blocks, "hidden_size": hidden,
        "context": next((int(v) for k, v in info.items() if k.endswith(".context_length")), 0),
    }


def pull(name: str, progress: Callable[[str, float], None] | None = None,
         should_stop: Callable[[], bool] | None = None) -> str:
    """Скачать модель в каталог Ollama. Возвращает статус последней строки."""
    last = ""
    with _request("/api/pull", {"model": name, "stream": True}, timeout=60.0) as resp:
        for raw in resp:
            if should_stop is not None and should_stop():
                raise KeyboardInterrupt("загрузка отменена")
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            last = str(event.get("status") or last)
            total, done = event.get("total"), event.get("completed")
            frac = (done / total) if (total and done is not None) else None
            if progress:
                progress(f"Ollama: {last}", frac if frac is not None else 0.0)
            if event.get("error"):
                raise RuntimeError(f"Ollama: {event['error']}")
    return last


# ---------------------------------------------------------------------------
# Бэкенд
# ---------------------------------------------------------------------------
class OllamaBackend(Backend):
    """Чат через локальный сервер Ollama. Активации недоступны — только генерация."""

    is_real = True
    is_demo = False
    supports_axes = False

    def __init__(self, spec: ModelSpec, keep_alive: str = "5m"):
        super().__init__(spec)
        self.device = "ollama"
        self.dtype_str = "gguf"
        self.keep_alive = keep_alive
        self.info_extra: dict = {}

    @property
    def model_name(self) -> str:
        return self.spec.repo_id[len(OLLAMA_PREFIX):]

    def load(self, progress: ProgressFn = None, **kwargs) -> None:
        if not server_available(timeout=3.0):
            raise RuntimeError(
                f"Сервер Ollama не отвечает на {host()}.\n"
                "Запустите его командой «ollama serve» (или откройте приложение Ollama) "
                "и повторите загрузку."
            )
        names = {m["name"] for m in list_models()[0]}
        if self.model_name not in names:
            self._report(progress, f"Модели «{self.model_name}» нет в каталоге — скачиваю…", 0.1)
            pull(self.model_name, progress=lambda m, f: self._report(progress, m, f))
        self.info_extra = show(self.model_name)
        self.n_layers = int(self.info_extra.get("n_layers") or 0)
        self.hidden_size = int(self.info_extra.get("hidden_size") or 0)
        self.dtype_str = self.info_extra.get("quant") or "gguf"
        self.loaded = True
        self._report(
            progress,
            "Ollama готова: " + (f"{self.info_extra.get('params', '')} "
                                 f"{self.dtype_str}, блоков {self.n_layers}"
                                 if self.info_extra else self.model_name)
            + ". Внимание: активации через Ollama недоступны — оси и ползунки не работают.",
            1.0,
        )

    def unload(self) -> None:
        """Попросить Ollama выгрузить модель из памяти."""
        try:
            _request("/api/generate", {"model": self.model_name, "keep_alive": 0}, timeout=10.0)
        except Exception:
            pass
        self.loaded = False

    # ------------------------------------------------------------- генерация
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 60,
        temperature: float = 0.0,
        top_p: float = 0.95,
        steer: SteerSpec | None = None,
        axis=None,
        use_chat: bool = True,
        progress: ProgressFn = None,
        messages: Sequence[dict] | None = None,
    ) -> GenResult:
        if steer is not None:
            raise RuntimeError("Ollama не отдаёт активации: стиринг для этих моделей невозможен.")
        msgs = [dict(m) for m in (messages or []) if str(m.get("content", "")).strip()]
        if not msgs:
            msgs = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model_name,
            "messages": msgs,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": float(temperature or 0.0), "num_predict": int(max_new_tokens)},
        }
        with _request("/api/chat", payload, timeout=600.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("error"):
            raise RuntimeError(f"Ollama: {data['error']}")
        text = (data.get("message") or {}).get("content", "")
        return GenResult(text=text, prompt=msgs[-1].get("content", ""),
                         n_new_tokens=int(data.get("eval_count") or 0), trace=[],
                         steered=False, alpha=0.0, layer=-1)

    # --------------------------------------------------------------- нечем
    def collect(self, texts, pooling="last", use_chat=False, batch_size=4,
                progress: ProgressFn = None):
        raise RuntimeError(
            "Ollama не отдаёт внутренние активации, поэтому оси, профиль по слоям и "
            "проба недоступны. Для осей нужна модель Hugging Face (вкладка «Hugging Face»)."
        )

    def score_continuations(self, prompt, continuations, steer=None, use_chat=False):
        raise RuntimeError(
            "Ollama не отдаёт лог-вероятности, поэтому проба «кнопки облегчения» для "
            "этих моделей недоступна."
        )
