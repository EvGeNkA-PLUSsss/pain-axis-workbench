# SPDX-FileCopyrightText: 2026 EvGeNkA-PLUSsss
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Каталог Hugging Face: поиск любых моделей, их реальные параметры и кэш метаданных.

Что делает модуль
-----------------
* ``search(query)`` — ищет модели в каталоге Hugging Face (сортировка по загрузкам),
  возвращает id, число параметров (из safetensors, если автор его публикует),
  загрузки, лайки, теги и признак gated;
* ``model_meta(repo)`` — читает ``config.json`` прямо из репозитория, чтобы узнать
  число слоёв, размер скрытого состояния и контекст — без скачивания весов;
* ``load_custom()/save_custom()`` — помнит добавленные вручную модели между
  запусками (``cache/custom_models.json``);
* ``load_meta_cache()/cache_meta()`` — кэширует всё найденное, чтобы список
  открывался мгновенно и работал без сети.

Сеть не обязательна: при отсутствии доступа функции возвращают пустой результат или
данные из кэша, а интерфейс показывает, что каталог недоступен.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

CACHE_DIR = Path("cache")
CUSTOM_FILE = CACHE_DIR / "custom_models.json"
META_FILE = CACHE_DIR / "models_meta.json"

# id моделей, которые заведомо не языковые генераторы — отфильтровываем в поиске
_SKIP_TAGS = {"diffusers", "text-to-image", "text-to-video", "audio", "image-classification",
              "object-detection", "segmentation", "automatic-speech-recognition",
              "text-to-speech", "image-to-text", "depth-estimation"}


def _hub():
    try:
        from huggingface_hub import HfApi, hf_hub_download  # noqa: F401

        return HfApi(), hf_hub_download
    except Exception:
        return None, None


def hub_available() -> bool:
    api, _ = _hub()
    return api is not None


# ---------------------------------------------------------------------------
# Кэш метаданных
# ---------------------------------------------------------------------------
def load_meta_cache() -> dict:
    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def cache_meta(repo: str, meta: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = load_meta_cache()
        data[repo] = {**meta, "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
        META_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


def params_from_config(cfg: dict) -> float:
    """Грубая оценка числа параметров (млрд) по конфигурации модели."""
    try:
        layers = int(cfg.get("num_hidden_layers") or cfg.get("n_layer") or 0)
        hidden = int(cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model") or 0)
        inter = int(cfg.get("intermediate_size") or cfg.get("n_inner") or hidden * 4)
        vocab = int(cfg.get("vocab_size") or 32000)
        heads = int(cfg.get("num_attention_heads") or 1)
        kv_heads = int(cfg.get("num_key_value_heads") or heads)
        if not layers or not hidden:
            return 0.0
        # внимание + MLP на слой, делённое на число «групп» при GQA
        attn = 2 * hidden * hidden + 2 * hidden * (hidden // max(1, heads)) * kv_heads
        mlp = 3 * hidden * inter
        per_layer = attn + mlp
        total = layers * per_layer + 2 * vocab * hidden
        return round(total / 1e9, 3)
    except Exception:
        return 0.0


def model_meta(repo: str, use_network: bool = True) -> dict:
    """Метаданные модели: параметры, слои, hidden, контекст, лицензия, gated."""
    cache = load_meta_cache()
    if repo in cache:
        return cache[repo]
    api, download = _hub()
    meta: dict = {"repo_id": repo, "params_b": 0.0, "n_layers": 0, "hidden_size": 0,
                  "context": 0, "gated": False, "license": "", "source": "кэш"}
    if not use_network or api is None:
        return meta
    try:
        info = api.model_info(repo, files_metadata=True)
        meta["gated"] = bool(getattr(info, "gated", False))
        meta["downloads"] = int(getattr(info, "downloads", 0) or 0)
        meta["likes"] = int(getattr(info, "likes", 0) or 0)
        tags = list(getattr(info, "tags", []) or [])
        meta["tags"] = tags[:12]
        card = getattr(info, "card_data", None)
        if card is not None:
            lic = getattr(card, "license", None) or (card.get("license") if isinstance(card, dict) else "")
            meta["license"] = str(lic or "")
        st = getattr(info, "safetensors", None)
        if st is not None:
            total = st.get("total") if isinstance(st, dict) else getattr(st, "total", None)
            if total:
                meta["params_b"] = round(float(total) / 1e9, 3)
        meta["source"] = "каталог"
    except Exception as exc:  # noqa: BLE001
        meta["error"] = f"{type(exc).__name__}: {exc}"
    if download is not None:
        try:
            cfg_path = download(repo, "config.json")
            cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
            if not meta["params_b"]:
                meta["params_b"] = params_from_config(cfg)
            meta["n_layers"] = int(cfg.get("num_hidden_layers") or cfg.get("n_layer") or 0)
            meta["hidden_size"] = int(cfg.get("hidden_size") or cfg.get("n_embd")
                                      or cfg.get("d_model") or 0)
            meta["context"] = int(cfg.get("max_position_embeddings")
                                  or cfg.get("n_positions") or 0)
            meta["architectures"] = list(cfg.get("architectures") or [])[:2]
        except Exception:
            pass
    if meta.get("params_b") or meta.get("n_layers"):
        meta.pop("error", None)
        cache_meta(repo, meta)
    return meta


# ---------------------------------------------------------------------------
# Поиск по каталогу
# ---------------------------------------------------------------------------
def search(query: str, limit: int = 40, sort: str = "downloads",
           progress: Callable[[str, float], None] | None = None,
           only_llm: bool = True) -> list[dict]:
    """Поиск моделей в каталоге Hugging Face.

    Через ``expand=['safetensors', …]`` каталог сразу отдаёт число параметров, поэтому
    колонка «помещается» заполняется без отдельного запроса на каждую модель.
    """
    api, _ = _hub()
    if api is None:
        raise RuntimeError(
            "huggingface_hub недоступен: поиск по каталогу возможен только там, где "
            "установлены torch и huggingface_hub (см. вкладку «Hugging Face»)."
        )
    wanted = max(limit * 3, 60)
    if progress:
        progress(f"Ищу в каталоге: {query}…", 0.2)
    found: list = []
    for kwargs in ({"search": query, "limit": wanted, "sort": sort,
                    "expand": ["safetensors", "downloads", "likes", "gated", "tags"]},
                   {"search": query, "limit": wanted, "sort": sort},
                   {"search": query, "limit": wanted}):
        try:
            found = list(api.list_models(**kwargs))
            break
        except Exception:  # noqa: BLE001
            continue
    out: list[dict] = []
    for m in found:
        tags = set(getattr(m, "tags", []) or [])
        if only_llm and (tags & _SKIP_TAGS):
            continue
        if only_llm and "text-generation" not in tags and "text2text-generation" not in tags:
            continue
        st = getattr(m, "safetensors", None)
        total = None
        if isinstance(st, dict):
            total = st.get("total")
        elif st is not None:
            total = getattr(st, "total", None)
        out.append({
            "repo_id": getattr(m, "id", "") or getattr(m, "modelId", ""),
            "params_b": round(float(total) / 1e9, 3) if total else 0.0,
            "downloads": int(getattr(m, "downloads", 0) or 0),
            "likes": int(getattr(m, "likes", 0) or 0),
            "gated": bool(getattr(m, "gated", False)),
            "tags": sorted(tags)[:8],
        })
        if len(out) >= limit:
            break
    if progress:
        progress(f"Найдено моделей: {len(out)}", 1.0)
    return out


# ---------------------------------------------------------------------------
# Добавленные вручную модели
# ---------------------------------------------------------------------------
def load_custom() -> list[str]:
    try:
        data = json.loads(CUSTOM_FILE.read_text(encoding="utf-8"))
        return [str(x) for x in data if str(x).strip()]
    except Exception:
        return []


def save_custom(repos: list[str]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CUSTOM_FILE.write_text(json.dumps(list(dict.fromkeys(repos)), ensure_ascii=False, indent=1),
                               encoding="utf-8")
    except Exception:
        pass


def add_custom(repo: str) -> list[str]:
    repos = load_custom()
    if repo not in repos:
        repos.append(repo)
        save_custom(repos)
    return repos


def remove_custom(repo: str) -> list[str]:
    repos = [r for r in load_custom() if r != repo]
    save_custom(repos)
    return repos
