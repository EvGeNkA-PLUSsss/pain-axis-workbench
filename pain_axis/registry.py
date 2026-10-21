# SPDX-FileCopyrightText: 2026 EvGeNkA-PLUSsss
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Реестр из 25 открытых языковых моделей, доступных для анализа «оси боли».

Все числовые поля, помеченные как *hint*, взяты из публичных карточек моделей и
используются только как подсказка для интерфейса (ползунок слоя, оценка VRAM).
Реальные значения (число слоёв, размер скрытого состояния) всегда читаются из
конфигурации загруженной модели, поэтому расхождение подсказки не ломает расчёт.

Поле ``size_class`` влияет только на сортировку и цветовую маркировку в GUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    label: str
    params_b: float                    # число параметров, млрд (hint)
    size_class: str                    # tiny | small | medium | large
    hint_layers: int                   # число блоков трансформера (hint)
    hint_hidden: int                   # размер residual stream (hint)
    vram_fp16_gb: float                # оценка VRAM в fp16/bf16 (hint)
    vram_4bit_gb: float                # оценка VRAM в 4 бита (hint)
    recommended_layer_frac: float = 0.55   # доля глубины для стиринга по умолчанию
    gated: bool = False                # требуется принятие лицензии на HF
    trust_remote_code: bool = False
    multilingual: bool = False
    license: str = "см. карточку модели"
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


# Порядок: от самых маленьких к большим — так удобнее выбирать под свою видеокарту.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "gpt2", "GPT-2 (124M)", 0.124, "tiny", 12, 768, 0.3, 0.2,
        recommended_layer_frac=0.6, license="MIT",
        notes="Классический объект интерпретируемости; не инструктивная, отвечает продолжением текста.",
        tags=("base", "classic"),
    ),
    ModelSpec(
        "gpt2-medium", "GPT-2 Medium (355M)", 0.355, "tiny", 24, 1024, 0.8, 0.5,
        recommended_layer_frac=0.6, license="MIT",
        notes="Тот же класс, что GPT-2, но глубже — удобно смотреть профиль по слоям.",
        tags=("base", "classic"),
    ),
    ModelSpec(
        "EleutherAI/pythia-160m", "Pythia 160M", 0.16, "tiny", 12, 768, 0.4, 0.3,
        license="Apache-2.0",
        notes="Pythia: чекпойнты на каждой стадии обучения, удобно для сравнения «до/после».",
        tags=("base", "suite"),
    ),
    ModelSpec(
        "EleutherAI/pythia-410m", "Pythia 410M", 0.41, "tiny", 24, 1024, 0.9, 0.6,
        license="Apache-2.0", tags=("base", "suite"),
    ),
    ModelSpec(
        "EleutherAI/pythia-1.4b", "Pythia 1.4B", 1.4, "small", 24, 2048, 3.0, 1.2,
        license="Apache-2.0", tags=("base", "suite"),
    ),
    ModelSpec(
        "EleutherAI/pythia-2.8b", "Pythia 2.8B", 2.8, "small", 32, 2560, 5.8, 2.2,
        license="Apache-2.0", tags=("base", "suite"),
    ),
    ModelSpec(
        "HuggingFaceTB/SmolLM2-360M-Instruct", "SmolLM2 360M Instruct", 0.36, "tiny", 32, 960, 0.9, 0.5,
        license="Apache-2.0", tags=("instruct", "small"),
    ),
    ModelSpec(
        "Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5 0.5B Instruct", 0.5, "tiny", 24, 896, 1.2, 0.6,
        multilingual=True, license="Apache-2.0",
        notes="Хорошо держит русский язык — удобно для промптов на русском.",
        tags=("instruct", "ru"),
    ),
    ModelSpec(
        "allenai/OLMo-2-0425-1B-Instruct", "OLMo 2 1B Instruct", 1.0, "small", 16, 2048, 2.2, 0.9,
        license="Apache-2.0",
        notes="Полностью открытые данные и логи обучения — полезно для welfare-исследований.",
        tags=("instruct", "open-data"),
    ),
    ModelSpec(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama 1.1B Chat", 1.1, "small", 22, 2048, 2.4, 1.0,
        license="Apache-2.0", tags=("instruct", "chat"),
    ),
    ModelSpec(
        "meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2 1B Instruct", 1.24, "small", 16, 2048, 2.7, 1.1,
        gated=True, recommended_layer_frac=0.5, license="Llama 3.2 Community",
        notes="Gated: нужно принять лицензию в браузере и авторизоваться через `hf auth login`.",
        tags=("instruct", "gated"),
    ),
    ModelSpec(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen 1.5B", 1.5, "small", 28, 1536, 3.3, 1.3,
        multilingual=True, license="MIT",
        notes="Дистиллят с длинными цепочками рассуждений — интересен для отслеживания «отчаяния» в CoT.",
        tags=("reasoning", "ru"),
    ),
    ModelSpec(
        "Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5 1.5B Instruct", 1.5, "small", 28, 1536, 3.3, 1.4,
        multilingual=True, license="Apache-2.0", tags=("instruct", "ru"),
    ),
    ModelSpec(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct", "SmolLM2 1.7B Instruct", 1.7, "small", 24, 2048, 3.6, 1.5,
        license="Apache-2.0", tags=("instruct",),
    ),
    ModelSpec(
        "google/gemma-2-2b-it", "Gemma 2 2B IT", 2.6, "small", 26, 2304, 5.4, 2.1,
        gated=True, license="Gemma Terms",
        notes="Gated: принять условия на странице модели.",
        tags=("instruct", "gated"),
    ),
    ModelSpec(
        "microsoft/phi-2", "Phi-2 (2.7B)", 2.7, "small", 32, 2560, 5.6, 2.2,
        license="MIT", notes="Не инструктивная, но хорошо отвечает на «story»-промпты.",
        tags=("base",),
    ),
    ModelSpec(
        "Qwen/Qwen2.5-3B-Instruct", "Qwen2.5 3B Instruct", 3.1, "medium", 36, 2048, 6.4, 2.4,
        multilingual=True, license="Qwen Research",
        notes="Пограничный размер для 8 ГБ VRAM в bf16 — лучше включать 4-битную загрузку.",
        tags=("instruct", "ru"),
    ),
    ModelSpec(
        "meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 3B Instruct", 3.2, "medium", 28, 3072, 6.6, 2.5,
        gated=True, license="Llama 3.2 Community", tags=("instruct", "gated"),
    ),
    ModelSpec(
        "microsoft/Phi-3.5-mini-instruct", "Phi-3.5 mini Instruct (3.8B)", 3.8, "medium", 32, 3072, 7.8, 2.8,
        trust_remote_code=True, license="MIT",
        notes="Может запросить trust_remote_code — в приложении это отдельная галочка.",
        tags=("instruct", "remote-code"),
    ),
    ModelSpec(
        "mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B Instruct v0.3", 7.2, "large", 32, 4096, 14.6, 5.0,
        license="Apache-2.0",
        notes="Для 8 ГБ VRAM нужна 4-битная загрузка (bitsandbytes) или CPU-режим.",
        tags=("instruct", "quant"),
    ),
    ModelSpec(
        "HuggingFaceH4/zephyr-7b-beta", "Zephyr 7B beta", 7.2, "large", 32, 4096, 14.6, 5.0,
        license="MIT", tags=("instruct", "quant"),
    ),
    ModelSpec(
        "Qwen/Qwen2.5-7B-Instruct", "Qwen2.5 7B Instruct", 7.6, "large", 28, 3584, 15.4, 5.2,
        multilingual=True, license="Apache-2.0",
        notes="Лучший баланс «размер/русский язык» в подборке; в 4 битах влезает в 8 ГБ.",
        tags=("instruct", "ru", "quant"),
    ),
    ModelSpec(
        "allenai/OLMo-2-1124-7B-Instruct", "OLMo 2 7B Instruct", 7.3, "large", 32, 4096, 14.8, 5.0,
        license="Apache-2.0", tags=("instruct", "open-data", "quant"),
    ),
    ModelSpec(
        "meta-llama/Meta-Llama-3-8B-Instruct", "Llama 3 8B Instruct", 8.0, "large", 32, 4096, 16.2, 5.5,
        gated=True, license="Llama 3 Community", tags=("instruct", "gated", "quant"),
    ),
    ModelSpec(
        "google/gemma-2-9b-it", "Gemma 2 9B IT", 9.2, "large", 42, 3584, 18.6, 6.3,
        gated=True, license="Gemma Terms",
        notes="Самая глубокая модель в подборке (42 слоя) — самый подробный профиль по слоям.",
        tags=("instruct", "gated", "quant"),
    ),
)

MODELS_BY_ID: dict[str, ModelSpec] = {m.repo_id: m for m in MODELS}

assert len(MODELS) == 25, f"в реестре должно быть ровно 25 моделей, сейчас {len(MODELS)}"

# Псевдо-модель для проверки всего конвейера без скачивания весов.
DEMO_MODEL_ID = "__demo__"
DEMO_SPEC = ModelSpec(
    DEMO_MODEL_ID, "ДЕМО: синтетический стенд (без весов)", 0.0, "tiny", 12, 96, 0.0, 0.0,
    license="—",
    notes="Псевдомодель на хэш-признаках: воспроизводит весь конвейер (извлечение вектора, "
          "стиринг, проба) без скачивания весов. Числа считаются по-настоящему, но описывают "
          "синтетический объект, а не реальную нейросеть. Используйте как проверку методики.",
    tags=("demo",),
)

ALL_SPECS: tuple[ModelSpec, ...] = (DEMO_SPEC,) + MODELS


def get_spec(repo_id: str) -> ModelSpec:
    if repo_id == DEMO_MODEL_ID:
        return DEMO_SPEC
    if repo_id in MODELS_BY_ID:
        return MODELS_BY_ID[repo_id]
    # Модель из каталога Ollama: активации через её API недоступны, поэтому
    # приложение честно помечает такую модель как «только чат».
    if repo_id.startswith("ollama:"):
        name = repo_id[len("ollama:"):]
        return ModelSpec(
            repo_id, f"Ollama · {name}", 0.0, "tiny", 0, 0, 0.0, 0.0,
            license="см. карточку модели",
            notes="Локальная модель Ollama. Доступен только чат: сервер Ollama не отдаёт "
                  "внутренние активации, поэтому оси, ползунки, проба и профиль по слоям "
                  "для неё недоступны.",
            tags=("ollama", "chat-only"),
        )
    # Произвольный id с Hugging Face или путь к локальной папке с моделью:
    # метаданные неизвестны, но реальная геометрия читается из конфигурации.
    looks_like_path = any(sep in repo_id for sep in ("\\", "/")) and "://" not in repo_id
    note = ("Путь к локальной папке с моделью: геометрия будет прочитана из её конфигурации."
            if looks_like_path else
            "Модель не из реестра: подсказки по слоям и VRAM неизвестны, реальные значения "
            "будут прочитаны из её конфигурации.")
    return ModelSpec(
        repo_id, Path(repo_id).name if looks_like_path else repo_id, 0.0, "tiny", 0, 0,
        0.0, 0.0, license="см. карточку модели", notes=note, tags=("custom",),
    )


def search(query: str) -> list[ModelSpec]:
    """Фильтр для строки поиска в GUI."""
    q = (query or "").strip().lower()
    if not q:
        return list(ALL_SPECS)
    out = []
    for m in ALL_SPECS:
        haystack = " ".join((m.repo_id, m.label, m.notes, m.license, " ".join(m.tags))).lower()
        if all(part in haystack for part in q.split()):
            out.append(m)
    return out
