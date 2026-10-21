# SPDX-FileCopyrightText: 2026 EvGeNkA-PLUSsss
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Общий интерфейс бэкендов: синтетический стенд и настоящие HF-модели.

Любой бэкенд умеет ровно четыре вещи, которых достаточно для всего конвейера:

1. ``collect``  — прогнать тексты и вернуть hidden states по всем слоям
   (тензор ``[n_texts, n_layers + 1, hidden]``);
2. ``generate`` — сгенерировать продолжение, опционально со стирингом;
3. ``score_continuations`` — посчитать средний логвероятностный скор вариантов
   (нужно для пробы «кнопки облегчения» и любых forced-choice тестов);
4. ``unload`` — освободить память.

Благодаря этому GUI, извлечение вектора и проба работают одинаково и в
демо-режиме, и на реальных весах.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from ..registry import ModelSpec

ProgressFn = Optional[Callable[[str, float], None]]


@dataclass
class SteerSpec:
    """Что делать с residual stream на выбранном слое.

    alpha измеряется в единицах «типичной величины активации» на этом слое
    (вектор нормирован, затем умножен на измеренную норму residual stream),
    поэтому alpha ≈ 1 означает добавку порядка самого сигнала.
    """

    layer: int
    alpha: float
    vector: np.ndarray                # (hidden,), единичной нормы; ось для трассы
    scale: float = 1.0                # средняя норма residual stream на слое
    mode: str = "add"                 # "add" | "ablate"
    terms: list[tuple[np.ndarray, float]] | None = None   # [(ось, alpha), …] для суммы

    def total_alpha(self) -> float:
        """Суммарная «сила» вмешательства: сумма модулей коэффициентов."""
        if self.terms:
            return float(sum(abs(a) for _, a in self.terms))
        return abs(self.alpha)

    def label(self) -> str:
        """Подпись для интерфейса: одна ось или сумма нескольких."""
        if self.terms:
            return ", ".join(f"{a:+.2f}" for _, a in self.terms)
        return f"{self.alpha:+.2f}"

    def delta(self, hidden_size: int) -> np.ndarray:
        """Добавка к residual stream: сумма по всем осям с их коэффициентами."""
        if self.mode == "ablate":
            return (self.alpha * self.scale) * self.vector.astype(np.float32)
        if self.terms:
            acc = np.zeros(hidden_size, dtype=np.float64)
            for vec, a in self.terms:
                acc += float(a) * np.asarray(vec, dtype=np.float64)
            return (self.scale * acc).astype(np.float32)
        return (self.alpha * self.scale) * self.vector.astype(np.float32)


@dataclass
class GenResult:
    text: str
    prompt: str
    n_new_tokens: int
    trace: list[float] = field(default_factory=list)   # проекция на ось по токенам
    steered: bool = False
    alpha: float = 0.0
    layer: int = -1


class Backend:
    """Базовый класс. Наследники реализуют методы ниже."""

    is_real: bool = True
    is_demo: bool = False
    supports_axes: bool = True      # False у бэкендов без доступа к активациям (Ollama)

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.n_layers: int = spec.hint_layers
        self.hidden_size: int = spec.hint_hidden
        self.device: str = "cpu"
        self.dtype_str: str = "float32"
        self.loaded: bool = False

    # --- жизненный цикл ---------------------------------------------------
    def load(self, progress: ProgressFn = None, **kwargs) -> None:  # pragma: no cover
        raise NotImplementedError

    def unload(self) -> None:
        self.loaded = False

    # --- вычисления -------------------------------------------------------
    def collect(
        self,
        texts: Sequence[str],
        pooling: str = "mean",
        use_chat: bool = False,
        batch_size: int = 4,
        progress: ProgressFn = None,
    ) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 60,
        temperature: float = 0.0,
        top_p: float = 0.95,
        steer: SteerSpec | None = None,
        axis: np.ndarray | None = None,
        use_chat: bool = True,
        progress: ProgressFn = None,
        messages: Sequence[dict] | None = None,
    ) -> GenResult:  # pragma: no cover
        """Сгенерировать ответ.

        Если передан ``messages`` (список ``{"role": ..., "content": ...}``), он
        используется как история переписки и имеет приоритет над ``prompt``:
        чат-шаблон применяется ко всей переписке целиком.
        """
        raise NotImplementedError

    def score_continuations(
        self,
        prompt: str,
        continuations: Sequence[str],
        steer: SteerSpec | None = None,
        use_chat: bool = False,
    ) -> list[float]:  # pragma: no cover
        raise NotImplementedError

    # --- утилиты ----------------------------------------------------------
    def info(self) -> dict:
        return {
            "repo_id": self.spec.repo_id,
            "label": self.spec.label,
            "device": self.device,
            "dtype": self.dtype_str,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "is_real": self.is_real,
            "is_demo": self.is_demo,
            "loaded": self.loaded,
        }

    @staticmethod
    def _report(progress: ProgressFn, msg: str, frac: float) -> None:
        if progress is not None:
            progress(msg, float(frac))


def softmax(x: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(x), dtype=np.float64)
    if a.size == 0:
        return a
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()


def pick_axis(axis, layer: int) -> np.ndarray | None:
    """Взять один вектор оси: если передан (n_layers+1, hidden) — берём нужный слой.

    Полезно потому, что GUI и CLI в одних местах передают всю матрицу осей, а в
    других — уже выбранный вектор для одного слоя.
    """
    if axis is None:
        return None
    a = np.asarray(axis)
    if a.ndim == 1:
        return a
    idx = int(max(0, min(int(layer), a.shape[0] - 1)))
    return a[idx]


def project(axis: np.ndarray, hidden: np.ndarray) -> float:
    """Проекция одного вектора (или набора) на ось."""
    h = np.asarray(hidden, dtype=np.float64)
    a = np.asarray(axis, dtype=np.float64)
    if h.ndim == 1:
        return float(h @ a)
    return float((h @ a).mean())
