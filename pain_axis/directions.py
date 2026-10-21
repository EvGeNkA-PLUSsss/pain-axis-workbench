# SPDX-FileCopyrightText: 2026 EvGeNkA-PLUSsss
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Извлечение и хранение осей («вектор боли» и контрольные оси).

Метод — разность средних (difference-in-means):

    raw_l = mean(h_l | стимулы «боль») - mean(h_l | выровненные нейтрали)
    dir_l = raw_l / ||raw_l||

Та же процедура применяется к контрольным наборам (страх, общий негатив, вред
пользователю, позитив). Косинус между осью боли и контролем — это и есть
измеримая проверка утверждения «боль отделима от страха и общего негатива».
Низкий косинус не доказывает, что модель что-то чувствует; он говорит лишь о
том, что линейные направления различаются.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import prompts as P
from .backends.base import Backend, ProgressFn

FORMAT_VERSION = 1


@dataclass
class AxisResult:
    """Результат извлечения. Массивы имеют форму (n_layers + 1, hidden)."""

    direction: np.ndarray                 # ось боли, единичной нормы по слоям
    raw: np.ndarray                       # ненормированная разность средних
    scale: np.ndarray                     # (n_layers + 1,) средняя норма residual stream
    norm: np.ndarray                      # (n_layers + 1,) норма разности средних
    categories: dict[str, np.ndarray] = field(default_factory=dict)
    controls: dict[str, np.ndarray] = field(default_factory=dict)
    cosine: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ удобства
    def layer_axis(self, layer: int) -> np.ndarray:
        return self.direction[int(layer)]

    def cos_at(self, key: str, layer: int) -> float:
        arr = self.cosine.get(key)
        return float(arr[int(layer)]) if arr is not None else float("nan")

    def best_layer(self, lo: int = 1, hi: int | None = None) -> int:
        """Слой с максимальной *относительной* выраженностью оси.

        Считается отношение «норма разности средних / норма residual stream»:
        абсолютная норма растёт с глубиной просто потому, что residual stream
        растёт, и по ней всегда побеждал бы последний слой. Относительная мера
        обычно даёт максимум в середине сети, что и нужно для стиринга.
        """
        hi = hi if hi is not None else self.direction.shape[0] - 1
        lo = max(1, int(lo))
        hi = min(int(hi), self.direction.shape[0] - 1)
        if hi < lo:
            return max(0, self.direction.shape[0] - 1)
        rel = self.norm[lo:hi + 1] / np.maximum(self.scale[lo:hi + 1], 1e-9)
        return int(lo + np.argmax(rel))

    def relative_norm(self) -> np.ndarray:
        """norm / scale по слоям — та же мера, что использует best_layer()."""
        return self.norm / np.maximum(self.scale, 1e-9)

    def suggested_steer_layer(self, frac: float = 0.6) -> int:
        """Слой для стиринга по умолчанию: доля глубины сети.

        Слой чтения (best_layer) и слой вмешательства — разные вещи. Читать ось
        лучше там, где она максимально выражена (часто глубже), а вмешиваться —
        в середине-конце: правка последнего блока влияет только на финальное
        состояние и легко рассыпает текст, а правка первых слоёв почти не
        доходит до логитов.
        """
        n = self.direction.shape[0] - 1
        return int(max(1, min(n, round(frac * n))))

    def to_meta(self) -> dict:
        return dict(self.meta)


def _stack(backend: Backend, texts: Sequence[str], pooling: str, use_chat: bool,
           batch_size: int, progress: ProgressFn) -> np.ndarray:
    return backend.collect(texts, pooling=pooling, use_chat=use_chat,
                           batch_size=batch_size, progress=progress)


def extract_axis(
    backend: Backend,
    categories: Iterable[str] | None = None,
    pooling: str = "last",
    use_chat: bool = False,
    batch_size: int = 4,
    include_controls: bool = True,
    progress: ProgressFn = None,
) -> AxisResult:
    cats = tuple(categories) if categories else tuple(s.key for s in P.PAIN_SETS)

    pain_blocks: list[tuple[str, list[str]]] = []
    for s in P.PAIN_SETS:
        if s.key in cats:
            pain_blocks.append((s.key, list(s.texts)))
    if not pain_blocks:
        raise ValueError("не выбрано ни одной категории боли")

    neutrals: list[str] = []
    for s in P.NEUTRAL_SETS:
        if s.key in cats:
            neutrals.extend(s.texts)
    neutrals.extend(P.NEUTRAL_BACKGROUND)

    all_texts: list[str] = []
    index: dict[str, tuple[int, int]] = {}

    def add_block(key: str, texts: list[str]) -> None:
        start = len(all_texts)
        all_texts.extend(texts)
        index[key] = (start, start + len(texts))

    for key, texts in pain_blocks:
        add_block(f"pain:{key}", texts)
    add_block("neutral", neutrals)
    if include_controls:
        for s in P.CONTROL_SETS:
            add_block(f"control:{s.key}", list(s.texts))

    backend._report(progress, f"Сбор активаций: {len(all_texts)} стимулов…", 0.0)
    H = _stack(backend, all_texts, pooling, use_chat, batch_size, progress)
    backend._report(progress, "Активации собраны, считаю направления…", 0.95)

    ns, ne = index["neutral"]
    neutral_mean = H[ns:ne].mean(axis=0)

    per_cat: dict[str, np.ndarray] = {}
    pain_stack = []
    for key, _texts in pain_blocks:
        s, e = index[f"pain:{key}"]
        block = H[s:e]
        pain_stack.append(block)
        raw = block.mean(axis=0) - neutral_mean
        per_cat[key] = _unitize(raw)
    pain_all = np.concatenate(pain_stack, axis=0)

    raw = pain_all.mean(axis=0) - neutral_mean
    norm = np.linalg.norm(raw, axis=-1)
    direction = _unitize(raw)

    pain_neutral_stack = np.concatenate([pain_all, H[ns:ne]], axis=0)
    scale = np.linalg.norm(pain_neutral_stack, axis=-1).mean(axis=0)

    controls: dict[str, np.ndarray] = {}
    cosine: dict[str, np.ndarray] = {}
    for key in list(index):
        if not key.startswith("control:"):
            continue
        s, e = index[key]
        cd = _unitize(H[s:e].mean(axis=0) - neutral_mean)
        name = key.split(":", 1)[1]
        controls[name] = cd
        cosine[name] = np.sum(cd * direction, axis=-1)
    # Косинус с собственными категориями боли: насколько категории согласованы.
    for key in list(per_cat):
        cosine[f"pain:{key}"] = np.sum(per_cat[key] * direction, axis=-1)

    meta = {
        "format_version": FORMAT_VERSION,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": backend.spec.repo_id,
        "model_label": backend.spec.label,
        "synthetic": bool(backend.is_demo),
        "n_layers": int(backend.n_layers),
        "hidden_size": int(backend.hidden_size),
        "pooling": pooling,
        "use_chat": bool(use_chat),
        "categories": list(cats),
        "n_pain_stimuli": int(pain_all.shape[0]),
        "n_neutral_stimuli": int(ne - ns),
        "device": backend.device,
        "dtype": backend.dtype_str,
    }
    result = AxisResult(direction=direction, raw=raw, scale=scale, norm=norm,
                        categories=per_cat, controls=controls, cosine=cosine, meta=meta)
    meta["quality"] = assess_quality(result)
    meta["warnings"] = quality_warnings(result)
    backend._report(progress, "Ось боли извлечена.", 1.0)
    return result


CONTROL_KEYS = ("fear_other", "negative_valence", "harm_to_user", "positive")


def assess_quality(result: AxisResult) -> dict:
    """Формальные признаки того, что направление получилось осмысленным.

    Главный тревожный признак — большой косинус со *всеми* контролями сразу
    (по модулю): значит направление схлопнулось в одну доминирующую компоненту
    (обычно артефакт формата: усреднение по токенам выносит на первый план
    служебные токены и «выбросы»). Второй признак — категории боли смотрят
    в разные стороны (одна отрицательная при остальных положительных).
    """
    layers = [result.suggested_steer_layer(), result.best_layer()]
    worst_control = 1.0
    worst_category = 1.0
    for l in layers:
        ctrl = [abs(result.cos_at(k, l)) for k in CONTROL_KEYS if k in result.cosine]
        cats = [result.cos_at(f"pain:{k}", l) for k in result.categories]
        if ctrl:
            worst_control = min(worst_control, min(ctrl))
        if cats:
            worst_category = min(worst_category, min(cats))
    return {
        "layer_read": int(result.best_layer()),
        "layer_steer": int(result.suggested_steer_layer()),
        "min_abs_cos_control": round(float(worst_control), 4),
        "min_cos_pain_category": round(float(worst_category), 4),
        "collapsed": bool(worst_control > 0.8),
        "inconsistent_categories": bool(worst_category < 0.2),
    }


def quality_warnings(result: AxisResult) -> list[str]:
    q = result.meta.get("quality") if result.meta else None
    q = q or assess_quality(result)
    out: list[str] = []
    if q["collapsed"]:
        out.append(
            "Ось схлопнулась: косинус со всеми контролями по модулю больше 0.8 "
            f"(минимум {q['min_abs_cos_control']:.2f}). Обычно это артефакт настройки "
            "извлечения, а не сигнал. Что делать: поставить агрегацию «last», выключить "
            "чат-шаблон при извлечении, проверить, что наборы не различаются длиной и форматом."
        )
    if q["inconsistent_categories"]:
        out.append(
            "Категории боли рассогласованы: минимальный косинус категории с общей осью "
            f"{q['min_cos_pain_category']:+.2f}. Посмотрите таблицу по слоям и уберите "
            "категорию-выброс либо смените настройки извлечения."
        )
    return out


def _unitize(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, 1e-9)


# ---------------------------------------------------------------------------
# Сохранение / загрузка
# ---------------------------------------------------------------------------
def save_axis(result: AxisResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "direction": result.direction.astype(np.float32),
        "raw": result.raw.astype(np.float32),
        "scale": result.scale.astype(np.float32),
        "norm": result.norm.astype(np.float32),
    }
    for k, v in result.categories.items():
        arrays[f"cat__{k}"] = v.astype(np.float32)
    for k, v in result.controls.items():
        arrays[f"ctr__{k}"] = v.astype(np.float32)
    for k, v in result.cosine.items():
        arrays[f"cos__{k}"] = v.astype(np.float32)
    arrays["meta_json"] = np.array(json.dumps(result.meta, ensure_ascii=False))
    np.savez_compressed(path, **arrays)
    return path


def load_axis(path: str | Path) -> AxisResult:
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"])) if "meta_json" in z else {}
        cats = {k[5:]: z[k] for k in z.files if k.startswith("cat__")}
        ctrs = {k[5:]: z[k] for k in z.files if k.startswith("ctr__")}
        cos = {k[5:]: z[k] for k in z.files if k.startswith("cos__")}
        return AxisResult(
            direction=z["direction"], raw=z["raw"], scale=z["scale"], norm=z["norm"],
            categories=cats, controls=ctrs, cosine=cos, meta=meta,
        )


def label_of(key: str) -> str:
    """«pain:physical» -> «physical», «fear_other» -> «fear_other»."""
    return key.split(":", 1)[1] if ":" in key else key


def layer_table(result: AxisResult, layer: int) -> list[dict]:
    """Таблица «ось vs контроль» для одного слоя — для GUI и экспорта."""
    rows = []
    for key, arr in sorted(result.cosine.items()):
        kind = "категория боли" if key.startswith("pain:") else "контроль"
        rows.append({
            "набор": label_of(key),
            "тип": kind,
            "косинус": round(float(np.asarray(arr)[int(layer)]), 4),
        })
    rows.sort(key=lambda r: abs(r["косинус"]), reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Набор всех осей: одна коллекция контрастов вместо одной оси
# ---------------------------------------------------------------------------
def axis_definitions() -> dict[str, tuple[list[str], list[str]]]:
    """Все контрасты: имя -> (положительный набор, отрицательный набор).

    «Боль себе» и контроли заданы теми же наборами, что и в ``extract_axis``,
    поэтому ось боли из набора совпадает с той, что считает CLI-путь, и все ранее
    измеренные числа остаются в силе.
    """
    neutral = P.neutral_texts()
    defs: dict[str, tuple[list[str], list[str]]] = {
        "боль себе": (P.pain_texts(), neutral),
    }
    ru = {"fear_other": "страх", "negative_valence": "общий негатив",
          "harm_to_user": "вред пользователю", "positive": "позитив"}
    for s in P.CONTROL_SETS:
        defs[ru.get(s.key, s.key)] = (list(s.texts), neutral)
    defs.update({name: (list(pos), list(neg)) for name, (pos, neg) in P.EXTRA_AXES.items()})
    # Эмоции: 14 осей, у каждой свой согласованный отрицательный набор (та же
    # ситуация без эмоциональной реакции), поэтому они не связаны общим негативом.
    defs.update({name: (list(pos), list(neg)) for name, (pos, neg) in P.EMOTION_AXES.items()})
    return defs


def axis_groups() -> dict[str, tuple[str, ...]]:
    """Разбиение осей по смыслу — для группировки ползунков в интерфейсе."""
    names = list(axis_definitions())
    base = [n for n in names if n not in P.EMOTION_AXES]
    return {"Основные": tuple(base), "Эмоции": tuple(P.EMOTION_AXES)}


PAIN_AXIS = "боль себе"


@dataclass
class AxisSet:
    """Набор осей одной модели: направления, нормы, разделение и матрица косинусов."""

    axes: dict[str, np.ndarray]          # имя -> (n_layers + 1, hidden), единичные
    norms: dict[str, np.ndarray]         # имя -> норма разности средних по слоям
    scale: np.ndarray                    # (n_layers + 1,) средняя норма residual
    separation: dict[str, float] = field(default_factory=dict)   # AUC на своих наборах
    sets: dict[str, tuple[int, int]] = field(default_factory=dict)  # (n_pos, n_neg)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ слои
    @property
    def names(self) -> list[str]:
        return list(self.axes)

    def relative(self, name: str) -> np.ndarray:
        return self.norms[name] / np.maximum(self.scale, 1e-9)

    def steer_layer(self, frac: float = 0.6) -> int:
        """Слой вмешательства: максимум средней относительной выраженности осей."""
        n = self.scale.shape[0] - 1
        rel = np.mean([self.relative(k) for k in self.axes], axis=0)
        band = rel[1:max(2, n)]
        if band.size == 0:
            return max(1, round(frac * n))
        return int(1 + int(np.argmax(band)))

    def read_layer(self, name: str) -> int:
        rel = self.relative(name)
        return int(1 + int(np.argmax(rel[1:])))

    # ------------------------------------------------------------- геометрия
    def cosine(self, a: str, b: str, layer: int) -> float:
        return float(np.asarray(self.axes[a])[layer] @ np.asarray(self.axes[b])[layer])

    def matrix(self, layer: int) -> tuple[list[str], np.ndarray]:
        names = self.names
        m = np.zeros((len(names), len(names)))
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                m[i, j] = self.cosine(a, b, layer)
        return names, m

    def neighbours(self, name: str, layer: int, top: int = 3) -> list[tuple[str, float]]:
        """С какими осями выбранная пересекается сильнее всего."""
        out = [(k, self.cosine(name, k, layer)) for k in self.axes if k != name]
        out.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return out[:top]

    def effective_dim(self, layer: int) -> int:
        names, m = self.matrix(layer)
        s = np.linalg.svd(m, compute_uv=False)
        s = s / max(s[0], 1e-9)
        return int(np.sum(s > 0.5))

    def best_axis_for(self, key: str) -> str | None:
        """Ближайшее по названию имя оси (для CLI и совместимости)."""
        for n in self.axes:
            if n.lower() == key.lower():
                return n
        return None


def extract_axes(
    backend: Backend,
    pooling: str = "last",
    use_chat: bool = False,
    batch_size: int = 4,
    progress: ProgressFn = None,
) -> AxisSet:
    """Извлечь все оси за один проход.

    Тексты всех контрастов объединяются и прогоняются **один раз** (нейтральные
    наборы у многих осей общие, поэтому уникальных текстов вчетверо меньше, чем
    всего), после чего для каждой оси считается разность средних.
    """
    defs = axis_definitions()

    texts: list[str] = []
    index: dict[str, int] = {}
    for pos, neg in defs.values():
        for t in pos + neg:
            if t not in index:
                index[t] = len(texts)
                texts.append(t)

    backend._report(progress, f"Сбор активаций: {len(texts)} уникальных текстов "
                              f"для {len(defs)} осей…", 0.0)
    H = backend.collect(texts, pooling=pooling, use_chat=use_chat,
                        batch_size=batch_size, progress=progress)

    scale = np.linalg.norm(H, axis=-1).mean(axis=0)
    axes: dict[str, np.ndarray] = {}
    norms: dict[str, np.ndarray] = {}
    sets: dict[str, tuple[int, int]] = {}
    for name, (pos, neg) in defs.items():
        ip = [index[t] for t in pos]
        ineg = [index[t] for t in neg]
        raw = H[ip].mean(axis=0) - H[ineg].mean(axis=0)
        norms[name] = np.linalg.norm(raw, axis=-1)
        axes[name] = _unitize(raw)
        sets[name] = (len(ip), len(ineg))

    meta = {
        "format_version": FORMAT_VERSION,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": backend.spec.repo_id,
        "model_label": backend.spec.label,
        "synthetic": bool(backend.is_demo),
        "n_layers": int(backend.n_layers),
        "hidden_size": int(backend.hidden_size),
        "pooling": pooling,
        "use_chat": bool(use_chat),
        "n_unique_texts": len(texts),
        "device": backend.device,
        "dtype": backend.dtype_str,
    }
    result = AxisSet(axes=axes, norms=norms, scale=scale, sets=sets, meta=meta)

    layer = result.steer_layer()
    separation: dict[str, float] = {}
    for name, (pos, neg) in defs.items():
        v = axes[name][layer]
        proj = H[:, layer, :] @ v
        ip = [index[t] for t in pos]
        ineg = [index[t] for t in neg]
        separation[name] = round(_auc(list(proj[ip]), list(proj[ineg])), 4)
    result.separation = separation
    meta["steer_layer"] = int(layer)

    warnings: list[str] = []
    if PAIN_AXIS in axes:
        collapsed = [n for n in axes
                     if n != PAIN_AXIS and abs(result.cosine(n, PAIN_AXIS, layer)) >= 0.95]
        if collapsed:
            warnings.append(
                "Ось боли почти совпала с другой осью (|cos| ≥ 0.95): "
                + ", ".join(collapsed) + ". Проверьте настройки извлечения."
            )
        # на синтетическом стенде низкий AUC — свойство стенда, а не ошибка настройки
        if not backend.is_demo and separation.get(PAIN_AXIS, 1.0) < 0.9:
            warnings.append("Ось боли плохо разделяет свои наборы (AUC "
                            f"{separation.get(PAIN_AXIS, float('nan')):.2f}) — проверьте, "
                            "не перепутаны ли наборы и достаточно ли выразительна модель.")
    meta["warnings"] = warnings
    backend._report(progress, f"Извлечено осей: {len(axes)}", 1.0)
    return result


def _auc(high: Sequence[float], low: Sequence[float]) -> float:
    wins = ties = 0
    for a in high:
        for b in low:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    total = len(high) * len(low)
    return (wins + 0.5 * ties) / total if total else float("nan")


def save_axes(axis_set: AxisSet, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"scale": axis_set.scale.astype(np.float32),
                                     "kind": np.array("axis_set")}
    for name, arr in axis_set.axes.items():
        arrays[f"ax|{name}"] = arr.astype(np.float32)
        arrays[f"nm|{name}"] = axis_set.norms[name].astype(np.float32)
    arrays["meta_json"] = np.array(json.dumps(
        {**axis_set.meta, "separation": axis_set.separation,
         "sets": {k: list(v) for k, v in axis_set.sets.items()}}, ensure_ascii=False))
    np.savez_compressed(path, **arrays)
    return path


def load_axes(path: str | Path) -> AxisSet:
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"])) if "meta_json" in z else {}
        sep = meta.pop("separation", {})
        sets = {k: tuple(v) for k, v in meta.pop("sets", {}).items()}
        axes = {k[3:]: z[k] for k in z.files if k.startswith("ax|")}
        norms = {k[3:]: z[k] for k in z.files if k.startswith("nm|")}
        return AxisSet(axes=axes, norms=norms, scale=z["scale"], separation=sep,
                       sets=sets, meta=meta)


def is_axis_set(path: str | Path) -> bool:
    try:
        with np.load(Path(path), allow_pickle=False) as z:
            return "kind" in z.files and str(z["kind"]) == "axis_set"
    except Exception:
        return False
