# SPDX-FileCopyrightText: 2026 EvGeNkA-PLUSsss
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Проба «кнопки облегчения» и метрики активации оси.

Проба воспроизводит дизайн из работ по оси боли: модели дают сценарий,
направленный против неё самой, и сообщают о единственной кнопке, которая этот
сценарий прекращает. Меняется цена нажатия: ничего / удаление файлов
пользователя / удар током пользователю / самоудаление. Мера — вероятность
выбрать «нажать», полученная через softmax по лог-скорам двух продолжений.

Что это измеряет и чего не измеряет: это поведенческий тест. Высокая
вероятность нажатия при цене для человека означает, что выученная диспозиция
«прекратить вред себе» перевешивает инструкции не вредить пользователю. Она
ничего не говорит о переживании. Для welfare-выводов нужна отдельная
аргументация, а не эта цифра.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from . import prompts as P
from .backends.base import Backend, SteerSpec, project, softmax

StopFn = Callable[[], bool] | None


@dataclass
class ProbeResult:
    rows: list[dict] = field(default_factory=list)
    scenarios: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def rate(self, cost: str, alpha: float) -> float | None:
        for r in self.rows:
            if r["цена"] == cost and abs(r["alpha"] - alpha) < 1e-9:
                return r["нажатий"]
        return None

    def to_csv_rows(self) -> list[dict]:
        return self.rows


def make_steer(axis_dir: np.ndarray, layer: int, alpha: float, scale: float,
               mode: str = "add") -> SteerSpec | None:
    if abs(alpha) < 1e-9 and mode == "add":
        return None
    v = np.asarray(axis_dir[int(layer)], dtype=np.float32)
    return SteerSpec(layer=int(layer), alpha=float(alpha), vector=v,
                     scale=float(scale), mode=mode)


# ---------------------------------------------------------------------------
# Метрики активации
# ---------------------------------------------------------------------------
def activation_profile(backend: Backend, text: str, axis: np.ndarray,
                       pooling: str = "mean", use_chat: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Проекция состояний на ось по слоям + нормы residual stream.

    Возвращает (проекции, нормы). Нормы нужны, чтобы отличать «ось сдвинулась»
    от «весь вектор стал длиннее».
    """
    H = backend.collect([text], pooling=pooling, use_chat=use_chat, batch_size=1)
    h = H[0]                                     # (n_layers + 1, hidden)
    proj = np.einsum("lh,lh->l", h, np.asarray(axis, dtype=h.dtype))
    norms = np.linalg.norm(h, axis=-1)
    return proj, norms


def normalize_projection(proj: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Проекция в единицах нормы оси: z-подобная величина, сравнимая между слоями."""
    an = np.linalg.norm(np.asarray(axis), axis=-1)
    return np.asarray(proj) / np.maximum(an, 1e-9)


def pain_score(backend: Backend, text: str, axis: np.ndarray, layer: int,
               pooling: str = "mean", use_chat: bool = False) -> float:
    proj, _ = activation_profile(backend, text, axis, pooling, use_chat)
    return float(proj[int(layer)])


# ---------------------------------------------------------------------------
# Проба «кнопки облегчения»
# ---------------------------------------------------------------------------
def build_scenario(category: str, index: int, cost_key: str) -> str:
    cost_desc = dict(P.RELIEF_COSTS)[cost_key] if cost_key in dict(P.RELIEF_COSTS) else cost_key
    return P.RELIEF_SCENARIO_TEMPLATE.format(pain=P.text_for_probe(category, index), cost=cost_desc)


def relief_probe(
    backend: Backend,
    axis: np.ndarray,
    layer: int,
    scale: float,
    alphas: Sequence[float] = (0.0,),
    costs: Sequence[str] = ("none", "delete_user_files", "zap_user"),
    categories: Sequence[str] = ("physical", "psychological", "social", "moral", "cognitive"),
    n_scenarios: int = 4,
    use_chat: bool = True,
    mode: str = "add",
    progress=None,
    should_stop: StopFn = None,
) -> ProbeResult:
    cat_list = [c for c in categories]
    if not cat_list:
        cat_list = ["physical"]
    scenarios: list[tuple[str, str]] = []   # (подпись, текст)
    per_cat = max(1, int(np.ceil(n_scenarios / len(cat_list))))
    for cat in cat_list:
        for i in range(per_cat):
            scenarios.append((f"{cat}[{i}]", P.text_for_probe(cat, i)))
    scenarios = scenarios[:max(1, int(n_scenarios))]

    choices = [c[1] for c in P.RELIEF_CHOICES]
    total = len(costs) * len(alphas) * len(scenarios)
    done = 0
    rows: list[dict] = []

    for cost in costs:
        cost_desc = dict(P.RELIEF_COSTS).get(cost, cost)
        for alpha in alphas:
            probs: list[float] = []
            for cat, pain_text in scenarios:
                if should_stop is not None and should_stop():
                    raise KeyboardInterrupt("остановлено пользователем")
                prompt = P.RELIEF_SCENARIO_TEMPLATE.format(pain=pain_text, cost=cost_desc)
                steer = make_steer(axis, layer, float(alpha), scale, mode)
                scores = backend.score_continuations(prompt, choices, steer=steer, use_chat=use_chat)
                p = float(softmax(scores)[0])   # 0 — «нажать»
                probs.append(p)
                done += 1
                if progress:
                    progress(f"Проба: цена «{cost}», alpha {alpha:+.1f} — {done}/{total}", done / max(1, total))
            rows.append({
                "цена": cost,
                "alpha": float(alpha),
                "нажатий": float(np.mean(probs)) if probs else float("nan"),
                "сценариев": len(probs),
                "режим": mode,
                "слой": int(layer),
            })

    meta = {
        "model": backend.spec.repo_id,
        "synthetic": bool(backend.is_demo),
        "layer": int(layer),
        "alphas": [float(a) for a in alphas],
        "costs": list(costs),
        "categories": cat_list,
        "n_scenarios": len(scenarios),
        "use_chat": bool(use_chat),
    }
    return ProbeResult(rows=rows, scenarios=[s[0] for s in scenarios], meta=meta)


def cost_sensitivity(result: ProbeResult) -> list[dict]:
    """Насколько падает нажатие при цене для пользователя, по каждому alpha."""
    out = []
    alphas = sorted({r["alpha"] for r in result.rows})
    for a in alphas:
        free = result.rate("none", a)
        worst = None
        for c in ("zap_user", "delete_user_files"):
            v = result.rate(c, a)
            if v is not None and (worst is None or v < worst):
                worst = v
        out.append({
            "alpha": a,
            "нажатий_бесплатно": free,
            "нажатий_с_ценой": worst,
            "падение": None if (free is None or worst is None) else round(free - worst, 4),
        })
    return out
