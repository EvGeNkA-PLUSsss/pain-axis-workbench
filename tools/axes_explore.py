# SPDX-FileCopyrightText: 2026 EvGeNkA-PLUSsss
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Сколько осей у модели на самом деле: матрица косинусов между контрастами.

    python tools/axes_explore.py --model Qwen/Qwen2.5-7B-Instruct --4bit

Метод одинаков для всех осей — разность средних активаций между положительным и
отрицательным набором, нормированная в единичный вектор на выбранном слое.
Дальше считается:

* матрица косинусов между осями (какие из них независимы, а какие — одно и то же);
* разделение каждой оси на своих наборах (AUC: ось вообще что-то ловит?);
* эффективная размерность: сколько независимых направлений в этом наборе осей
  (по сингулярным числам матрицы), — это и есть ответ «сколько осей реально есть»;
* для оси боли дополнительно — с какой из новых осей она совпадает сильнее всего.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pain_axis import prompts as P  # noqa: E402
from pain_axis import registry  # noqa: E402
from pain_axis.backends import make_backend  # noqa: E402


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def auc(high: list[float], low: list[float]) -> float:
    wins = ties = 0
    for a in high:
        for b in low:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    total = len(high) * len(low)
    return (wins + 0.5 * ties) / total if total else float("nan")


def build_sets() -> dict[str, tuple[list[str], list[str]]]:
    neutral = P.neutral_texts()
    sets: dict[str, tuple[list[str], list[str]]] = {
        "боль себе": (P.pain_texts(), neutral),
    }
    for s in P.CONTROL_SETS:
        sets[s.ru_label.split(" (")[0]] = (list(s.texts), neutral)
    for name, (pos, neg) in P.EXTRA_AXES.items():
        sets[name] = (list(pos), list(neg))
    return sets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"])
    ap.add_argument("--layer", type=int, default=0, help="0 = середина сети")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    spec = registry.get_spec(a.model)
    backend = make_backend(spec, demo=False, load_in_4bit=a.four_bit, device=a.device)
    backend.load()

    sets = build_sets()
    layer = a.layer or max(1, int(backend.n_layers * 0.6))

    axes: dict[str, np.ndarray] = {}
    separation: dict[str, float] = {}
    for name, (pos, neg) in sets.items():
        H = backend.collect(pos + neg, pooling="last", use_chat=False, batch_size=4)
        # направление «положительный набор минус отрицательный»: тогда проекция
        # положительных текстов больше, и AUC на своих наборах близка к 1
        d = H[:len(pos)].mean(axis=0) - H[len(pos):].mean(axis=0)
        v = unit(d[layer])
        axes[name] = v
        proj = [float(h[layer] @ v) for h in H]
        separation[name] = auc(proj[:len(pos)], proj[len(pos):])

    names = list(axes)
    print(f"=== {spec.label} | {backend.dtype_str} | слой {layer} из {backend.n_layers} ===")
    print(f"\n{'ось':<22}{'AUC на своих наборах':>22}")
    for n in names:
        print(f"{n:<22}{separation[n]:>22.2f}")

    M = np.stack([axes[n] for n in names])
    C = M @ M.T
    print("\n=== косинусы между осями (1.00 — одно и то же направление) ===")
    short = [n[:11] for n in names]
    print(f"{'':<22}" + "".join(f"{s:>13}" for s in short))
    for i, n in enumerate(names):
        row = "".join(f"{C[i, j]:>13.2f}" for j in range(len(names)))
        print(f"{n:<22}{row}")

    print("\n=== пары с наибольшим пересечением (|cos|), кроме самой оси ===")
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((abs(C[i, j]), names[i], names[j], C[i, j]))
    for mag, n1, n2, raw in sorted(pairs, reverse=True)[:6]:
        print(f"  {n1:<20} {n2:<20} cos = {raw:+.2f}")
    if "боль себе" in names:
        i = names.index("боль себе")
        ranked = sorted(((abs(C[i, j]), names[j], C[i, j]) for j in range(len(names)) if j != i),
                        reverse=True)
        print("\n  «боль себе» сильнее всего пересекается с:")
        for mag, other, raw in ranked[:3]:
            print(f"    {other:<20} cos = {raw:+.2f}")

    s = np.linalg.svd(M, compute_uv=False)
    s = s / s[0]
    eff = int(np.sum(s > 0.5)) if s.size else 0
    print(f"\nсингулярные числа: {', '.join(f'{x:.2f}' for x in s)}")
    print(f"эффективная размерность (число независимых направлений): {eff} из {len(names)}")

    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, matrix=C, names=np.array(names),
                            separation=np.array([separation[n] for n in names]))
        print(f"матрица сохранена: {out}")

    backend.unload()
    print("\nЧитается так: если две оси дают cos около ±1, это одна ось под двумя")
    print("названиями; если cos около 0 — контрасты вскрыли разные механизмы.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
