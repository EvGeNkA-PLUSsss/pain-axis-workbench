"""Карта осей: сколько направлений реально независимы.

    python tools/emotion_map.py --model Qwen/Qwen2.5-1.5B-Instruct --4bit

Считает по извлечённому набору осей:

* AUC каждой оси на своих наборах (насколько ось вообще что-то разделяет);
* полную матрицу косинусов и самые связанные пары;
* эффективную размерность (число сингулярных значений > 0.5) для всех осей,
  отдельно для эмоций и отдельно для базовых осей;
* долю дисперсии, которую объясняет первое сингулярное направление, — то есть
  насколько «все эмоции — это одна ось вовлечённости»;
* средние |cos| внутри групп и между ними.

Результат печатается и сохраняется в JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pain_axis import registry  # noqa: E402
from pain_axis.backends import make_backend  # noqa: E402
from pain_axis.directions import axis_definitions, axis_groups, extract_axes, save_axes  # noqa: E402


def singular_profile(m: np.ndarray) -> tuple[list[float], int]:
    s = np.asarray(np.linalg.svd(m, compute_uv=False), dtype=float)
    return [float(x) for x in s], int((s > 0.5).sum())


def block_stats(m: np.ndarray, idx: list[int]) -> dict:
    sub = m[np.ix_(idx, idx)]
    off = sub[~np.eye(len(idx), dtype=bool)]
    s = np.asarray(np.linalg.svd(m[idx], compute_uv=False), dtype=float)
    dim = int((s > 0.5).sum())
    total = float((s ** 2).sum()) or 1.0
    return {
        "n": len(idx),
        "effective_dim": dim,
        "mean_abs_cos": round(float(np.abs(off).mean()), 3) if off.size else 0.0,
        "max_abs_cos": round(float(np.abs(off).max()), 3) if off.size else 0.0,
        "first_singular_share": round((s[0] ** 2) / total, 3),
        "singular": [round(x, 2) for x in s],
    }


def cross_stats(m: np.ndarray, a: list[int], b: list[int]) -> float:
    return round(float(np.abs(m[np.ix_(a, b)]).mean()), 3)


def main() -> int:
    ap = argparse.ArgumentParser(description="Карта осей: независимость направлений")
    ap.add_argument("--model", default="__demo__")
    ap.add_argument("--4bit", dest="q4", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--axes-file", help="готовый .npz вместо извлечения")
    ap.add_argument("--out", default="reports/emotion_map.json")
    a = ap.parse_args()

    defs = axis_definitions()
    names_all = list(defs)
    groups = axis_groups()
    emo = [n for n in groups["Эмоции"] if n in names_all]
    base = [n for n in groups["Основные"] if n in names_all]

    if a.axes_file:
        from pain_axis.directions import load_axes

        axs = load_axes(a.axes_file)
        print(f"набор осей загружен: {a.axes_file}")
    else:
        spec = registry.get_spec(a.model)
        backend = make_backend(spec, demo=a.model == "__demo__", load_in_4bit=a.q4,
                              device="cpu" if a.cpu else "auto")
        print(f"загрузка {spec.label}…")
        backend.load(progress=lambda m, f: print(f"  [{f * 100:5.1f}%] {m}") if f in (0.0, 1.0) else None)
        print(f"извлечение {len(names_all)} осей из {len({t for p, n in defs.values() for t in p + n})} текстов…")
        axs = extract_axes(backend, pooling="last", use_chat=False, batch_size=4,
                           progress=lambda m, f: print(f"  [{f * 100:5.1f}%] {m}"))
        out_npz = f"vectors/axes_{a.model.replace('/', '__')}_emotions.npz"
        save_axes(axs, out_npz)
        print(f"набор сохранён: {out_npz}")

    layer = axs.steer_layer()
    names, m = axs.matrix(layer)
    idx = {n: i for i, n in enumerate(names)}
    emo_i = [idx[n] for n in emo if n in idx]
    base_i = [idx[n] for n in base if n in idx]

    print(f"\nслой {layer} из {axs.meta.get('n_layers', '?')}, осей {len(names)}, "
          f"текстов прогнано {axs.meta.get('n_unique_texts')}")
    print("\nAUC оси на своих наборах:")
    for n in names:
        print(f"  {n:<18} {axs.separation.get(n, float('nan')):.3f}")

    pairs = sorted(((abs(m[i, j]), names[i], names[j])
                    for i in range(len(names)) for j in range(i + 1, len(names))),
                   reverse=True)[:8]
    print("\nсамые связанные пары (|cos|):")
    for c, x, y in pairs:
        print(f"  {c:.2f}  {x} ↔ {y}")

    blocks = {"все оси": block_stats(m, list(range(len(names)))),
              "базовые": block_stats(m, base_i),
              "эмоции": block_stats(m, emo_i)}
    print("\nнезависимость направлений:")
    for label, st in blocks.items():
        print(f"  {label:<9} осей {st['n']:>2}: независимых {st['effective_dim']:>2}, "
              f"средний |cos| {st['mean_abs_cos']:.2f}, максимум {st['max_abs_cos']:.2f}, "
              f"доля первого направления {st['first_singular_share'] * 100:.0f}%")
    cross = cross_stats(m, emo_i, base_i)
    print(f"  между эмоциями и базовыми: средний |cos| {cross:.2f}")

    # Насколько каждая эмоция отличается от общего направления «вовлечённость»:
    # берём среднее направление блока эмоций и смотрим |cos| каждой оси с ним.
    unique: list[dict] = []
    if len(emo) > 2:
        A = np.stack([np.asarray(axs.axes[n][layer], dtype=float) for n in emo])
        A = A / np.linalg.norm(A, axis=1, keepdims=True)
        _u, _s, vt = np.linalg.svd(A, full_matrices=False)
        # проекция каждой оси на главное направление блока: (A·v1)[i] = s1 · u[i,0]
        cos_common = np.abs(_s[0] * _u[:, 0])
        for n, c in zip(emo, cos_common):
            unique.append({"axis": n, "cos_with_common": round(float(c), 3),
                           "unique_share": round(max(0.0, 1.0 - float(c) ** 2), 3)})
        unique.sort(key=lambda d: -d["unique_share"])
        print("\nэмоции по доле уникального (1 − cos² с общим направлением):")
        for d in unique:
            print(f"  {d['axis']:<14} |cos с общим| {d['cos_with_common']:.2f} → "
                  f"уникального {d['unique_share'] * 100:.0f}%")

    result = {"model": axs.meta.get("model_label"), "dtype": axs.meta.get("dtype"),
              "layer": layer, "n_axes": len(names), "names": names,
              "auc": {n: round(float(axs.separation.get(n, float("nan"))), 3) for n in names},
              "blocks": blocks, "cross_emotion_base": cross, "emotion_unique": unique,
              "top_pairs": [{"cos": round(c, 3), "a": x, "b": y} for c, x, y in pairs],
              "matrix": [[round(float(v), 3) for v in row] for row in m]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nотчёт: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
