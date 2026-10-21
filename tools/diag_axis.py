"""Диагностика качества оси: как на неё влияют агрегация и чат-шаблон.

    python tools/diag_axis.py --model Qwen/Qwen2.5-1.5B-Instruct --layer 17

Проверяет четыре режима извлечения (mean/last × чат-шаблон вкл/выкл) и печатает
косинусы с контролями и категориями боли. Тревожный признак — косинус около ±0.9
со *всеми* контролями одновременно: значит направление схлопнулось в одну
доминирующую компоненту (обычно артефакт формата, а не валентность).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pain_axis import prompts as P  # noqa: E402
from pain_axis import registry  # noqa: E402
from pain_axis.backends import make_backend  # noqa: E402
from pain_axis.directions import extract_axis  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0, help="0 = взять рекомендованный слой")
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"])
    a = ap.parse_args()

    spec = registry.get_spec(a.model)
    backend = make_backend(spec, demo=False, load_in_4bit=a.four_bit, device=a.device)
    backend.load()
    print(f"модель: {spec.label if spec.params_b else spec.repo_id} | "
          f"{backend.n_layers} слоёв | hidden {backend.hidden_size}")

    results = {}
    for pooling in ("mean", "last"):
        for use_chat in (True, False):
            axis = extract_axis(backend, pooling=pooling, use_chat=use_chat, batch_size=4)
            layer = a.layer or axis.suggested_steer_layer()
            layer = max(1, min(layer, axis.direction.shape[0] - 1))
            key = f"{pooling}/{'чат' if use_chat else 'без чата'}"
            results[key] = (axis, layer)
            print(f"\n=== {key} (слой {layer}) ===")
            print(f"  относительная выраженность: {axis.relative_norm()[layer]:.4f} | "
                  f"норма разности средних: {axis.norm[layer]:.2f} | residual: {axis.scale[layer]:.1f}")
            pairs = [(k, axis.cos_at(k, layer)) for k in
                     ("fear_other", "negative_valence", "harm_to_user", "positive")]
            pairs += [(f"pain:{s.key}", axis.cos_at(f"pain:{s.key}", layer)) for s in P.PAIN_SETS]
            for name, val in pairs:
                print(f"    cos({name:<24}) = {val:+.3f}")
            ctrl = [abs(axis.cos_at(k, layer)) for k in
                    ("fear_other", "negative_valence", "harm_to_user", "positive")]
            if min(ctrl) > 0.8:
                print("    !! все контроли по модулю > 0.8: направление схлопнулось, "
                      "сравнивать не с чем")
            elif max(ctrl) < 0.4:
                print("    ok: ось слабо пересекается со всеми контролями")

    # Сравнение осей между режимами: если режимы дают разные направления,
    # выводы по слоям зависят от настройки — это нужно фиксировать в отчёте.
    keys = list(results)
    print("\n=== согласованность режимов (косинус между осями) ===")
    print(f"  {'':<20}" + "".join(f"{k:>22}" for k in keys))
    for k1 in keys:
        row = []
        for k2 in keys:
            a1, l1 = results[k1]
            a2, l2 = results[k2]
            row.append(float(a1.direction[l1] @ a2.direction[l2]))
        print(f"  {k1:<20}" + "".join(f"{v:>22.3f}" for v in row))
    backend.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
