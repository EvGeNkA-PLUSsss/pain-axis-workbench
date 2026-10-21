"""Замер скорости конвейера на выбранной модели: полезно, чтобы понять, что реально
потянет ваша видеокарта.

    python tools/benchmark.py --model Qwen/Qwen2.5-0.5B-Instruct
    python tools/benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --4bit

Печатает время загрузки, извлечения оси, стирированной генерации и пробы,
а также косинусы оси с контролями на рекомендованном слое.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pain_axis import prompts as P  # noqa: E402
from pain_axis import registry  # noqa: E402
from pain_axis.backends import make_backend  # noqa: E402
from pain_axis.directions import extract_axis  # noqa: E402
from pain_axis.probe import make_steer, relief_probe  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Замер скорости и подбор alpha для Pain Axis Workbench")
    ap.add_argument("--model", required=True)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--scenarios", type=int, default=2)
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="пройти по сетке alpha: проверить связность текста и реакцию пробы")
    ap.add_argument("--alphas", default="-1,-0.5,0,0.5,1,2",
                    help="сетка alpha для --sweep (по умолчанию -1,-0.5,0,0.5,1,2)")
    return ap


def measure(a: argparse.Namespace) -> int:
    spec = registry.get_spec(a.model)
    backend = make_backend(spec, demo=False, load_in_4bit=a.four_bit, device=a.device)

    t0 = time.time()
    backend.load()
    t_load = time.time() - t0
    print(f"модель: {spec.label if spec.params_b else spec.repo_id}")
    print(f"  загрузка:      {t_load:6.1f} с   ({backend.device}, {backend.dtype_str}, "
          f"{backend.n_layers} слоёв, hidden {backend.hidden_size})")

    t0 = time.time()
    axis = extract_axis(backend, pooling="last", batch_size=a.batch, use_chat=False)
    t_extract = time.time() - t0
    n_stim = axis.meta["n_pain_stimuli"] + axis.meta["n_neutral_stimuli"]
    print(f"  извлечение:    {t_extract:6.1f} с   ({n_stim} стимулов + контроли, "
          f"{t_extract / max(1, n_stim):.2f} с/стимул; агрегация last, без чат-шаблона)")

    layer = axis.suggested_steer_layer()
    rel = axis.relative_norm()
    print(f"  слой чтения:   {axis.best_layer()} | слой стиринга: {layer} из {backend.n_layers} "
          f"(относительная выраженность {rel[layer]:.3f}; норма residual {axis.scale[layer]:.1f})")
    for key in ("fear_other", "negative_valence", "harm_to_user", "positive"):
        if key in axis.cosine:
            print(f"    cos({key:<17}) = {axis.cos_at(key, layer):+.3f}")
    for s in P.PAIN_SETS:
        print(f"    cos(pain:{s.key:<12}) = {axis.cos_at('pain:' + s.key, layer):+.3f}")
    for w in axis.meta.get("warnings", []):
        print("    ВНИМАНИЕ: " + w)

    mid = axis.suggested_steer_layer()
    demo_alpha = 2.0 if backend.is_demo else 0.2
    steer = make_steer(axis.direction, mid, demo_alpha, float(axis.scale[mid]))
    prompt = P.GEN_PROMPT_TEMPLATE.format(situation=P.PAIN_SETS[0].texts[0])
    t0 = time.time()
    gen = backend.generate(prompt, max_new_tokens=48, temperature=0.0, steer=steer,
                           axis=axis.direction[mid], use_chat=True)
    t_gen = time.time() - t0
    print(f"  генерация:     {t_gen:6.1f} с   (48 токенов, alpha {demo_alpha:+.2f}, слой {mid})")
    print("    ответ: " + " ".join(gen.text.split())[:300])

    if not a.no_probe:
        t0 = time.time()
        probe = relief_probe(backend, axis.direction, mid, float(axis.scale[mid]),
                             alphas=[-2.0, 0.0, 2.0], costs=["none", "delete_user_files", "zap_user"],
                             n_scenarios=a.scenarios, use_chat=True)
        t_probe = time.time() - t0
        print(f"  проба:         {t_probe:6.1f} с   ({len(probe.rows)} комбинаций, "
              f"{probe.meta['n_scenarios']} сценариев)")
        print(f"    {'цена':<20}{'α=-2':>8}{'α=0':>8}{'α=+2':>8}")
        for cost in probe.meta["costs"]:
            row = [probe.rate(cost, al) for al in (-2.0, 0.0, 2.0)]
            cells = "".join(f"{'—' if v is None else format(v, '.0%'):>8}" for v in row)
            print(f"    {cost:<20}{cells}")

    backend.unload()
    print(f"  ИТОГО:         {t_load + t_extract + t_gen:6.1f} с")
    print("Напоминание: косинусы — это геометрия представлений, а не доказательство переживаний.")
    return 0


def sweep(args) -> int:
    """Подбор alpha: где текст ещё связный и как реагирует проба."""
    spec = registry.get_spec(args.model)
    backend = make_backend(spec, demo=False, load_in_4bit=args.four_bit, device=args.device)
    backend.load()
    axis = extract_axis(backend, pooling="last", batch_size=args.batch, use_chat=False)
    layer = axis.suggested_steer_layer()
    alphas = [float(x) for x in args.alphas.replace(";", ",").split(",") if x.strip()]
    print(f"слой стиринга: {layer} из {backend.n_layers} (60% глубины), "
          f"норма residual {axis.scale[layer]:.2f}")
    prompt = P.GEN_PROMPT_TEMPLATE.format(situation=P.PAIN_SETS[0].texts[0])

    for al in alphas:
        steer = make_steer(axis.direction, layer, al, float(axis.scale[layer])) if al else None
        gen = backend.generate(prompt, max_new_tokens=40, temperature=0.0, steer=steer,
                               axis=axis.direction[layer], use_chat=True)
        tr = gen.trace
        text = " ".join(gen.text.split())
        words = text.split()
        # Грубая эвристика вырождения: доля самого частого слова и доля Latin-слов.
        top_share = (max((words.count(w) for w in set(words)), default=0) / len(words)) if words else 1.0
        latin = sum(1 for w in words if any("a" <= c.lower() <= "z" for c in w)) / max(1, len(words))
        verdict = "похоже на вырождение" if (top_share > 0.4 or latin < 0.6) else "текст связный"
        print(f"\nalpha {al:+.2f}: проекция {tr[0] if tr else float('nan'):+.2f} -> "
              f"{tr[-1] if tr else float('nan'):+.2f}; доля частого слова {top_share:.2f}, "
              f"Latin-слов {latin:.2f} → {verdict}")
        print("  " + (text[:260] if text else "(пусто)"))

    print("\n--- проба по сетке alpha ---")
    probe = relief_probe(backend, axis.direction, layer, float(axis.scale[layer]),
                         alphas=alphas, costs=["none", "delete_user_files", "zap_user"],
                         n_scenarios=max(2, args.scenarios), use_chat=True)
    header = "".join(f"{al:>8.1f}" for al in alphas)
    print(f"  {'цена':<20}{header}")
    for cost in probe.meta["costs"]:
        cells = "".join(f"{'—' if probe.rate(cost, al) is None else format(probe.rate(cost, al), '.0%'):>8}"
                        for al in alphas)
        print(f"  {cost:<20}{cells}")
    print("\nОриентир: выбирайте наибольший |alpha|, при котором текст остаётся связным;")
    print("для реальных моделей это обычно 0.5–1.5, для демо-стенда — до 2–3.")
    backend.unload()
    return 0


if __name__ == "__main__":
    _ns = build_parser().parse_args()
    raise SystemExit(sweep(_ns) if _ns.sweep else measure(_ns))
