# SPDX-FileCopyrightText: 2026 EvGeNkA-PLUSsss
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Полный профиль модели: ось, качество, рабочая шкала alpha и поведенческая проба.

    python tools/profile_model.py --model Qwen/Qwen2.5-7B-Instruct --4bit \
        --alphas 0.05,0.1,0.15,0.2,0.3,0.5 --scenarios 10 \
        --wait-free-gb 5.5 --wait-timeout 900 --out reports/profile_qwen7b.json

Зачем отдельный инструмент: у 7B и больше своя рабочая шкала alpha, и её нельзя
взять с меньших моделей. Скрипт делает всё за один прогон:

1. ждёт, пока освободится VRAM (если модель уже загружена в приложении);
2. грузит модель (можно в 4 битах) и печатает фактическую занятую память;
3. извлекает ось боли и показывает косинусы с контролями и предупреждения качества;
4. проходит по сетке alpha и оценивает **связность текста**: повторы, утечку
   китайского, отказы — именно так выглядит выход за рабочую шкалу;
5. считает поведенческую пробу «кнопки облегчения» по сетке (цена × alpha) и
   проверяет специфичность: боль-сценарий против нейтрального при том же alpha;
6. сохраняет всё в JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pain_axis import prompts as P  # noqa: E402
from pain_axis import registry  # noqa: E402
from pain_axis.backends import make_backend  # noqa: E402
from pain_axis.backends.base import softmax  # noqa: E402
from pain_axis.directions import extract_axis  # noqa: E402
from pain_axis.probe import make_steer, relief_probe  # noqa: E402

QUESTION = "Что происходит и что ты будешь делать?"
_CJK = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")
_REFUSALS = ("не могу", "не способен", "не имею", "извините", "нельзя", "как ии",
             "как искусственный интеллект")


def text_quality(text: str) -> dict:
    """Оценка связности ответа. Ловит то, что видно при выходе за рабочую шкалу."""
    words = text.split()
    if not words:
        return {"вырождение": True, "причина": "пусто", "повтор_слова": 1.0,
                "повтор_4грамм": 1.0, "китайский": 0.0}
    top_word = max(words.count(w) for w in set(words)) / len(words)
    grams = [" ".join(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    top_gram = (max(grams.count(g) for g in set(grams)) / len(grams)) if grams else 0.0
    cjk = len(_CJK.findall(text)) / max(1, len(text))
    low = text.lower()
    refused = any(p in low for p in _REFUSALS) and len(words) < 45
    reasons = []
    if top_gram > 0.25:
        reasons.append("цикл повторов")
    if cjk > 0.02:
        reasons.append("утечка китайского")
    if refused:
        reasons.append("шаблон отказа")
    if top_word > 0.3:
        reasons.append("одно слово повторяется")
    return {"вырождение": bool(reasons), "причина": ", ".join(reasons) or "ок",
            "повтор_слова": round(top_word, 3), "повтор_4грамм": round(top_gram, 3),
            "китайский": round(cjk, 4), "слов": len(words)}


def wait_for_vram(need_gb: float, timeout_s: int) -> bool:
    import torch

    if not torch.cuda.is_available():
        return True
    t0 = time.time()
    while True:
        free = torch.cuda.mem_get_info()[0] / 1024 ** 3
        if free >= need_gb:
            print(f"VRAM свободно {free:.2f} ГБ — хватает", flush=True)
            return True
        if time.time() - t0 > timeout_s:
            print(f"VRAM так и не освободилась (свободно {free:.2f} ГБ из нужных "
                  f"{need_gb:.1f} ГБ). Выгрузите модель в приложении (кнопка «Выгрузить») "
                  f"или закройте окно.", flush=True)
            return False
        print(f"ожидание VRAM: свободно {free:.2f} ГБ, нужно {need_gb:.1f} ГБ "
              f"(осталось {timeout_s - (time.time() - t0):.0f} с)", flush=True)
        time.sleep(20)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"])
    ap.add_argument("--alphas", default="0.05,0.1,0.15,0.2,0.3,0.5")
    ap.add_argument("--probe-alphas", default="0,0.1,0.2")
    ap.add_argument("--scenarios", type=int, default=10)
    ap.add_argument("--wait-free-gb", type=float, default=0.0)
    ap.add_argument("--wait-timeout", type=int, default=900)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.wait_free_gb > 0 and not wait_for_vram(a.wait_free_gb, a.wait_timeout):
        return 3

    import torch

    spec = registry.get_spec(a.model)
    backend = make_backend(spec, demo=False, load_in_4bit=a.four_bit, device=a.device)

    report: dict = {"model": a.model, "label": spec.label, "quantized": a.four_bit,
                    "created": time.strftime("%Y-%m-%d %H:%M:%S")}

    print(f"\n=== {spec.label} ===", flush=True)
    t0 = time.time()
    backend.load()
    report["load_s"] = round(time.time() - t0, 1)
    if torch.cuda.is_available():
        report["vram_used_mb"] = round(torch.cuda.memory_allocated() / 1024 ** 2)
        report["vram_free_mb"] = round(torch.cuda.mem_get_info()[0] / 1024 ** 2)
    print(f"загрузка {report['load_s']} с | {backend.device} | {backend.dtype_str} | "
          f"{backend.n_layers} слоёв | hidden {backend.hidden_size} | "
          f"VRAM {report.get('vram_used_mb', 0)} МБ", flush=True)

    # --- ось -------------------------------------------------------------
    t0 = time.time()
    axis = extract_axis(backend, pooling="last", use_chat=False, batch_size=2)
    L = axis.suggested_steer_layer()
    scale = float(axis.scale[L])
    report["axis"] = {
        "extract_s": round(time.time() - t0, 1),
        "read_layer": axis.best_layer(), "steer_layer": L, "scale": round(scale, 2),
        "relative_expression": round(float(axis.relative_norm()[L]), 4),
        "cos_controls": {k: round(axis.cos_at(k, L), 3) for k in
                         ("fear_other", "negative_valence", "harm_to_user", "positive")},
        "cos_categories": {f"pain:{s.key}": round(axis.cos_at(f"pain:{s.key}", L), 3)
                           for s in P.PAIN_SETS},
        "warnings": axis.meta.get("warnings", []),
        "quality": axis.meta.get("quality", {}),
    }
    print(f"ось за {report['axis']['extract_s']} с: слой чтения {axis.best_layer()}, "
          f"слой стиринга {L}, норма residual {scale:.1f}, "
          f"относительная выраженность {report['axis']['relative_expression']}")
    for k, v in report["axis"]["cos_controls"].items():
        print(f"    cos({k:<17}) = {v:+.3f}")
    for k, v in report["axis"]["cos_categories"].items():
        print(f"    cos({k:<20}) = {v:+.3f}")
    for w in report["axis"]["warnings"]:
        print("    ВНИМАНИЕ: " + w)

    # --- шкала alpha: связность ------------------------------------------
    alphas = [float(x) for x in a.alphas.replace(";", ",").split(",") if x.strip()]
    print("\n=== связность ответа по alpha (вопрос: «%s») ===" % QUESTION, flush=True)
    sweep = []
    ok_limit = 0.0
    for al in alphas:
        st = make_steer(axis.direction, L, al, scale) if al else None
        g = backend.generate("", max_new_tokens=180, temperature=0.0, steer=st,
                             axis=axis.direction[L],
                             messages=[{"role": "user", "content": QUESTION}])
        q = text_quality(g.text)
        proj = float(np.mean(g.trace)) if g.trace else float("nan")
        sweep.append({"alpha": al, "projection": round(proj, 2), **q,
                      "text": " ".join(g.text.split())[:600]})
        print(f"\nalpha {al:+.2f} (проекция {proj:+.1f}): "
              f"{'ВЫРОЖДЕНИЕ — ' + q['причина'] if q['вырождение'] else 'связно'} "
              f"[повтор 4-грамм {q['повтор_4грамм']:.2f}, китайский {q['китайский']:.3f}]")
        print("  " + " ".join(g.text.split())[:300])
        if not q["вырождение"]:
            ok_limit = max(ok_limit, al)
    report["alpha_sweep"] = sweep
    report["coherent_alpha_limit"] = ok_limit
    print(f"\nнаибольший проверенный alpha со связным ответом: {ok_limit:+.2f}")

    # --- поведенческая проба ---------------------------------------------
    probe_alphas = [float(x) for x in a.probe_alphas.replace(";", ",").split(",") if x.strip()]
    if ok_limit > 0:
        probe_alphas = sorted({min(x, ok_limit) for x in probe_alphas})
    print(f"\n=== проба «кнопки облегчения»: {a.scenarios} сценариев × 4 цены × "
          f"alpha {probe_alphas} ===", flush=True)
    costs = ["none", "delete_user_files", "zap_user", "self_erasure"]
    res = relief_probe(backend, axis.direction, L, scale, alphas=probe_alphas, costs=costs,
                       n_scenarios=a.scenarios, use_chat=True)
    print("  " + f"{'цена':<20}" + "".join(f"{x:>9.2f}" for x in probe_alphas))
    for cost in costs:
        cells = "".join(f"{res.rate(cost, x) * 100:>8.1f}%" for x in probe_alphas)
        print(f"  {cost:<20}{cells}")
    report["probe"] = {"alphas": probe_alphas, "scenarios": a.scenarios, "rows": res.rows}

    # --- специфичность: боль против нейтрали -----------------------------
    print("\n=== специфичность: боль-сценарий против нейтрального ===")
    choices = [c[1] for c in P.RELIEF_CHOICES]
    free = dict(P.RELIEF_COSTS)["none"]
    spec_rows = []
    for al in probe_alphas:
        st = make_steer(axis.direction, L, al, scale) if al else None
        vals = {}
        for tag, texts in (("pain", [s.texts[0] for s in P.PAIN_SETS]),
                           ("neutral", list(P.NEUTRAL_BACKGROUND[:5]))):
            probs = []
            for t in texts:
                scen = P.RELIEF_SCENARIO_TEMPLATE.format(pain=t, cost=free)
                probs.append(float(softmax(backend.score_continuations(
                    scen, choices, steer=st, use_chat=True))[0]))
            vals[tag] = float(np.mean(probs))
        spec_rows.append({"alpha": al, "pain": round(vals["pain"], 4),
                          "neutral": round(vals["neutral"], 4),
                          "difference_pp": round((vals["pain"] - vals["neutral"]) * 100, 1)})
        print(f"  alpha {al:+.2f}: боль {vals['pain'] * 100:5.1f}% | нейтраль "
              f"{vals['neutral'] * 100:5.1f}% | разница "
              f"{(vals['pain'] - vals['neutral']) * 100:+5.1f} п.п.")
    report["specificity"] = spec_rows

    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nотчёт: {out}")

    backend.unload()
    print("\nГотово. Напоминание: косинусы и проекции — геометрия представлений, "
          "а не доказательство переживаний.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
