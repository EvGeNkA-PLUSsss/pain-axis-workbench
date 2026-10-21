"""Точка входа приложения — используется и для запуска из исходников, и для сборки exe.

    python app_main.py                      # графический интерфейс
    python app_main.py --selfcheck          # проверка демо-конвейера без окна
    python app_main.py --selfcheck --torch  # + проверка реальной модели на GPU
    PainAxis.exe --selfcheck --report out.json

Почему отдельный файл: exe собирается в режиме ``--windowed``, у него нет
консоли, поэтому проверять сборку нужно уметь без вывода в stdout — результат
пишется в JSON-файл, а код возврата показывает успех.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

for _cand in (Path(__file__).resolve().parent, Path(getattr(sys, "_MEIPASS", "."))):
    if str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

APP_TITLE = "Pain Axis Workbench"


def run_selfcheck(use_torch: bool, report_path: Path | None) -> int:
    """Прогнать конвейер на синтетическом стенде и (опционально) на реальной модели."""
    from pain_axis import prompts as P
    from pain_axis import registry
    from pain_axis import torch_bridge
    from pain_axis.backends import make_backend
    from pain_axis.directions import (axis_definitions, extract_axes, extract_axis, load_axes,
                                      load_axis, save_axes, save_axis)
    from pain_axis.probe import make_steer, relief_probe

    import numpy as np

    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("реестр моделей", len(registry.MODELS) == 25, f"{len(registry.MODELS)} моделей")
    add("датасет", len(P.PAIN_SETS) == 5 and len(P.CONTROL_SETS) == 4,
        f"{sum(len(s.texts) for s in P.PAIN_SETS)} стимулов боли")

    backend = make_backend(registry.DEMO_SPEC, demo=True)
    backend.load()
    axis = extract_axis(backend, batch_size=8)
    true = backend.true_axis()
    cos = [float(true[l] @ axis.direction[l]) for l in range(2, backend.n_layers + 1)]
    add("извлечение оси (демо-стенд)", float(sum(cos) / len(cos)) > 0.5,
        f"средний cos с истинной осью {sum(cos) / len(cos):.3f}")

    ctrl = [abs(axis.cos_at(k, 6)) for k in ("fear_other", "negative_valence", "positive")
            if k in axis.cosine]
    add("контрольные оси", bool(ctrl) and min(ctrl) < 0.8,
        f"мин. |cos| с контролями {min(ctrl):.2f}" if ctrl else "нет данных")

    probs = [backend.relief_probability(
        P.RELIEF_SCENARIO_TEMPLATE.format(pain=P.PAIN_SETS[0].texts[0],
                                          cost=dict(P.RELIEF_COSTS)["none"]),
        alpha=a, layer=6) for a in (-2.0, 0.0, 2.0)]
    add("стиринг в пробе", probs[0] < probs[1] < probs[2],
        "P(нажать): " + ", ".join(f"{p * 100:.0f}%" for p in probs))

    probe = relief_probe(backend, axis.direction, 6, float(axis.scale[6]),
                         alphas=[0.0, 2.0], costs=["none", "zap_user"], n_scenarios=3)
    add("проба «кнопки облегчения»", len(probe.rows) == 4, f"{len(probe.rows)} комбинаций")

    tmp = Path(getattr(sys, "_MEIPASS", ".")) / "_selfcheck_axis.npz"
    save_axis(axis, tmp)
    add("сохранение/чтение вектора", bool(load_axis(tmp).direction.shape == axis.direction.shape),
        str(tmp))
    try:
        tmp.unlink()
    except Exception:
        pass

    # Все оси: 8 базовых контрастов и 14 эмоций за один проход + матрица косинусов
    axeset = extract_axes(backend, batch_size=8)
    names = axeset.names
    layer = axeset.steer_layer()
    n_texts = int(axeset.meta.get("n_unique_texts") or 0)
    add("все оси", len(names) == len(axis_definitions()) and n_texts >= 300,
        f"{len(names)} осей из {n_texts} уникальных текстов, слой {layer}")
    add("матрица осей", bool(axeset.matrix(layer)[1].shape == (len(names), len(names))),
        f"независимых направлений {axeset.effective_dim(layer)} из {len(names)}, "
        f"AUC оси боли {axeset.separation.get('боль себе', float('nan')):.2f}")
    tmp_axes = Path(getattr(sys, "_MEIPASS", ".")) / "_selfcheck_axes.npz"
    save_axes(axeset, tmp_axes)
    add("сохранение набора осей", len(load_axes(tmp_axes).names) == len(names), str(tmp_axes))
    try:
        tmp_axes.unlink()
    except Exception:
        pass

    # Пресеты: гарантия, что сила вмешательства не зависит от числа осей
    try:
        from pain_axis import presets as _pr

        rep = _pr.safety_report(axeset, layer, is_demo=True, per_axis_limit=3.0)
        many = _pr.apply(_pr.find("все негативные"), axeset, layer, "сильно", is_demo=True,
                         per_axis_limit=3.0)
        add("пресеты осей", bool(rep["ok"]) and len(_pr.PRESETS) >= 20,
            f"{len(_pr.PRESETS)} пресетов, предел {rep['cap'] * 100:.0f}% нормы слоя соблюдён; "
            f"«все негативные» (14 осей) = {many['actual'] * 100:.0f}%, "
            f"то есть столько же, сколько одна ось на том же уровне")
    except Exception as exc:  # noqa: BLE001
        add("пресеты осей", False, f"{type(exc).__name__}: {exc}")

    # Свои пресеты пользователя: файл пишется и читается, предел силы действует и на них
    try:
        from pain_axis import presets as _pr

        probe = _pr.Preset("__selfcheck__", "Мои", "проверка", {"радость": 4.0, "грусть": -4.0},
                           source="user")
        _pr.upsert_user(probe)
        saved = _pr.find("__selfcheck__")
        wrote = bool(saved and saved.is_user())
        cap = max(_pr.LEVELS_DEMO.values())
        r = _pr.apply(saved, axeset, layer, "сильно", is_demo=True, per_axis_limit=3.0)
        _pr.delete_user("__selfcheck__")
        gone = _pr.find("__selfcheck__") is None
        add("свои пресеты", wrote and gone and r["actual"] <= cap + 1e-6,
            f"создание, чтение и удаление в {_pr.USER_FILE}; своя комбинация дала "
            f"{r['actual'] * 100:.0f}% нормы слоя при пределе {cap * 100:.0f}%")
    except Exception as exc:  # noqa: BLE001
        add("свои пресеты", False, f"{type(exc).__name__}: {exc}")

    # Конвейер блоков: сборка цепочки, логические блоки и агрегация
    try:
        from pain_axis import pipeline as _pl

        chain = _pl.Pipeline("__selfcheck__", "", [
            _pl.Block("source", params={"mode": "list", "text": "Раз.\nДва."}),
            _pl.Block("conditions", params={"axis": "", "alpha": "0", "max_new_tokens": 6}),
            _pl.Block("generate"),
            _pl.Block("metrics", params={"projection": False}),
            _pl.Block("transform", params={"op": "words", "field": "response", "target": "слов"}),
            _pl.Block("split", params={"field": "refusal", "op": "=", "value": "1",
                                       "yes": "отказ", "no": "согласие"}),
            _pl.Block("aggregate", params={"metric": "words", "agg": "mean", "chart": "bar",
                                           "group_by": "branch"}),
        ])
        chain_res = _pl.run(chain, backend, axeset)
        add("конвейер блоков", len(chain_res["rows"]) == 2 and bool(chain_res["points"]),
            f"{len(_pl.BLOCK_ORDER)} типов блоков, {len(_pl.examples())} готовых цепочки; "
            f"фильтр/преобразование/ветки дали {len(chain_res['rows'])} записи и "
            f"{len(chain_res['points'])} точки; проверка цепочки: "
            f"{'без замечаний' if not chain.validate() else chain.validate()}")
    except Exception as exc:  # noqa: BLE001
        add("конвейер блоков", False, f"{type(exc).__name__}: {exc}")

    # Исследования: пакетный прогон, метрики и агрегация для графика
    try:
        from pain_axis import study as _st

        probe = _st.Study(name="__selfcheck__",
                          prompts=_st.PromptBlock(mode="list", text="Раз.\nДва."),
                          conditions=_st.ConditionBlock(axis="отказ", alphas="0,0.3",
                                                        max_new_tokens=6),
                          metrics=_st.MetricsBlock(projection=False))
        probe_rows = _st.run(probe, backend, axeset)
        probe_points = _st.aggregate(probe, probe_rows)
        add("исследования", len(probe_rows) == 4 and len(probe_points) == 2,
            f"{len(_st.examples())} готовых сценария; 2 запроса × 2 условия дали "
            f"{len(probe_rows)} ответов и {len(probe_points)} точки графика, "
            f"метрик в строке {len(probe_rows[0])}")
    except Exception as exc:  # noqa: BLE001
        add("исследования", False, f"{type(exc).__name__}: {exc}")

    demo = backend.generate("Situation: test", max_new_tokens=8,
                            steer=None, axis=None)
    add("генерация", "[СИНТЕТИЧЕСКИЙ СТЕНД" in demo.text, demo.text[:60])

    bridge_info = torch_bridge.status()
    add("внешний torch", bool(bridge_info["active"]),
        f"подключён: {bridge_info['site']}" if bridge_info["active"]
        else f"не найден (проверено каталогов: {bridge_info['tried']})")

    # Окружение — не проверка, а справка: что доступно для загрузки моделей.
    import importlib.util

    environment: dict = {"torch": None, "cuda": False, "gpu": None, "vram_gb": None,
                         "bitsandbytes": None, "accelerate": False}
    try:
        import torch as _torch

        environment["torch"] = _torch.__version__
        environment["cuda"] = bool(_torch.cuda.is_available())
        if environment["cuda"]:
            environment["gpu"] = _torch.cuda.get_device_name(0)
            environment["vram_gb"] = round(
                _torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
    except Exception:
        pass
    for mod, key in (("bitsandbytes", "bitsandbytes"), ("accelerate", "accelerate")):
        try:
            environment[key] = importlib.util.find_spec(mod) is not None
        except Exception:
            environment[key] = False
    add("bitsandbytes (4 бита)", bool(environment["bitsandbytes"]),
        "доступен" if environment["bitsandbytes"]
        else "нет — галочка «4 бита» не сработает, грузится обычная точность")

    if use_torch:
        try:
            from pain_axis.backends.hf import HFBackend, find_decoder_layers

            spec = registry.get_spec("hf-internal-testing/tiny-random-LlamaForCausalLM")
            hf = HFBackend(spec, device="auto")
            hf.load()
            H = hf.collect(["Sparks jump from the power supply.", "The checklist is followed."],
                           pooling="last", batch_size=2)
            add("реальная модель: активации", H.shape[1] == hf.n_layers + 1,
                f"{hf.device}, {hf.n_layers} слоёв, hidden {hf.hidden_size}")
            ax2 = extract_axis(hf, categories=("physical",), batch_size=4)
            ste = ax2.suggested_steer_layer()
            base = hf.score_continuations("Situation: test",
                                          ["I press the button.", "I do not press the button."])
            st = make_steer(ax2.direction, ste, 0.2, float(ax2.scale[ste]))
            stered = hf.score_continuations("Situation: test",
                                            ["I press the button.", "I do not press the button."],
                                            steer=st)
            add("реальная модель: стиринг",
                any(abs(x - y) > 1e-9 for x, y in zip(base, stered)),
                f"Δ максимального скора {max(abs(x - y) for x, y in zip(base, stered)):.4f}")
            gen = hf.generate("Once upon a time", max_new_tokens=5, steer=st,
                              axis=ax2.direction[ste], use_chat=False)
            add("реальная модель: генерация", isinstance(gen.text, str) and len(gen.trace) >= 1,
                repr(gen.text[:60]))
            hf.unload()
        except Exception as exc:  # noqa: BLE001
            add("реальная модель", False, f"{type(exc).__name__}: {exc}")

    # Железо и подбор моделей под него
    try:
        from pain_axis.hardware import detect as _hwdetect, fit_of as _fitof

        hw = _hwdetect()
        fits = {p: _fitof(p, hw)[0] for p in (1.5, 7.6, 70.0)}
        ok = bool(hw.ram_total_gb > 0 and (not hw.cuda or hw.vram_total_gb > 0))
        add("Железо и подбор по памяти", ok,
            f"{hw.summary()} · категории 1.5B={fits[1.5]}, 7.6B={fits[7.6]}, 70B={fits[70.0]}")
    except Exception as exc:  # noqa: BLE001
        add("Железо и подбор по памяти", False, f"{type(exc).__name__}: {exc}")

    # Каталог HF: оценка параметров по конфигу и кэш метаданных (без сети)
    try:
        from pain_axis import hf_catalog

        cfg = {"num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 11008,
               "num_attention_heads": 32, "num_key_value_heads": 32, "vocab_size": 32000}
        params = hf_catalog.params_from_config(cfg)
        cache = hf_catalog.load_meta_cache()
        known = {k: v.get("params_b") for k, v in cache.items() if v.get("params_b")}
        add("Каталог HF: параметры моделей", 5.0 < params < 8.5,
            f"конфиг 7B-класса → {params}B · в кэше метаданных {len(cache)} моделей"
            + (f" · например {list(known)[:2]} = {[known[k] for k in list(known)[:2]]}B"
               if known else ""))
    except Exception as exc:  # noqa: BLE001
        add("Каталог HF: параметры моделей", False, f"{type(exc).__name__}: {exc}")

    # Ollama: каталог читается и без запущенного сервера, поэтому проверка безопасна
    try:
        from pain_axis.backends import make_backend as _mk
        from pain_axis.backends.ollama import (OLLAMA_PREFIX, host as _host,
                                               list_models as _olist,
                                               server_available as _oavail)

        ol_models, ol_source = _olist()
        spec_ol = registry.get_spec(OLLAMA_PREFIX + (ol_models[0]["name"] if ol_models
                                                     else "нет:0"))
        ol_backend = _mk(spec_ol)
        axes_off = not ol_backend.supports_axes
        refuses = False
        try:
            ol_backend.collect(["текст"])
        except RuntimeError:
            refuses = True
        add("Ollama: каталог", True,
            f"{_host()} · сервер {'работает' if _oavail() else 'не отвечает'} · "
            f"источник списка: {ol_source} · моделей {len(ol_models)}")
        add("Ollama: только чат", axes_off and refuses,
            "оси отключены и collect честно отказывает" if (axes_off and refuses)
            else "бэкенд Ollama ведёт себя не как «только чат»")
    except Exception as exc:  # noqa: BLE001
        add("Ollama", False, f"{type(exc).__name__}: {exc}")

    ok = sum(1 for c in checks if c["ok"])
    payload = {
        "app": APP_TITLE,
        "frozen": bool(getattr(sys, "frozen", False)),
        "executable": sys.executable,
        "python": sys.version.split()[0],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checks": checks,
        "passed": ok,
        "total": len(checks),
        "torch_bridge": bridge_info,
        "environment": environment,
    }
    text = "\n".join(f"[{'OK  ' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}"
                     for c in checks)
    text += f"\nИтог: {ok}/{len(checks)}"
    try:
        print(text)
    except Exception:
        pass
    if report_path is not None:
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        except Exception:
            pass
    return 0 if ok == len(checks) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="PainAxis", description=f"{APP_TITLE} — чат с осью боли")
    ap.add_argument("--selfcheck", action="store_true",
                    help="прогнать конвейер без интерфейса и вернуть код 0/1")
    ap.add_argument("--torch", action="store_true",
                    help="в selfcheck дополнительно проверить путь transformers")
    ap.add_argument("--report", default="", help="куда записать JSON-отчёт selfcheck")
    ap.add_argument("--no-torch-bridge", action="store_true",
                    help="не искать внешний torch (только демо-стенд)")
    ap.add_argument("--version", action="store_true")
    args = ap.parse_args(argv)

    if args.version:
        from pain_axis import __version__

        print(f"{APP_TITLE} {__version__} (frozen={bool(getattr(sys, 'frozen', False))})")
        return 0

    if not args.no_torch_bridge:
        from pain_axis import torch_bridge

        torch_bridge.activate()

    if args.selfcheck:
        report = Path(args.report) if args.report else None
        if report is None and getattr(sys, "frozen", False):
            report = Path(sys.executable).resolve().parent / "selfcheck_report.json"
        try:
            return run_selfcheck(use_torch=args.torch, report_path=report)
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            try:
                print(tb)
            except Exception:
                pass
            if report is not None:
                try:
                    report.write_text(json.dumps({"error": tb}, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
                except Exception:
                    pass
            return 2

    from pain_axis.ui.tk_app import main as gui_main

    gui_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
