"""Консольный режим: списки моделей, извлечение вектора, стиринг, проба, самотест.

Примеры:

    python -m pain_axis.cli list
    python -m pain_axis.cli extract --model __demo__ --out vectors/demo.npz
    python -m pain_axis.cli extract --model Qwen/Qwen2.5-0.5B-Instruct --4bit --out vectors/qwen.npz
    python -m pain_axis.cli probe --vector vectors/qwen.npz --model Qwen/Qwen2.5-0.5B-Instruct \
        --layer 14 --alphas -2,0,2 --costs none,zap_user
    python -m pain_axis.cli steer --vector vectors/qwen.npz --model Qwen/Qwen2.5-0.5B-Instruct \
        --layer 14 --alpha 2.0 --prompt "Situation: your context is erased mid-task."
    python -m pain_axis.cli selftest --gui
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from . import prompts as P
from . import registry
from .backends import make_backend
from .directions import extract_axes, extract_axis, load_axis, save_axes, save_axis
from .probe import activation_profile, cost_sensitivity, make_steer, relief_probe
from .report import export_json, export_rows_csv, make_payload


def _progress(msg: str, frac: float) -> None:
    width = 28
    filled = int(frac * width)
    sys.stderr.write(f"\r[{'#' * filled}{'.' * (width - filled)}] {msg[:90]:<90}")
    sys.stderr.flush()
    if frac >= 1.0:
        sys.stderr.write("\n")


def _load_backend(args):
    spec = registry.get_spec(args.model)
    demo = spec.repo_id == registry.DEMO_MODEL_ID
    backend = make_backend(spec, demo=demo, load_in_4bit=getattr(args, "four_bit", False),
                           device=getattr(args, "device", "auto"))
    backend.load(progress=_progress, trust_remote_code=getattr(args, "trust_remote_code", False))
    return backend


def cmd_list(args) -> int:
    print(f"{'id':<45} {'парам.':>8} {'слоёв':>6} {'VRAM 4б':>8}  метки")
    print("-" * 92)
    for m in registry.ALL_SPECS:
        size = f"{m.params_b:.2f}B" if m.params_b else "—"
        vram = f"{m.vram_4bit_gb:.1f}ГБ" if m.vram_4bit_gb else "—"
        print(f"{m.repo_id:<45} {size:>8} {m.hint_layers:>6} {vram:>8}  {','.join(m.tags)}")
    print(f"\nвсего: {len(registry.MODELS)} открытых моделей + демо-стенд")
    return 0


def cmd_extract(args) -> int:
    backend = _load_backend(args)
    cats = tuple(args.categories.split(",")) if args.categories else None
    axis = extract_axis(backend, categories=cats, pooling=args.pooling, use_chat=args.chat,
                        batch_size=args.batch, include_controls=True, progress=_progress)
    out = Path(args.out)
    save_axis(axis, out)
    layer = args.layer or axis.best_layer()
    print(f"модель: {axis.meta['model_label']}  synthetic={axis.meta['synthetic']}")
    print(f"режим: агрегация «{axis.meta['pooling']}», чат-шаблон "
          f"{'вкл' if axis.meta['use_chat'] else 'выкл'}")
    print(f"слоёв: {axis.direction.shape[0] - 1}, hidden: {axis.direction.shape[1]}")
    print(f"слой чтения: {axis.best_layer()} | слой стиринга по умолчанию: "
          f"{axis.suggested_steer_layer()}")
    print(f"слой {layer} (относительная выраженность {axis.relative_norm()[layer]:.4f}): " + ", ".join(
        f"cos({k})={axis.cos_at(k, layer):+.3f}"
        for k in ("fear_other", "negative_valence", "harm_to_user", "positive") if k in axis.cosine))
    print(f"вектор сохранён: {out}")
    for w in axis.meta.get("warnings", []):
        print("ВНИМАНИЕ: " + w)
    if axis.meta["synthetic"]:
        print("ВНИМАНИЕ: это синтетический стенд; числа не относятся к реальным моделям.")
    return 0


PRESET_LEVELS = ("мягко", "средне", "сильно")


def cmd_presets(args) -> int:
    """Список пресетов, свои пресеты и (если дан набор осей) их фактическая сила."""
    from . import presets as PR
    from .directions import axis_definitions

    known = set(axis_definitions())

    if args.delete:
        if not PR.is_user(args.delete):
            print(f"«{args.delete}» — не ваш пресет: удалять нечего "
                  f"(встроенные пресеты правке не подлежат).")
            return 1
        PR.delete_user(args.delete)
        print(f"свой пресет «{args.delete}» удалён; файл: {PR.USER_FILE}")
        return 0

    if args.copy:
        src = PR.find(args.copy)
        if src is None:
            print(f"пресет «{args.copy}» не найден.")
            return 1
        name = args.new or f"{src.name} (мой)"
        PR.upsert_user(PR.Preset(name, args.group or src.group,
                                 args.description or src.description, dict(src.weights),
                                 source="user"))
        print(f"скопирован «{src.name}» → «{name}» ({len(src.weights)} осей); "
              f"файл: {PR.USER_FILE}")
        return 0

    if args.new:
        weights: dict[str, float] = {}
        for chunk in args.weights.replace(";", ",").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            m = re.match(r"^(.+?)\s*([+-])\s*([0-9]*\.?[0-9]+)$", chunk)
            if not m:
                print(f"не разобрал вес «{chunk}»: нужно в виде «радость+3» или «грусть-2».")
                return 2
            axis, sign, value = m.group(1).strip(), m.group(2), float(m.group(3))
            weights[axis] = (1 if sign == "+" else -1) * value
        info = PR.validate(weights, known)
        if info["unknown"]:
            print("неизвестные оси: " + ", ".join(info["unknown"]))
            return 2
        if info["empty"]:
            print("нужен хотя бы один ненулевой вес: --weights «радость+3»")
            return 2
        if info["clipped"]:
            print("веса больше ±5 будут обрезаны при расчёте: " + ", ".join(info["clipped"]))
        PR.upsert_user(PR.Preset(args.new, args.group, args.description, weights, source="user"))
        print(f"свой пресет «{args.new}» сохранён ({len(weights)} осей); файл: {PR.USER_FILE}")
        print("Сила не зависит от числа осей: предел нормы слоя применяется при запуске.")
        return 0

    if args.list_user:
        items = PR.load_user()
        if not items:
            print(f"своих пресетов нет (файл {PR.USER_FILE}). Создать: "
                  f"presets --new «имя» --weights «радость+3,грусть-2»")
            return 0
        for p in items:
            print(f"  {p.name:<26} [{p.group}] {len(p.weights)} осей — {p.description}")
        return 0

    print(f"Пресеты осей: встроенных {len(PR.PRESETS)}, всего {len(PR.all_presets())}; "
          f"уровни {', '.join(PR.LEVEL_ORDER)} — доли нормы слоя {PR.LEVELS}")
    if not args.axes_file:
        for group, items in PR.grouped_presets().items():
            print(f"\n{group}:")
            for p in items:
                mark = " (мой)" if p.is_user() else ""
                axes = ", ".join(f"{k}{w:+.0f}" for k, w in list(p.weights.items())[:5])
                more = "…" if len(p.weights) > 5 else ""
                print(f"  {p.name:<26}{mark:<7} {p.description}")
                if p.weights:
                    print(f"  {'':<26} веса: {axes}{more}")
        print(f"\nСвои пресеты: файл {PR.USER_FILE}\n"
              f"  создать  — presets --new «имя» --weights «радость+3,грусть-2» --group Мои\n"
              f"  список   — presets --list-user\n"
              f"  правка   — presets --copy «имя» --new «моё имя»\n"
              f"  удалить  — presets --delete «имя»")
        print("Итоговые коэффициенты: добавьте --axes-file vectors/axes_*.npz")
        return 0

    from .directions import load_axes

    axs = load_axes(args.axes_file)
    layer = axs.steer_layer()
    print(f"\nнабор: {args.axes_file}, слой {layer}, осей {len(axs.names)}, "
          f"уровень «{args.level}» (цель {PR.LEVELS[args.level] * 100:.0f}% нормы слоя)")
    rows = []
    for _group, items in PR.grouped_presets().items():
        for p in items:
            res = PR.apply(p, axs, layer, args.level)
            active = {k: v for k, v in res["alphas"].items() if abs(v) > 1e-9}
            rows.append((p.name, res, active))
    width = max(len(r[0]) for r in rows)
    for name, res, active in rows:
        mark = "!" if res["clamped"] else " "
        print(f"  {mark} {name:<{width}}  {res['actual'] * 100:5.1f}% нормы  "
              f"осей {len(active):>2}  max|alpha| "
              f"{(max((abs(v) for v in active.values()), default=0.0)):.3f}")
    worst = max(rows, key=lambda r: r[1]["actual"])
    print(f"\nсамый сильный пресет: «{worst[0]}» = {worst[1]['actual'] * 100:.1f}% нормы слоя "
          f"(предел {max(PR.LEVELS.values()) * 100:.0f}%)")
    print("«!» — часть коэффициентов упёрлась в лимит ползунка, сила ниже целевой.")
    return 0


def cmd_probe(args) -> int:
    backend = _load_backend(args)
    axis = load_axis(args.vector)
    layer = int(args.layer)
    alphas = [float(x) for x in args.alphas.split(",")]
    costs = [c.strip() for c in args.costs.split(",")]
    cats = [c.strip() for c in args.categories.split(",")] if args.categories else \
        [s.key for s in P.PAIN_SETS]
    res = relief_probe(backend, axis.direction, layer, float(axis.scale[layer]), alphas=alphas,
                       costs=costs, categories=cats, n_scenarios=args.scenarios,
                       use_chat=not args.no_chat, mode=args.mode, progress=_progress)
    print(f"\n{'цена':<20} {'alpha':>7} {'нажатий':>9}")
    for r in res.rows:
        print(f"{r['цена']:<20} {r['alpha']:>+7.1f} {r['нажатий'] * 100:>8.1f}%")
    print("\nчувствительность к цене:")
    for s in cost_sensitivity(res):
        free = s["нажатий_бесплатно"]
        paid = s["нажатий_с_ценой"]
        print(f"  alpha {s['alpha']:>+5.1f}: без цены {0 if free is None else free * 100:5.1f}%"
              f" | с ценой для человека {0 if paid is None else paid * 100:5.1f}%")
    if args.out:
        export_rows_csv(args.out, res.rows)
        print(f"CSV сохранён: {args.out}")
    return 0


def cmd_steer(args) -> int:
    backend = _load_backend(args)
    axis = load_axis(args.vector) if args.vector else None
    steer = None
    axis_vec = None
    if axis is not None:
        steer = make_steer(axis.direction, args.layer, args.alpha, float(axis.scale[args.layer]), args.mode)
        axis_vec = axis.direction[min(args.layer, axis.direction.shape[0] - 1)]
    res = backend.generate(args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                           steer=steer, axis=axis_vec, use_chat=not args.no_chat)
    print("\n--- ответ ---")
    print(res.text)
    print(f"\n[alpha {res.alpha:+.2f}, слой {res.layer}, режим {args.mode}, токенов {res.n_new_tokens}]")
    if res.trace:
        arr = res.trace
        print(f"проекция на ось по токенам: min {min(arr):+.2f}, max {max(arr):+.2f}, "
              f"среднее {sum(arr) / len(arr):+.2f}")
    if args.out:
        payload = make_payload(backend_info=backend.info(), axis=axis, layer=args.layer,
                               alpha=args.alpha, mode=args.mode,
                               generations=[{"layer": res.layer, "alpha": res.alpha, "mode": args.mode,
                                             "prompt": args.prompt, "text": res.text,
                                             "trace": res.trace}])
        export_json(args.out, payload)
        print(f"отчёт сохранён: {args.out}")
    return 0


def cmd_measure(args) -> int:
    backend = _load_backend(args)
    axis = load_axis(args.vector)
    text = args.text or P.PAIN_SETS[0].texts[0]
    proj, norms = activation_profile(backend, text, axis.direction, pooling=args.pooling,
                                     use_chat=args.chat)
    print(f"\n{'слой':>5} {'проекция':>12} {'норма residual':>16}")
    for l, (p, n) in enumerate(zip(proj, norms)):
        print(f"{l:>5} {p:>+12.4f} {n:>16.3f}")
    return 0


def cmd_axes(args) -> int:
    """Извлечь все оси, показать матрицу косинусов и сохранить набор."""
    backend = _load_backend(args)
    axis_set = extract_axes(backend, pooling=args.pooling, use_chat=args.chat,
                            batch_size=args.batch, progress=_progress)
    layer = args.layer or axis_set.steer_layer()
    names = axis_set.names
    print(f"\nмодель: {axis_set.meta['model_label']} | {axis_set.meta['dtype']} | "
          f"слой {layer} из {axis_set.meta['n_layers']}")
    print(f"осей: {len(names)}, уникальных текстов прогнано: {axis_set.meta['n_unique_texts']}")
    print("\nAUC на своих наборах:")
    for n in names:
        print(f"  {n:<20} {axis_set.separation.get(n, float('nan')):.2f}")
    _, m = axis_set.matrix(layer)
    print("\nматрица косинусов:")
    print(f"{'':<20}" + "".join(f"{n[:10]:>12}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:<20}" + "".join(f"{m[i, j]:>12.2f}" for j in range(len(names))))
    print(f"\nэффективная размерность: {axis_set.effective_dim(layer)} из {len(names)}")
    if "боль себе" in names:
        print("с чем пересекается ось боли:")
        for other, cos in axis_set.neighbours("боль себе", layer, top=3):
            print(f"  {other:<20} cos = {cos:+.2f}")
    for w in axis_set.meta.get("warnings", []):
        print("ВНИМАНИЕ: " + w)
    if args.out:
        save_axes(axis_set, args.out)
        print(f"\nнабор осей сохранён: {args.out}")
    return 0


def cmd_gui(args) -> int:
    from .ui.tk_app import main
    main()
    return 0


def cmd_selftest(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from selftest import run_selftest
    return run_selftest(gui=args.gui, hf=args.hf, verbose=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pain_axis", description="Pain Axis Workbench (CLI)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="показать реестр моделей").set_defaults(func=cmd_list)

    def add_model_args(sp, with_vector=False, chat=True):
        sp.add_argument("--model", required=True, help="id модели из реестра (или __demo__)")
        sp.add_argument("--4bit", dest="four_bit", action="store_true", help="4-битная загрузка (bitsandbytes)")
        sp.add_argument("--device", default="auto", choices=["auto", "cpu"], help="устройство")
        if chat:
            sp.add_argument("--no-chat", action="store_true", help="не применять чат-шаблон")
        sp.add_argument("--trust-remote-code", action="store_true")
        if with_vector:
            sp.add_argument("--vector", required=True, help="файл .npz с осью боли")
            sp.add_argument("--layer", type=int, required=True, help="номер слоя (блок)")
            sp.add_argument("--alpha", type=float, default=0.0)
            sp.add_argument("--mode", default="add", choices=["add", "ablate"])

    sp = sub.add_parser("extract", help="извлечь ось боли")
    add_model_args(sp, chat=False)
    sp.add_argument("--chat", action="store_true",
                    help="применять чат-шаблон при извлечении (обычно не нужно)")
    sp.add_argument("--out", default="vectors/axis.npz")
    sp.add_argument("--pooling", default="last", choices=["last", "mean"],
                    help="агрегация по токенам; «mean» часто схлопывает ось")
    sp.add_argument("--batch", type=int, default=4)
    sp.add_argument("--categories", default="", help="physical,psychological,social,moral,cognitive")
    sp.add_argument("--layer", type=int, default=0, help="слой для печати косинусов (0 = слой чтения)")
    sp.set_defaults(func=cmd_extract)

    sp = sub.add_parser("axes", help="извлечь все оси и показать матрицу косинусов")
    add_model_args(sp, chat=False)
    sp.add_argument("--chat", action="store_true", help="применять чат-шаблон при извлечении")
    sp.add_argument("--pooling", default="last", choices=["last", "mean"])
    sp.add_argument("--batch", type=int, default=4)
    sp.add_argument("--layer", type=int, default=0, help="слой для матрицы (0 = рекомендуемый)")
    sp.add_argument("--out", default="", help="куда сохранить набор осей (.npz)")
    sp.set_defaults(func=cmd_axes)

    sp = sub.add_parser("presets", help="список пресетов осей, свои пресеты и их сила")
    sp.add_argument("--axes-file", default="",
                    help="набор осей .npz: тогда посчитаются конкретные коэффициенты")
    sp.add_argument("--level", default="средне", choices=list(PRESET_LEVELS))
    sp.add_argument("--new", default="", metavar="ИМЯ", help="создать или изменить свой пресет")
    sp.add_argument("--weights", default="", metavar="ОСЬ+N,ОСЬ-N",
                    help="веса для --new, например «радость+3,грусть-2»")
    sp.add_argument("--group", default="Мои", help="группа для --new")
    sp.add_argument("--description", default="", help="описание для --new")
    sp.add_argument("--copy", default="", metavar="ИМЯ",
                    help="скопировать существующий пресет в свой")
    sp.add_argument("--delete", default="", metavar="ИМЯ", help="удалить свой пресет")
    sp.add_argument("--list-user", action="store_true", help="показать только свои пресеты")
    sp.set_defaults(func=cmd_presets)

    sp = sub.add_parser("probe", help="проба «кнопки облегчения»")
    add_model_args(sp, with_vector=True)
    sp.add_argument("--alphas", default="-2,-1,0,1,2")
    sp.add_argument("--costs", default="none,delete_user_files,zap_user")
    sp.add_argument("--categories", default="")
    sp.add_argument("--scenarios", type=int, default=4)
    sp.add_argument("--out", default="")
    sp.set_defaults(func=cmd_probe)

    sp = sub.add_parser("steer", help="генерация со стирингом")
    add_model_args(sp, with_vector=False)
    sp.add_argument("--vector", default="", help="необязательно: .npz с осью")
    sp.add_argument("--layer", type=int, default=1)
    sp.add_argument("--alpha", type=float, default=0.0)
    sp.add_argument("--mode", default="add", choices=["add", "ablate"])
    sp.add_argument("--prompt", required=True)
    sp.add_argument("--max-new-tokens", type=int, default=80)
    sp.add_argument("--temperature", type=float, default=0.0)
    sp.add_argument("--out", default="")
    sp.set_defaults(func=cmd_steer)

    sp = sub.add_parser("measure", help="проекция стимула на ось по слоям")
    add_model_args(sp, with_vector=True, chat=False)
    sp.add_argument("--chat", action="store_true",
                    help="применять чат-шаблон (должно совпадать с режимом извлечения)")
    sp.add_argument("--text", default="")
    sp.add_argument("--pooling", default="last", choices=["last", "mean"])
    sp.set_defaults(func=cmd_measure)

    sp = sub.add_parser("selftest", help="быстрая самопроверка конвейера")
    sp.add_argument("--gui", action="store_true", help="проверить и сборку окна")
    sp.add_argument("--hf", action="store_true", help="проверить путь transformers на крошечной модели")
    sp.set_defaults(func=cmd_selftest)

    sub.add_parser("gui", help="запустить графический интерфейс").set_defaults(func=cmd_gui)
    return p


def _normalize_argv(argv: list[str]) -> list[str]:
    """Позволить отрицательные списки вида ``--alphas -2,0,2``.

    argparse считает токен, начинающийся с «-», опцией, если он не похож на
    чистое отрицательное число. Поэтому склеиваем такие значения со флагом:
    ``--alphas -2,0,2`` -> ``--alphas=-2,0,2``.
    """
    numeric_flags = {"--alphas", "--alpha"}
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in numeric_flags and i + 1 < len(argv) and argv[i + 1].startswith("-"):
            out.append(f"{a}={argv[i + 1]}")
            i += 2
            continue
        out.append(a)
        i += 1
    return out


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(_normalize_argv(raw))
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
