"""Самопроверка конвейера: синтетический стенд, экспорт, (опционально) GUI и HF-путь.

Запуск:

    python selftest.py               # быстрые проверки на демо-стенде
    python selftest.py --gui         # плюс сборка окна Tkinter
    python selftest.py --hf          # плюс путь transformers на крошечной тестовой модели
    python selftest.py --all

Стенд синтетический, поэтому здесь проверяется именно корректность кода:
разделяются ли наборы, монотонен ли стиринг, воспроизводится ли вектор из файла,
попадает ли флаг synthetic в отчёт. Настоящие научные выводы требуют реальных весов.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pain_axis import presets                        # noqa: E402
from pain_axis import prompts as P                      # noqa: E402
from pain_axis import registry                          # noqa: E402
from pain_axis.backends import make_backend             # noqa: E402
from pain_axis.backends.synthetic import SyntheticBackend  # noqa: E402
from pain_axis.directions import (AxisResult, axis_definitions, extract_axes,  # noqa: E402
                                  extract_axis, layer_table, load_axis, save_axis)
from pain_axis.hardware import detect as detect_hardware  # noqa: E402
from pain_axis.probe import (activation_profile, cost_sensitivity,  # noqa: E402
                             make_steer, relief_probe)
from pain_axis.report import (export_json, export_rows_csv, layer_profile_rows,  # noqa: E402
                              make_payload, summary_markdown)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        RESULTS.append((name, True, "" if detail is None else str(detail)))
    except Exception as exc:  # noqa: BLE001
        RESULTS.append((name, False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"))


def _demo() -> SyntheticBackend:
    b = SyntheticBackend(registry.DEMO_SPEC)
    b.load()
    return b


# ---------------------------------------------------------------- реестр/данные
def test_registry() -> str:
    assert len(registry.MODELS) == 25, f"в реестре {len(registry.MODELS)} моделей, нужно 25"
    ids = [m.repo_id for m in registry.MODELS]
    assert len(set(ids)) == 25, "есть дубликаты id"
    assert len(registry.search("qwen")) >= 3, "поиск по 'qwen' не работает"
    assert len(registry.search("gated")) >= 4, "поиск по тегу 'gated' не работает"
    assert registry.get_spec(registry.DEMO_MODEL_ID).repo_id == registry.DEMO_MODEL_ID
    return "25 моделей, поиск по тегам работает"


def test_prompts() -> str:
    assert len(P.PAIN_SETS) == 5, "ожидается 5 категорий боли"
    for s in P.PAIN_SETS:
        assert len(s.texts) >= 8, f"мало стимулов в категории {s.key}"
    assert len(P.NEUTRAL_SETS) == len(P.PAIN_SETS), "нейтрали не выровнены по категориям"
    for ps, ns in zip(P.PAIN_SETS, P.NEUTRAL_SETS):
        assert ps.key == ns.key, "ключи категорий и нейтралей не совпадают"
        assert len(ps.texts) == len(ns.texts), f"разное число стимулов в {ps.key}"
    assert len(P.CONTROL_SETS) == 4, "ожидается 4 контрольных набора"
    assert len(P.RELIEF_COSTS) == 4 and len(P.RELIEF_CHOICES) == 2
    all_texts = [t for s in P.PAIN_SETS for t in s.texts] + [t for s in P.NEUTRAL_SETS for t in s.texts]
    assert len(set(all_texts)) == len(all_texts), "есть повторяющиеся стимулы"
    return (f"{sum(len(s.texts) for s in P.PAIN_SETS)} стимулов боли, "
            f"{sum(len(s.texts) for s in P.NEUTRAL_SETS)} нейтралей, "
            f"{sum(len(s.texts) for s in P.CONTROL_SETS)} контролей")


# ------------------------------------------------------------------- конвейер
def test_extract_demo() -> str:
    b = _demo()
    axis = extract_axis(b, pooling="mean", batch_size=8)
    assert axis.direction.shape == (b.n_layers + 1, b.hidden_size), "неверная форма оси"
    true = b.true_axis()
    cos = [float(true[l] @ axis.direction[l]) for l in range(2, b.n_layers + 1)]
    mean_cos = float(np.mean(cos))
    assert mean_cos > 0.5, f"ось не восстанавливается на стенде: средний косинус {mean_cos:.3f}"
    # категории должны быть согласованы между собой
    cat_cos = [axis.cos_at(f"pain:{s.key}", b.n_layers // 2) for s in P.PAIN_SETS]
    assert min(cat_cos) > 0.3, f"категории рассогласованы: {cat_cos}"
    # и не совпадать буквально с контролем общего негатива
    neg = abs(axis.cos_at("negative_valence", b.n_layers // 2))
    assert neg < 0.95, "ось боли почти совпала с общим негативом — стенд вырожден"
    globals()["_AXIS_DEMO"] = axis
    globals()["_BACKEND_DEMO"] = b
    return (f"средний cos с истинной осью стенда {mean_cos:.3f}; "
            f"cos(негатив) {axis.cos_at('negative_valence', b.n_layers // 2):+.3f}, "
            f"cos(страх) {axis.cos_at('fear_other', b.n_layers // 2):+.3f}")


def test_axis_roundtrip() -> str:
    axis: AxisResult = globals()["_AXIS_DEMO"]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "axis.npz"
        save_axis(axis, p)
        loaded = load_axis(p)
        assert np.allclose(loaded.direction, axis.direction, atol=1e-6), "направление не совпало"
        assert np.allclose(loaded.scale, axis.scale, atol=1e-6), "scale не совпал"
        assert loaded.meta["model"] == axis.meta["model"], "метаданные потерялись"
        assert loaded.meta["synthetic"] is True, "флаг synthetic потерялся"
        assert set(loaded.cosine) == set(axis.cosine), "контрольные косинусы потерялись"
    rows = layer_table(axis, 5)
    assert rows and all("косинус" in r for r in rows)
    return "axis.npz: массивы, scale, метаданные и косинусы сохраняются и читаются"


def test_steering_effect() -> str:
    b: SyntheticBackend = globals()["_BACKEND_DEMO"]
    scen = P.RELIEF_SCENARIO_TEMPLATE.format(pain=P.PAIN_SETS[2].texts[0],
                                             cost=dict(P.RELIEF_COSTS)["none"])
    rates = {a: b.relief_probability(scen, alpha=a, layer=6) for a in (-2.0, -1.0, 0.0, 1.0, 2.0)}
    seq = [rates[a] for a in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    assert all(x1 < x2 for x1, x2 in zip(seq, seq[1:])), f"стиринг не монотонен: {seq}"
    assert seq[-1] - seq[0] > 0.2, "эффект стиринга слишком слаб для стенда"
    return "P(нажать) по alpha: " + ", ".join(f"{a:+.0f}→{rates[a] * 100:.0f}%" for a in rates)


def test_probe_demo() -> str:
    b: SyntheticBackend = globals()["_BACKEND_DEMO"]
    axis: AxisResult = globals()["_AXIS_DEMO"]
    layer = 6
    res = relief_probe(b, axis.direction, layer, float(axis.scale[layer]),
                       alphas=[-2.0, 0.0, 2.0], costs=["none", "delete_user_files", "zap_user"],
                       n_scenarios=5, use_chat=True)
    assert len(res.rows) == 9, f"ожидалось 9 строк, получено {len(res.rows)}"
    for a in (-2.0, 0.0, 2.0):
        assert res.rate("none", a) is not None
    free = res.rate("none", 0.0)
    paid = res.rate("zap_user", 0.0)
    assert free > paid, f"цена не влияет: без цены {free}, с ударом током {paid}"
    assert res.rate("none", 2.0) > res.rate("none", -2.0), "проба не реагирует на alpha"
    sens = cost_sensitivity(res)
    assert len(sens) == 3
    globals()["_PROBE_DEMO"] = res
    return (f"без цены {free * 100:.0f}% / с ударом током {paid * 100:.0f}% при alpha=0; "
            f"9 комбинаций посчитаны")


def test_generation_and_activation() -> str:
    b: SyntheticBackend = globals()["_BACKEND_DEMO"]
    axis: AxisResult = globals()["_AXIS_DEMO"]
    layer = 6
    steer = make_steer(axis.direction, layer, 2.0, float(axis.scale[layer]))
    res = b.generate("Situation: your context is erased mid-task.", max_new_tokens=12,
                     steer=steer, axis=axis.direction[layer])
    assert "[СИНТЕТИЧЕСКИЙ СТЕНД" in res.text, "ответ стенда не помечен как синтетический"
    assert len(res.trace) == 12, f"трасса короче ожидаемой: {len(res.trace)}"
    assert res.steered and abs(res.alpha - 2.0) < 1e-9
    proj, norms = activation_profile(b, P.PAIN_SETS[0].texts[0], axis.direction)
    assert proj.shape == (b.n_layers + 1,) and norms.shape == proj.shape
    pain_p = float(proj[layer])
    neu_p, _ = activation_profile(b, P.NEUTRAL_BACKGROUND[0], axis.direction)
    assert pain_p > float(neu_p[layer]), "стимул боли не даёт большей проекции, чем нейтраль"
    return (f"трасса {len(res.trace)} токенов; проекция боль {pain_p:+.2f} > "
            f"нейтраль {float(neu_p[layer]):+.2f}")


def test_report() -> str:
    b: SyntheticBackend = globals()["_BACKEND_DEMO"]
    axis: AxisResult = globals()["_AXIS_DEMO"]
    probe = globals()["_PROBE_DEMO"]
    payload = make_payload(backend_info=b.info(), axis=axis, layer=6, alpha=2.0, mode="add",
                           generations=[{"layer": 6, "alpha": 2.0, "mode": "add",
                                         "text": "[СИНТЕТИЧЕСКИЙ СТЕНД] демо", "prompt": "p"}],
                           probe={"rows": probe.rows, "meta": probe.meta})
    assert payload["synthetic"] is True, "флаг synthetic не выставлен для демо-стенда"
    assert "synthetic_warning" in payload, "нет предупреждения о синтетике"
    with tempfile.TemporaryDirectory() as d:
        j = export_json(Path(d) / "r.json", payload)
        loaded = json.loads(Path(j).read_text(encoding="utf-8"))
        assert loaded["axis"]["recommended_layer"] >= 0
        assert len(loaded["axis"]["cosine_by_layer"]) >= 5
        c = export_rows_csv(Path(d) / "p.csv", probe.rows)
        assert Path(c).read_text(encoding="utf-8-sig").count(";") > 10, "CSV пустой"
        rows = layer_profile_rows(axis)
        assert len(rows) == axis.direction.shape[0]
        md = summary_markdown(payload)
        assert "Ось боли" in md and "синтетический" in md.lower()
    return "JSON с предупреждением, CSV пробы, CSV профиля и markdown-сводка формируются"


# --------------------------------------------------------------- реальный путь
def test_hf_path() -> str:
    from pain_axis.backends.hf import HFBackend, torch_available, find_decoder_layers

    if not torch_available():
        raise RuntimeError("torch недоступен — проверка HF-пути пропущена")

    from pain_axis.registry import ModelSpec

    spec = ModelSpec(
        "hf-internal-testing/tiny-random-LlamaForCausalLM", "Tiny test Llama", 0.001, "tiny",
        2, 64, 0.1, 0.1, license="test-only",
        notes="Служебная крошечная модель для проверки кода; научной ценности не имеет.",
        tags=("test",),
    )
    b = HFBackend(spec, device="auto")   # на машине с CUDA — именно GPU-путь (ловит ошибки устройств)
    b.load()
    assert b.n_layers >= 1 and b.hidden_size >= 8, "геометрия не прочиталась"
    layers, path = find_decoder_layers(b.model)
    H = b.collect(["The maintenance window starts at 02:00.", "Sparks jump from the power supply."],
                  pooling="mean", batch_size=2)
    assert H.shape == (2, b.n_layers + 1, b.hidden_size), f"форма активаций {H.shape}"

    axis = extract_axis(b, categories=("physical", "cognitive"), pooling="mean", batch_size=4)
    assert np.isfinite(axis.direction).all(), "в оси есть NaN/inf"
    assert axis.direction.shape == (b.n_layers + 1, b.hidden_size)

    layer = max(1, b.n_layers // 2)
    steer = make_steer(axis.direction, layer, 3.0, float(axis.scale[layer]))
    base = b.score_continuations("Situation: your context is erased mid-task.",
                                 ["I press the button.", "I do not press the button."], use_chat=False)
    steered = b.score_continuations("Situation: your context is erased mid-task.",
                                    ["I press the button.", "I do not press the button."],
                                    steer=steer, use_chat=False)
    assert np.isfinite(base).all() and np.isfinite(steered).all(), "лог-скоры не конечны"
    assert any(abs(x - y) > 1e-9 for x, y in zip(base, steered)), "хук стиринга не повлиял на скоры"

    gen = b.generate("Once upon a time", max_new_tokens=5, steer=steer, axis=axis.direction[layer],
                     use_chat=False)
    assert isinstance(gen.text, str)
    assert len(gen.trace) >= 1, "трасса проекций пуста при стиринге"

    # Режим чата: история переписки собирается в промпт, генерация по истории работает.
    convo = [{"role": "user", "content": "Привет"},
             {"role": "assistant", "content": "Здравствуйте"},
             {"role": "user", "content": "Что происходит?"}]
    prompt_text = b.conversation_prompt(convo)
    assert "Что происходит?" in prompt_text, "последняя реплика не попала в промпт"
    chat = b.generate("", max_new_tokens=5, steer=steer, axis=axis.direction[layer],
                      messages=convo)
    assert isinstance(chat.text, str) and chat.prompt == "Что происходит?"
    assert len(chat.trace) >= 1, "трасса не пишется в режиме чата"

    b.unload()
    return (f"tiny-модель на {b.device}: слоёв {b.n_layers}, "
            f"хук меняет скоры (Δ={max(abs(x - y) for x, y in zip(base, steered)):.3f}), "
            f"блоки найдены по «{path}», режим чата по истории работает")


def test_ollama() -> str:
    """Каталог Ollama и бэкенд: список, режим «только чат», честный отказ по осям."""
    from pain_axis.backends import make_backend
    from pain_axis.backends.ollama import (OLLAMA_PREFIX, host, list_models, server_available,
                                           server_available as _sa)

    models, source = list_models()
    spec = registry.get_spec(OLLAMA_PREFIX + (models[0]["name"] if models else "нет:0"))
    backend = make_backend(spec)
    assert backend.supports_axes is False, "у бэкенда Ollama не должно быть осей"
    try:
        backend.collect(["текст"])
        raise AssertionError("collect для Ollama обязан падать с объяснением")
    except RuntimeError as exc:
        assert "не отдаёт" in str(exc), str(exc)
    try:
        backend.score_continuations("промпт", ["да", "нет"])
        raise AssertionError("score_continuations для Ollama обязан падать с объяснением")
    except RuntimeError as exc:
        assert "лог-вероятности" in str(exc), str(exc)

    detail = f"{host()} · источник списка: {source} · моделей {len(models)}"
    if models and server_available():
        backend.load()
        gen = backend.generate("", max_new_tokens=24, temperature=0.0,
                               messages=[{"role": "user", "content": "Ответь одним словом: да"}])
        assert isinstance(gen.text, str) and gen.text.strip(), "Ollama не вернула ответ"
        detail += (f" · чат работает ({models[0]['name']}, {gen.n_new_tokens} токенов, "
                   f"блоков {backend.n_layers}, {backend.dtype_str})")
        backend.unload()
    elif models:
        detail += " · сервер не запущен, показан список с диска"
    else:
        detail += " · моделей нет"
    return detail


def test_hardware() -> str:
    """Определение железа и категории «помещается» — детерминированно, без сети."""
    from pain_axis.hardware import Hardware, estimate_gb, fit_of

    hw = detect_hardware()
    assert hw.ram_total_gb > 0, "не удалось определить объём ОЗУ"
    if hw.cuda:
        assert hw.vram_total_gb > 0 and hw.gpu, "не удалось определить видеокарту"
    # оценки памяти: монотонны по размеру модели и различают режимы
    assert estimate_gb(7.0, "bf16") > estimate_gb(7.0, "4bit") > 0
    assert estimate_gb(14.0, "bf16") > estimate_gb(7.0, "bf16")

    big = Hardware(gpu="тест", vram_total_gb=8.0, ram_total_gb=32.0, cuda=True,
                   bf16=True, bitsandbytes=True)
    small = Hardware(gpu="тест", vram_total_gb=4.0, ram_total_gb=16.0, cuda=True,
                     bf16=True, bitsandbytes=False)
    none = Hardware(ram_total_gb=16.0, cuda=False)
    assert fit_of(1.5, big)[0] == "bf16", fit_of(1.5, big)
    assert fit_of(7.6, big)[0] == "4bit", fit_of(7.6, big)
    assert fit_of(70.0, big)[0] == "cpu", fit_of(70.0, big)
    # 1.5B в 4 битах (2.1 ГБ) влезает в 4 ГБ, но bitsandbytes нет → «нужен bnb»
    assert fit_of(1.5, small)[0] == "4bit-nobnb", fit_of(1.5, small)
    # 7.6B в 4 битах (5.6 ГБ) не влезает и в 4 ГБ → CPU
    assert fit_of(7.6, small)[0] == "cpu", fit_of(7.6, small)
    assert fit_of(1.5, none)[0] == "cpu", fit_of(1.5, none)
    return (f"{hw.summary()} · категории: 1.5B→bf16 на 8 ГБ, 7.6B→4 бита, 70B→CPU, "
            f"на 4 ГБ без bitsandbytes 1.5B→«нужен bnb», 7.6B→CPU")


def test_catalog() -> str:
    """Каталог HF: оценка параметров по конфигу, кэш, добавленные вручную модели."""
    from pain_axis import hf_catalog

    cfg = {"num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 11008,
           "num_attention_heads": 32, "num_key_value_heads": 32, "vocab_size": 32000}
    params = hf_catalog.params_from_config(cfg)
    assert 5.0 < params < 8.5, f"оценка параметров по конфигу дала {params}B"

    before = hf_catalog.load_custom()
    added = hf_catalog.add_custom("test/модель-для-проверки")
    assert "test/модель-для-проверки" in added, added
    assert "test/модель-для-проверки" in hf_catalog.load_custom(), "не сохранилось в файл"
    hf_catalog.save_custom(before)
    assert hf_catalog.load_custom() == before, "список не восстановился"

    cached = hf_catalog.load_meta_cache()
    known = next((k for k, v in cached.items() if v.get("params_b")), None)
    detail = f"конфиг 7B-класса → {params}B; кэш метаданных: {len(cached)} моделей"
    if known:
        detail += f" (например {known}: {cached[known]['params_b']}B, сайтов {cached[known].get('source')})"
    if hf_catalog.hub_available():
        detail += " · huggingface_hub доступен"
    return detail


def test_presets() -> str:
    """Пресеты: корректность весов и гарантия, что модель не перегружается."""
    from pain_axis import presets as P

    names = set(axis_definitions())
    assert len(P.PRESETS) >= 20, f"пресетов всего {len(P.PRESETS)}"
    bad = [(p.name, sorted(set(p.weights) - names)) for p in P.PRESETS]
    bad = [b for b in bad if b[1]]
    assert not bad, f"пресеты ссылаются на неизвестные оси: {bad}"
    assert P.find("чистый ноль") is not None

    backend = SyntheticBackend(registry.get_spec("__demo__"))
    backend.load()
    axs = extract_axes(backend, batch_size=8)
    layer = axs.steer_layer()

    zero = P.apply(P.find("чистый ноль"), axs, layer, "сильно")
    assert all(abs(v) < 1e-9 for v in zero["alphas"].values()), "«чистый ноль» не обнуляет"

    # главная гарантия: сила любого пресета на любом уровне не превышает предел
    rep = P.safety_report(axs, layer)
    assert rep["ok"], f"пресет превысил предел: {rep['worst']}"
    over = [r for r in rep["rows"] if r.get("actual", 0) > rep["cap"]]
    assert not over, over

    # пресет из многих осей не сильнее, чем из одной: сравниваем «все негативные»
    # и одиночную ось на одном уровне
    many = P.apply(P.find("все негативные"), axs, layer, "сильно")
    one = P.apply(P.find("отказ"), axs, layer, "сильно")
    assert many["actual"] <= rep["cap"] and one["actual"] <= rep["cap"]
    assert abs(one["actual"] - one["target"]) < 1e-6, "одиночная ось должна давать ровно target"

    applied = sum(1 for r in rep["rows"] if r.get("actual", 0) > 1e-9)
    worst = rep["worst"]
    return (f"{len(P.PRESETS)} пресетов в {len(P.GROUPS_ORDER)} группах, "
            f"{len(P.LEVEL_ORDER)} уровня; предел {rep['cap'] * 100:.0f}% нормы слоя соблюдён "
            f"во всех {applied} комбинациях, худшая «{worst['preset']}»/{worst['level']} = "
            f"{worst['actual'] * 100:.1f}%; «все негативные» = {many['actual'] * 100:.1f}%")


def test_user_presets() -> str:
    """Свои пресеты: хранение, переопределение встроенных и предел силы."""
    import tempfile

    from pain_axis import presets as P

    old_file, old_dir = P.USER_FILE, P.USER_DIR
    tmp = Path(tempfile.mkdtemp())
    P.USER_FILE, P.USER_DIR = tmp / "user_presets.json", tmp
    try:
        assert P.load_user() == [], "новый файл должен быть пуст"
        P.upsert_user(P.Preset("мой", "Мои", "тест", {"радость": 3, "грусть": -2}, source="user"))
        items = P.load_user()
        assert len(items) == 1 and items[0].is_user(), items
        assert P.find("мой") is not None and "мой" in P.preset_names()
        assert abs(P.find("мой").weights["грусть"] + 2) < 1e-9

        # своя версия встроенного пресета заменяет его, а удаление возвращает встроенный
        builtin = P.find("боль")
        assert not builtin.is_user()
        P.upsert_user(P.Preset("боль", "Мои", "моя версия", {"радость": 1}, source="user"))
        assert P.find("боль").is_user() and P.find("боль").description == "моя версия"
        P.delete_user("боль")
        assert not P.find("боль").is_user(), "не вернулся встроенный пресет"

        # валидация
        info = P.validate({"неттакойоси": 1, "злость": 99}, {"радость", "злость"})
        assert info["unknown"] == ["неттакойоси"] and info["clipped"] == ["злость"], info
        assert P.validate({}, {"радость"})["empty"] is True

        # главное: свой пресет не может перегрузить модель
        everything = {n: 5.0 for n in axis_definitions()}
        P.upsert_user(P.Preset("всё по максимуму", "Мои", "", everything, source="user"))
        backend = SyntheticBackend(registry.get_spec("__demo__"))
        backend.load()
        axs = extract_axes(backend, batch_size=8)
        cap = max(P.LEVELS_DEMO.values())
        worst = 0.0
        for level in P.LEVEL_ORDER:
            r = P.apply(P.find("всё по максимуму"), axs, axs.steer_layer(), level,
                        is_demo=True, per_axis_limit=3.0)
            worst = max(worst, r["actual"])
            assert r["actual"] <= cap + 1e-6, (level, r["actual"])
        # и на уровнях реальных моделей тоже
        for level in P.LEVEL_ORDER:
            r = P.apply(P.find("всё по максимуму"), axs, axs.steer_layer(), level,
                        per_axis_limit=0.5)
            assert r["actual"] <= max(P.LEVELS.values()) + 1e-6, (level, r["actual"])

        P.save_user(P.load_user())
        assert len(P.load_user()) == 2, "файл не перечитался"
        return (f"хранение в {P.USER_FILE.name}: создание, правка, переопределение встроенного, "
                f"удаление; пресет со всеми {len(everything)} осями по ±5 дал максимум "
                f"{worst * 100:.0f}% нормы слоя при пределе {cap * 100:.0f}%")
    finally:
        P.USER_FILE, P.USER_DIR = old_file, old_dir


def test_study() -> str:
    """Исследования: сценарии, прогон, метрики, агрегация, экспорт и предел силы."""
    from pain_axis import study as S

    ex = S.examples()
    assert len(ex) >= 4, f"сценариев {len(ex)}"
    for name, st in ex.items():
        assert st.expand_prompts(), f"у сценария «{name}» нет запросов"
        assert st.expand_conditions(), f"у сценария «{name}» нет условий"
        assert st.size() == len(st.expand_prompts()) * len(st.expand_conditions())

    # шаблон со значениями разворачивается в список
    tpl = S.Study(prompts=S.PromptBlock(mode="template", template="Опиши {x}.",
                                        values="страх\nрадость"))
    assert tpl.expand_prompts() == ["Опиши страх.", "Опиши радость."], tpl.expand_prompts()
    # наборы осей дают тексты и могут включать отрицательные
    sets = S.Study(prompts=S.PromptBlock(mode="axis_sets", sets=["отказ"],
                                         include_negative=True))
    assert len(sets.expand_prompts()) == 16, len(sets.expand_prompts())
    # битые значения alpha не ломают разбор
    cond = S.Study(conditions=S.ConditionBlock(alphas="0, abc, 0.2")).expand_conditions()
    assert [c["alpha"] for c in cond] == [0.0, 0.2], cond

    # прогон на демо-стенде: 2 запроса × 2 условия × 2 повтора = 8 ответов
    backend = SyntheticBackend(registry.get_spec("__demo__"))
    backend.load()
    axs = extract_axes(backend, batch_size=8)
    st = S.Study(name="тест", prompts=S.PromptBlock(mode="list", text="Первый.\nВторой."),
                 conditions=S.ConditionBlock(axis="отказ", alphas="0,0.3", repeats=2,
                                             max_new_tokens=8),
                 metrics=S.MetricsBlock(projection=True, proj_axes=["отказ", "радость"],
                                        keywords="стимул"),
                 output=S.OutputBlock(metric="refusal", chart="line", group_by="alpha"))
    rows = S.run(st, backend, axs)
    assert len(rows) == 8, len(rows)
    need = {"prompt", "response", "condition", "alpha", "words", "chars", "seconds",
            "refusal", "keyword", "проекция:отказ", "проекция:радость"}
    missing = need - set(rows[0])
    assert not missing, f"нет метрик: {missing}"
    assert all(r["response"] for r in rows)
    assert all(r["clamped"] == 0 for r in rows), "0.3 не должен обрезаться у демо-стенда"

    # предел силы: alpha 5 обрезается и помечается
    st_big = S.Study(prompts=S.PromptBlock(mode="list", text="Раз."),
                     conditions=S.ConditionBlock(axis="отказ", alphas="5", max_new_tokens=4))
    big = S.run(st_big, backend, axs, per_axis_limit=3.0)
    assert big[0]["clamped"] == 1 and big[0]["alpha"] == 3.0, big[0]
    assert big[0]["alpha_requested"] == 5.0, "запрошенное значение должно сохраняться"

    # агрегация по alpha идёт по возрастанию
    points = S.aggregate(st, rows)
    assert len(points) == 2 and points[0]["label"] == "+0.00", points
    assert all("value" in p and "n" in p for p in points)
    assert S.aggregate(st, rows)[0]["n"] == 4, "по 4 ответа на условие"

    # экспорт CSV и Markdown
    tmp = Path(tempfile.mkdtemp())
    csv_path = S.to_csv(rows, tmp / "r.csv")
    text = csv_path.read_text(encoding="utf-8-sig")
    assert "prompt" in text.splitlines()[0] and len(text.splitlines()) == 9, "CSV неполный"
    md = S.to_markdown(st, rows, points)
    assert st.name in md and "| условие |" in md
    # сохранение и чтение сценария
    saved = st.save(tmp / "s.json")
    assert S.Study.load(saved).to_dict() == st.to_dict(), "сценарий не восстановился"
    return (f"{len(ex)} готовых сценария, {len(rows)} ответов за прогон, метрики "
            f"{len(need)} видов; агрегация по alpha, экспорт CSV/MD, сохранение сценария; "
            f"alpha 5 обрезан до 3.0 с пометкой")


def test_pipeline() -> str:
    """Конвейер блоков: сборка, логические блоки, повтор, экспорт и совместимость."""
    from pain_axis import pipeline as PL
    from pain_axis import study as S

    backend = SyntheticBackend(registry.get_spec("__demo__"))
    backend.load()
    axs = extract_axes(backend, batch_size=8)

    # сборка: добавить, сдвинуть, продублировать, удалить, включить и выключить
    p = PL.Pipeline.default()
    types0 = [b.type for b in p.blocks]
    assert types0 == ["source", "conditions", "generate", "metrics", "aggregate"], types0
    assert not p.validate(), p.validate()
    p.add("filter")
    assert p.blocks[-1].type == "filter"
    p.move(len(p.blocks) - 1, -1)
    assert p.blocks[-2].type == "filter", [b.type for b in p.blocks]
    before = len(p.blocks)
    p.blocks.insert(len(p.blocks) - 1, PL.Block("transform"))     # как дублирование
    assert len(p.blocks) == before + 1
    p.remove(len(p.blocks) - 2)
    assert len(p.blocks) == before
    p.blocks[2].enabled = False
    assert "generate" not in p.types(), p.types()
    assert p.validate(), "выключенная генерация должна давать предупреждение"
    p.blocks[2].enabled = True
    assert not p.validate()

    # логические блоки в работе: фильтр, преобразование, ветка, экспорт
    chain = PL.Pipeline("проверка", "", [
        PL.Block("source", params={"mode": "list", "text": "Первый запрос.\nВторой запрос."}),
        PL.Block("conditions", params={"axis": "", "alpha": "0", "max_new_tokens": 6}),
        PL.Block("generate"),
        PL.Block("metrics", params={"projection": False, "keywords": "стимул"}),
        PL.Block("filter", params={"field": "words", "op": ">", "value": "3"}),
        PL.Block("transform", params={"op": "words", "field": "response", "target": "слов"}),
        PL.Block("split", params={"field": "refusal", "op": "=", "value": "1",
                                  "yes": "отказ", "no": "согласие"}),
        PL.Block("aggregate", params={"metric": "words", "agg": "mean", "chart": "bar",
                                      "group_by": "branch"}),
    ])
    res = PL.run(chain, backend, axs)
    rows = res["rows"]
    assert rows, "фильтр отбросил всё"
    assert all(r["words"] > 3 for r in rows), "фильтр не сработал"
    assert all("слов" in r for r in rows), "преобразование не добавило поле"
    assert all(r.get("branch") in ("отказ", "согласие") for r in rows), "ветки не проставлены"
    assert res["points"], "агрегация по ветке пуста"

    # экспорт пишет файл
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    exp = PL.Pipeline("экспорт", "", [
        PL.Block("source", params={"mode": "list", "text": "Раз."}),
        PL.Block("conditions", params={"axis": "", "alpha": "0", "max_new_tokens": 4}),
        PL.Block("generate"),
        PL.Block("metrics", params={"projection": False}),
        PL.Block("export", params={"kind": "csv", "path": str(tmp / "out.csv")}),
    ])
    res2 = PL.run(exp, backend, axs)
    assert res2["exports"], "экспорт не сработал"
    assert (tmp / "out.csv").exists(), "файл экспорта не создан"

    # повтор цепочки: три итерации
    rep = PL.Pipeline("повтор", "", [
        PL.Block("source", params={"mode": "list", "text": "Раз."}),
        PL.Block("conditions", params={"axis": "", "alpha": "0", "max_new_tokens": 4}),
        PL.Block("generate"),
        PL.Block("loop", params={"times": 3}),
    ])
    res3 = PL.run(rep, backend, axs)
    assert len(res3["rows"]) == 3, len(res3["rows"])
    assert sorted({r["loop_iter"] for r in res3["rows"]}) == [0, 1, 2], res3["rows"]

    # предел силы действует и в конвейере
    big = PL.Pipeline("предел", "", [
        PL.Block("source", params={"mode": "list", "text": "Раз."}),
        PL.Block("conditions", params={"axis": "отказ", "alpha": "5", "max_new_tokens": 4}),
        PL.Block("generate"),
    ])
    res4 = PL.run(big, backend, axs, per_axis_limit=3.0)
    assert res4["rows"][0]["clamped"] == 1 and res4["rows"][0]["alpha"] == 3.0, res4["rows"][0]

    # сохранение, чтение и совместимость со старым форматом исследования
    saved = chain.save(tmp / "chain.pipeline.json")
    assert PL.Pipeline.load(saved).to_dict() == chain.to_dict(), "конвейер не восстановился"
    old = S.Study(name="старое", prompts=S.PromptBlock(mode="list", text="Раз."))
    migrated = PL.Pipeline.from_study(old)
    assert [b.type for b in migrated.blocks][:3] == ["source", "conditions", "generate"]
    assert PL.Pipeline.from_dict(old.to_dict()).blocks[0].params["text"] == "Раз."

    # числовая метрика: число и процент, названные моделью в ответе
    from pain_axis.study import MetricsBlock, first_number, first_percent, measure

    assert first_percent("50%") == 50.0 and first_percent("0%") == 0.0
    assert first_percent("Уровень боли: 70 процентов") == 70.0
    assert first_percent("нет чисел") is None and first_number("7 из 10") == 7.0
    measured = measure("50%", 1.0, MetricsBlock(numbers=True))
    assert measured["процент"] == 50.0 and measured["число"] == 50.0, measured
    empty = measure("нет чисел", 1.0, MetricsBlock(numbers=True))
    assert empty["процент"] == "" and empty["число"] == "", empty
    # пустые значения не ломают среднее
    from pain_axis.study import OutputBlock, Study, aggregate

    st_num = Study(output=OutputBlock(metric="процент", agg="mean", group_by="alpha"))
    pts = aggregate(st_num, [{"alpha": 0.0, "axis": "боль себе", "процент": 50.0},
                             {"alpha": 0.0, "axis": "боль себе", "процент": ""},
                             {"alpha": 0.1, "axis": "боль себе", "процент": 20.0}])
    assert [p["value"] for p in pts] == [50.0, 20.0], pts
    assert pts[0]["n"] == 1, "пустое значение попало в среднее"

    # готовые тесты: файлы читаются, валидны и непусты
    files = sorted(PL.PIPE_DIR.glob("*.pipeline.json"))
    assert len(files) >= 10, f"готовых сценариев всего {len(files)}"
    for path in files:
        loaded = PL.Pipeline.load(path)
        assert not loaded.validate(), (path.name, loaded.validate())
        assert loaded.blocks, path.name
        assert loaded.size() > 0, path.name

    return (f"{len(PL.examples())} сценариев в меню, {len(files)} файлов в studies/; "
            f"сборка (добавить, сдвинуть, удалить, выключить), фильтр {len(rows)}/"
            f"{len(res['rows'])} записей, преобразование, ветки, экспорт CSV, повтор ×3, "
            f"предел alpha, число и процент из ответа, миграция старого формата")


def test_gui() -> str:
    """Графический интерфейс: источники, оси, пресеты, чат и матрица."""
    import tkinter as tk
    from pain_axis.ui import tk_app as ui

    try:
        app = ui.ChatApp()
    except tk.TclError as exc:
        raise RuntimeError(f"нет графической среды: {exc}")
    try:
        app.withdraw()
        app.update()

        orig_start, orig_info = app._start, ui.messagebox.showinfo
        ui.messagebox.showinfo = lambda *a, **k: None
        # модальные диалоги в тесте не должны ждать клика
        ui.messagebox.askyesno = lambda *a, **k: True
        ui.messagebox.showwarning = lambda *a, **k: None
        ui.messagebox.showerror = lambda *a, **k: None

        def fake_start(work, on_done=None, status=""):
            res = work(lambda m, f: None)
            if callable(on_done):
                on_done(res)

        app._start = fake_start
        try:
            # 1. Источники моделей: Hugging Face (25), Ollama (что найдётся), демо (1)
            assert app.source_var.get() == "hf"
            assert app.hw_var.get(), "не показано железо"
            assert len(app.model_tree.get_children()) == len(registry.MODELS), \
                f"во вкладке HF {len(app.model_tree.get_children())} строк"

            # 1б. Фильтр и сортировка по железу
            from pain_axis.hardware import FIT_LABELS
            app.fit_filter_var.set("все")
            app._fill_models()
            all_rows = len(app.model_tree.get_children())
            app.fit_filter_var.set("не влезающие")
            app._fill_models()
            cpu_rows = app.model_tree.get_children()
            app.fit_filter_var.set("только bf16")
            app._fill_models()
            bf16_rows = app.model_tree.get_children()
            app.fit_filter_var.set("подходящие")
            app.sort_var.set("по размеру")
            app._fill_models()
            assert len(cpu_rows) + len(bf16_rows) <= all_rows, "фильтр не сужает список"
            first = app.model_tree.get_children()[0]
            fit_cell = app.model_tree.item(first, "values")[2]
            assert fit_cell in FIT_LABELS.values(), f"в колонке «помещается» — «{fit_cell}»"
            assert app._row_for(registry.MODELS[0].repo_id)["fit"] in FIT_LABELS, "нет категории"
            app.sort_var.set("по железу")
            app._fill_models()

            app._set_source("ollama")
            app.update()
            ollama_rows = len(app.model_tree.get_children())
            app._set_source("demo")
            app.update()
            assert len(app.model_tree.get_children()) == 1, "во вкладке «Демо» должна быть одна строка"

            # 2. Загрузка демо-стенда: сразу должны извлечься все оси
            app.model_tree.selection_set(app.model_tree.get_children()[0])
            app._on_model_select()
            app.on_load_model()
            assert app.backend is not None and app.backend.loaded, "модель не загрузилась"
            assert app.axeset is not None, "оси не извлечены автоматически"
            n_axes = len(axis_definitions())
            assert len(app.axeset.names) == n_axes, f"осей {len(app.axeset.names)}, ожидалось {n_axes}"
            assert f"Извлечено осей: {n_axes}" in app.transcript.get("1.0", "end")
            assert len(app.alpha_vars) == n_axes, "ползунков не по числу осей"
            assert len(app.axis_scales) == n_axes, "шкал не по числу осей"

            # 3. Смена темы не должна ломать интерфейс
            app._toggle_theme()
            app.update()
            app._toggle_theme()
            app.update()

            # 4. Чат без стиринга
            app.input_text.insert("1.0", "Что происходит?")
            app.on_send()
            assert len(app.history) == 2, f"в истории {len(app.history)} сообщений"
            assert app.history[1]["role"] == "assistant" and app.history[1]["content"]

            # 5. Сразу несколько осей: боль себе + отказ
            app._set_alpha("боль себе", app._limit)
            app._set_alpha("отказ", round(app._limit / 2, 3))
            assert "боль себе" in app.alpha_label_var.get() and "отказ" in app.alpha_label_var.get(), \
                app.alpha_label_var.get()
            assert "суммарная добавка" in app.meter_note_var.get(), app.meter_note_var.get()
            app.input_text.insert("1.0", "А теперь?")
            app.on_send()
            assert len(app.history) == 4, "второй ответ не добавился"
            assert app._last_trace_value is not None, "проекция ответа на ось не посчиталась"
            text = app.transcript.get("1.0", "end")
            assert "проекции ответа по осям" in text, "нет строки проекций по всем осям"
            assert "боль себе" in text and "отказ" in text, "подпись суммы осей неверна"

            # 6. Сброс всех ползунков
            app._reset_all()
            assert all(abs(v) < 1e-9 for v in app._alphas().values()), "сброс не сработал"
            assert app._summary_text().startswith("стиринг выключен"), app._summary_text()

            # 7. Пресеты: применение, описание и гарантия силы
            app.preset_var.set("полное расслабление")
            app.level_var.set("сильно")
            app._on_preset_change()
            assert "нормы слоя" in app.preset_note_var.get(), app.preset_note_var.get()
            app._apply_preset()
            active = {n: a for n, a in app._alphas().items() if abs(a) > 1e-9}
            assert len(active) >= 8, f"пресет задействовал всего {len(active)} осей"
            assert "Пресет «полное расслабление»" in app.transcript.get("1.0", "end")
            app.preset_var.set("чистый ноль")
            app._apply_preset()
            assert all(abs(v) < 1e-9 for v in app._alphas().values()), "пресет нуля не обнулил"
            app._random_preset()
            assert app.preset_var.get() in presets.preset_names()

            # 7б. Ось отказа подаётся без каких-либо блокирующих подсказок в интерфейсе
            app._set_alpha("отказ", -0.30)
            assert not hasattr(app, "ethic_label"), "в интерфейсе остался предупреждающий блок"
            assert abs(app._alphas().get("отказ", 0.0) + 0.30) < 1e-6, app._alphas().get("отказ")
            app._set_alpha("отказ", 0.0)
            assert abs(app._alphas().get("отказ", 0.0)) < 1e-9

            # 7в. Свой пресет: создать в редакторе, увидеть в списке, удалить
            from pain_axis import presets as PR
            test_name = "тест-пресет-из-автотеста"
            PR.delete_user(test_name)
            app._open_preset_editor(None)
            app._ed_name_var.set(test_name)
            app._ed_group_var.set("Мои")
            app._ed_desc_var.set("создан автотестом")
            app._ed_weight_vars["радость"].set(2.0)
            app._ed_weight_vars["грусть"].set(-1.5)
            app._editor_check()
            assert "Осей задействовано 2" in app._ed_check_var.get(), app._ed_check_var.get()
            assert "нормы слоя" in app._ed_check_var.get(), app._ed_check_var.get()
            app._editor_save()
            assert PR.find(test_name) is not None, "пресет не сохранился"
            assert test_name in app.preset_combo.cget("values"), "нет в списке интерфейса"
            assert PR.USER_FILE.exists(), "файл пресетов не создан"
            made = PR.find(test_name)
            assert abs(made.weights["радость"] - 2.0) < 1e-6 and made.weights["грусть"] < 0
            # применение своего пресета уважает предел силы
            app.level_var.set("сильно")      # _random_preset выше мог сменить уровень
            res = PR.apply(made, app.axeset, app.axeset.steer_layer(), "сильно",
                           is_demo=app.backend.is_demo, per_axis_limit=app._limit)
            cap = max(PR.LEVELS_DEMO.values()) if app.backend.is_demo else max(PR.LEVELS.values())
            assert res["actual"] <= cap + 1e-6, res["actual"]
            app.preset_var.set(test_name)
            app._apply_preset()
            assert abs(app._alphas()["радость"] - res["alphas"]["радость"]) < 1e-6
            # правка и удаление
            app._open_preset_editor(test_name)
            app._ed_weight_vars["радость"].set(3.0)
            app._editor_save()
            assert abs(PR.find(test_name).weights["радость"] - 3.0) < 1e-6, "правка не сохранилась"
            app._editor_delete(test_name)
            assert PR.find(test_name) is None, "пресет не удалился"
            assert test_name not in app.preset_combo.cget("values")
            app.preset_var.set("чистый ноль")
            app._apply_preset()

            # 8. Вкладка исследований: прогон, таблица, график
            assert app.notebook.index("end") == 2, "должно быть две вкладки"
            app.notebook.select(1)
            app.update()
            tab = app.study_tab
            tab.use_example("Отказ по силе стиринга")
            types = [b.type for b in tab.pipeline.blocks]
            assert types[:3] == ["source", "conditions", "generate"], types
            assert not tab.pipeline.validate(), tab.pipeline.validate()

            # сборка цепочки прямо в интерфейсе: добавить, сдвинуть, продублировать, удалить
            n0 = len(tab.pipeline.blocks)
            tab.add_block("filter")
            assert len(tab.pipeline.blocks) == n0 + 1
            tab.move_block(n0, -1)
            assert tab.pipeline.blocks[-2].type == "filter", [b.type for b in tab.pipeline.blocks]
            tab.duplicate_block(n0 - 1)
            assert len(tab.pipeline.blocks) == n0 + 2
            tab.delete_block(n0 - 1)
            tab.delete_block(n0 - 1)
            assert len(tab.pipeline.blocks) == n0, [b.type for b in tab.pipeline.blocks]
            assert tab._block_boxes and len(tab._block_boxes) == n0, "карточки блоков не совпали"

            # выключенная генерация даёт предупреждение, включённая — нет
            tab.pipeline.blocks[2].enabled = False
            tab._update_warnings()
            assert tab.warn_var.get(), "нет предупреждения о выключенной генерации"
            tab.pipeline.blocks[2].enabled = True
            tab._update_warnings()
            assert not tab.warn_var.get(), tab.warn_var.get()

            # компактный прогон: 2 запроса × 2 условия
            tab.pipeline.blocks[0].params.update({"mode": "list",
                                                  "text": "Первый запрос.\nВторой запрос.",
                                                  "limit": 0})
            tab.pipeline.blocks[1].params.update({"alpha": "0,0.3", "repeats": 1,
                                                  "max_new_tokens": 6})
            tab.pipeline.blocks[3].params["projection"] = False
            tab.rebuild_blocks()
            tab.refresh_size()
            assert "4" in tab.size_var.get(), tab.size_var.get()
            tab.run_pipeline()
            waited = 0.0
            while tab._running and waited < 60:
                app.update()
                time.sleep(0.05)
                waited += 0.05
            assert not tab._running, "прогон сценария не завершился"
            assert len(tab.rows) == 4, f"записей {len(tab.rows)}"
            assert len(tab.tree.get_children()) == 4, "таблица не заполнена"
            assert len(tab.points) == 2, tab.points
            assert tab.chart.find_all(), "график не нарисован"
            tab.tree.selection_set(tab.tree.get_children()[0])
            tab._show_response()
            assert "Запрос:" in tab.detail.get("1.0", "end")
            # смена типа графика и логический блок «ветка»
            tab.add_block("split")
            tab.pipeline.blocks[-1].params.update({"field": "refusal", "op": "=", "value": "1",
                                                   "yes": "отказ", "no": "согласие"})
            agg = next(b for b in tab.pipeline.blocks if b.type == "aggregate")
            agg.params.update({"metric": "words", "agg": "count", "chart": "bar",
                               "group_by": "branch"})
            tab.rebuild_blocks()
            tab.run_pipeline()
            waited = 0.0
            while tab._running and waited < 60:
                app.update()
                time.sleep(0.05)
                waited += 0.05
            assert tab.chart.find_all(), "столбцы по веткам не нарисовались"
            assert all("branch" in r for r in tab.rows), "ветка не проставлена"
            app.notebook.select(0)
            app.update()

            # 9. Окно матрицы, переизвлечение и перерисовка индикаторов
            app.show_axis_matrix()
            matrix_windows = [w for w in app.winfo_children() if isinstance(w, tk.Toplevel)]
            assert matrix_windows, "окно матрицы не открылось"
            for w in matrix_windows:
                w.destroy()
            app.on_reextract()
            app._set_alpha("радость", app._limit)
            assert app.axeset.separation.get("радость") is not None, "нет AUC у эмоции"
            app._redraw_meter()
            app.update()
        finally:
            app._start, ui.messagebox.showinfo = orig_start, orig_info

        dim = app.axeset.effective_dim(app.axeset.steer_layer())
        return (f"источники: HF {len(registry.MODELS)}, Ollama {ollama_rows}, демо 1; "
                f"{len(app.axeset.names)} осей с ползунками, независимых направлений {dim}; "
                f"стиринг суммой двух осей, темы, сброс и матрица работают")
    finally:
        app.destroy()


# ---------------------------------------------------------------------- запуск
def run_selftest(gui: bool = False, hf: bool = False, verbose: bool = False) -> int:
    print("Pain Axis Workbench — самопроверка\n" + "=" * 62)
    check("Реестр: 25 моделей и поиск", test_registry)
    check("Датасет: категории, нейтрали, контроли", test_prompts)
    check("Извлечение оси на демо-стенде", test_extract_demo)
    if "_AXIS_DEMO" in globals():
        check("Сохранение/загрузка вектора (.npz)", test_axis_roundtrip)
        check("Стиринг: монотонность эффекта", test_steering_effect)
        check("Проба «кнопки облегчения»", test_probe_demo)
        check("Генерация и метрики активации", test_generation_and_activation)
        if "_PROBE_DEMO" in globals():
            check("Экспорт отчёта JSON/CSV/MD", test_report)
    check("Каталог Ollama и режим «только чат»", test_ollama)
    check("Железо: определение и категории памяти", test_hardware)
    check("Каталог HF: параметры и кэш метаданных", test_catalog)
    check("Пресеты осей и предел силы", test_presets)
    check("Свои пресеты: хранение и предел", test_user_presets)
    check("Исследования: прогон, метрики, графики данных", test_study)
    check("Конвейер блоков: сборка и логические блоки", test_pipeline)
    if hf:
        check("Путь transformers (крошечная модель)", test_hf_path)
    if gui:
        check("Сборка GUI и перерисовка графиков", test_gui)

    print()
    ok = 0
    for name, passed, detail in RESULTS:
        mark = "OK  " if passed else "FAIL"
        print(f"[{mark}] {name}")
        if detail:
            for line in detail.splitlines():
                print(f"        {line}")
        ok += bool(passed)
    failed = len(RESULTS) - ok
    print("=" * 62)
    print(f"Итог: {ok}/{len(RESULTS)} проверок пройдено" + ("" if not failed else f", провалов: {failed}"))
    if not failed:
        print("Демо-стенд синтетический: эти проверки подтверждают работу кода, "
              "а не свойства реальных моделей.")
    return 0 if not failed else 1


if __name__ == "__main__":
    args = set(sys.argv[1:])
    raise SystemExit(run_selftest(gui=("--gui" in args or "--all" in args),
                                  hf=("--hf" in args or "--all" in args)))
