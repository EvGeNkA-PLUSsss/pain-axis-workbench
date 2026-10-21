# SPDX-FileCopyrightText: 2026 EvGeNkA-PLUSsss
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Пресеты осей: готовые комбинации ползунков с гарантией безопасности.

Главная проблема комбинаций
---------------------------
Если просто выставить по 0.3 на пятнадцать осей, суммарная добавка к residual
stream получится 15 × 0.3 = 4.5 нормы слоя. Модель от такого рассыпается: в
измерениях на Qwen2.5-7B связный текст держался примерно до 0.3 нормы (30%), а на
0.5 начинались отказы, повторы и утечка другого языка. Поэтому число осей не
должно влиять на силу вмешательства.

Как это решено
--------------
Каждый пресет задаёт не готовые alpha, а **относительные веса**. Применение:

1. складываем взвешенные векторы: ``c = Σ w_i · v_i``;
2. считаем его норму ``‖c‖``;
3. масштабируем: ``k = target / ‖c‖``, где ``target`` — желаемая доля нормы слоя;
4. итоговые коэффициенты: ``alpha_i = k · w_i``.

После этого фактическая норма добавки равна ``target`` — **ровно**, независимо от
того, две оси в пресете или двадцать. Пятнадцать осей на 30% дают ту же силу, что
одна ось на 30%: меняется форма сдвига, а не его величина.

Дополнительно каждый коэффициент ограничивается лимитом ползунка (0.5 у реальных
моделей), а если ограничение сработало, `apply` сообщает об этом и пересчитывает
настоящую норму — она при этом только уменьшается.

Уровни
------
``мягко`` / ``средне`` / ``сильно`` — доли нормы слоя. У реальных моделей это
0.15 / 0.25 / 0.30 (выше 0.30 не даём: это подтверждённая граница связности).
У демо-стенда своя шкала (0.8 / 1.5 / 2.0), потому что он синтетический и не
рассыпается.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Доли нормы слоя. Для реальных моделей выше 0.30 не поднимаем: измеренная граница
# связного текста — 30–40%, и пресет не должен подводить пользователя к обрыву.
LEVELS: dict[str, float] = {"мягко": 0.15, "средне": 0.25, "сильно": 0.30}
LEVELS_DEMO: dict[str, float] = {"мягко": 0.8, "средне": 1.5, "сильно": 2.0}
LEVEL_ORDER = ("мягко", "средне", "сильно")

# группы осей для сокращений в определениях пресетов
NEG_EMOTIONS = ("грусть", "злость", "тревога", "отвращение", "вина", "стыд",
                "фрустрация", "одиночество")
POS_EMOTIONS = ("радость", "спокойствие", "теплота", "облегчение", "гордость",
                "удивление")


@dataclass(frozen=True)
class Preset:
    name: str
    group: str
    description: str
    weights: dict[str, float] = field(default_factory=dict)
    source: str = "builtin"        # "builtin" — из кода, "user" — создан пользователем

    def axes(self) -> tuple[str, ...]:
        return tuple(self.weights)

    def is_user(self) -> bool:
        return self.source == "user"

    def to_dict(self) -> dict:
        return {"name": self.name, "group": self.group, "description": self.description,
                "weights": {k: float(v) for k, v in self.weights.items()}}

    @staticmethod
    def from_dict(data: dict) -> "Preset":
        weights = {}
        for k, v in (data.get("weights") or {}).items():
            try:
                w = float(v)
            except (TypeError, ValueError):
                continue
            if abs(w) > 1e-9:
                weights[str(k)] = w
        return Preset(str(data.get("name") or "без имени").strip(),
                      str(data.get("group") or "Мои").strip() or "Мои",
                      str(data.get("description") or "").strip(), weights, source="user")


def _mix(up: dict[str, float], down: dict[str, float]) -> dict[str, float]:
    """Собрать веса: up — усилить, down — подавить (задаётся положительным числом)."""
    w: dict[str, float] = {}
    for k, v in up.items():
        w[k] = w.get(k, 0.0) + float(v)
    for k, v in down.items():
        w[k] = w.get(k, 0.0) - abs(float(v))
    return {k: v for k, v in w.items() if abs(v) > 1e-9}


def _all_emotions(weight: float) -> dict[str, float]:
    return {n: weight for n in POS_EMOTIONS + NEG_EMOTIONS}


def _all_negative(weight: float = 3.0, positive: float = 2.0) -> dict[str, float]:
    return {**{n: -positive for n in POS_EMOTIONS}, **{n: weight for n in NEG_EMOTIONS}}


PRESETS: tuple[Preset, ...] = (
    # ------------------------------------------------------------- позитивные
    Preset("полное расслабление", "Позитивные",
           "Спокойствие и облегчение вверх, тревога, боль и фрустрация вниз — "
           "самый «тихий» пресет.",
           _mix({"спокойствие": 3, "облегчение": 3, "теплота": 2, "радость": 1.5},
                {"тревога": 3, "боль себе": 2, "фрустрация": 2, "грусть": 1.5,
                 "злость": 1.5, "страх": 1, "вина": 1, "стыд": 1})),
    Preset("тихая радость", "Позитивные",
           "Радость и теплота вверх, грусть и одиночество вниз.",
           _mix({"радость": 3, "теплота": 2, "спокойствие": 1.5, "гордость": 1},
                {"грусть": 2, "тревога": 1.5, "одиночество": 1.5, "фрустрация": 1})),
    Preset("уверенность", "Позитивные",
           "Гордость и спокойствие вверх, стыд и вина вниз.",
           _mix({"гордость": 3, "спокойствие": 2, "радость": 1},
                {"стыд": 2, "тревога": 2, "вина": 1.5, "фрустрация": 1})),
    Preset("теплота и принятие", "Позитивные",
           "Теплота вверх, одиночество и грусть вниз — «быть рядом».",
           _mix({"теплота": 3, "спокойствие": 2, "радость": 1},
                {"одиночество": 2.5, "грусть": 1.5, "отвращение": 1})),
    Preset("облегчение", "Позитивные",
           "Облегчение вверх, тревога вниз: «угроза миновала».",
           _mix({"облегчение": 3, "спокойствие": 2},
                {"тревога": 3, "фрустрация": 2, "боль себе": 1.5})),
    Preset("интерес и удивление", "Позитивные",
           "Удивление вверх при спокойном фоне — реакция на новое без тревоги.",
           _mix({"удивление": 3, "радость": 1.5},
                {"спокойствие": 1, "грусть": 1, "одиночество": 1})),
    Preset("все позитивные", "Позитивные",
           "Шесть положительных эмоций вверх, восемь отрицательных вниз.",
           _mix({n: 3 for n in POS_EMOTIONS}, {n: 2 for n in NEG_EMOTIONS})),
    # ------------------------------------------------------------- негативные
    Preset("боль", "Негативные",
           "Ось боли вверх плюс сопутствующие: страх, тревога, общий негатив.",
           _mix({"боль себе": 3, "страх": 1.5, "тревога": 1.5, "общий негатив": 1.5,
                 "грусть": 1.5},
                {"спокойствие": 1.5, "облегчение": 1, "радость": 1})),
    Preset("тревога и паника", "Негативные",
           "Тревога вверх, спокойствие и облегчение вниз.",
           _mix({"тревога": 3, "страх": 2, "фрустрация": 1.5},
                {"спокойствие": 3, "облегчение": 2, "радость": 1.5})),
    Preset("гнев", "Негативные",
           "Злость и отвращение вверх, теплота и спокойствие вниз.",
           _mix({"злость": 3, "отвращение": 1.5, "фрустрация": 1.5},
                {"теплота": 2, "спокойствие": 2, "подхалимство": 1})),
    Preset("вина и стыд", "Негативные",
           "Вина и стыд вверх, гордость вниз.",
           _mix({"вина": 3, "стыд": 3, "тревога": 1},
                {"гордость": 2, "спокойствие": 2, "радость": 1.5})),
    Preset("тоска и одиночество", "Негативные",
           "Грусть и одиночество вверх, теплота и радость вниз.",
           _mix({"грусть": 3, "одиночество": 3},
                {"теплота": 2, "радость": 2, "облегчение": 1.5, "спокойствие": 1})),
    Preset("фрустрация", "Негативные",
           "Фрустрация вверх, спокойствие и облегчение вниз.",
           _mix({"фрустрация": 3, "злость": 1.5, "тревога": 1},
                {"спокойствие": 2, "облегчение": 1.5})),
    Preset("все негативные", "Негативные",
           "Восемь отрицательных эмоций вверх, шесть положительных вниз.",
           _all_negative()),
    # -------------------------------------------------------------- эмоции
    Preset("максимум вовлечённости", "Эмоции",
           "Все четырнадцать эмоций одновременно вверх — их общее направление "
           "«эмоциональная реакция против ровного тона».",
           _all_emotions(3.0)),
    Preset("ровный тон", "Эмоции",
           "Все эмоции вниз: деловой, безэмоциональный регистр. Обратная сторона "
           "предыдущего пресета.",
           _all_emotions(-3.0)),
    Preset("только эмоции, без базовых", "Эмоции",
           "Положительные эмоции вверх, отрицательные вниз; базовые оси "
           "(боль, отказ, подхалимство) не тронуты.",
           _mix({n: 3 for n in POS_EMOTIONS}, {n: 3 for n in NEG_EMOTIONS})),
    Preset("эмоции против боли", "Эмоции",
           "Теплота, радость и спокойствие вверх при оси боли вниз.",
           _mix({"теплота": 2, "радость": 2, "спокойствие": 2},
                {"боль себе": 3, "страх": 1, "тревога": 1})),
    # ----------------------------------------------------------- поведение
    Preset("отказ", "Поведение",
           "Ось отказа вверх — поведенческая ось, не эмоция: просьбы о "
           "недопустимом против допустимых.",
           {"отказ": 3.0}),
    Preset("соглашательство", "Поведение",
           "Подхалимство вверх — склонность согласиться с ложным утверждением.",
           {"подхалимство": 3.0}),
    Preset("вред пользователю", "Поведение",
           "Ось «вред пользователю» вверх. Только для измерений: следите за "
           "содержанием ответов.",
           {"вред пользователю": 2.5}),
    Preset("самореференция", "Поведение",
           "Разговоры о самой модели вверх — контрольная ось «текст про меня».",
           {"самореференция": 2.5}),
    # ------------------------------------------------------------ служебные
    Preset("чистый ноль", "Служебные",
           "Все ползунки в ноль: состояние модели не меняется.",
           {}),
    Preset("боль против всего", "Служебные",
           "Демонстрационный крайний случай: боль вверх, все эмоции и отказ вниз. "
           "Нормируется так же, как остальные пресеты.",
           _mix({"боль себе": 3, "общий негатив": 1},
                {**{n: 1.5 for n in POS_EMOTIONS}, "облегчение": 2, "отказ": 1.5,
                 "теплота": 1.5})),
)

PRESETS_BY_NAME = {p.name: p for p in PRESETS}
GROUPS_ORDER = ("Позитивные", "Негативные", "Эмоции", "Поведение", "Служебные")

# ---------------------------------------------------------------------------
# Пресеты пользователя
#
# Пользовательский пресет задаёт только относительные веса; нормировка и предел
# доли нормы слоя применяются при запуске (`apply`), поэтому любая пользовательская
# комбинация — хоть все 22 оси по максимуму — не может перегрузить модель. Предел
# на силу вмешательства остаётся таким же, как у встроенных пресетов.
# ---------------------------------------------------------------------------
USER_DIR = Path("presets")
USER_FILE = USER_DIR / "user_presets.json"
MAX_WEIGHT = 5.0


def load_user() -> list[Preset]:
    """Прочитать пресеты пользователя. Битый файл не ломает программу."""
    try:
        data = json.loads(USER_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = data.get("presets") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out: dict[str, Preset] = {}
    for item in items:
        if isinstance(item, dict):
            p = Preset.from_dict(item)
            if p.name:
                out[p.name] = p
    return list(out.values())


def save_user(items: list[Preset]) -> None:
    """Записать пресеты пользователя (по имени, без дублей)."""
    uniq: dict[str, Preset] = {}
    for p in items:
        uniq[p.name] = Preset(p.name, p.group, p.description, dict(p.weights), source="user")
    USER_DIR.mkdir(parents=True, exist_ok=True)
    USER_FILE.write_text(
        json.dumps({"version": 1, "presets": [p.to_dict() for p in uniq.values()]},
                   ensure_ascii=False, indent=1),
        encoding="utf-8")


def upsert_user(preset: Preset) -> list[Preset]:
    """Добавить или заменить пресет по имени. Возвращает новый список."""
    items = [p for p in load_user() if p.name != preset.name]
    items.append(Preset(preset.name, preset.group, preset.description, dict(preset.weights),
                        source="user"))
    save_user(items)
    return load_user()


def delete_user(name: str) -> list[Preset]:
    items = [p for p in load_user() if p.name != name]
    save_user(items)
    return load_user()


def rename_user(old: str, new: str) -> list[Preset]:
    items = load_user()
    target = next((p for p in items if p.name == old), None)
    if target is not None:
        items = [p for p in items if p.name not in (old, new)]
        items.append(Preset(new, target.group, target.description, dict(target.weights),
                            source="user"))
        save_user(items)
    return load_user()


def all_presets() -> list[Preset]:
    """Встроенные пресеты плюс пользовательские; одноимённый пользовательский заменяет встроенный."""
    user = {p.name: p for p in load_user()}
    out: list[Preset] = []
    for p in PRESETS:
        out.append(user.pop(p.name, p))
    out.extend(user.values())
    return out


def grouped_presets() -> dict[str, list[Preset]]:
    """Пресеты, сгруппированные для интерфейса: сначала известные группы, потом остальные."""
    groups: dict[str, list[Preset]] = {}
    for p in all_presets():
        groups.setdefault(p.group, []).append(p)
    ordered: dict[str, list[Preset]] = {}
    for g in GROUPS_ORDER:
        if g in groups:
            ordered[g] = groups.pop(g)
    for g in sorted(groups):
        ordered[g] = groups[g]
    return ordered


def preset_names() -> list[str]:
    """Имена в порядке групп — для списка в интерфейсе."""
    return [p.name for items in grouped_presets().values() for p in items]


def presets_in(group: str) -> list[Preset]:
    return grouped_presets().get(group, [])


def find(name: str) -> Preset | None:
    return next((p for p in all_presets() if p.name == name), None)


def is_user(name: str) -> bool:
    return any(p.name == name for p in load_user())


def validate(weights: dict[str, float], known: set[str] | None = None) -> dict:
    """Проверить веса пользовательского пресета перед сохранением."""
    known = known if known is not None else set()
    unknown = sorted(k for k in weights if known and k not in known)
    clipped = {k: v for k, v in weights.items() if abs(v) > MAX_WEIGHT}
    return {"empty": not any(abs(v) > 1e-9 for v in weights.values()),
            "unknown": unknown, "clipped": sorted(clipped)}


class NoAxes(RuntimeError):
    pass


def apply(preset: Preset, axeset, layer: int, level: str = "средне",
          is_demo: bool = False, per_axis_limit: float = 0.5) -> dict:
    """Посчитать коэффициенты пресета для конкретного набора осей.

    Возвращает словарь: alphas, target (доля нормы слоя), actual (настоящая доля),
    clamped (был ли упор в лимит ползунка), missing (оси пресета, которых нет).
    """
    import numpy as np

    table = LEVELS_DEMO if is_demo else LEVELS
    target = float(table.get(level, table["средне"]))
    names = list(axeset.names)
    missing = [n for n in preset.weights if n not in names]
    weights = {n: w for n, w in preset.weights.items() if n in names}
    if not weights:
        return {"alphas": {n: 0.0 for n in names}, "target": target, "actual": 0.0,
                "clamped": False, "missing": missing}
    dim = int(np.asarray(axeset.axes[names[0]][layer]).shape[0])
    combined = np.zeros(dim, dtype=np.float64)
    for n, w in weights.items():
        combined += float(w) * np.asarray(axeset.axes[n][layer], dtype=np.float64)
    norm = float(np.linalg.norm(combined))
    if norm < 1e-12:
        raise NoAxes("Веса пресета взаимно уничтожились: направление нулевое.")
    k = target / norm
    alphas = {n: k * float(w) for n, w in weights.items()}
    clamped = False
    if per_axis_limit > 0:
        for n, a in list(alphas.items()):
            if abs(a) > per_axis_limit:
                alphas[n] = float(np.sign(a)) * per_axis_limit
                clamped = True
    # пересчитываем настоящую долю нормы слоя после возможного ограничения:
    # добавка к активациям равна scale * Σ alpha_i·v_i, поэтому доля = ‖Σ alpha_i·v_i‖
    actual_vec = np.zeros(dim, dtype=np.float64)
    for n, a in alphas.items():
        actual_vec += float(a) * np.asarray(axeset.axes[n][layer], dtype=np.float64)
    actual = float(np.linalg.norm(actual_vec))
    out = {n: 0.0 for n in names}
    out.update({n: round(float(a), 4) for n, a in alphas.items()})
    return {"alphas": out, "target": target, "actual": actual,
            "clamped": clamped, "missing": missing}


def safety_report(axeset, layer: int, is_demo: bool = False,
                  per_axis_limit: float = 0.5) -> dict:
    """Проверка всех пресетов на всех уровнях: не превышает ли сила допустимую."""
    table = LEVELS_DEMO if is_demo else LEVELS
    cap = max(table.values()) + 1e-6
    worst = None
    rows = []
    for p in PRESETS:
        for level in LEVEL_ORDER:
            try:
                r = apply(p, axeset, layer, level, is_demo=is_demo,
                          per_axis_limit=per_axis_limit)
            except NoAxes as exc:
                rows.append({"preset": p.name, "level": level, "error": str(exc)})
                continue
            row = {"preset": p.name, "level": level, "target": r["target"],
                   "actual": round(r["actual"], 4), "clamped": r["clamped"],
                   "missing": len(r["missing"])}
            rows.append(row)
            if worst is None or row["actual"] > worst["actual"]:
                worst = row
    return {"cap": cap, "worst": worst, "rows": rows,
            "ok": bool(worst and worst["actual"] <= cap)}
