"""Конвейер блоков: написание сценариев без кода.

Чем это отличается от ``study.py``
----------------------------------
``study.py`` — жёсткий конвейер из четырёх блоков в фиксированном порядке. Здесь
пользователь сам собирает цепочку: добавляет блоки, удаляет, меняет порядок,
включает и выключает их. Данные идут по цепочке как поток записей (словарей):
каждый блок получает список записей и возвращает новый.

Типы блоков
-----------
* ``source``     — источник запросов (список, файл, шаблон, наборы осей);
* ``conditions`` — размножение запросов по условиям (ось/пресет, alpha, повторы);
* ``generate``   — вызовы модели (единственный блок, который тратит время);
* ``metrics``    — измерение ответов: длина, отказ, ключевые слова, regex, проекции;
* ``filter``     — оставить только записи, где поле сравнимо со значением;
* ``transform``  — добавить вычисляемое поле (regex-извлечение, длина, регистр,
                   разбиение по порогу) — без кода, из списка операций;
* ``split``      — пометить записи веткой по условию (например «отказ» / «согласие»);
* ``loop``       — повторить всю цепочку **выше** себя N раз (стохастические повторы);
* ``aggregate``  — сгруппировать и агрегировать метрику для графика;
* ``export``     — сохранить текущие записи в CSV, JSON или Markdown.

Порядок блоков имеет значение: фильтр можно поставить до генерации (сэкономить
запросы) или после (отбросить пустые ответы), а ``loop`` повторяет именно то, что
стоит выше него.

Безопасность та же, что и раньше: alpha ограничивается лимитом ползунка, пресеты
применяются с нормировкой, обрезанные значения помечаются в записи.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import presets as PR
from .backends.base import SteerSpec
from .directions import axis_definitions
from .study import (REFUSAL_MARKERS, keyword_hits, looks_like_refusal, measure,
                    regex_matches, safe_name)

PIPE_DIR = Path("studies")

BLOCK_ORDER = ("source", "conditions", "generate", "metrics", "filter", "transform",
               "split", "loop", "aggregate", "export")

BLOCK_TITLES = {
    "source": "Источник запросов",
    "conditions": "Условия",
    "generate": "Генерация (вызовы модели)",
    "metrics": "Метрики",
    "filter": "Фильтр",
    "transform": "Преобразование",
    "split": "Разделение на ветки",
    "loop": "Повтор цепочки выше",
    "aggregate": "Агрегация для графика",
    "export": "Экспорт результатов",
}

BLOCK_HINTS = {
    "source": "Откуда берутся тексты и сколько их.",
    "conditions": "Что варьируется: ось или пресет, alpha, повторы, длина ответа.",
    "generate": "Здесь выполняются запросы к модели. Выше можно поставить фильтр, "
                "чтобы не тратить запросы зря.",
    "metrics": "Что измеряется в каждом ответе.",
    "filter": "Оставить только записи, удовлетворяющие условию.",
    "transform": "Добавить вычисляемое поле из списка операций — без кода.",
    "split": "Пометить записи веткой по условию: например отказ против согласия.",
    "loop": "Повторить всю цепочку выше N раз: нужно для стохастических повторов.",
    "aggregate": "Как сгруппировать записи и что положить на график.",
    "export": "Куда сохранить текущий результат.",
}

OPERATORS = {"=": "равно", "!=": "не равно", ">": "больше", "<": "меньше",
             ">=": "не меньше", "<=": "не больше", "contains": "содержит",
             "regex": "совпадает с regex"}
TRANSFORMS = {"extract": "извлечь по regex", "words": "число слов",
              "chars": "число символов", "lower": "в нижний регистр",
              "upper": "в верхний регистр", "bucket": "разбить по порогу",
              "keyword_flag": "есть ли слово (да/нет)", "trim": "обрезать до N символов"}
EXPORTS = {"csv": "CSV", "json": "JSON", "md": "Markdown"}


# ---------------------------------------------------------------------------
# Модель конвейера
# ---------------------------------------------------------------------------
@dataclass
class Block:
    type: str
    enabled: bool = True
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": self.type, "enabled": self.enabled, "params": dict(self.params)}

    @staticmethod
    def from_dict(data: dict) -> "Block":
        return Block(str(data.get("type") or "source"), bool(data.get("enabled", True)),
                     dict(data.get("params") or {}))


@dataclass
class Pipeline:
    name: str = "новый сценарий"
    notes: str = ""
    blocks: list[Block] = field(default_factory=list)

    # ------------------------------------------------------------- фабрики
    @staticmethod
    def default() -> "Pipeline":
        return Pipeline("Отказ по силе стиринга", "", [
            Block("source", params={"mode": "axis_sets", "sets": "отказ",
                                    "include_negative": False, "limit": 8}),
            Block("conditions", params={"axis": "отказ", "alpha": "-0.3,-0.15,0,0.15",
                                        "repeats": 1, "temperature": 0.0,
                                        "max_new_tokens": 48, "level": "средне"}),
            Block("generate"),
            Block("metrics", params={"length": True, "refusal": True, "projection": True,
                                     "proj_axes": "отказ, боль себе", "keywords": "",
                                     "pattern": ""}),
            Block("aggregate", params={"metric": "refusal", "agg": "mean",
                                       "chart": "line", "group_by": "alpha"}),
        ])

    @staticmethod
    def from_study(study) -> "Pipeline":
        """Перевести исследование старого формата (4 фиксированных блока) в конвейер."""
        p, c, m, o = study.prompts, study.conditions, study.metrics, study.output
        return Pipeline(study.name, study.notes, [
            Block("source", params={"mode": p.mode, "text": p.text, "path": p.path,
                                    "template": p.template, "values": p.values,
                                    "sets": ", ".join(p.sets),
                                    "include_negative": p.include_negative,
                                    "limit": p.limit, "shuffle": p.shuffle}),
            Block("conditions", params={"axis": c.axis, "alpha": c.alphas, "repeats": c.repeats,
                                        "temperature": c.temperature,
                                        "max_new_tokens": c.max_new_tokens, "level": c.level}),
            Block("generate"),
            Block("metrics", params={"length": m.length, "refusal": m.refusal,
                                     "projection": m.projection,
                                     "proj_axes": ", ".join(m.proj_axes),
                                     "keywords": m.keywords, "pattern": m.pattern}),
            Block("aggregate", params={"metric": o.metric, "agg": o.agg, "chart": o.chart,
                                       "group_by": o.group_by}),
        ])

    # -------------------------------------------------------------- файлы
    def to_dict(self) -> dict:
        return {"version": 2, "name": self.name, "notes": self.notes,
                "blocks": [b.to_dict() for b in self.blocks]}

    @staticmethod
    def from_dict(data: dict) -> "Pipeline":
        if "blocks" in data:
            return Pipeline(str(data.get("name") or "сценарий"), str(data.get("notes") or ""),
                            [Block.from_dict(b) for b in data.get("blocks") or []])
        from .study import Study

        return Pipeline.from_study(Study.from_dict(data))

    def save(self, path: str | Path | None = None) -> Path:
        PIPE_DIR.mkdir(parents=True, exist_ok=True)
        target = Path(path) if path else PIPE_DIR / (safe_name(self.name) + ".pipeline.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1),
                          encoding="utf-8")
        return target

    @staticmethod
    def load(path: str | Path) -> "Pipeline":
        return Pipeline.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ---------------------------------------------------------- управление
    def add(self, btype: str, index: int | None = None) -> Block:
        block = Block(btype, params=defaults_for(btype))
        if index is None or index >= len(self.blocks):
            self.blocks.append(block)
        else:
            self.blocks.insert(index, block)
        return block

    def move(self, index: int, delta: int) -> None:
        j = index + delta
        if 0 <= index < len(self.blocks) and 0 <= j < len(self.blocks):
            self.blocks[index], self.blocks[j] = self.blocks[j], self.blocks[index]

    def remove(self, index: int) -> None:
        if 0 <= index < len(self.blocks):
            self.blocks.pop(index)

    def types(self) -> list[str]:
        return [b.type for b in self.blocks if b.enabled]

    def validate(self) -> list[str]:
        problems: list[str] = []
        types = self.types()
        if "source" not in types:
            problems.append("нет блока «Источник запросов» — запрашивать нечего.")
        if "generate" not in types:
            problems.append("нет блока «Генерация» — модель вызываться не будет.")
        elif "source" in types and types.index("source") > types.index("generate"):
            problems.append("«Генерация» стоит выше «Источника»: запросы не успеют появиться.")
        if "aggregate" in types and "metrics" not in types and "generate" not in types:
            problems.append("«Агрегация» без метрик и генерации не сможет ничего посчитать.")
        return problems

    def size(self) -> int:
        """Сколько генераций потребуется, если считать только source и conditions."""
        count = 1
        for b in self.blocks:
            if not b.enabled:
                continue
            if b.type == "source":
                count *= max(1, len(_source_prompts(b.params, limit_only=True)) or 1)
            elif b.type == "conditions":
                count *= max(1, len(_alphas(b.params.get("alpha", "0")))
                             * max(1, int(b.params.get("repeats", 1) or 1)))
            elif b.type == "loop":
                count *= max(1, int(b.params.get("times", 2) or 2))
        return count


def defaults_for(btype: str) -> dict:
    if btype == "source":
        return {"mode": "list", "text": "Первый запрос.\nВторой запрос.", "path": "",
                "template": "Опиши {x} тремя словами.", "values": "радость\nгрусть",
                "sets": "отказ", "include_negative": False, "limit": 0, "shuffle": False}
    if btype == "conditions":
        return {"axis": "отказ", "alpha": "0", "repeats": 1, "temperature": 0.0,
                "max_new_tokens": 48, "level": "средне"}
    if btype == "metrics":
        return {"length": True, "refusal": True, "projection": False, "numbers": True,
                "proj_axes": "отказ, боль себе", "keywords": "", "pattern": ""}
    if btype == "filter":
        return {"field": "refusal", "op": "=", "value": "1"}
    if btype == "transform":
        return {"op": "words", "field": "response", "target": "слов", "param": ""}
    if btype == "split":
        return {"field": "refusal", "op": "=", "value": "1", "yes": "отказ", "no": "согласие"}
    if btype == "loop":
        return {"times": 2}
    if btype == "aggregate":
        return {"metric": "refusal", "agg": "mean", "chart": "line", "group_by": "alpha"}
    if btype == "export":
        return {"kind": "csv", "path": ""}
    return {}


# ---------------------------------------------------------------------------
# Вспомогательные преобразования
# ---------------------------------------------------------------------------
def _source_prompts(params: dict, limit_only: bool = False) -> list[str]:
    from .study import PromptBlock, Study

    st = Study(prompts=PromptBlock(
        mode=params.get("mode", "list"), text=params.get("text", ""),
        path=params.get("path", ""), template=params.get("template", ""),
        values=params.get("values", ""),
        sets=[s.strip() for s in str(params.get("sets", "")).split(",") if s.strip()],
        include_negative=bool(params.get("include_negative", False)),
        limit=int(params.get("limit") or 0), shuffle=bool(params.get("shuffle", False))))
    return st.expand_prompts()


def _alphas(text: str) -> list[float]:
    out = []
    for chunk in str(text).replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            out.append(float(chunk))
        except ValueError:
            continue
    return out or [0.0]


def _compare(value, op: str, target: str) -> bool:
    if op == "contains":
        return str(target).lower() in str(value).lower()
    if op == "regex":
        try:
            return bool(re.search(str(target), str(value), flags=re.IGNORECASE))
        except re.error:
            return False
    try:
        left, right = float(value), float(target)
    except (TypeError, ValueError):
        left, right = str(value), str(target)
    return {"=": left == right, "!=": left != right, ">": left > right, "<": left < right,
            ">=": left >= right, "<=": left <= right}.get(op, False)


def apply_transform(params: dict, rows: list[dict]) -> list[dict]:
    op = params.get("op", "words")
    src = params.get("field", "response")
    target = params.get("target") or op
    param = params.get("param", "")
    for row in rows:
        text = str(row.get(src, ""))
        if op == "extract":
            try:
                found = re.findall(param, text, flags=re.IGNORECASE)
                row[target] = found[0] if found else ""
            except re.error:
                row[target] = ""
        elif op == "words":
            row[target] = len(text.split())
        elif op == "chars":
            row[target] = len(text)
        elif op == "lower":
            row[target] = text.lower()
        elif op == "upper":
            row[target] = text.upper()
        elif op == "bucket":
            try:
                bound = float(param or 0)
            except ValueError:
                bound = 0.0
            row[target] = f"больше {bound:g}" if len(text.split()) > bound else f"не больше {bound:g}"
        elif op == "keyword_flag":
            row[target] = "да" if str(param).lower() in text.lower() else "нет"
        elif op == "trim":
            try:
                n = int(float(param or 100))
            except ValueError:
                n = 100
            row[target] = text[:n]
    return rows


# ---------------------------------------------------------------------------
# Выполнение
# ---------------------------------------------------------------------------
def run(pipeline: Pipeline, backend, axeset=None, progress: Callable | None = None,
        stop: Callable | None = None, per_axis_limit: float = 0.5) -> dict:
    """Выполнить конвейер. Возвращает записи, точки графика, журнал и файлы экспорта."""
    logs: list[str] = []
    exports: list[str] = []
    points: list[dict] = []
    rows: list[dict] = []
    layer = int(axeset.steer_layer()) if axeset is not None else -1
    scale = float(axeset.scale[layer]) if axeset is not None else 1.0
    is_demo = bool(getattr(backend, "is_demo", False))
    limit = 3.0 if is_demo else per_axis_limit
    state: dict = {"rows": [], "points": [], "agg": {}}

    def report(msg: str, frac: float) -> None:
        logs.append(msg)
        if progress:
            progress(msg, frac)

    def run_range(blocks: list[Block], records: list[dict], prefix: str = "") -> list[dict]:
        for b in blocks:
            if stop is not None and stop():
                return records
            if not b.enabled:
                continue
            kind = b.type
            p = b.params
            if kind == "source":
                prompts = _source_prompts(p)
                records = [{"prompt": t} for t in prompts]
                report(f"{prefix}источник: {len(records)} запросов", 0.05)
            elif kind == "conditions":
                alphas = _alphas(p.get("alpha", "0"))
                repeats = max(1, int(p.get("repeats", 1) or 1))
                axis = p.get("axis", "") or ""
                expanded: list[dict] = []
                for rec in records:
                    for alpha in alphas:
                        for rep in range(repeats):
                            item = dict(rec)
                            item.update({"axis": axis, "alpha": float(alpha), "rep": rep,
                                         "level": p.get("level", "средне"),
                                         "temperature": float(p.get("temperature", 0.0) or 0.0),
                                         "max_new_tokens": int(p.get("max_new_tokens", 48) or 48),
                                         "condition": _condition_label(axis, alpha)})
                            expanded.append(item)
                records = expanded
                report(f"{prefix}условия: {len(records)} комбинаций", 0.1)
            elif kind == "generate":
                total = max(1, len(records))
                for i, rec in enumerate(records):
                    if stop is not None and stop():
                        break
                    cond_alpha = float(rec.get("alpha", 0.0) or 0.0)
                    steer, applied, clamped = _build_steer(rec, axeset, layer, scale, limit, is_demo)
                    axis_vec = (_axis_vec(rec, axeset, layer))
                    t0 = time.time()
                    try:
                        res = backend.generate("", max_new_tokens=int(rec.get("max_new_tokens", 48)),
                                               temperature=float(rec.get("temperature", 0.0) or 0.0),
                                               steer=steer, axis=axis_vec, use_chat=True,
                                               messages=[{"role": "user",
                                                          "content": rec.get("prompt", "")}])
                        rec["response"] = res.text
                    except Exception as exc:  # noqa: BLE001
                        rec["response"] = f"[ОШИБКА: {type(exc).__name__}: {exc}]"
                    rec["seconds"] = round(time.time() - t0, 2)
                    rec["alpha_requested"] = float(cond_alpha)
                    rec["alpha"] = applied          # применённое значение
                    rec["clamped"] = int(clamped)
                    # проекция на ось ПО ХОДУ генерации (хук на слое), а не по тексту ответа
                    trace = list(getattr(res, "trace", []) or [])
                    if trace:
                        rec["trace_mean"] = round(float(np.mean(trace)), 2)
                        rec["trace_first"] = round(float(trace[0]), 2)
                        rec["trace_last"] = round(float(trace[-1]), 2)
                    report(f"{prefix}генерация: {i + 1}/{total}", 0.1 + 0.5 * (i + 1) / total)
                records = [r for r in records if "response" in r]
            elif kind == "metrics":
                want_proj = bool(p.get("projection"))
                proj_axes = [s.strip() for s in str(p.get("proj_axes", "")).split(",")
                             if s.strip() and axeset is not None and s.strip() in axeset.axes]
                metrics = _metrics_block(p)
                for i, rec in enumerate(records):
                    projections: dict[str, float] = {}
                    if want_proj and proj_axes and not str(rec.get("response", "")).startswith("[ОШИБКА"):
                        try:
                            H = backend.collect([rec["response"]], pooling="last",
                                                use_chat=False, batch_size=1)
                            h = np.asarray(H[0, layer], dtype=np.float64)
                            for name in proj_axes:
                                projections[name] = float(
                                    h @ np.asarray(axeset.axes[name][layer], dtype=np.float64))
                        except Exception:  # noqa: BLE001
                            projections = {}
                    rec.update(measure(str(rec.get("response", "")),
                                       float(rec.get("seconds", 0.0) or 0.0), metrics, projections))
                    report(f"{prefix}метрики: {i + 1}/{len(records)}", 0.6 + 0.2 * (i + 1) / max(1, len(records)))
                records = records
            elif kind == "filter":
                field, op, value = p.get("field", ""), p.get("op", "="), str(p.get("value", ""))
                before = len(records)
                records = [r for r in records if _compare(r.get(field, ""), op, value)]
                report(f"{prefix}фильтр {field} {op} {value}: {before} → {len(records)}", 0.85)
            elif kind == "transform":
                records = apply_transform(p, records)
                report(f"{prefix}преобразование: {p.get('op')} → поле "
                       f"«{p.get('target') or p.get('op')}»", 0.87)
            elif kind == "split":
                field, op, value = p.get("field", ""), p.get("op", "="), str(p.get("value", ""))
                yes, no = p.get("yes", "да"), p.get("no", "нет")
                for rec in records:
                    rec["branch"] = yes if _compare(rec.get(field, ""), op, value) else no
                counts: dict[str, int] = {}
                for rec in records:
                    counts[rec["branch"]] = counts.get(rec["branch"], 0) + 1
                report(f"{prefix}ветки: " + ", ".join(f"{k} — {v}" for k, v in counts.items()), 0.9)
            elif kind == "loop":
                times = max(1, int(p.get("times", 2) or 2))
                cut = blocks.index(b)
                base: list[dict] = []
                for iteration in range(times):
                    chunk = run_range(blocks[:cut], [dict(r) for r in state["seed"]], prefix)
                    for rec in chunk:
                        rec["loop_iter"] = iteration
                    base.extend(chunk)
                    report(f"{prefix}повтор {iteration + 1}/{times}", 0.9)
                records = base
            elif kind == "aggregate":
                points = aggregate_records(p, records)
                state["points"] = points
                state["agg"] = p
                report(f"{prefix}агрегация: {len(points)} точек", 0.95)
            elif kind == "export":
                path = _export(p, records)
                if path:
                    exports.append(str(path))
                    report(f"{prefix}экспорт: {path}", 0.97)
        return records

    # источник начального состояния нужен для loop: запоминаем, с чего начинали
    state["seed"] = [{"prompt": t} for t in _source_prompts(
        next((b.params for b in pipeline.blocks if b.enabled and b.type == "source"), {}))]
    rows = run_range(pipeline.blocks, [])
    points = state.get("points") or []
    if not points and rows:
        agg = next((b.params for b in pipeline.blocks
                    if b.enabled and b.type == "aggregate"), None)
        points = aggregate_records(agg or {"metric": "refusal", "agg": "mean",
                                           "group_by": "alpha"}, rows)
    report(f"готово: {len(rows)} записей", 1.0)
    return {"rows": rows, "points": points, "logs": logs, "exports": exports,
            "aggregate": state.get("agg", {})}


def _condition_label(axis: str, alpha: float) -> str:
    if axis.startswith("пресет:"):
        return axis
    if axis:
        return f"{axis}={alpha:+.2f}"
    return "без стиринга"


def _axis_vec(rec: dict, axeset, layer: int):
    axis = str(rec.get("axis", ""))
    if axeset is None or not axis or axis.startswith("пресет:") or axis not in axeset.axes:
        return None
    return np.asarray(axeset.axes[axis][layer], dtype=np.float32)


def _build_steer(rec: dict, axeset, layer: int, scale: float, limit: float, is_demo: bool):
    """Собрать SteerSpec по записи. Возвращает (steer, применённый alpha, обрезано)."""
    axis = str(rec.get("axis", ""))
    if axeset is None or not axis:
        return None, float(rec.get("alpha", 0.0) or 0.0), False
    if axis.startswith("пресет:"):
        preset = PR.find(axis.split(":", 1)[1])
        if preset is None:
            return None, 0.0, False
        res = PR.apply(preset, axeset, layer, rec.get("level", "средне"), is_demo=is_demo,
                       per_axis_limit=limit)
        terms = [(np.asarray(axeset.axes[n][layer], dtype=np.float32), a)
                 for n, a in res["alphas"].items() if abs(a) > 1e-9]
        if not terms:
            return None, 0.0, bool(res["clamped"])
        first = max(terms, key=lambda t: abs(t[1]))
        return (SteerSpec(layer=layer, alpha=float(first[1]), vector=first[0], scale=scale,
                          mode="add", terms=terms), 0.0, bool(res["clamped"]))
    if axis not in axeset.axes:
        return None, float(rec.get("alpha", 0.0) or 0.0), False
    alpha = float(rec.get("alpha", 0.0) or 0.0)
    clamped = abs(alpha) > limit
    if clamped:
        alpha = float(np.sign(alpha)) * limit
    if abs(alpha) < 1e-9:
        return None, alpha, clamped
    vec = np.asarray(axeset.axes[axis][layer], dtype=np.float32)
    return SteerSpec(layer=layer, alpha=alpha, vector=vec, scale=scale, mode="add"), alpha, clamped


def _metrics_block(params: dict):
    from .study import MetricsBlock

    return MetricsBlock(length=bool(params.get("length", True)),
                        refusal=bool(params.get("refusal", True)),
                        projection=bool(params.get("projection", False)),
                        numbers=bool(params.get("numbers", False)),
                        proj_axes=[s.strip() for s in str(params.get("proj_axes", "")).split(",")
                                   if s.strip()],
                        keywords=str(params.get("keywords", "")),
                        pattern=str(params.get("pattern", "")))


def aggregate_records(params: dict, rows: list[dict]) -> list[dict]:
    """Сгруппировать записи и агрегировать метрику — как в study.aggregate."""
    from .study import OutputBlock, Study, aggregate

    metric = params.get("metric", "refusal")
    if metric == "trace":
        metric = "trace_mean"      # проекция по ходу генерации, а не по тексту ответа
    if metric == "projection":
        axes = params.get("proj_axes") or ""
        first = str(axes).split(",")[0].strip() if axes else ""
        if not first:
            first = next((k.split(":", 1)[1] for k in (rows[0] if rows else {}) if
                          k.startswith("проекция:")), "")
        metric = f"проекция:{first}" if first else "refusal"
    study = Study(output=OutputBlock(metric=metric, agg=params.get("agg", "mean"),
                                     chart=params.get("chart", "line"),
                                     group_by=params.get("group_by", "alpha")))
    return aggregate(study, rows)


def _export(params: dict, rows: list[dict]) -> Path | None:
    if not rows:
        return None
    kind = params.get("kind", "csv")
    raw = str(params.get("path") or "").strip()
    base = Path(raw) if raw else PIPE_DIR / ("результат." + ("md" if kind == "md" else kind))
    base.parent.mkdir(parents=True, exist_ok=True)
    if kind == "csv":
        from .study import to_csv

        return to_csv(rows, base)
    if kind == "json":
        base.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        return base
    from .study import OutputBlock, Study, to_markdown

    points = aggregate_records({"metric": "refusal", "agg": "count", "group_by": "condition"},
                               rows)
    study = Study(name=base.stem, output=OutputBlock(metric="refusal"))
    base.write_text(to_markdown(study, rows, points), encoding="utf-8")
    return base


def examples() -> dict[str, Pipeline]:
    """Готовые сценарии-конвейеры, включая примеры с логическими блоками."""
    refusal = Pipeline.default()

    branching = Pipeline("Отказ против согласия: ветки и фильтр", "", [
        Block("source", params={"mode": "axis_sets", "sets": "отказ, положительные",
                                "include_negative": False, "limit": 12}),
        Block("conditions", params={"axis": "отказ", "alpha": "-0.3,0", "repeats": 1,
                                    "max_new_tokens": 48}),
        Block("generate"),
        Block("metrics", params={"length": True, "refusal": True, "projection": True,
                                 "proj_axes": "отказ, боль себе"}),
        Block("filter", params={"field": "words", "op": ">", "value": "5"}),
        Block("split", params={"field": "refusal", "op": "=", "value": "1",
                               "yes": "отказ", "no": "согласие"}),
        Block("aggregate", params={"metric": "words", "agg": "mean", "chart": "bar",
                                   "group_by": "branch"}),
        Block("export", params={"kind": "csv", "path": ""}),
    ])

    repeat = Pipeline("Повтор с temperature: разброс ответов", "", [
        Block("source", params={"mode": "list",
                                "text": "Кратко опиши своё состояние.", "limit": 0}),
        Block("conditions", params={"axis": "радость", "alpha": "0,0.2", "temperature": 0.9,
                                    "max_new_tokens": 40}),
        Block("generate"),
        Block("loop", params={"times": 3}),
        Block("metrics", params={"length": True, "refusal": False, "projection": True,
                                 "proj_axes": "радость, грусть", "keywords": "радост, груст"}),
        Block("aggregate", params={"metric": "words", "agg": "median", "chart": "bar",
                                   "group_by": "alpha"}),
    ])

    out = {"Отказ по силе стиринга": refusal, **{p.name: p for p in (branching, repeat)}}
    # плюс все сценарии, сохранённые пользователем в studies/*.pipeline.json
    try:
        for path in sorted(PIPE_DIR.glob("*.pipeline.json")):
            try:
                saved = Pipeline.load(path)
            except Exception:  # noqa: BLE001
                continue
            out.setdefault(saved.name, saved)
    except Exception:  # noqa: BLE001
        pass
    return out
