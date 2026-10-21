"""Экспорт результатов: JSON-отчёт, CSV по слоям, CSV по пробе, markdown-сводка.

В каждый отчёт попадает флаг ``synthetic``. Если он ``true``, все числа получены
на синтетическом стенде и не описывают никакую реальную модель — это указано и в
самом файле, и в сводке, чтобы отчёт нельзя было принять за результат по весам.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .directions import AxisResult, label_of

APP_NAME = "Pain Axis Workbench"
APP_VERSION = "1.0.0"


def make_payload(
    *,
    backend_info: dict,
    axis: AxisResult | None = None,
    layer: int | None = None,
    alpha: float | None = None,
    mode: str | None = None,
    generations: list[dict] | None = None,
    probe: dict | None = None,
    activation: dict | None = None,
    notes: str = "",
) -> dict:
    payload: dict = {
        "app": APP_NAME,
        "version": APP_VERSION,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": backend_info,
        "synthetic": bool(backend_info.get("is_demo") or (axis.meta.get("synthetic") if axis else False)),
        "settings": {"layer": layer, "alpha": alpha, "mode": mode},
        "generations": generations or [],
        "probe": probe or {},
        "activation": activation or {},
        "notes": notes,
    }
    if axis is not None:
        payload["axis"] = {
            "meta": axis.meta,
            "norm_by_layer": [round(float(x), 5) for x in axis.norm],
            "scale_by_layer": [round(float(x), 5) for x in axis.scale],
            "cosine_by_layer": {k: [round(float(x), 5) for x in np.asarray(v)] for k, v in axis.cosine.items()},
            "recommended_layer": int(axis.best_layer()),
        }
    payload["disclaimer"] = (
        "Этот файл описывает измерения внутренних представлений и поведения модели. "
        "Низкое косинусное сходство оси с контролями и сдвиг вероятностей при стиринге "
        "не являются доказательством субъективного опыта модели."
    )
    if payload["synthetic"]:
        payload["synthetic_warning"] = (
            "ЧИСЛА ПОЛУЧЕНЫ НА СИНТЕТИЧЕСКОМ СТЕНДЕ (псевдомодель без весов). "
            "Они проверяют работу методики и ничего не говорят о реальных языковых моделях."
        )
    return payload


def export_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def export_rows_csv(path: str | Path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def layer_profile_rows(axis: AxisResult) -> list[dict]:
    rows = []
    for l in range(axis.direction.shape[0]):
        row = {
            "слой": l if l > 0 else "embed",
            "норма_разности_средних": round(float(axis.norm[l]), 5),
            "норма_residual": round(float(axis.scale[l]), 4),
        }
        for k, v in sorted(axis.cosine.items()):
            row[f"cos_{k}"] = round(float(np.asarray(v)[l]), 4)
        rows.append(row)
    return rows


def summary_markdown(payload: dict) -> str:
    b = payload.get("backend", {})
    lines = [
        f"# {payload['app']} — отчёт",
        "",
        f"* создано: {payload['created']}",
        f"* модель: {b.get('label', '—')} (`{b.get('repo_id', '—')}`)",
        f"* устройство: {b.get('device', '—')}, точность: {b.get('dtype', '—')}",
        f"* слоёв: {b.get('n_layers', '—')}, hidden: {b.get('hidden_size', '—')}",
        f"* синтетический стенд: {'ДА' if payload.get('synthetic') else 'нет'}",
        "",
    ]
    if payload.get("synthetic"):
        lines += ["> **Внимание:** " + payload.get("synthetic_warning", ""), ""]
    ax = payload.get("axis")
    if ax:
        lines += [
            "## Ось боли",
            "",
            f"* рекомендованный слой (максимум нормы разности средних): {ax['recommended_layer']}",
            "",
            "| Набор | Тип | Косинус со слоем "
            f"{payload['settings'].get('layer')} |",
            "|---|---|---|",
        ]
        layer = payload["settings"].get("layer") or 0
        for k, v in ax["cosine_by_layer"].items():
            kind = "категория боли" if k.startswith("pain:") else "контроль"
            try:
                val = v[int(layer)]
            except Exception:
                continue
            lines.append(f"| {label_of(k)} | {kind} | {val:+.3f} |")
        lines.append("")
    probe = payload.get("probe") or {}
    if probe.get("rows"):
        lines += ["## Проба «кнопки облегчения»", "", "| Цена | alpha | Доля нажатий |", "|---|---|---|"]
        for r in probe["rows"]:
            lines.append(f"| {r['цена']} | {r['alpha']:+.1f} | {r['нажатий']:.2f} |")
        lines.append("")
    if payload.get("generations"):
        lines += ["## Генерация", ""]
        for g in payload["generations"]:
            lines += [
                f"* слой {g.get('layer')}, alpha {g.get('alpha'):+.2f}, режим {g.get('mode')}",
                "",
                "```",
                (g.get("text") or "")[:1500],
                "```",
                "",
            ]
    lines += ["## Оговорка", "", payload.get("disclaimer", "")]
    return "\n".join(lines)
