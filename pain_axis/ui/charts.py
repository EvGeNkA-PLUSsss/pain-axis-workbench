"""Графики на чистом tkinter: линия, столбцы, точки.

Matplotlib в сборку exe не входит (это десятки мегабайт ради одной картинки),
поэтому графики рисуются прямо на Canvas: сетка, оси, подписи, значения над
точками. Этого достаточно для результатов исследований — «метрика против
условия», а не публикационные рисунки.
"""

from __future__ import annotations

import tkinter as tk


def _nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(1, count)
    mag = 10 ** int(f"{raw:e}".split("e")[1]) if raw > 0 else 1.0
    for mult in (1, 2, 2.5, 5, 10):
        step = mag * mult
        if step >= raw:
            break
    start = step * int(lo / step) if step else lo
    ticks = []
    v = start
    while v <= hi + step * 0.5 and len(ticks) < 12:
        if v >= lo - step * 0.5:
            ticks.append(v)
        v += step
    return ticks or [lo, hi]


def draw_chart(canvas: tk.Canvas, points: list[dict], *, kind: str = "line",
               title: str = "", y_label: str = "", x_label: str = "",
               theme=None) -> None:
    """Нарисовать график по точкам [{'label': str, 'value': float, 'n': int}, …]."""
    bg = getattr(theme, "card", "#ffffff")
    fg = getattr(theme, "text", "#0f172a")
    muted = getattr(theme, "muted", "#64748b")
    accent = getattr(theme, "accent", "#4f46e5")
    border = getattr(theme, "border", "#e2e8f0")

    canvas.delete("all")
    canvas.configure(bg=bg)
    w = max(canvas.winfo_width(), 320)
    h = max(canvas.winfo_height(), 200)
    left, right, top, bottom = 62, 18, 34, 58
    plot_w = max(10, w - left - right)
    plot_h = max(10, h - top - bottom)

    if title:
        canvas.create_text(left, 14, text=title, fill=fg, anchor="w",
                           font=("Segoe UI", 10, "bold"))
    if not points:
        canvas.create_text(w / 2, h / 2, text="нет данных: выполните исследование",
                           fill=muted, font=("Segoe UI", 9))
        return

    values = [float(p["value"]) for p in points]
    lo, hi = min(values), max(values)
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    pad = (hi - lo) * 0.12
    lo, hi = lo - pad, hi + pad
    ticks = _nice_ticks(lo, hi)

    def y_of(v: float) -> float:
        return top + plot_h - (v - lo) / (hi - lo) * plot_h

    def x_of(i: int) -> float:
        if len(points) == 1:
            return left + plot_w / 2
        return left + plot_w * i / (len(points) - 1)

    # сетка и подписи оси Y
    for t in ticks:
        y = y_of(t)
        canvas.create_line(left, y, left + plot_w, y, fill=border)
        canvas.create_text(left - 8, y, text=f"{t:g}", fill=muted, anchor="e",
                           font=("Segoe UI", 8))
    canvas.create_line(left, top, left, top + plot_h, fill=muted)
    canvas.create_line(left, top + plot_h, left + plot_w, top + plot_h, fill=muted)
    if y_label:
        canvas.create_text(14, top + plot_h / 2, text=y_label, fill=muted, angle=90,
                           font=("Segoe UI", 8))
    if x_label:
        canvas.create_text(left + plot_w / 2, h - 8, text=x_label, fill=muted,
                           font=("Segoe UI", 8))

    # столбики: своя геометрия
    if kind == "bar":
        step = plot_w / max(1, len(points))
        width = max(6.0, step * 0.6)
        for i, p in enumerate(points):
            x = left + step * (i + 0.5)
            y = y_of(float(p["value"]))
            base = y_of(max(lo, 0.0))
            canvas.create_rectangle(x - width / 2, min(y, base), x + width / 2, max(y, base),
                                    fill=accent, outline="")
            canvas.create_text(x, y - 8, text=f"{p['value']:.2f}", fill=fg,
                               font=("Segoe UI", 8))
        for i, p in enumerate(points):
            x = left + step * (i + 0.5)
            label = str(p["label"])
            canvas.create_text(x, top + plot_h + 14, text=label[:12], fill=muted,
                               font=("Segoe UI", 8))
        return

    coords = [(x_of(i), y_of(float(p["value"]))) for i, p in enumerate(points)]
    if kind == "line" and len(coords) > 1:
        flat = [c for xy in coords for c in xy]
        canvas.create_line(*flat, fill=accent, width=2, smooth=False)
    for (x, y), p in zip(coords, points):
        r = 4
        canvas.create_oval(x - r, y - r, x + r, y + r, fill=accent, outline=bg)
        canvas.create_text(x, y - 12, text=f"{p['value']:.2f}", fill=fg,
                           font=("Segoe UI", 8))
    for (x, _y), p in zip(coords, points):
        canvas.create_text(x, top + plot_h + 14, text=str(p["label"])[:12], fill=muted,
                           font=("Segoe UI", 8))
