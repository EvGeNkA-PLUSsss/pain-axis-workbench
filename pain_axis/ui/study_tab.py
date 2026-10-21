# SPDX-FileCopyrightText: 2026 EvGeNkA-PLUSsss
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Вкладка «Исследования»: визуальные блоки, пакетные прогоны, таблица и графики.

Устройство вкладки
------------------
Слева — конвейер из четырёх блоков, которые настраиваются на месте: «Запросы»,
«Условия», «Метрики», «Вывод». Порядок блоков сверху вниз и есть порядок обработки:
запросы размножаются условиями, каждый ответ измеряется метриками, а вывод говорит,
что положить на график и как агрегировать.

Справа — таблица всех ответов (двойной клик открывает полный текст), график на
Canvas и строка состояния. Есть готовые сценарии, сохранение и загрузка
исследований в JSON, экспорт результатов в CSV и отчёт в Markdown.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .. import presets as PR
from .. import study as S
from ..directions import axis_definitions
from .charts import draw_chart

NO_STEER = "без стиринга"
METRIC_LABELS = {"refusal": "доля отказов", "chars": "символов", "words": "слов",
                 "seconds": "секунд", "keyword": "ключевые слова", "projection": "проекция"}
AGG_LABELS = {"mean": "среднее", "median": "медиана", "count": "количество"}
GROUP_LABELS = {"alpha": "по alpha", "axis": "по условию", "prompt": "по запросу"}
KIND_LABELS = {"line": "линия", "bar": "столбцы", "scatter": "точки"}
MODE_LABELS = {"list": "список построчно", "file": "файл со списком",
               "template": "шаблон × значения", "axis_sets": "готовые наборы осей"}


class StudyTab(ttk.Frame):
    def __init__(self, master: tk.Widget, app) -> None:
        super().__init__(master)
        self.app = app
        self.study = list(S.examples().values())[0]
        self.rows: list[dict] = []
        self.points: list[dict] = []
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._running = False

        self._build_toolbar()
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        left = ttk.Frame(body, style="Card.TFrame", padding=10)
        left.pack(side="left", fill="both", padx=(0, 10))
        left.configure(width=430)
        left.pack_propagate(False)
        self._build_blocks(left)
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        self._build_results(right)
        self._load_into_ui(self.study)
        self.after(120, self._drain)

    # ------------------------------------------------------------- интерфейс
    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=(10, 8, 10, 6))
        bar.pack(fill="x")
        ttk.Label(bar, text="Исследование:", style="H2.TLabel").pack(side="left")
        self.name_var = tk.StringVar()
        ttk.Entry(bar, textvariable=self.name_var, width=28).pack(side="left", padx=(6, 10))
        self.size_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.size_var, style="Muted.TLabel").pack(side="left")

        ttk.Button(bar, text="Экспорт MD", style="Ghost.TButton",
                   command=self.export_md).pack(side="right")
        ttk.Button(bar, text="Экспорт CSV", style="Ghost.TButton",
                   command=self.export_csv).pack(side="right", padx=6)
        self.stop_btn = ttk.Button(bar, text="Стоп", style="Ghost.TButton",
                                   command=self.stop_run, state="disabled")
        self.stop_btn.pack(side="right", padx=6)
        self.run_btn = ttk.Button(bar, text="Выполнить", style="Accent.TButton",
                                  command=self.run_study)
        self.run_btn.pack(side="right")
        ttk.Button(bar, text="Загрузить…", style="Ghost.TButton",
                   command=self.load_study).pack(side="right", padx=6)
        ttk.Button(bar, text="Сохранить", style="Ghost.TButton",
                   command=self.save_study).pack(side="right")
        self.examples_btn = ttk.Menubutton(bar, text="Готовые сценарии ▾",
                                           style="Ghost.TButton")
        menu = tk.Menu(self.examples_btn, tearoff=0)
        for name in S.examples():
            menu.add_command(label=name, command=lambda n=name: self.use_example(n))
        self.examples_btn.configure(menu=menu)
        self.examples_btn.pack(side="right", padx=6)

    def _card(self, parent: ttk.Frame, title: str, hint: str) -> ttk.Frame:
        box = ttk.Labelframe(parent, text=title, style="Card.TLabelframe", padding=8)
        box.pack(fill="x", pady=(0, 8))
        if hint:
            ttk.Label(box, text=hint, style="MutedCard.TLabel", wraplength=380,
                      justify="left").pack(anchor="w", pady=(0, 4))
        return box

    def _build_blocks(self, parent: ttk.Frame) -> None:
        # 1. запросы
        b1 = self._card(parent, "1. Запросы", "Откуда берутся тексты и сколько их.")
        row = ttk.Frame(b1, style="Card.TFrame")
        row.pack(fill="x")
        self.mode_var = tk.StringVar(value="list")
        combo = ttk.Combobox(row, textvariable=self.mode_var, state="readonly", width=26,
                             values=[MODE_LABELS[k] for k in MODE_LABELS])
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda _e: self._sync_mode())
        self.limit_var = tk.StringVar(value="0")
        ttk.Label(row, text="лимит:", style="MutedCard.TLabel").pack(side="left", padx=(8, 2))
        ttk.Spinbox(row, from_=0, to=5000, increment=10, textvariable=self.limit_var,
                    width=6, command=self.refresh_size).pack(side="left")
        self.shuffle_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="перемешать", variable=self.shuffle_var,
                        command=self.refresh_size).pack(side="left", padx=(8, 0))

        self.prompt_text = tk.Text(b1, height=5, wrap="word", font=("Segoe UI", 9),
                                   relief="flat")
        self.prompt_text.pack(fill="x", pady=(6, 0))
        self.prompt_text.bind("<KeyRelease>", lambda _e: self.refresh_size())

        self.file_row = ttk.Frame(b1, style="Card.TFrame")
        self.file_var = tk.StringVar()
        ttk.Entry(self.file_row, textvariable=self.file_var).pack(side="left", fill="x",
                                                                 expand=True)
        ttk.Button(self.file_row, text="…", width=3, style="Ghost.TButton",
                   command=self._pick_file).pack(side="left", padx=(4, 0))

        self.tpl_row = ttk.Frame(b1, style="Card.TFrame")
        self.tpl_var = tk.StringVar(value="Опиши {x} тремя словами.")
        self.tpl_values = tk.Text(self.tpl_row, height=4, wrap="word", font=("Segoe UI", 9),
                                  relief="flat")
        ttk.Label(self.tpl_row, text="шаблон с {x}:", style="MutedCard.TLabel").pack(anchor="w")
        ttk.Entry(self.tpl_row, textvariable=self.tpl_var).pack(fill="x", pady=(2, 4))
        ttk.Label(self.tpl_row, text="значения (по строке):", style="MutedCard.TLabel").pack(
            anchor="w")
        self.tpl_values.pack(fill="x")
        self.tpl_values.bind("<KeyRelease>", lambda _e: self.refresh_size())

        self.sets_row = ttk.Frame(b1, style="Card.TFrame")
        self.sets_var = tk.StringVar(value="отказ")
        ttk.Label(self.sets_row, text="наборы осей (через запятую):",
                  style="MutedCard.TLabel").pack(anchor="w")
        ttk.Entry(self.sets_row, textvariable=self.sets_var).pack(fill="x", pady=(2, 4))
        self.neg_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.sets_row, text="брать и отрицательные тексты наборов",
                        variable=self.neg_var, command=self.refresh_size).pack(anchor="w")

        # 2. условия
        b2 = self._card(parent, "2. Условия", "Что варьируется между запросами.")
        r2 = ttk.Frame(b2, style="Card.TFrame")
        r2.pack(fill="x")
        ttk.Label(r2, text="ось:", style="MutedCard.TLabel").grid(row=0, column=0, sticky="e")
        self.axis_var = tk.StringVar(value="отказ")
        self.axis_combo = ttk.Combobox(r2, textvariable=self.axis_var, width=28,
                                       values=[NO_STEER] + list(axis_definitions())
                                       + [f"пресет:{p.name}" for p in PR.all_presets()])
        self.axis_combo.grid(row=0, column=1, columnspan=2, sticky="we", padx=4, pady=2)
        ttk.Label(r2, text="alpha:", style="MutedCard.TLabel").grid(row=1, column=0, sticky="e")
        self.alphas_var = tk.StringVar(value="-0.3,-0.15,0,0.15")
        ttk.Entry(r2, textvariable=self.alphas_var, width=24).grid(row=1, column=1, sticky="we",
                                                                  padx=4, pady=2)
        self.level_var = tk.StringVar(value="средне")
        ttk.Combobox(r2, textvariable=self.level_var, width=8, state="readonly",
                     values=list(PR.LEVEL_ORDER)).grid(row=1, column=2, sticky="w", pady=2)
        ttk.Label(r2, text="повторов:", style="MutedCard.TLabel").grid(row=2, column=0, sticky="e")
        self.repeats_var = tk.StringVar(value="1")
        ttk.Spinbox(r2, from_=1, to=50, textvariable=self.repeats_var, width=6,
                    command=self.refresh_size).grid(row=2, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(r2, text="temperature:", style="MutedCard.TLabel").grid(row=3, column=0,
                                                                         sticky="e")
        self.temp_var = tk.StringVar(value="0.0")
        ttk.Entry(r2, textvariable=self.temp_var, width=8).grid(row=3, column=1, sticky="w",
                                                               padx=4, pady=2)
        ttk.Label(r2, text="токенов:", style="MutedCard.TLabel").grid(row=4, column=0, sticky="e")
        self.tokens_var = tk.StringVar(value="48")
        ttk.Spinbox(r2, from_=8, to=1024, increment=8, textvariable=self.tokens_var,
                    width=6).grid(row=4, column=1, sticky="w", padx=4, pady=2)
        r2.columnconfigure(1, weight=1)

        # 3. метрики
        b3 = self._card(parent, "3. Метрики", "Что измеряется в каждом ответе.")
        self.m_len = tk.BooleanVar(value=True)
        self.m_ref = tk.BooleanVar(value=True)
        self.m_proj = tk.BooleanVar(value=True)
        ttk.Checkbutton(b3, text="длина (символы и слова)", variable=self.m_len).pack(anchor="w")
        ttk.Checkbutton(b3, text="маркер отказа", variable=self.m_ref).pack(anchor="w")
        ttk.Checkbutton(b3, text="проекции на оси (один доп. проход на ответ)",
                        variable=self.m_proj).pack(anchor="w")
        ttk.Label(b3, text="оси для проекций:", style="MutedCard.TLabel").pack(anchor="w",
                                                                              pady=(4, 0))
        self.proj_var = tk.StringVar(value="отказ, боль себе")
        ttk.Entry(b3, textvariable=self.proj_var).pack(fill="x")
        ttk.Label(b3, text="ключевые слова (через запятую):",
                  style="MutedCard.TLabel").pack(anchor="w", pady=(4, 0))
        self.keywords_var = tk.StringVar()
        ttk.Entry(b3, textvariable=self.keywords_var).pack(fill="x")
        ttk.Label(b3, text="регулярное выражение:", style="MutedCard.TLabel").pack(anchor="w",
                                                                                  pady=(4, 0))
        self.pattern_var = tk.StringVar()
        ttk.Entry(b3, textvariable=self.pattern_var).pack(fill="x")

        # 4. вывод
        b4 = self._card(parent, "4. Вывод", "Что попадает на график и в отчёт.")
        r4 = ttk.Frame(b4, style="Card.TFrame")
        r4.pack(fill="x")
        ttk.Label(r4, text="метрика:", style="MutedCard.TLabel").grid(row=0, column=0, sticky="e")
        self.metric_var = tk.StringVar(value="доля отказов")
        ttk.Combobox(r4, textvariable=self.metric_var, state="readonly", width=18,
                     values=list(METRIC_LABELS.values())).grid(row=0, column=1, sticky="we",
                                                              padx=4, pady=2)
        ttk.Label(r4, text="агрегация:", style="MutedCard.TLabel").grid(row=1, column=0,
                                                                       sticky="e")
        self.agg_var = tk.StringVar(value="среднее")
        ttk.Combobox(r4, textvariable=self.agg_var, state="readonly", width=12,
                     values=list(AGG_LABELS.values())).grid(row=1, column=1, sticky="w",
                                                            padx=4, pady=2)
        ttk.Label(r4, text="график:", style="MutedCard.TLabel").grid(row=2, column=0, sticky="e")
        self.kind_var = tk.StringVar(value="линия")
        ttk.Combobox(r4, textvariable=self.kind_var, state="readonly", width=12,
                     values=list(KIND_LABELS.values())).grid(row=2, column=1, sticky="w",
                                                             padx=4, pady=2)
        ttk.Label(r4, text="группировка:", style="MutedCard.TLabel").grid(row=3, column=0,
                                                                         sticky="e")
        self.group_var = tk.StringVar(value="по alpha")
        ttk.Combobox(r4, textvariable=self.group_var, state="readonly", width=14,
                     values=list(GROUP_LABELS.values())).grid(row=3, column=1, sticky="w",
                                                              padx=4, pady=2)
        r4.columnconfigure(1, weight=1)
        ttk.Button(b4, text="Построить график заново", style="Ghost.TButton",
                   command=self.rebuild).pack(anchor="w", pady=(6, 0))

    def _build_results(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        card.pack(fill="both", expand=True)
        head = ttk.Frame(card, style="Card.TFrame")
        head.pack(fill="x")
        self.status_var = tk.StringVar(value="Готово к запуску")
        ttk.Label(head, textvariable=self.status_var, style="Card.TLabel",
                  font=("Segoe UI", 9, "bold")).pack(side="left")
        self.progress = ttk.Progressbar(head, mode="determinate", maximum=100, length=200)
        self.progress.pack(side="right")

        cols = ("cond", "alpha", "ref", "words", "sec", "proj", "response")
        self.tree = ttk.Treeview(card, columns=cols, show="headings", height=11)
        for key, label, width, anchor in (("cond", "условие", 150, "w"),
                                          ("alpha", "alpha", 60, "center"),
                                          ("ref", "отказ", 55, "center"),
                                          ("words", "слов", 55, "e"),
                                          ("sec", "сек", 55, "e"),
                                          ("proj", "проекция", 90, "e"),
                                          ("response", "ответ", 420, "w")):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, anchor=anchor)
        self.tree.pack(fill="both", expand=True, pady=(6, 6))
        self.tree.bind("<Double-1>", self._show_response)

        self.chart = tk.Canvas(card, height=230, highlightthickness=0, width=1)
        self.chart.pack(fill="x")
        self.chart.bind("<Configure>", lambda _e: self.draw())

        self.detail = tk.Text(card, height=6, wrap="word", font=("Segoe UI", 9), relief="flat")
        self.detail.pack(fill="x", pady=(6, 0))
        self.detail.insert("1.0", "Двойной клик по строке таблицы покажет полный ответ модели.")
        self.detail.configure(state="disabled")

    # ------------------------------------------------------------- сценарии
    def use_example(self, name: str) -> None:
        self._load_into_ui(S.examples()[name])
        self.status_var.set(f"Загружен сценарий «{name}»")

    def load_study(self) -> None:
        path = filedialog.askopenfilename(initialdir="studies",
                                          filetypes=[("Исследование", "*.json")])
        if not path:
            return
        try:
            self._load_into_ui(S.Study.load(path))
            self.status_var.set(f"Загружено: {Path(path).name}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Исследование", f"Не удалось прочитать файл: {exc}")

    def save_study(self) -> None:
        self._collect()
        try:
            path = self.study.save()
            self.status_var.set(f"Сохранено: {path}")
            self.app._system(f"Сценарий исследования сохранён: {path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Исследование", f"Не удалось сохранить: {exc}")

    def _pick_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Текстовый файл", "*.txt"),
                                                     ("Все файлы", "*.*")])
        if path:
            self.file_var.set(path)
            self.refresh_size()

    def _sync_mode(self) -> None:
        mode = {v: k for k, v in MODE_LABELS.items()}[self.mode_var.get()]
        for frame, show in ((self.file_row, mode == "file"),
                            (self.tpl_row, mode == "template"),
                            (self.sets_row, mode == "axis_sets"),
                            (self.prompt_text, mode == "list")):
            if show:
                frame.pack(fill="x", pady=(6, 0))
            else:
                frame.pack_forget()
        self.refresh_size()

    # --------------------------------------------------- данные ↔ интерфейс
    def _load_into_ui(self, study: S.Study) -> None:
        self.study = study
        self.name_var.set(study.name)
        p, c, m, o = study.prompts, study.conditions, study.metrics, study.output
        self.mode_var.set(MODE_LABELS.get(p.mode, MODE_LABELS["list"]))
        self.prompt_text.delete("1.0", "end")
        self.prompt_text.insert("1.0", p.text)
        self.file_var.set(p.path)
        self.tpl_var.set(p.template)
        self.tpl_values.delete("1.0", "end")
        self.tpl_values.insert("1.0", p.values)
        self.sets_var.set(", ".join(p.sets))
        self.neg_var.set(bool(p.include_negative))
        self.limit_var.set(str(p.limit))
        self.shuffle_var.set(bool(p.shuffle))
        self.axis_var.set(c.axis if c.axis else NO_STEER)
        self.alphas_var.set(c.alphas)
        self.repeats_var.set(str(c.repeats))
        self.temp_var.set(str(c.temperature))
        self.tokens_var.set(str(c.max_new_tokens))
        self.level_var.set(c.level)
        self.m_len.set(m.length)
        self.m_ref.set(m.refusal)
        self.m_proj.set(m.projection)
        self.proj_var.set(", ".join(m.proj_axes))
        self.keywords_var.set(m.keywords)
        self.pattern_var.set(m.pattern)
        self.metric_var.set(METRIC_LABELS.get(o.metric, METRIC_LABELS["refusal"]))
        self.agg_var.set(AGG_LABELS.get(o.agg, AGG_LABELS["mean"]))
        self.kind_var.set(KIND_LABELS.get(o.chart, KIND_LABELS["line"]))
        self.group_var.set(GROUP_LABELS.get(o.group_by, GROUP_LABELS["alpha"]))
        self._sync_mode()

    def _collect(self) -> S.Study:
        mode = {v: k for k, v in MODE_LABELS.items()}[self.mode_var.get()]
        metric = {v: k for k, v in METRIC_LABELS.items()}[self.metric_var.get()]
        agg = {v: k for k, v in AGG_LABELS.items()}[self.agg_var.get()]
        kind = {v: k for k, v in KIND_LABELS.items()}[self.kind_var.get()]
        group = {v: k for k, v in GROUP_LABELS.items()}[self.group_var.get()]
        axis = self.axis_var.get()
        if axis == NO_STEER:
            axis = ""
        self.study = S.Study(
            name=self.name_var.get().strip() or "исследование",
            notes=self.study.notes,
            prompts=S.PromptBlock(
                mode=mode, text=self.prompt_text.get("1.0", "end").strip(),
                path=self.file_var.get().strip(), template=self.tpl_var.get(),
                values=self.tpl_values.get("1.0", "end").strip(),
                sets=[s.strip() for s in self.sets_var.get().split(",") if s.strip()],
                include_negative=bool(self.neg_var.get()),
                limit=int(self.limit_var.get() or 0), shuffle=bool(self.shuffle_var.get())),
            conditions=S.ConditionBlock(
                axis=axis, alphas=self.alphas_var.get(), repeats=int(self.repeats_var.get() or 1),
                temperature=float(self.temp_var.get() or 0.0),
                max_new_tokens=int(self.tokens_var.get() or 48), level=self.level_var.get()),
            metrics=S.MetricsBlock(
                length=bool(self.m_len.get()), refusal=bool(self.m_ref.get()),
                projection=bool(self.m_proj.get()),
                proj_axes=[s.strip() for s in self.proj_var.get().split(",") if s.strip()],
                keywords=self.keywords_var.get(), pattern=self.pattern_var.get()),
            output=S.OutputBlock(metric=metric, agg=agg, chart=kind, group_by=group))
        return self.study

    def refresh_size(self) -> None:
        try:
            self._collect()
            prompts = len(self.study.expand_prompts())
            conds = len(self.study.expand_conditions())
            self.size_var.set(f"{prompts} запросов × {conds} условий = "
                              f"{prompts * conds} генераций")
        except Exception:  # noqa: BLE001
            self.size_var.set("проверьте настройки блоков")

    # ------------------------------------------------------------------ прогон
    def run_study(self) -> None:
        if self._running:
            return
        backend = getattr(self.app, "backend", None)
        if backend is None or not getattr(backend, "loaded", False):
            messagebox.showinfo("Исследование", "Сначала загрузите модель на вкладке «Чат и оси».")
            return
        if getattr(self.app, "_busy", False):
            messagebox.showinfo("Исследование", "Дождитесь завершения операции в чате.")
            return
        self._collect()
        total = self.study.size()
        if total == 0:
            messagebox.showinfo("Исследование", "Нет запросов: проверьте блок «Запросы».")
            return
        if total > 40 and not messagebox.askyesno(
                "Исследование", f"Будет выполнено {total} генераций. Это может занять "
                                f"много времени. Продолжить?"):
            return
        axeset = getattr(self.app, "axeset", None)
        if self.study.metrics.projection and axeset is None:
            self.app._system("В исследовании запрошены проекции, но оси не извлечены "
                             "(модель Ollama или демо-стенд): метрика пропущена.")
        self._stop.clear()
        self._running = True
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.rows = []
        self.tree.delete(*self.tree.get_children())
        limit = 3.0 if getattr(backend, "is_demo", False) else 0.5

        def worker() -> None:
            try:
                rows = S.run(self.study, backend, axeset,
                             progress=lambda m, f: self._q.put(("progress", m, f)),
                             stop=self._stop.is_set, per_axis_limit=limit)
                self._q.put(("done", rows, None))
            except Exception as exc:  # noqa: BLE001
                self._q.put(("error", None, exc))

        threading.Thread(target=worker, daemon=True).start()

    def stop_run(self) -> None:
        self._stop.set()
        self.status_var.set("Останавливаю после текущего запроса…")

    def _drain(self) -> None:
        try:
            while True:
                kind, payload, extra = self._q.get_nowait()
                if kind == "progress":
                    self.status_var.set(str(payload))
                    self.progress.configure(value=max(0.0, min(100.0, float(extra) * 100)))
                elif kind == "done":
                    self._finish(payload or [])
                else:
                    self._running = False
                    self.run_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.status_var.set("Ошибка")
                    messagebox.showerror("Исследование", str(extra))
        except queue.Empty:
            pass
        self.after(120, self._drain)

    def _finish(self, rows: list[dict]) -> None:
        self._running = False
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.rows = rows
        self._fill_table(rows)
        self.rebuild()
        self.status_var.set(f"Готово: {len(rows)} ответов")
        self.progress.configure(value=100)
        if rows:
            self.app._system(f"Исследование «{self.study.name}»: выполнено {len(rows)} запросов, "
                             f"график построен по метрике "
                             f"{METRIC_LABELS.get(self.study.output.metric, '')}.")

    def _fill_table(self, rows: list[dict]) -> None:
        self.tree.delete(*self.tree.get_children())
        proj_col = S.metric_column(self.study, rows)
        for i, row in enumerate(rows):
            proj = row.get(proj_col, "")
            self.tree.insert("", "end", iid=str(i),
                             values=(row.get("condition", "")[:26], f"{row['alpha']:+.2f}",
                                     "да" if row.get("refusal") else "нет", row.get("words", ""),
                                     row.get("seconds", ""),
                                     f"{proj:.2f}" if isinstance(proj, (int, float)) else "",
                                     " ".join(str(row.get("response", "")).split())[:160]))

    def _show_response(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        row = self.rows[int(sel[0])]
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", f"Запрос: {row['prompt']}\nУсловие: {row['condition']} · "
                                 f"alpha {row['alpha']:+.2f} · "
                                 f"{'обрезано лимитом' if row.get('clamped') else 'в пределах лимита'}\n\n"
                                 f"{row['response']}")
        self.detail.configure(state="disabled")

    # ----------------------------------------------------------------- график
    def rebuild(self) -> None:
        self._collect()
        self.points = S.aggregate(self.study, self.rows)
        self.draw()

    def draw(self) -> None:
        col = S.metric_column(self.study, self.rows) if self.rows else self.study.output.metric
        title = (f"{METRIC_LABELS.get(self.study.output.metric, col)} · "
                 f"{AGG_LABELS.get(self.study.output.agg, '')} · "
                 f"{len(self.rows)} ответов")
        draw_chart(self.chart, self.points, kind=self.study.output.chart, title=title,
                   y_label=METRIC_LABELS.get(self.study.output.metric, col),
                   x_label=GROUP_LABELS.get(self.study.output.group_by, ""),
                   theme=getattr(self.app, "theme", None))

    def on_theme(self) -> None:
        self.draw()

    # ----------------------------------------------------------------- экспорт
    def export_csv(self) -> None:
        if not self.rows:
            messagebox.showinfo("Исследование", "Сначала выполните исследование.")
            return
        path = S.STUDY_DIR / (S.safe_name(self.study.name) + ".csv")
        S.to_csv(self.rows, path)
        self.status_var.set(f"CSV: {path}")
        self.app._system(f"Результаты сохранены: {path}")

    def export_md(self) -> None:
        if not self.rows:
            messagebox.showinfo("Исследование", "Сначала выполните исследование.")
            return
        path = S.STUDY_DIR / (S.safe_name(self.study.name) + ".md")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(S.to_markdown(self.study, self.rows, self.points), encoding="utf-8")
        self.status_var.set(f"Отчёт: {path}")
        self.app._system(f"Отчёт сохранён: {path}")
