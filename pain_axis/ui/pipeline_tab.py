"""Вкладка «Исследования»: сборка сценария из блоков без кода.

Слева — цепочка блоков. Их можно добавлять, удалять, включать и выключать,
менять местами (▲ ▼) и дублировать (⧉). Порядок блоков и есть порядок обработки:
записи идут по цепочке сверху вниз. Логические блоки — фильтр, преобразование,
разделение на ветки и повтор — позволяют собирать сценарии без единой строки кода.

Справа — таблица ответов, график и полный текст выбранного ответа.
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .. import pipeline as PL
from .. import presets as PR
from .. import study as S
from ..directions import axis_definitions
from .charts import draw_chart

METRIC_LABELS = {"trace": "проекция по ходу генерации", "процент": "процент из ответа",
                 "число": "число из ответа", "refusal": "доля отказов", "chars": "символов",
                 "words": "слов", "seconds": "секунд", "keyword": "ключевые слова",
                 "projection": "проекция", "regex": "совпадений regex", "branch": "ветка"}
AGG_LABELS = {"mean": "среднее", "median": "медиана", "count": "количество"}
GROUP_LABELS = {"alpha": "по alpha", "axis": "по условию", "prompt": "по запросу",
                "branch": "по ветке", "condition": "по условию"}
KIND_LABELS = {"line": "линия", "bar": "столбцы", "scatter": "точки"}
MODE_LABELS = {"list": "список построчно", "file": "файл со списком",
               "template": "шаблон × значения", "axis_sets": "готовые наборы осей"}
_LABEL_MAPS = {"mode": MODE_LABELS, "metric": METRIC_LABELS, "agg": AGG_LABELS,
               "chart": KIND_LABELS, "group_by": GROUP_LABELS, "op": PL.OPERATORS,
               "kind": PL.EXPORTS}
FIELD_SUGGESTIONS = ("response", "prompt", "branch", "condition", "refusal", "words", "chars",
                     "seconds", "keyword", "regex")


class PipelineTab(ttk.Frame):
    def __init__(self, master: tk.Widget, app) -> None:
        super().__init__(master)
        self.app = app
        self.pipeline = PL.Pipeline.default()
        self.rows: list[dict] = []
        self.points: list[dict] = []
        self._block_vars: list[dict] = []
        self._block_boxes: list[ttk.Frame] = []
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._running = False

        self._build_toolbar()
        body = ttk.Frame(self, padding=(10, 0, 10, 8))
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, style="Card.TFrame", padding=8)
        left.pack(side="left", fill="both", padx=(0, 10))
        left.configure(width=470)
        left.pack_propagate(False)
        self._build_block_column(left)
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        self._build_results(right)
        self.rebuild_blocks()
        self.after(120, self._drain)

    # --------------------------------------------------------------- каркас
    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=(10, 8, 10, 6))
        bar.pack(fill="x")
        ttk.Label(bar, text="Сценарий:", style="H2.TLabel").pack(side="left")
        self.name_var = tk.StringVar(value=self.pipeline.name)
        ttk.Entry(bar, textvariable=self.name_var, width=26).pack(side="left", padx=(6, 10))
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
                                  command=self.run_pipeline)
        self.run_btn.pack(side="right")
        ttk.Button(bar, text="Загрузить…", style="Ghost.TButton",
                   command=self.load_pipeline).pack(side="right", padx=6)
        ttk.Button(bar, text="Сохранить", style="Ghost.TButton",
                   command=self.save_pipeline).pack(side="right")
        self.examples_btn = ttk.Menubutton(bar, text="Готовые сценарии ▾",
                                           style="Ghost.TButton")
        menu = tk.Menu(self.examples_btn, tearoff=0)
        for name in PL.examples():
            menu.add_command(label=name, command=lambda n=name: self.use_example(n))
        self.examples_btn.configure(menu=menu)
        self.examples_btn.pack(side="right", padx=6)

    def _build_block_column(self, parent: ttk.Frame) -> None:
        head = ttk.Frame(parent, style="Card.TFrame")
        head.pack(fill="x")
        ttk.Label(head, text="Блоки сценария", style="H2.TLabel").pack(side="left")
        add_btn = ttk.Menubutton(head, text="＋ блок ▾", style="Ghost.TButton")
        menu = tk.Menu(add_btn, tearoff=0)
        for btype in PL.BLOCK_ORDER:
            menu.add_command(label=PL.BLOCK_TITLES[btype],
                             command=lambda t=btype: self.add_block(t))
        add_btn.configure(menu=menu)
        add_btn.pack(side="right")

        wrap = ttk.Frame(parent, style="Card.TFrame")
        wrap.pack(fill="both", expand=True, pady=(6, 0))
        self.canvas = tk.Canvas(wrap, highlightthickness=0, width=1)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.blocks_frame = ttk.Frame(self.canvas, style="Card.TFrame")
        window = self.canvas.create_window((0, 0), window=self.blocks_frame, anchor="nw")
        self.blocks_frame.bind("<Configure>", lambda _e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(window, width=e.width))
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-e.delta / 120),
                                                                           "units"))
        self.warn_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.warn_var, style="WarnCard.TLabel", wraplength=430,
                  justify="left").pack(fill="x", pady=(6, 0))

    def _build_results(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        card.pack(fill="both", expand=True)
        head = ttk.Frame(card, style="Card.TFrame")
        head.pack(fill="x")
        self.status_var = tk.StringVar(value="Соберите цепочку блоков и нажмите «Выполнить»")
        ttk.Label(head, textvariable=self.status_var, style="Card.TLabel",
                  font=("Segoe UI", 9, "bold")).pack(side="left")
        self.progress = ttk.Progressbar(head, mode="determinate", maximum=100, length=200)
        self.progress.pack(side="right")

        cols = ("cond", "branch", "alpha", "ref", "words", "sec", "extra", "response")
        self.tree = ttk.Treeview(card, columns=cols, show="headings", height=11)
        for key, label, width, anchor in (("cond", "условие", 140, "w"),
                                          ("branch", "ветка", 80, "center"),
                                          ("alpha", "alpha", 55, "center"),
                                          ("ref", "отказ", 50, "center"),
                                          ("words", "слов", 50, "e"),
                                          ("sec", "сек", 50, "e"),
                                          ("extra", "метрика", 80, "e"),
                                          ("response", "ответ", 380, "w")):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, anchor=anchor)
        self.tree.pack(fill="both", expand=True, pady=(6, 6))
        self.tree.bind("<Double-1>", self._show_response)

        self.chart = tk.Canvas(card, height=220, highlightthickness=0, width=1)
        self.chart.pack(fill="x")
        self.chart.bind("<Configure>", lambda _e: self.draw())

        self.detail = tk.Text(card, height=6, wrap="word", font=("Segoe UI", 9), relief="flat")
        self.detail.pack(fill="x", pady=(6, 0))
        self.detail.insert("1.0", "Двойной клик по строке таблицы покажет полный ответ модели.")
        self.detail.configure(state="disabled")
        self.log = tk.Text(card, height=3, wrap="word", font=("Segoe UI", 8), relief="flat")
        self.log.pack(fill="x", pady=(4, 0))
        self.log.configure(state="disabled")

    # ------------------------------------------------------------- блоки UI
    def rebuild_blocks(self) -> None:
        """Перерисовать цепочку блоков по текущей модели конвейера."""
        for w in self.blocks_frame.winfo_children():
            w.destroy()
        self._block_vars = []
        self._block_boxes = []
        for i, block in enumerate(self.pipeline.blocks):
            self._build_block_card(i, block)
        self._update_warnings()
        self.canvas.yview_moveto(0.0)

    def _build_block_card(self, index: int, block: PL.Block) -> None:
        box = ttk.Labelframe(self.blocks_frame, text=f"{index + 1}. {PL.BLOCK_TITLES[block.type]}",
                             style="Card.TLabelframe", padding=8)
        box.pack(fill="x", pady=(0, 6))
        self._block_boxes.append(box)

        head = ttk.Frame(box, style="Card.TFrame")
        head.pack(fill="x")
        enabled = tk.BooleanVar(value=block.enabled)
        ttk.Checkbutton(head, text="включён", variable=enabled,
                        command=lambda i=index: self._toggle(i)).pack(side="left")
        ttk.Button(head, text="✕", width=2, style="Ghost.TButton",
                   command=lambda i=index: self.delete_block(i)).pack(side="right")
        ttk.Button(head, text="⧉", width=2, style="Ghost.TButton",
                   command=lambda i=index: self.duplicate_block(i)).pack(side="right", padx=2)
        ttk.Button(head, text="▼", width=2, style="Ghost.TButton",
                   command=lambda i=index: self.move_block(i, 1)).pack(side="right")
        ttk.Button(head, text="▲", width=2, style="Ghost.TButton",
                   command=lambda i=index: self.move_block(i, -1)).pack(side="right")
        hint = PL.BLOCK_HINTS.get(block.type, "")
        if hint:
            ttk.Label(box, text=hint, style="MutedCard.TLabel", wraplength=420,
                      justify="left").pack(anchor="w", pady=(2, 4))
        variables = self._build_form(box, block)
        variables["enabled"] = enabled
        self._block_vars.append(variables)

    def _build_form(self, box: ttk.Frame, block: PL.Block) -> dict:
        p = block.params
        v: dict[str, tk.Variable] = {}
        grid = ttk.Frame(box, style="Card.TFrame")
        grid.pack(fill="x")

        def lab(text: str, row: int) -> None:
            ttk.Label(grid, text=text, style="MutedCard.TLabel").grid(row=row, column=0,
                                                                     sticky="e", padx=(0, 4))

        def entry(key: str, row: int, col: int = 1, width: int = 26, default="") -> None:
            var = tk.StringVar(value=str(p.get(key, default)))
            v[key] = var
            ttk.Entry(grid, textvariable=var, width=width).grid(row=row, column=col,
                                                               columnspan=2, sticky="we",
                                                               pady=2)

        def combo(key: str, values: list[str], row: int, default="", state="readonly",
                  width: int = 24) -> None:
            raw = str(p.get(key, default))
            shown = raw if raw in values else default      # код → подпись, если нужно
            var = tk.StringVar(value=shown)
            v[key] = var
            cb = ttk.Combobox(grid, textvariable=var, values=values, width=width, state=state)
            cb.grid(row=row, column=1, columnspan=2, sticky="we", pady=2)
            return cb

        def check(key: str, text: str, row: int, default=False) -> None:
            var = tk.BooleanVar(value=bool(p.get(key, default)))
            v[key] = var
            ttk.Checkbutton(grid, text=text, variable=var).grid(row=row, column=1, columnspan=2,
                                                               sticky="w")

        t = block.type
        if t == "source":
            lab("режим:", 0)
            combo("mode", [MODE_LABELS[k] for k in MODE_LABELS], 0,
                  MODE_LABELS.get(p.get("mode", "list"), MODE_LABELS["list"]))
            lab("текст:", 1)
            text = tk.Text(grid, height=4, wrap="word", font=("Segoe UI", 9), relief="flat")
            text.grid(row=1, column=1, columnspan=2, sticky="we", pady=2)
            text.insert("1.0", str(p.get("text", "")))
            v["_text"] = text
            lab("файл:", 2)
            entry("path", 2)
            lab("шаблон:", 3)
            entry("template", 3)
            tpl_vals = tk.Text(grid, height=3, wrap="word", font=("Segoe UI", 9), relief="flat")
            lab("значения:", 4)
            tpl_vals.grid(row=4, column=1, columnspan=2, sticky="we", pady=2)
            tpl_vals.insert("1.0", str(p.get("values", "")))
            v["_values"] = tpl_vals
            lab("наборы:", 5)
            entry("sets", 5)
            check("include_negative", "и отрицательные тексты наборов", 6)
            lab("лимит:", 7)
            entry("limit", 7, width=8)
            check("shuffle", "перемешать", 8)
        elif t == "conditions":
            lab("ось:", 0)
            combo("axis", [""] + list(axis_definitions())
                  + [f"пресет:{x.name}" for x in PR.all_presets()], 0, width=30, state="normal")
            lab("alpha:", 1)
            entry("alpha", 1)
            lab("уровень:", 2)
            combo("level", list(PR.LEVEL_ORDER), 2, "средне", width=10)
            lab("повторов:", 3)
            entry("repeats", 3, width=8)
            lab("temperature:", 4)
            entry("temperature", 4, width=8)
            lab("токенов:", 5)
            entry("max_new_tokens", 5, width=8)
        elif t == "generate":
            ttk.Label(grid, text="Этот блок делает запросы к модели. Один проход на запись.",
                      style="MutedCard.TLabel", wraplength=420).grid(row=0, column=0,
                                                                    columnspan=3, sticky="w")
        elif t == "metrics":
            check("length", "длина (символы и слова)", 0, True)
            check("refusal", "маркер отказа", 1, True)
            check("projection", "проекции на оси (доп. проход на ответ)", 2, False)
            check("numbers", "число и процент, названные моделью в ответе", 6, True)
            lab("оси проекций:", 3)
            entry("proj_axes", 3)
            lab("ключевые слова:", 4)
            entry("keywords", 4)
            lab("regex:", 5)
            entry("pattern", 5)
        elif t in ("filter", "split"):
            lab("поле:", 0)
            combo("field", list(FIELD_SUGGESTIONS), 0, "refusal", state="normal", width=20)
            lab("условие:", 1)
            combo("op", [PL.OPERATORS[k] for k in PL.OPERATORS], 1, PL.OPERATORS["="], width=12)
            lab("значение:", 2)
            entry("value", 2, width=14)
            if t == "split":
                lab("ветка «да»:", 3)
                entry("yes", 3, width=18)
                lab("ветка «нет»:", 4)
                entry("no", 4, width=18)
        elif t == "transform":
            lab("операция:", 0)
            combo("op", [PL.TRANSFORMS[k] for k in PL.TRANSFORMS], 0, PL.TRANSFORMS["words"], width=22)
            lab("из поля:", 1)
            combo("field", list(FIELD_SUGGESTIONS), 1, "response", state="normal", width=20)
            lab("в поле:", 2)
            entry("target", 2, width=18)
            lab("параметр:", 3)
            entry("param", 3, width=18)
        elif t == "loop":
            lab("повторов:", 0)
            entry("times", 0, width=8)
            ttk.Label(grid, text="Повторяет всю цепочку блоков выше. При temperature > 0 даёт "
                                 "разные ответы на тот же запрос.", style="MutedCard.TLabel",
                      wraplength=420).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))
        elif t == "aggregate":
            lab("метрика:", 0)
            combo("metric", list(METRIC_LABELS.values()), 0, METRIC_LABELS["refusal"], width=18)
            lab("агрегация:", 1)
            combo("agg", list(AGG_LABELS.values()), 1, AGG_LABELS["mean"], width=14)
            lab("график:", 2)
            combo("chart", list(KIND_LABELS.values()), 2, KIND_LABELS["line"], width=14)
            lab("группировка:", 3)
            combo("group_by", list(GROUP_LABELS.values()), 3, GROUP_LABELS["alpha"], width=16)
        elif t == "export":
            lab("формат:", 0)
            combo("kind", [PL.EXPORTS[k] for k in PL.EXPORTS], 0, PL.EXPORTS["csv"], width=12)
            lab("файл:", 1)
            entry("path", 1, width=28)
            ttk.Label(grid, text="Пусто — имя по названию сценария в папке studies.",
                      style="MutedCard.TLabel", wraplength=420).grid(row=2, column=0,
                                                                    columnspan=3, sticky="w")
        grid.columnconfigure(1, weight=1)
        return v

    # ------------------------------------------------- управление блоками
    def add_block(self, btype: str) -> None:
        self._collect()
        self.pipeline.add(btype)
        self.rebuild_blocks()
        self.refresh_size()

    def delete_block(self, index: int) -> None:
        self._collect()
        self.pipeline.remove(index)
        self.rebuild_blocks()
        self.refresh_size()

    def move_block(self, index: int, delta: int) -> None:
        self._collect()
        self.pipeline.move(index, delta)
        self.rebuild_blocks()
        self.refresh_size()

    def duplicate_block(self, index: int) -> None:
        self._collect()
        if 0 <= index < len(self.pipeline.blocks):
            src = self.pipeline.blocks[index]
            self.pipeline.blocks.insert(index + 1, PL.Block(src.type, src.enabled,
                                                            dict(src.params)))
        self.rebuild_blocks()
        self.refresh_size()

    def _toggle(self, index: int) -> None:
        self._collect()
        if 0 <= index < len(self.pipeline.blocks):
            self.pipeline.blocks[index].enabled = bool(
                self._block_vars[index].get("enabled").get())
        self._update_warnings()
        self.refresh_size()

    # ------------------------------------------------------ данные ↔ формы
    def _collect(self) -> PL.Pipeline:
        self.pipeline.name = self.name_var.get().strip() or "сценарий"
        for i, block in enumerate(self.pipeline.blocks):
            if i >= len(self._block_vars):
                break
            v = self._block_vars[i]
            block.enabled = bool(v["enabled"].get()) if "enabled" in v else block.enabled
            for key, var in v.items():
                if key.startswith("_") or key == "enabled":
                    continue
                if isinstance(var, tk.Text):
                    continue
                value = var.get()
                if key in _LABEL_MAPS:      # «доля отказов» → refusal и т. п.
                    table = _LABEL_MAPS[key]
                    value = {label: code for code, label in table.items()}.get(value, value)
                block.params[key] = value
            if "_text" in v:
                block.params["text"] = v["_text"].get("1.0", "end").strip()
            if "_values" in v:
                block.params["values"] = v["_values"].get("1.0", "end").strip()
            if "limit" in block.params:
                try:
                    block.params["limit"] = int(float(block.params["limit"] or 0))
                except (TypeError, ValueError):
                    block.params["limit"] = 0
            for key in ("repeats", "times", "max_new_tokens"):
                if key in block.params:
                    try:
                        block.params[key] = int(float(block.params[key] or 1))
                    except (TypeError, ValueError):
                        block.params[key] = 1
            for key in ("temperature",):
                if key in block.params:
                    try:
                        block.params[key] = float(block.params[key] or 0.0)
                    except (TypeError, ValueError):
                        block.params[key] = 0.0
        return self.pipeline

    def _update_warnings(self) -> None:
        problems = self.pipeline.validate()
        self.warn_var.set("  ".join(problems))

    def refresh_size(self) -> None:
        try:
            self._collect()
            self.size_var.set(f"по текущей цепочке: ~{self.pipeline.size()} генераций")
        except Exception:  # noqa: BLE001
            self.size_var.set("")

    # --------------------------------------------------------------- запуск
    def run_pipeline(self) -> None:
        if self._running:
            return
        self._collect()
        backend = getattr(self.app, "backend", None)
        if backend is None or not getattr(backend, "loaded", False):
            messagebox.showinfo("Исследование", "Сначала загрузите модель на вкладке «Чат и оси».")
            return
        problems = self.pipeline.validate()
        if problems:
            messagebox.showwarning("Сценарий", "  ".join(problems))
            return
        expected = self.pipeline.size()
        if expected > 40 and not messagebox.askyesno(
                "Исследование", f"По цепочке будет примерно {expected} генераций. Продолжить?"):
            return
        axeset = getattr(self.app, "axeset", None)
        self._stop.clear()
        self._running = True
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.tree.delete(*self.tree.get_children())
        self.rows = []
        limit = 3.0 if getattr(backend, "is_demo", False) else 0.5

        def worker() -> None:
            try:
                result = PL.run(self.pipeline, backend, axeset,
                                progress=lambda m, f: self._q.put(("progress", m, f)),
                                stop=self._stop.is_set, per_axis_limit=limit)
                self._q.put(("done", result, None))
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
                    if payload:
                        self._log_line(str(payload))
                elif kind == "done":
                    self._finish(payload or {})
                else:
                    self._running = False
                    self.run_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.status_var.set("Ошибка")
                    messagebox.showerror("Исследование", str(extra))
        except queue.Empty:
            pass
        self.after(120, self._drain)

    def _log_line(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _finish(self, result: dict) -> None:
        self._running = False
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.rows = result.get("rows") or []
        self.points = result.get("points") or []
        self._fill_table(self.rows)
        self.draw()
        extra = ("; файлы: " + ", ".join(result.get("exports") or [])) if result.get("exports") else ""
        self.status_var.set(f"Готово: {len(self.rows)} записей{extra}")
        self.progress.configure(value=100)
        if self.rows:
            self.app._system(f"Сценарий «{self.pipeline.name}»: {len(self.rows)} записей, "
                             f"{len(self.points)} точки графика.")

    def _fill_table(self, rows: list[dict]) -> None:
        self.tree.delete(*self.tree.get_children())
        extra_col = ""
        agg = next((b.params for b in self.pipeline.blocks
                    if b.enabled and b.type == "aggregate"), {})
        metric = agg.get("metric", "")
        if metric == "projection" and rows:
            extra_col = next((k for k in rows[0] if k.startswith("проекция:")), "")
        elif metric == "trace":
            extra_col = "trace_mean"
        elif metric in ("процент", "число"):
            extra_col = metric
        for i, row in enumerate(rows):
            value = row.get(extra_col, "") if extra_col else ""
            self.tree.insert("", "end", iid=str(i),
                             values=(str(row.get("condition", ""))[:22],
                                     str(row.get("branch", "")),
                                     f"{float(row.get('alpha', 0.0)):+.2f}",
                                     "да" if row.get("refusal") else "нет",
                                     row.get("words", ""), row.get("seconds", ""),
                                     f"{value:.2f}" if isinstance(value, (int, float)) else "",
                                     " ".join(str(row.get("response", "")).split())[:140]))

    def _show_response(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        row = self.rows[int(sel[0])]
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        fields = " · ".join(f"{k}={v}" for k, v in row.items()
                            if k not in ("response", "prompt") and not isinstance(v, dict))
        self.detail.insert("1.0", f"Запрос: {row.get('prompt', '')}\n{fields}\n\n"
                                 f"{row.get('response', '')}")
        self.detail.configure(state="disabled")

    # -------------------------------------------------------------- график
    def draw(self) -> None:
        metric = "refusal"
        agg = next((b.params for b in self.pipeline.blocks
                    if b.enabled and b.type == "aggregate"), {})
        metric = agg.get("metric", "refusal")
        title = f"{METRIC_LABELS.get(metric, metric)} · {len(self.rows)} записей"
        draw_chart(self.chart, self.points, kind=agg.get("chart", "line"), title=title,
                   y_label=METRIC_LABELS.get(metric, metric),
                   x_label=GROUP_LABELS.get(agg.get("group_by", ""), ""),
                   theme=getattr(self.app, "theme", None))

    def on_theme(self) -> None:
        self.draw()

    # -------------------------------------------------------------- файлы
    def use_example(self, name: str) -> None:
        self.pipeline = PL.examples()[name]
        self.name_var.set(self.pipeline.name)
        self.rebuild_blocks()
        self.refresh_size()
        self.status_var.set(f"Загружен сценарий «{name}»")

    def save_pipeline(self) -> None:
        self._collect()
        try:
            path = self.pipeline.save()
            self.status_var.set(f"Сохранено: {path}")
            self.app._system(f"Сценарий сохранён: {path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Сценарий", f"Не удалось сохранить: {exc}")

    def load_pipeline(self) -> None:
        path = filedialog.askopenfilename(initialdir="studies",
                                          filetypes=[("Сценарий", "*.json")])
        if not path:
            return
        try:
            self.pipeline = PL.Pipeline.load(path)
            self.name_var.set(self.pipeline.name)
            self.rebuild_blocks()
            self.refresh_size()
            self.status_var.set(f"Загружено: {Path(path).name}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Сценарий", f"Не удалось прочитать: {exc}")

    def export_csv(self) -> None:
        if not self.rows:
            messagebox.showinfo("Исследование", "Сначала выполните сценарий.")
            return
        path = PL.PIPE_DIR / (S.safe_name(self.pipeline.name) + ".csv")
        S.to_csv(self.rows, path)
        self.status_var.set(f"CSV: {path}")

    def export_md(self) -> None:
        if not self.rows:
            messagebox.showinfo("Исследование", "Сначала выполните сценарий.")
            return
        path = PL.PIPE_DIR / (S.safe_name(self.pipeline.name) + ".md")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(S.to_markdown(S.Study(name=self.pipeline.name), self.rows, self.points),
                        encoding="utf-8")
        self.status_var.set(f"Отчёт: {path}")
