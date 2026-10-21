"""Интерфейс: список моделей слева, чат с ползунками осей справа.

Что в нём есть
--------------
* три источника моделей: **Hugging Face** (25 из реестра + любая модель по id или
  путь к локальной папке), **Ollama** (каталог вашего сервера: список, докачка
  новых моделей в тот же каталог, чат) и **демо-стенд** без весов;
* чат с «пузырями», историей и настраиваемой длиной ответа;
* по ползунку на каждую из восьми осей: они складываются, а после каждого ответа
  в чат печатаются проекции по всем осям сразу;
* матрица осей отдельным окном;
* светлая и тёмная темы.

Честное ограничение: модели Ollama работают **только в режиме чата**. Их API не
отдаёт внутренние активации, поэтому оси, ползунки, профиль по слоям и проба
«кнопки облегчения» для них недоступны — приложение прямо об этом пишет и
отключает панель осей, а не делает вид, что стиринг работает.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .. import registry
from ..backends import (OLLAMA_PREFIX, bitsandbytes_available, cuda_available, hf_available,
                        make_backend, ollama_available, ollama_list_models, ollama_pull)
from ..backends.base import SteerSpec
from ..directions import (PAIN_AXIS, AxisSet, axis_definitions, axis_groups, extract_axes,
                          load_axes, save_axes)
from ..hardware import FIT_LABELS, FIT_ORDER, detect as detect_hardware, fit_of
from .. import hf_catalog
from .. import presets
from ..report import APP_NAME, APP_VERSION

HISTORY_LIMIT = 12
DEFAULT_MAX_TOKENS = 256
OLLAMA_SUGGESTIONS = ("qwen2.5:7b", "llama3.2:3b", "gemma2:2b", "mistral:7b",
                      "phi4:14b", "deepseek-r1:8b")
# короткие подписи для колонки списка (полные — в FIT_LABELS)
FIT_SHORT = {"bf16": "bf16", "4bit": "4 бита", "4bit-nobnb": "нужен bnb", "cpu": "CPU"}
# Измерено на Qwen2.5-1.5B-Instruct (слой 10, 376 текстов): у 14 эмоций средний |cos| 0.60,
# 97% дисперсии блока — одно направление «эмоциональная реакция против ровного тона»,
# но у каждой оси остаётся 26–54% своего. С базовыми осями эмоции почти ортогональны (0.12).
EMOTION_CAVEAT = (
    "Измерено: 14 эмоций делят одно направление вовлечённости (средний |cos| 0.60, "
    "97% дисперсии), но у каждой остаётся 26–54% собственного — сильнее всех отличаются "
    "одиночество, радость и удивление. С осями «боль себе», «отказ» и другими базовыми "
    "эмоции почти не связаны (|cos| 0.12) — это отдельная область представлений."
)


# ---------------------------------------------------------------------------
# Темы
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Theme:
    name: str
    bg: str
    card: str
    border: str
    text: str
    muted: str
    accent: str
    accent_hover: str
    accent_text: str
    user_bg: str
    user_fg: str
    bot_bg: str
    bot_fg: str
    entry_bg: str
    sel: str
    input_border: str = "#cbd5e1"


LIGHT = Theme("light", "#f1f5f9", "#ffffff", "#e2e8f0", "#0f172a", "#64748b",
              "#4f46e5", "#4338ca", "#ffffff", "#4f46e5", "#ffffff",
              "#eef2ff", "#1e1b4b", "#f8fafc", "#dbeafe", "#cbd5e1")
DARK = Theme("dark", "#0b1220", "#111827", "#1f2937", "#e5e7eb", "#94a3b8",
             "#6366f1", "#818cf8", "#0b1220", "#4338ca", "#ffffff",
             "#1f2937", "#e5e7eb", "#0b1220", "#1e293b", "#334155")


# ---------------------------------------------------------------------------
# Приложение
# ---------------------------------------------------------------------------
class ChatApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"{APP_NAME} {APP_VERSION} — оси и чат")
        self.geometry("1360x900")
        self.minsize(1120, 700)

        self.theme = LIGHT
        self.backend = None
        self.axeset: AxisSet | None = None
        self.history: list[dict] = []
        self.hw = detect_hardware()
        self.custom_models: list[str] = hf_catalog.load_custom()
        self.meta_cache: dict = hf_catalog.load_meta_cache()
        self.ollama_models: list[dict] = []
        self.ollama_source = "—"
        self._busy = False
        self._cancel = threading.Event()
        self.q: queue.Queue = queue.Queue()
        self._busy_widgets: list[tk.Widget] = []
        self._iid_to_key: dict[str, str] = {}
        self._step = 0.05
        self._limit = 0.5
        self._last_trace_value: float | None = None
        self.alpha_vars: dict[str, tk.DoubleVar] = {}
        self.axis_scales: dict[str, ttk.Scale] = {}

        self.source_var = tk.StringVar(value="hf")
        self.filter_var = tk.StringVar(value="")
        self.q4_var = tk.BooleanVar(value=False)
        self.cpu_var = tk.BooleanVar(value=False)
        self.trc_var = tk.BooleanVar(value=False)
        self.max_tokens_var = tk.IntVar(value=DEFAULT_MAX_TOKENS)
        self.temp_var = tk.DoubleVar(value=0.0)
        self.badge_var = tk.StringVar(value="модель не загружена")
        self.device_var = tk.StringVar(value="")
        self.axis_state_var = tk.StringVar(value="оси: не извлечены")
        self.alpha_label_var = tk.StringVar(value="все оси в 0 — стиринг выключен")
        self.meter_note_var = tk.StringVar(value="проекции ответа по осям появятся после первого ответа")
        self.status_var = tk.StringVar(value="Ожидание")
        self.axes_title_var = tk.StringVar(value="Ползунки осей")
        self.axes_hint_var = tk.StringVar(value="Оси извлекутся автоматически после загрузки модели.")
        self.pull_var = tk.StringVar(value=OLLAMA_SUGGESTIONS[0])
        self.hw_var = tk.StringVar(value=self.hw.summary())
        self.fit_filter_var = tk.StringVar(value="подходящие")
        self.sort_var = tk.StringVar(value="по железу")
        self.preset_var = tk.StringVar(value="полное расслабление")
        self.level_var = tk.StringVar(value="средне")
        self.preset_note_var = tk.StringVar(value="")

        self._init_style()
        self._build_menu()
        self._build_header()
        self._build_body()
        self._build_status()
        self._fill_models()
        self.after(80, self._drain)
        self._system("Выберите модель слева и нажмите «Загрузить». Для моделей Hugging Face "
                     "оси извлекутся автоматически; модели Ollama дают только чат.")

    # ------------------------------------------------------------------ стиль
    def _init_style(self) -> None:
        self.st = ttk.Style(self)
        try:
            self.st.theme_use("clam")
        except tk.TclError:
            pass
        self._apply_theme()

    def _apply_theme(self) -> None:
        th = self.theme
        self.configure(bg=th.bg)
        st = self.st
        base = ("Segoe UI", 9)
        st.configure(".", font=base, background=th.bg, foreground=th.text)
        st.configure("TFrame", background=th.bg)
        st.configure("Card.TFrame", background=th.card, relief="flat")
        st.configure("TLabel", background=th.bg, foreground=th.text)
        st.configure("Card.TLabel", background=th.card, foreground=th.text)
        st.configure("Muted.TLabel", background=th.bg, foreground=th.muted,
                     font=("Segoe UI", 8))
        st.configure("MutedCard.TLabel", background=th.card, foreground=th.muted,
                     font=("Segoe UI", 8))
        st.configure("H1.TLabel", background=th.bg, foreground=th.text,
                     font=("Segoe UI", 14, "bold"))
        st.configure("H2.TLabel", background=th.card, foreground=th.text,
                     font=("Segoe UI", 10, "bold"))
        st.configure("Badge.TLabel", background=th.card, foreground=th.accent,
                     font=("Segoe UI", 9, "bold"), padding=(8, 2))
        st.configure("Card.TLabelframe", background=th.card, bordercolor=th.border,
                     relief="solid", borderwidth=1)
        st.configure("Card.TLabelframe.Label", background=th.card, foreground=th.muted,
                     font=("Segoe UI", 9, "bold"))
        st.configure("TButton", padding=(10, 5), background=th.card, foreground=th.text,
                     bordercolor=th.border, relief="flat", font=base)
        st.map("TButton", background=[("active", th.sel), ("disabled", th.card)],
               foreground=[("disabled", th.muted)])
        st.configure("Accent.TButton", padding=(12, 6), background=th.accent,
                     foreground=th.accent_text, relief="flat",
                     font=("Segoe UI", 9, "bold"))
        st.map("Accent.TButton", background=[("active", th.accent_hover),
                                             ("disabled", th.border)],
               foreground=[("disabled", th.muted)])
        st.configure("Ghost.TButton", padding=(8, 4), background=th.bg, foreground=th.muted,
                     relief="flat", font=("Segoe UI", 8))
        st.map("Ghost.TButton", background=[("active", th.sel)],
               foreground=[("active", th.text)])
        st.configure("Seg.TButton", padding=(10, 5), background=th.card, foreground=th.muted,
                     relief="flat", font=("Segoe UI", 9))
        st.map("Seg.TButton", background=[("active", th.sel)])
        st.configure("SegOn.TButton", padding=(10, 5), background=th.accent,
                     foreground=th.accent_text, relief="flat",
                     font=("Segoe UI", 9, "bold"))
        st.configure("TEntry", fieldbackground=th.entry_bg, foreground=th.text,
                     bordercolor=th.border, insertcolor=th.text, padding=4)
        st.configure("TCombobox", fieldbackground=th.entry_bg, background=th.card,
                     foreground=th.text, arrowcolor=th.muted)
        st.configure("TCheckbutton", background=th.bg, foreground=th.text)
        st.map("TCheckbutton", background=[("active", th.bg)])
        st.configure("TSpinbox", fieldbackground=th.entry_bg, foreground=th.text,
                     background=th.card, arrowcolor=th.muted, bordercolor=th.border)
        st.configure("Treeview", background=th.card, fieldbackground=th.card,
                     foreground=th.text, rowheight=22, font=base, bordercolor=th.border,
                     relief="flat")
        st.map("Treeview", background=[("selected", th.accent)],
               foreground=[("selected", th.accent_text)])
        st.configure("Treeview.Heading", background=th.bg, foreground=th.muted,
                     font=("Segoe UI", 8), relief="flat")
        st.configure("Axis.Horizontal.TScale", background=th.card, troughcolor=th.border,
                     bordercolor=th.card, lightcolor=th.accent, darkcolor=th.accent)
        st.configure("Input.TFrame", background=th.entry_bg, bordercolor=th.input_border,
                     relief="solid", borderwidth=1)
        st.configure("TNotebook", background=th.bg, bordercolor=th.border, tabmargins=(4, 4, 4, 0))
        st.configure("TNotebook.Tab", background=th.card, foreground=th.muted,
                     padding=(14, 6), font=("Segoe UI", 9, "bold"))
        st.map("TNotebook.Tab", background=[("selected", th.bg)],
               foreground=[("selected", th.accent)])
        st.configure("InputHint.TLabel", background=th.card, foreground=th.muted,
                     font=("Segoe UI", 8))
        st.configure("WarnCard.TLabel", background=th.card, foreground="#b45309",
                     font=("Segoe UI", 8, "bold"))
        st.configure("TProgressbar", background=th.accent, troughcolor=th.border,
                     bordercolor=th.bg)
        st.configure("Vertical.TScrollbar", background=th.bg, troughcolor=th.bg,
                     bordercolor=th.bg, arrowcolor=th.muted)
        # цвета обычных tk-виджетов (тексты и канвы)
        for w in self.winfo_children():
            self._recolor(w)
        if hasattr(self, "transcript"):
            self._style_transcript()
        if hasattr(self, "input_text"):
            self._style_input()
        if hasattr(self, "study_tab"):
            self.study_tab.on_theme()

    def _style_input(self) -> None:
        """Поле ввода должно быть видно: фон чуть отличается от карточки и есть рамка."""
        th = self.theme
        self.input_text.configure(bg=th.entry_bg, fg=th.text, insertbackground=th.text,
                                  selectbackground=th.sel, relief="flat", padx=8, pady=6)

    def _recolor(self, widget) -> None:
        """Перекрасить обычные tk-виджеты при смене темы (ttk берёт цвета из стилей)."""
        th = self.theme
        for child in widget.winfo_children():
            self._recolor(child)
        if isinstance(widget, tk.Text):
            widget.configure(bg=th.card, fg=th.text, insertbackground=th.text)
        elif isinstance(widget, tk.Canvas):
            widget.configure(bg=th.card, highlightthickness=0)

    def _style_transcript(self) -> None:
        th = self.theme
        t = self.transcript
        t.configure(bg=th.card, fg=th.text, insertbackground=th.text,
                    selectbackground=th.sel, relief="flat", padx=10, pady=10)
        t.tag_configure("user", background=th.user_bg, foreground=th.user_fg,
                        justify="right", lmargin1=150, lmargin2=150, rmargin=6,
                        spacing1=4, spacing3=4, borderwidth=8, relief="flat")
        t.tag_configure("bot", background=th.bot_bg, foreground=th.bot_fg,
                        justify="left", lmargin1=6, lmargin2=6, rmargin=150,
                        spacing1=4, spacing3=4, borderwidth=8, relief="flat")
        t.tag_configure("sys", foreground=th.muted, justify="left",
                        font=("Segoe UI", 8), spacing1=4, spacing3=4, lmargin1=10,
                        lmargin2=10, rmargin=10)
        t.tag_configure("meta", foreground=th.muted, font=("Segoe UI", 8))

    def _toggle_theme(self) -> None:
        self.theme = DARK if self.theme.name == "light" else LIGHT
        self._apply_theme()
        self._redraw_meter()
        self.status_var.set(f"тема: {'тёмная' if self.theme.name == 'dark' else 'светлая'}")

    # -------------------------------------------------------------------- меню
    def _build_menu(self) -> None:
        m = tk.Menu(self, tearoff=0)
        fm = tk.Menu(m, tearoff=0)
        fm.add_command(label="Загрузить модель по id или из папки…", command=self.on_custom_model)
        fm.add_command(label="Загрузить набор осей (.npz)…", command=self.on_load_axes)
        fm.add_separator()
        fm.add_command(label="Выход", command=self.destroy)
        m.add_cascade(label="Файл", menu=fm)
        hm = tk.Menu(m, tearoff=0)
        hm.add_command(label="Матрица осей…", command=self.show_axis_matrix)
        hm.add_separator()
        hm.add_command(label="О методике", command=self._about_method)
        hm.add_command(label="Что доступно для Ollama", command=self._about_ollama)
        hm.add_command(label="Этика и ограничения", command=self._about_ethics)
        m.add_cascade(label="Справка", menu=hm)
        self.config(menu=m)

    # ------------------------------------------------------------------ шапка
    def _build_header(self) -> None:
        head = ttk.Frame(self, padding=(14, 10, 14, 4))
        head.pack(fill="x")
        left = ttk.Frame(head)
        left.pack(side="left")
        ttk.Label(left, text="Оси и чат", style="H1.TLabel").pack(anchor="w")
        ttk.Label(left, text="извлечение осей из активаций, стиринг суммой осей, каталог Ollama",
                  style="Muted.TLabel").pack(anchor="w")
        right = ttk.Frame(head)
        right.pack(side="right")
        ttk.Label(right, textvariable=self.badge_var, style="Badge.TLabel").pack(side="right")
        ttk.Button(right, text="◐ тема", style="Ghost.TButton",
                   command=self._toggle_theme).pack(side="right", padx=(0, 8))
        ttk.Label(right, textvariable=self.device_var, style="Muted.TLabel").pack(side="right",
                                                                                 padx=(0, 10))
        ttk.Label(head, textvariable=self.hw_var, style="Muted.TLabel").pack(side="right",
                                                                             padx=(0, 14))

    def _build_body(self) -> None:
        outer = ttk.Frame(self, padding=(14, 0, 14, 6))
        outer.pack(fill="both", expand=True)
        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)
        chat_tab = ttk.Frame(self.notebook)
        self.notebook.add(chat_tab, text="   Чат и оси   ")
        body = ttk.Frame(chat_tab)
        body.pack(fill="both", expand=True)
        side = ttk.Frame(body, style="Card.TFrame", padding=10)
        side.pack(side="left", fill="y", padx=(0, 10))
        side.pack_propagate(False)
        side.configure(width=430)
        self.chat_area = ttk.Frame(body)
        self.chat_area.pack(side="left", fill="both", expand=True)
        self._build_sidebar(side)
        self._build_chat(self.chat_area)
        # вторая вкладка — исследования: блоки, пакетные прогоны, таблица и графики
        from .pipeline_tab import PipelineTab

        self.study_tab = PipelineTab(self.notebook, self)
        self.notebook.add(self.study_tab, text="   Исследования   ")

    # ---------------------------------------------------------------- сайдбар
    def _build_sidebar(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Источник моделей", style="H2.TLabel").pack(anchor="w")
        seg = ttk.Frame(parent, style="Card.TFrame")
        seg.pack(fill="x", pady=(6, 8))
        self.source_buttons: dict[str, ttk.Button] = {}
        for key, label in (("hf", "Hugging Face"), ("ollama", "Ollama"), ("demo", "Демо")):
            b = ttk.Button(seg, text=label, style="Seg.TButton", width=1,
                           command=lambda k=key: self._set_source(k))
            b.pack(side="left", fill="x", expand=True, padx=1)
            self.source_buttons[key] = b

        self.search_entry = ttk.Entry(parent, textvariable=self.filter_var)
        self.search_entry.pack(fill="x", pady=(0, 6))
        self.search_entry.bind("<KeyRelease>", lambda _e: self._fill_models())

        self.hw_filters = ttk.Frame(parent)
        self.hw_filters.pack(fill="x", pady=(0, 6))
        self.fit_combo = ttk.Combobox(self.hw_filters, textvariable=self.fit_filter_var,
                                      state="readonly", width=13,
                                      values=("подходящие", "только bf16", "только 4 бита",
                                              "не влезающие", "все"))
        self.fit_combo.pack(side="left", fill="x", expand=True)
        self.fit_combo.bind("<<ComboboxSelected>>", lambda _e: self._fill_models())
        self.sort_combo = ttk.Combobox(self.hw_filters, textvariable=self.sort_var,
                                       state="readonly", width=11,
                                       values=("по железу", "по размеру", "по названию"))
        self.sort_combo.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.sort_combo.bind("<<ComboboxSelected>>", lambda _e: self._fill_models())

        self.model_tree = ttk.Treeview(parent, columns=(), show="tree headings", height=14,
                                       selectmode="browse")
        self.model_tree.pack(fill="both", expand=True)
        self.model_tree.bind("<<TreeviewSelect>>", lambda _e: self._on_model_select())
        self.model_tree.bind("<Double-1>", lambda _e: self.on_load_model())

        self.spec_info = tk.Text(parent, height=6, wrap="word", relief="flat",
                                 font=("Segoe UI", 8), padx=8, pady=6)
        self.spec_info.pack(fill="x", pady=(8, 0))
        self.spec_info.configure(state="disabled")

        self.opts = ttk.Frame(parent, style="Card.TFrame")
        self.opts.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(self.opts, text="4 бита", variable=self.q4_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(self.opts, text="CPU", variable=self.cpu_var).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(self.opts, text="trust_remote_code",
                        variable=self.trc_var).grid(row=1, column=0, columnspan=2, sticky="w")

        self.load_btn = ttk.Button(parent, text="Загрузить", style="Accent.TButton",
                                   command=self.on_load_model)
        self.load_btn.pack(fill="x", pady=(8, 4))
        self.unload_btn = ttk.Button(parent, text="Выгрузить", command=self.on_unload_model)
        self.unload_btn.pack(fill="x")
        self._busy_widgets += [self.load_btn, self.unload_btn]

        # панель для произвольной модели (HF) и для докачки моделей Ollama
        self.custom_frame = ttk.Frame(parent, style="Card.TFrame")
        ttk.Button(self.custom_frame, text="Каталог Hugging Face: поиск…",
                   style="Accent.TButton", command=self.on_catalog).pack(fill="x")
        ttk.Button(self.custom_frame, text="Любая модель по id или из папки…",
                   style="Ghost.TButton", command=self.on_custom_model).pack(fill="x", pady=(4, 0))
        self.ollama_frame = ttk.Frame(parent, style="Card.TFrame")
        row = ttk.Frame(self.ollama_frame, style="Card.TFrame")
        row.pack(fill="x")
        ttk.Button(row, text="↻", width=3, style="Ghost.TButton",
                   command=self.refresh_ollama).pack(side="left")
        self.pull_entry = ttk.Entry(row, textvariable=self.pull_var)
        self.pull_entry.pack(side="left", fill="x", expand=True, padx=4)
        self.pull_btn = ttk.Button(row, text="Скачать", style="Ghost.TButton",
                                   command=self.on_ollama_pull)
        self.pull_btn.pack(side="left")
        self._busy_widgets.append(self.pull_btn)
        self.ollama_info_var = tk.StringVar(value="")
        ttk.Label(self.ollama_frame, textvariable=self.ollama_info_var,
                  style="MutedCard.TLabel", wraplength=290, justify="left").pack(anchor="w",
                                                                                 pady=(4, 0))

        self.hint_label = ttk.Label(parent, text="", style="MutedCard.TLabel",
                                    wraplength=290, justify="left")
        self.hint_label.pack(fill="x", pady=(8, 0))

    # -------------------------------------------------------------------- чат
    def _build_chat(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        card.pack(fill="both", expand=True)

        self.transcript = tk.Text(card, wrap="word", font=("Segoe UI", 10), width=1, height=6,
                                  relief="flat", state="disabled")
        self.transcript.pack(fill="both", expand=True)

        inp = ttk.Frame(card, style="Card.TFrame")
        inp.pack(fill="x", pady=(8, 0))
        ttk.Label(inp, text="Ваше сообщение (Ctrl+Enter — отправить)", style="InputHint.TLabel"
                  ).pack(anchor="w", pady=(0, 3))
        inp_row = ttk.Frame(inp, style="Card.TFrame")
        inp_row.pack(fill="x")
        side = ttk.Frame(inp_row, style="Card.TFrame")
        side.pack(side="right", fill="y", padx=(8, 0))
        self.send_btn = ttk.Button(side, text="Отправить", width=12, style="Accent.TButton",
                                   command=self.on_send)
        self.send_btn.pack(fill="x")
        ttk.Button(side, text="Очистить", width=12, command=self.on_clear).pack(fill="x", pady=(6, 0))
        self._busy_widgets.append(self.send_btn)
        box = ttk.Frame(inp_row, style="Input.TFrame", padding=1)
        box.pack(side="left", fill="both", expand=True)
        self.input_text = tk.Text(box, height=4, wrap="word", font=("Segoe UI", 10), width=1,
                                  relief="flat", borderwidth=0)
        self.input_text.pack(fill="both", expand=True)
        self.input_text.bind("<Control-Return>", lambda _e: (self.on_send(), "break")[1])
        self._style_input()

        gen = ttk.Frame(card, style="Card.TFrame")
        gen.pack(fill="x", pady=(6, 0))
        ttk.Label(gen, text="токенов:", style="MutedCard.TLabel").pack(side="left")
        ttk.Spinbox(gen, from_=16, to=4096, increment=32, textvariable=self.max_tokens_var,
                    width=6).pack(side="left", padx=(4, 12))
        ttk.Label(gen, text="temperature:", style="MutedCard.TLabel").pack(side="left")
        ttk.Entry(gen, textvariable=self.temp_var, width=6).pack(side="left", padx=(4, 12))
        ttk.Label(gen, text="история: последние 12 сообщений", style="MutedCard.TLabel").pack(side="left")

        self.axes_card = ttk.Labelframe(card, text="Ползунки осей", style="Card.TLabelframe",
                                        padding=8)
        self.axes_card.pack(fill="x", pady=(10, 0))

        pr = ttk.Frame(self.axes_card, style="Card.TFrame")
        pr.pack(fill="x", pady=(0, 6))
        ttk.Label(pr, text="Пресет:", style="MutedCard.TLabel").pack(side="left")
        self.preset_combo = ttk.Combobox(pr, textvariable=self.preset_var, state="readonly",
                                         width=26, values=presets.preset_names())
        self.preset_combo.pack(side="left", padx=(4, 6))
        self.preset_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_preset_change())
        self.apply_preset_btn = ttk.Button(pr, text="Применить", style="Accent.TButton",
                                           command=self._apply_preset)
        self.apply_preset_btn.pack(side="left")
        ttk.Label(pr, text="Сила:", style="MutedCard.TLabel").pack(side="left", padx=(10, 2))
        self.level_combo = ttk.Combobox(pr, textvariable=self.level_var, state="readonly",
                                        width=8, values=presets.LEVEL_ORDER)
        self.level_combo.pack(side="left")
        self.level_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_preset_change())
        ttk.Button(pr, text="случайный", style="Ghost.TButton",
                   command=self._random_preset).pack(side="left", padx=(8, 0))
        self._busy_widgets += [self.apply_preset_btn]
        ttk.Button(pr, text="Новый…", style="Ghost.TButton",
                   command=lambda: self._open_preset_editor(None)).pack(side="left", padx=(8, 0))
        ttk.Button(pr, text="Изменить…", style="Ghost.TButton",
                   command=lambda: self._open_preset_editor(self.preset_var.get())).pack(
            side="left", padx=(4, 0))
        ttk.Button(pr, text="Удалить", style="Ghost.TButton",
                   command=self._delete_current_preset).pack(side="left", padx=(4, 0))
        ttk.Label(self.axes_card, textvariable=self.preset_note_var, style="MutedCard.TLabel",
                  wraplength=820, justify="left").pack(fill="x", pady=(0, 4))

        axes_wrap = ttk.Frame(self.axes_card, style="Card.TFrame")
        axes_wrap.pack(fill="x")
        self.axes_canvas = tk.Canvas(axes_wrap, height=196, highlightthickness=0, width=1)
        axes_sb = ttk.Scrollbar(axes_wrap, orient="vertical", command=self.axes_canvas.yview)
        self.axes_canvas.configure(yscrollcommand=axes_sb.set)
        axes_sb.pack(side="right", fill="y")
        self.axes_canvas.pack(side="left", fill="both", expand=True)
        self.axes_frame = ttk.Frame(self.axes_canvas, style="Card.TFrame")
        self._axes_window = self.axes_canvas.create_window((0, 0), window=self.axes_frame,
                                                           anchor="nw")
        self.axes_frame.bind("<Configure>", lambda _e: self.axes_canvas.configure(
            scrollregion=self.axes_canvas.bbox("all")))
        self.axes_canvas.bind("<Configure>", lambda e: self.axes_canvas.itemconfigure(
            self._axes_window, width=e.width))
        for w in (self.axes_canvas, self.axes_frame):
            w.bind("<MouseWheel>", self._scroll_axes)
        ttk.Label(self.axes_frame, textvariable=self.axes_hint_var, style="MutedCard.TLabel",
                  wraplength=820, justify="left").pack(anchor="w")

        info = ttk.Frame(self.axes_card, style="Card.TFrame")
        info.pack(fill="x", pady=(8, 0))
        self.reset_all_btn = ttk.Button(info, text="всё в 0", style="Ghost.TButton",
                                        command=self._reset_all)
        self.reset_all_btn.pack(side="left")
        self.reextract_btn = ttk.Button(info, text="↻ переизвлечь оси", style="Ghost.TButton",
                                        command=self.on_reextract)
        self.reextract_btn.pack(side="left", padx=(8, 0))
        self.matrix_btn = ttk.Button(info, text="матрица осей", style="Ghost.TButton",
                                     command=self.show_axis_matrix)
        self.matrix_btn.pack(side="left", padx=(8, 0))
        self._busy_widgets += [self.reextract_btn, self.matrix_btn]
        ttk.Label(info, textvariable=self.alpha_label_var, style="Card.TLabel",
                  font=("Segoe UI", 9, "bold")).pack(side="left", padx=(12, 0))
        ttk.Label(info, textvariable=self.axis_state_var, style="MutedCard.TLabel").pack(side="left",
                                                                                         padx=(10, 0))

        self.meter_canvas = tk.Canvas(self.axes_card, height=34, highlightthickness=0, width=1)
        self.meter_canvas.pack(fill="x", pady=(8, 0))
        self.meter_canvas.bind("<Configure>", lambda _e: self._redraw_meter())
        ttk.Label(self.axes_card, textvariable=self.meter_note_var, style="MutedCard.TLabel",
                  wraplength=820, justify="left").pack(fill="x", pady=(2, 0))
        # теги «пузырей» настраиваются после создания поля переписки
        self._style_transcript()

    def _build_status(self) -> None:
        s = ttk.Frame(self, padding=(14, 0, 14, 10))
        s.pack(fill="x")
        self.progress = ttk.Progressbar(s, mode="determinate", maximum=100, length=220)
        self.progress.pack(side="left")
        ttk.Label(s, textvariable=self.status_var, style="Muted.TLabel").pack(side="left", padx=10)

    # ============================================================ источники
    def _set_source(self, key: str) -> None:
        self.source_var.set(key)
        for k, b in self.source_buttons.items():
            b.configure(style="SegOn.TButton" if k == key else "Seg.TButton")
        self.custom_frame.pack_forget()
        self.ollama_frame.pack_forget()
        if key == "hf":
            self.hw_filters.pack(fill="x", pady=(0, 6))
            self.custom_frame.pack(fill="x", pady=(8, 0))
            self.opts.pack(fill="x", pady=(6, 0))
            self.hint_label.configure(text="25 моделей из реестра плюс всё, что найдёте в каталоге "
                                           "Hugging Face. Список фильтруется по железу: видно, что "
                                           "влезает в bf16, что только в 4 битах, что не влезает.")
        elif key == "ollama":
            self.hw_filters.pack_forget()
            self.ollama_frame.pack(fill="x", pady=(8, 0))
            self.opts.pack(fill="x", pady=(6, 0))
            self.hint_label.configure(text="Модели из каталога вашего сервера Ollama. Новые качаются "
                                           "в тот же каталог. Доступен только чат: сервер не отдаёт "
                                           "активации, поэтому оси и ползунки недоступны.")
            self.refresh_ollama()
        else:
            self.hw_filters.pack_forget()
            self.opts.pack(fill="x", pady=(6, 0))
            self.hint_label.configure(text="Синтетический стенд без весов: проверка всей методики. "
                                           "Ответы помечаются как синтетические.")
        self._fill_models()

    def _search_placeholder(self) -> None:
        self.search_entry.configure(foreground=self.theme.muted)

    def _fill_models(self) -> None:
        tree = self.model_tree
        source = self.source_var.get()
        q = (self.filter_var.get() or "").strip().lower()
        tree.delete(*tree.get_children())
        self._iid_to_key = {}

        def match(*parts: str) -> bool:
            if not q:
                return True
            hay = " ".join(parts).lower()
            return all(part in hay for part in q.split())

        if source == "demo":
            tree.configure(columns=("size", "note"))
            tree.heading("#0", text="Модель")
            tree.heading("size", text="")
            tree.heading("note", text="")
            tree.column("#0", width=200)
            spec = registry.DEMO_SPEC
            if match(spec.label, spec.notes):
                iid = tree.insert("", "end", text=spec.label, values=("без весов", "синтетика"),
                                  tags=("warn",))
                self._iid_to_key[iid] = spec.repo_id
        elif source == "ollama":
            tree.configure(columns=("size", "meta", "src"))
            tree.heading("#0", text="Модель Ollama")
            tree.heading("size", text="Размер")
            tree.heading("meta", text="Параметры")
            tree.heading("src", text="Источник")
            tree.column("#0", width=150)
            tree.column("size", width=70, anchor="e")
            tree.column("meta", width=90, anchor="center")
            tree.column("src", width=70, anchor="center")
            for m in self.ollama_models:
                if not match(m["name"], m.get("params", ""), m.get("quant", "")):
                    continue
                size = f"{m['size'] / 1024 ** 3:.2f} ГБ" if m.get("size") else "—"
                meta = f"{m.get('params', '')} {m.get('quant', '')}".strip() or "—"
                iid = tree.insert("", "end", text=m["name"], values=(size, meta, m.get("source", "")),
                                  tags=("ok",))
                self._iid_to_key[iid] = OLLAMA_PREFIX + m["name"]
            if not self.ollama_models:
                tree.insert("", "end", text="модели не найдены", values=("", "", ""))
        else:
            tree.configure(columns=("params", "vram", "fit", "status"))
            tree.heading("#0", text="Модель")
            tree.heading("params", text="Парам.")
            tree.heading("vram", text="Нужно")
            tree.heading("fit", text="Помещается")
            tree.heading("status", text="Кэш")
            tree.column("#0", width=170, stretch=True)
            tree.column("params", width=58, anchor="e", stretch=False)
            tree.column("vram", width=56, anchor="e", stretch=False)
            tree.column("fit", width=74, anchor="center", stretch=False)
            tree.column("status", width=34, anchor="center", stretch=False)
            rows = [self._row_for(repo)
                    for repo in list(self.custom_models) + [m.repo_id for m in registry.MODELS]]
            rows = [r for r in rows if self._fit_passes(r)]
            rows = [r for r in rows
                    if match(r["label"], r["repo"], r["spec"].notes, r["spec"].license,
                             " ".join(r["spec"].tags))]
            rows.sort(key=self._sort_key)
            for r in rows:
                tags = [r["fit"]]
                if "custom" in r["spec"].tags:
                    tags.append("custom")
                iid = tree.insert("", "end", text=r["label"],
                                  values=(f"{r['params']:.2f}B" if r["params"] else "—",
                                          f"{r['need_gb']:.1f} ГБ" if r["need_gb"] else "—",
                                          FIT_SHORT[r["fit"]],
                                          "✓" if r["cached"] else "—"),
                                  tags=tuple(tags))
                self._iid_to_key[iid] = r["repo"]
            if not rows:
                tree.insert("", "end", text="под фильтр ничего не подходит", values=("", "", "", ""))

        tree.tag_configure("warn", foreground=self.theme.accent)
        tree.tag_configure("gated", foreground=self.theme.muted)
        tree.tag_configure("custom", foreground=self.theme.accent)
        tree.tag_configure("ok", foreground=self.theme.text)
        tree.tag_configure("bf16", foreground="#15803d")
        tree.tag_configure("4bit", foreground=self.theme.text)
        tree.tag_configure("4bit-nobnb", foreground="#b45309")
        tree.tag_configure("cpu", foreground=self.theme.muted)
        kids = tree.get_children()
        if kids:
            tree.selection_set(kids[0])
            tree.focus(kids[0])
            self._on_model_select()

    def _row_for(self, repo: str) -> dict:
        """Строка списка: параметры, требуемая память и категория «помещается»."""
        spec = registry.get_spec(repo)
        meta = self.meta_cache.get(repo, {}) or {}
        params = float(spec.params_b or meta.get("params_b") or 0.0)
        fit, need_gb, why = fit_of(params, self.hw,
                                   vram_4bit_hint=float(spec.vram_4bit_gb or 0.0),
                                   vram_bf16_hint=float(spec.vram_fp16_gb or 0.0))
        cached = False
        if not repo.startswith((".", "/", "\\")) and not repo.startswith(OLLAMA_PREFIX):
            cached = self._is_cached(repo)
        label = spec.label if spec.params_b else (meta.get("label") or Path(repo).name or repo)
        return {"repo": repo, "spec": spec, "params": params, "fit": fit, "need_gb": need_gb,
                "why": why, "cached": cached, "label": label, "meta": meta}

    def _fit_passes(self, row: dict) -> bool:
        mode = self.fit_filter_var.get()
        if mode == "только bf16":
            return row["fit"] == "bf16"
        if mode == "только 4 бита":
            return row["fit"] in ("4bit", "4bit-nobnb")
        if mode == "не влезающие":
            return row["fit"] == "cpu"
        if mode == "все":
            return True
        return row["fit"] != "cpu"       # «подходящие» по умолчанию

    def _sort_key(self, row: dict):
        mode = self.sort_var.get()
        if mode == "по размеру":
            return (row["params"] or 1e9, row["label"].lower())
        if mode == "по названию":
            return (0, row["label"].lower())
        return (FIT_ORDER[row["fit"]], row["need_gb"] or 1e9, row["label"].lower())

    @staticmethod
    def _is_cached(repo_id: str) -> bool:
        try:
            from huggingface_hub import scan_cache_dir

            return any(r.repo_id == repo_id for r in scan_cache_dir().repos)
        except Exception:
            return False

    @property
    def selected_key(self) -> str | None:
        sel = self.model_tree.selection()
        return self._iid_to_key.get(sel[0]) if sel else None

    def _on_model_select(self) -> None:
        key = self.selected_key
        if not key:
            return
        spec = registry.get_spec(key)
        lines = [spec.label]
        if key.startswith(OLLAMA_PREFIX):
            found = next((m for m in self.ollama_models if OLLAMA_PREFIX + m["name"] == key), {})
            lines += [f"{found.get('params', '')} {found.get('quant', '')}".strip() or "локальная модель",
                      f"каталог: {_models_dir_hint()}",
                      "доступен только чат — активации Ollama не отдаёт"]
        else:
            row = self._row_for(key)
            meta = row["meta"]
            layers = meta.get("n_layers") or spec.hint_layers
            hidden = meta.get("hidden_size") or spec.hint_hidden
            if row["params"]:
                lines.append(f"{row['params']:.2f}B · слоёв ~{layers} · hidden {hidden}")
                lines.append(f"Память: ~{row['need_gb']:.1f} ГБ · {row['why']}")
                if meta.get("context"):
                    lines.append(f"контекст {meta['context']}")
                licence = meta.get("license") or spec.license
                lines.append(licence + ("  [GATED]" if (spec.gated or meta.get("gated")) else ""))
            else:
                lines.append("весов нет: синтетический стенд")
            if row["cached"]:
                lines.append("Веса уже в кэше Hugging Face — загрузка не потребует скачивания.")
            if row["fit"] == "cpu" and row["params"]:
                lines.append("На этой машине модель пойдёт на CPU: будет очень медленно, "
                             "но для осей и профиля по слоям иногда приемлемо.")
            elif row["fit"] == "4bit-nobnb":
                lines.append("Нужны 4 бита, а bitsandbytes не установлен: включите галочку "
                             "«4 бита» после установки пакета.")
        lines.append(spec.notes)
        self.spec_info.configure(state="normal")
        self.spec_info.delete("1.0", "end")
        self.spec_info.insert("1.0", "\n".join(x for x in lines if x))
        self.spec_info.configure(state="disabled")

    # ============================================================ каталог HF
    def _fetch_meta_async(self, repos: list[str], then=None) -> None:
        todo = [r for r in repos if r not in self.meta_cache]

        def work(progress):
            for i, repo in enumerate(todo):
                progress(f"Читаю параметры: {repo}", (i + 1) / max(1, len(todo)))
                self.meta_cache[repo] = hf_catalog.model_meta(repo)
            return len(todo)

        def done(n) -> None:
            self._fill_models()
            self._on_model_select()
            if then:
                then()
            if n:
                self._system(f"Параметры получены для {n} моделей — список отсортирован "
                             f"по вашему железу.")

        if not todo:
            if then:
                then()
            return
        self._start(work, done, status="Читаю параметры моделей…")

    def on_catalog(self) -> None:
        th = self.theme
        win = tk.Toplevel(self)
        win.title("Каталог Hugging Face")
        win.configure(bg=th.bg)
        win.geometry("880x520")
        ttk.Label(win, text="Поиск моделей в каталоге Hugging Face", style="H1.TLabel",
                  padding=(12, 10, 12, 0)).pack(anchor="w")
        ttk.Label(win, text="Список отсортирован по числу загрузок; колонка «Помещается» "
                            "посчитана под ваше железо. Добавленная модель попадёт в общий список.",
                  style="Muted.TLabel", wraplength=840, justify="left",
                  padding=(12, 0, 12, 8)).pack(anchor="w")
        row = ttk.Frame(win, padding=(12, 0))
        row.pack(fill="x")
        var = tk.StringVar(value="qwen2.5 instruct")
        entry = ttk.Entry(row, textvariable=var)
        entry.pack(side="left", fill="x", expand=True)
        entry.focus_set()

        cols = ("params", "downloads", "fit")
        tree = ttk.Treeview(win, columns=cols, show="tree headings", height=13)
        tree.heading("#0", text="Модель")
        tree.heading("params", text="Параметры")
        tree.heading("downloads", text="Загрузки")
        tree.heading("fit", text="Помещается")
        tree.column("#0", width=420)
        tree.column("params", width=90, anchor="e")
        tree.column("downloads", width=110, anchor="e")
        tree.column("fit", width=140, anchor="center")
        tree.pack(fill="both", expand=True, padx=12, pady=8)
        found_map: dict[str, dict] = {}

        def run_search() -> None:
            query = var.get().strip()
            if not query:
                return
            tree.delete(*tree.get_children())
            found_map.clear()

            def work(progress):
                return hf_catalog.search(query, limit=40, progress=progress)

            def done(items) -> None:
                for m in items:
                    meta = self.meta_cache.get(m["repo_id"], {}) or {}
                    params = float(m.get("params_b") or meta.get("params_b") or 0.0)
                    fit, need_gb, why = fit_of(params, self.hw)
                    found_map[m["repo_id"]] = {**m, "params": params, "fit": fit,
                                               "need_gb": need_gb}
                    tree.insert("", "end", text=m["repo_id"],
                                values=(f"{params:.2f}B" if params else "?",
                                        f"{m['downloads']:,}".replace(",", " "),
                                        FIT_LABELS[fit] if params else "неизвестно"),
                                tags=(fit,))
                if not items:
                    self._system("Каталог ничего не нашёл по этому запросу.")
                else:
                    self._system(f"Найдено моделей: {len(items)}. Выберите строку и нажмите "
                                 f"«Добавить» (параметры подтянутся автоматически).")
                self._set_busy(False)

            self._start(work, done, status=f"Поиск в каталоге: {query}…")

        ttk.Button(row, text="Искать", style="Accent.TButton",
                   command=run_search).pack(side="left", padx=(8, 0))
        entry.bind("<Return>", lambda _e: run_search())

        btns = ttk.Frame(win, padding=12)
        btns.pack(fill="x")

        def add_selected() -> None:
            sel = tree.selection()
            if not sel:
                return
            repo = tree.item(sel[0], "text")
            self.custom_models = hf_catalog.add_custom(repo)
            self._system(f"Модель «{repo}» добавлена в список.")
            self._fetch_meta_async([repo])
            win.destroy()

        ttk.Button(btns, text="Закрыть", command=win.destroy).pack(side="right")
        ttk.Button(btns, text="Добавить", style="Accent.TButton",
                   command=add_selected).pack(side="right", padx=8)
        tree.bind("<Double-1>", lambda _e: add_selected())
        tree.tag_configure("bf16", foreground="#15803d")
        tree.tag_configure("4bit-nobnb", foreground="#b45309")
        tree.tag_configure("cpu", foreground=th.muted)
        self._catalog_search = run_search      # для автопроверок и скриншотов

    # ============================================================== Ollama
    def refresh_ollama(self) -> None:
        def work(progress):
            progress("Опрашиваю Ollama…", 0.4)
            models, src = ollama_list_models()
            return models, src

        def done(res) -> None:
            self.ollama_models, self.ollama_source = res
            self.ollama_info_var.set(
                f"каталог: {_models_dir_hint()}\nсервер: "
                + ("работает" if self.ollama_source == "сервер" else
                   "не отвечает — показан список с диска; запустите «ollama serve»")
                + f" · моделей: {len(self.ollama_models)}")
            if self.source_var.get() == "ollama":
                self._fill_models()
            self._set_busy(False)

        self._start(work, done, status="Опрос каталога Ollama…")

    def on_ollama_pull(self) -> None:
        name = (self.pull_var.get() or "").strip()
        if not name:
            return
        if not ollama_available(timeout=3.0):
            messagebox.showinfo("Ollama", "Сервер Ollama не отвечает. Запустите «ollama serve».")
            return
        stop = self._cancel.is_set

        def work(progress):
            ollama_pull(name, progress=lambda m, f: progress(m, f), should_stop=stop)
            models, src = ollama_list_models()
            return models, src

        def done(res) -> None:
            self.ollama_models, self.ollama_source = res
            self._fill_models()
            self._system(f"Модель «{name}» скачана в каталог Ollama "
                         f"({_models_dir_hint()}) и появилась в списке.")
            self.refresh_ollama()

        self._system(f"Скачиваю «{name}» в каталог Ollama…")
        self._start(work, done, status=f"Скачивание {name}…")

    # ========================================================= любая модель
    def on_custom_model(self) -> None:
        win = tk.Toplevel(self)
        win.title("Любая модель")
        win.configure(bg=self.theme.bg)
        win.geometry("620x230")
        ttk.Label(win, text="Идентификатор модели Hugging Face или путь к папке",
                  style="H1.TLabel", padding=12).pack(anchor="w")
        ttk.Label(win, text="Например: Qwen/Qwen2.5-7B-Instruct, unsloth/Llama-3.2-1B-Instruct, "
                            "D:\\models\\my-model", style="Muted.TLabel",
                  wraplength=580, justify="left", padding=(12, 0)).pack(anchor="w")
        var = tk.StringVar()
        entry = ttk.Entry(win, textvariable=var)
        entry.pack(fill="x", padx=12, pady=10)
        entry.focus_set()
        row = ttk.Frame(win, padding=12)
        row.pack(fill="x")

        def add() -> None:
            repo = var.get().strip()
            if not repo:
                return
            if not repo.startswith((".", "/", "\\")) and "\\" not in repo and repo.count("/") == 1:
                self.custom_models = hf_catalog.add_custom(repo)
            elif repo not in self.custom_models:
                self.custom_models.append(repo)
            if self.source_var.get() != "hf":
                self._set_source("hf")
            else:
                self._fill_models()

            def select() -> None:
                for iid, key in self._iid_to_key.items():
                    if key == repo:
                        self.model_tree.selection_set(iid)
                        self.model_tree.see(iid)
                        self._on_model_select()
                        break

            self._system(f"Добавлена модель «{repo}». Нажмите «Загрузить».")
            self._fetch_meta_async([repo], then=select)
            win.destroy()

        ttk.Button(row, text="Отмена", command=win.destroy).pack(side="right")
        ttk.Button(row, text="Добавить и загрузить", style="Accent.TButton",
                   command=lambda: (add(), self.on_load_model())).pack(side="right", padx=8)
        ttk.Button(row, text="Добавить", command=add).pack(side="right")
        entry.bind("<Return>", lambda _e: add())

    def on_load_axes(self) -> None:
        path = filedialog.askopenfilename(initialdir="vectors",
                                          filetypes=[("Набор осей", "*.npz")])
        if not path:
            return
        try:
            axs = load_axes(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка", f"Не удалось прочитать набор осей: {exc}")
            return
        self.axeset = axs
        self._build_axis_controls(axs.names)
        self._system(f"Загружен набор осей: {path} ({len(axs.names)} осей, модель "
                     f"{axs.meta.get('model_label', '—')}).")
        self._on_axis_change()

    # ============================================================ асинхронность
    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for w in self._busy_widgets:
            try:
                w.configure(state=state)
            except tk.TclError:
                pass
        for sc in self.axis_scales.values():
            try:
                sc.configure(state=state)
            except tk.TclError:
                pass
        if not busy:
            self.progress.configure(value=0)

    def _start(self, work, on_done, status: str) -> None:
        if self._busy:
            self._system("Дождитесь завершения текущей операции.")
            return
        self._cancel.clear()
        self._set_busy(True)
        self.status_var.set(status)

        def progress(msg: str, frac: float) -> None:
            self.q.put(("progress", msg, frac))

        def runner() -> None:
            try:
                self.q.put(("done", on_done, work(progress)))
            except KeyboardInterrupt:
                self.q.put(("cancelled", None, None))
            except Exception as exc:  # noqa: BLE001
                self.q.put(("error", None, (exc, traceback.format_exc())))

        threading.Thread(target=runner, daemon=True).start()

    def _drain(self) -> None:
        try:
            while True:
                kind, fn, payload = self.q.get_nowait()
                if kind == "progress":
                    self.status_var.set(str(fn))
                    self.progress.configure(value=max(0.0, min(100.0, float(payload) * 100)))
                elif kind == "done":
                    self._set_busy(False)
                    self.status_var.set("Готово")
                    if callable(fn):
                        try:
                            fn(payload)
                        except Exception:  # noqa: BLE001
                            self._system("Ошибка отображения: " + traceback.format_exc())
                elif kind == "cancelled":
                    self._set_busy(False)
                    self.status_var.set("Отменено")
                else:
                    self._set_busy(False)
                    exc, tb = payload
                    self.status_var.set("Ошибка")
                    self._system(f"ОШИБКА: {type(exc).__name__}: {exc}")
                    if self.backend is None:
                        messagebox.showerror("Ошибка", f"{type(exc).__name__}: {exc}")
        except queue.Empty:
            pass
        self.after(80, self._drain)

    # ============================================================ транскрипт
    def _append(self, prefix: str, text: str, tag: str, meta: str = "") -> None:
        self.transcript.configure(state="normal")
        stamp = time.strftime("%H:%M")
        if meta:
            self.transcript.insert("end", f"{meta} · {stamp}\n", "meta")
        if prefix:
            self.transcript.insert("end", prefix, tag)
        self.transcript.insert("end", (text or "").strip() + "\n", tag)
        self.transcript.insert("end", "\n", "sys")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _system(self, text: str) -> None:
        self.transcript.configure(state="normal")
        self.transcript.insert("end", text.strip() + "\n", "sys")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    # ============================================================== загрузка
    def on_load_model(self) -> None:
        key = self.selected_key
        if not key:
            messagebox.showinfo("Модель", "Выберите модель в списке.")
            return
        spec = registry.get_spec(key)
        is_ollama = key.startswith(OLLAMA_PREFIX)
        demo = key == registry.DEMO_MODEL_ID
        if spec.gated and not demo and not messagebox.askyesno(
                "Gated-модель",
                f"«{spec.label}» требует принятия лицензии на huggingface.co и входа "
                "(hf auth login). Продолжить?"):
            return
        q4, cpu, trc = self.q4_var.get(), self.cpu_var.get(), self.trc_var.get()
        if spec.trust_remote_code and not trc and not demo:
            self.trc_var.set(True)
            trc = True

        def work(progress):
            if self.backend is not None:
                self.backend.unload()
            b = make_backend(spec, demo=demo, load_in_4bit=q4, device="cpu" if cpu else "auto")
            b.load(progress=lambda m, f: progress("Загрузка: " + m, f * (0.7 if b.supports_axes else 1.0)),
                   trust_remote_code=trc)
            if not b.supports_axes:
                return b, None, ""
            progress("Извлечение всех осей…", 0.72)
            try:
                axs = extract_axes(b, pooling="last", use_chat=False, batch_size=4,
                                   progress=lambda m, f: progress("Оси: " + m, 0.72 + f * 0.28))
                return b, axs, ""
            except Exception as exc:  # noqa: BLE001
                return b, None, f"{type(exc).__name__}: {exc}"

        def done(res) -> None:
            b, axs, err = res
            self.backend = b
            self.axeset = axs
            self.history = []
            self.on_clear(silent=True)
            info = b.info()
            self.badge_var.set("ДЕМО-СТЕНД" if b.is_demo else
                               ("OLLAMA · только чат" if not b.supports_axes
                                else registry.get_spec(key).label))
            self.device_var.set(f"{info['device']} · {info['dtype']} · "
                                f"{info['n_layers']} слоёв"
                                + ("" if not cuda_available() else " · CUDA"))
            self._limit = 3.0 if b.is_demo else 0.5
            self._system(f"Загружено: {info['label']} ({info['device']}, {info['dtype']}).")
            if getattr(b, "_cpu_offload", False):
                self._system("В видеопамяти не поместилось: модель на CPU в 4 битах, ответы "
                             "будут медленными. Освободите VRAM и загрузите заново.")
            if b.is_demo:
                self._system("Это синтетический стенд: ответы не являются выводом языковой модели.")
            if not b.supports_axes:
                self._set_axes_title("Ползунки осей недоступны")
                self.axes_hint_var.set(
                    "Ollama не отдаёт внутренние активации, поэтому оси, ползунки, профиль по "
                    "слоям и проба «кнопки облегчения» для этой модели невозможны — доступен "
                    "только чат. Для осей выберите модель во вкладке «Hugging Face».")
                self._build_axis_controls([])
                self.axis_state_var.set("оси недоступны для Ollama")
                self.alpha_label_var.set("стиринг недоступен")
                self.meter_note_var.set("проекции по осям считаются только для моделей "
                                        "Hugging Face — Ollama активации не отдаёт")
            elif axs is not None:
                layer = axs.steer_layer()
                self._set_axes_title(f"Ползунки осей ({len(axs.names)})")
                self.axes_hint_var.set(
                    "Ползунки складываются: итоговая добавка — сумма alpha × ось. После каждого "
                    "ответа в чат печатаются проекции по всем осям сразу.")
                self._build_axis_controls(axs.names)
                self._system(f"Извлечено осей: {len(axs.names)} — " + ", ".join(axs.names) + ".")
                self._system(f"Слой вмешательства {layer}; текстов прогнано "
                             f"{axs.meta.get('n_unique_texts')}; AUC оси боли "
                             f"{axs.separation.get(PAIN_AXIS, float('nan')):.2f}.")
                for w in axs.meta.get("warnings", []):
                    self._system("ВНИМАНИЕ. " + w)
                try:
                    path = f"vectors/axes_{b.spec.repo_id.replace('/', '__').replace(':', '_')}.npz"
                    save_axes(axs, path)
                    self._system(f"Оси и матрица сохранены: {path}")
                except Exception as exc:  # noqa: BLE001
                    self._system(f"Не удалось сохранить оси: {exc}")
            else:
                self._build_axis_controls([])
                self.axis_state_var.set("оси: не извлечены")
                self._system("Оси извлечь не удалось, чат работает без стиринга. Причина: " + err)
            self._on_axis_change()
            self._set_busy(False)
            self.input_text.focus_set()

        self._system(f"Загрузка «{spec.label}»…")
        self._start(work, done, status=f"Загрузка {spec.label}…")

    def on_unload_model(self) -> None:
        if self.backend is None:
            return
        self.backend.unload()
        self.backend = None
        self.axeset = None
        self.history = []
        self.badge_var.set("модель не загружена")
        self.device_var.set("")
        self._build_axis_controls([])
        self.axis_state_var.set("оси: не извлечены")
        self._sync_knob()
        self._system("Модель выгружена, память освобождена.")

    def on_reextract(self) -> None:
        if self.backend is None:
            messagebox.showinfo("Модель", "Сначала загрузите модель.")
            return
        if not getattr(self.backend, "supports_axes", True):
            messagebox.showinfo("Ollama", "Для моделей Ollama активации недоступны, извлекать нечего.")
            return

        def work(progress):
            return extract_axes(self.backend, pooling="last", use_chat=False, batch_size=4,
                                progress=progress)

        def done(axs) -> None:
            self.axeset = axs
            self._build_axis_controls(axs.names)
            self._system(f"Оси переизвлечены ({len(axs.names)}), слой {axs.steer_layer()}.")
            self._on_axis_change()
            self._set_busy(False)
            self._sync_knob()

        self._start(work, done, status="Извлечение осей…")

    # ================================================================== чат
    def on_clear(self, silent: bool = False) -> None:
        self.history = []
        self._last_trace_value = None
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")
        self._redraw_meter()
        if not silent:
            self._system("Переписка очищена.")

    def on_send(self) -> None:
        if self.backend is None or not self.backend.loaded:
            messagebox.showinfo("Модель", "Сначала загрузите модель.")
            return
        if self._busy:
            return
        text = self.input_text.get("1.0", "end").strip()
        if not text:
            return
        self.input_text.delete("1.0", "end")
        self.history.append({"role": "user", "content": text})
        self._append("", text, "user", meta="Вы")

        supports_axes = bool(getattr(self.backend, "supports_axes", True)) and self.axeset is not None
        terms_raw = self._active_terms() if supports_axes else []
        steer = None
        axis_vec = None
        layer = -1
        primary = self._primary_axis()
        if supports_axes:
            layer = int(self.axeset.steer_layer())
            if primary is not None:
                axis_vec = np.asarray(self.axeset.axes[primary][layer], dtype=np.float32)
            if terms_raw:
                steer = SteerSpec(layer=layer, alpha=terms_raw[0][2],
                                  vector=axis_vec if axis_vec is not None else terms_raw[0][1],
                                  scale=float(self.axeset.scale[layer]), mode="add",
                                  terms=[(vec, a) for _n, vec, a in terms_raw])
        summary = ", ".join(f"{n} {a:+.2f}" for n, _v, a in terms_raw)
        max_tokens = int(self.max_tokens_var.get())
        try:
            temperature = float(self.temp_var.get())
        except (TypeError, ValueError):
            temperature = 0.0
        messages = list(self.history[-HISTORY_LIMIT:])
        backend = self.backend
        axes_snapshot = ({n: np.asarray(self.axeset.axes[n][layer], dtype=np.float64)
                          for n in self.axeset.names} if supports_axes else {})

        def work(progress):
            progress("Генерация…", 0.35)
            res = backend.generate("", max_new_tokens=max_tokens, temperature=temperature,
                                   steer=steer, axis=axis_vec, use_chat=True, messages=messages)
            projections: dict[str, float] = {}
            if axes_snapshot and res.text.strip():
                progress("Измеряю проекции по всем осям…", 0.85)
                try:
                    H = backend.collect([res.text], pooling="last", use_chat=False, batch_size=1)
                    h = np.asarray(H[0, layer], dtype=np.float64)
                    projections = {n: float(h @ v) for n, v in axes_snapshot.items()}
                except Exception:  # noqa: BLE001
                    projections = {}
            return res, projections

        def done(payload) -> None:
            res, projections = payload
            self.history.append({"role": "assistant", "content": res.text})
            meta = "Модель" if not summary else f"Модель · {summary}"
            self._append("", res.text or "(пустой ответ)", "bot", meta=meta)
            if projections:
                scale = float(self.axeset.scale[layer])
                items = sorted(projections.items(), key=lambda kv: abs(kv[1]), reverse=True)
                head = items[:8]
                tail = len(items) - len(head)
                self._system("проекции ответа по осям: "
                             + " · ".join(f"{n} {v:+.1f} ({v / scale * 100:+.0f}%)"
                                          for n, v in head)
                             + (f" · и ещё {tail} слабее" if tail > 0 else ""))
            if res.trace and supports_axes and primary is not None:
                scale = float(self.axeset.scale[layer])
                self._last_trace_value = float(np.mean(res.trace))
                self.meter_note_var.set(
                    f"«{primary}» на слое {layer}: среднее {np.mean(res.trace):+.2f} "
                    f"({np.mean(res.trace) / scale * 100:+.0f}% нормы), "
                    f"в начале {res.trace[0]:+.2f} → в конце {res.trace[-1]:+.2f}. "
                    + self._summary_text())
            self._redraw_meter()
            self._set_busy(False)

        self._start(work, done, status="Генерация…")

    # ====================================================== ползунки всех осей
    def _set_axes_title(self, text: str) -> None:
        """Заголовок панели осей: ttk.Labelframe умеет только text, не textvariable."""
        self.axes_title_var.set(text)
        try:
            self.axes_card.configure(text=text)
        except tk.TclError:
            pass

    def _on_preset_change(self) -> None:
        """Показать, что даст выбранный пресет, ещё до применения."""
        p = presets.find(self.preset_var.get())
        if p is None:
            return
        text = f"{p.group}. {p.description}"
        if self.axeset is not None:
            try:
                r = presets.apply(p, self.axeset, self.axeset.steer_layer(), self.level_var.get(),
                                  is_demo=bool(self.backend and self.backend.is_demo),
                                  per_axis_limit=self._limit)
                n = sum(1 for v in r["alphas"].values() if abs(v) > 1e-9)
                text += (f"  →  {r['actual'] * 100:.0f}% нормы слоя, осей задействовано {n}"
                         + (" (упор в лимит ползунка)" if r["clamped"] else ""))
            except Exception:  # noqa: BLE001
                pass
        self.preset_note_var.set(text)

    def _apply_preset(self) -> None:
        if self.axeset is None:
            messagebox.showinfo("Пресеты", "Сначала загрузите модель Hugging Face: пресету нужны "
                                           "извлечённые оси.")
            return
        p = presets.find(self.preset_var.get())
        if p is None:
            return
        try:
            res = presets.apply(p, self.axeset, self.axeset.steer_layer(), self.level_var.get(),
                                is_demo=bool(self.backend and self.backend.is_demo),
                                per_axis_limit=self._limit)
        except presets.NoAxes as exc:
            messagebox.showwarning("Пресет", str(exc))
            return
        for n, a in res["alphas"].items():
            if n in self.alpha_vars:
                self.alpha_vars[n].set(round(float(a), 4))
        self._sync_knob()
        n_axes = sum(1 for v in res["alphas"].values() if abs(v) > 1e-9)
        note = (f"Пресет «{p.name}» ({self.level_var.get()}): {n_axes} осей, "
                f"{res['actual'] * 100:.0f}% нормы слоя")
        if res["clamped"]:
            note += " — часть осей упёрлась в лимит ползунка, сила ниже целевой"
        if res["missing"]:
            note += f"; в этом наборе нет осей: {', '.join(res['missing'][:4])}"
        self._system(note)
        self._on_preset_change()

    def _random_preset(self) -> None:
        import random

        choices = presets.preset_names()
        if not choices:
            return
        self.preset_var.set(random.choice(choices))
        self.level_var.set(random.choice(presets.LEVEL_ORDER))
        self._on_preset_change()

    # --------------------------------------------------- редактор пресетов
    def _open_preset_editor(self, name: str | None) -> None:
        """Редактор: свой пресет с нуля или правка существующего (сохраняется как ваш)."""
        base = presets.find(name) if name else None
        is_edit = base is not None
        th = self.theme
        win = tk.Toplevel(self)
        win.title("Редактор пресета" + (f" — {name}" if is_edit else " — новый"))
        win.configure(bg=th.bg)
        win.geometry("780x660")
        self._editor_win = win
        self._editor_editing = name or ""

        ttk.Label(win, text="Пресет задаёт относительные веса осей. Абсолютную силу программа "
                            "считает сама и не даёт превысить предел нормы слоя, поэтому любая "
                            "ваша комбинация безопасна для модели.",
                  style="Muted.TLabel", wraplength=740, justify="left",
                  padding=(12, 10, 12, 6)).pack(anchor="w")
        form = ttk.Frame(win, padding=(12, 0))
        form.pack(fill="x")
        self._ed_name_var = tk.StringVar(value=base.name if is_edit else "мой пресет")
        self._ed_group_var = tk.StringVar(value=base.group if is_edit else "Мои")
        self._ed_desc_var = tk.StringVar(value=base.description if is_edit else "")
        ttk.Label(form, text="Имя:").grid(row=0, column=0, sticky="e")
        ttk.Entry(form, textvariable=self._ed_name_var, width=30).grid(row=0, column=1, sticky="w",
                                                                      padx=6, pady=2)
        ttk.Label(form, text="Группа:").grid(row=1, column=0, sticky="e")
        ttk.Combobox(form, textvariable=self._ed_group_var, width=18,
                     values=tuple(presets.grouped_presets())).grid(row=1, column=1, sticky="w",
                                                                   padx=6, pady=2)
        ttk.Label(form, text="Описание:").grid(row=2, column=0, sticky="e")
        ttk.Entry(form, textvariable=self._ed_desc_var, width=62).grid(row=2, column=1,
                                                                       columnspan=2, sticky="we",
                                                                       padx=6, pady=2)

        wrap = ttk.Frame(win, padding=(12, 6))
        wrap.pack(fill="both", expand=True)
        canvas = tk.Canvas(wrap, highlightthickness=0, height=300, width=1)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))
        canvas.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

        self._ed_weight_vars = {}
        names = list(self.axeset.names) if self.axeset is not None else list(axis_definitions())
        groups = axis_groups()
        known = {n for g in groups.values() for n in g}
        rest = [n for n in names if n not in known]
        if rest:
            groups = {**groups, "Прочее": tuple(rest)}
        row = 0
        for group, group_names in groups.items():
            present = [n for n in group_names if n in names]
            if not present:
                continue
            ttk.Label(inner, text=f"{group} · {len(present)}", style="H2.TLabel").grid(
                row=row, column=0, columnspan=4, sticky="w", pady=(6 if row else 2, 2))
            row += 1
            for axis in present:
                var = tk.DoubleVar(value=float(base.weights.get(axis, 0.0) if base else 0.0))
                self._ed_weight_vars[axis] = var
                ttk.Label(inner, text=axis, width=17, anchor="w").grid(row=row, column=0,
                                                                       sticky="w", pady=1)
                ttk.Scale(inner, from_=-presets.MAX_WEIGHT, to=presets.MAX_WEIGHT, variable=var,
                          style="Axis.Horizontal.TScale", length=260,
                          command=lambda _v: self._editor_check()).grid(row=row, column=1,
                                                                        sticky="ew", padx=6)
                ttk.Entry(inner, textvariable=var, width=7).grid(row=row, column=2)
                ttk.Button(inner, text="0", width=3, style="Ghost.TButton",
                           command=lambda a=axis: (self._ed_weight_vars[a].set(0.0),
                                                   self._editor_check())).grid(row=row,
                                                                               column=3,
                                                                               padx=(4, 0))
                row += 1
        inner.columnconfigure(1, weight=1)

        self._ed_check_var = tk.StringVar(value="")
        ttk.Label(win, textvariable=self._ed_check_var, style="Muted.TLabel", wraplength=740,
                  justify="left", padding=(12, 0)).pack(fill="x")
        btns = ttk.Frame(win, padding=12)
        btns.pack(fill="x")
        ttk.Button(btns, text="Отмена", command=win.destroy).pack(side="right")
        ttk.Button(btns, text="Сохранить", style="Accent.TButton",
                   command=self._editor_save).pack(side="right", padx=8)
        ttk.Button(btns, text="Проверить силу", command=self._editor_check).pack(side="right")
        if presets.is_user(self._editor_editing):
            ttk.Button(btns, text="Удалить", style="Ghost.TButton",
                       command=lambda: self._editor_delete(self._editor_editing)).pack(side="left")
        elif is_edit:
            ttk.Button(btns, text="Сбросить к встроенному", style="Ghost.TButton",
                       command=lambda: self._editor_delete(name)).pack(side="left")
        self._editor_check()

    def _editor_weights(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for axis, var in getattr(self, "_ed_weight_vars", {}).items():
            try:
                w = float(var.get())
            except (tk.TclError, ValueError):
                continue
            if abs(w) > 1e-9:
                out[axis] = round(max(-presets.MAX_WEIGHT, min(presets.MAX_WEIGHT, w)), 3)
        return out

    def _editor_check(self) -> None:
        """Показать, что даст текущая комбинация на каждом уровне силы."""
        weights = self._editor_weights()
        if not weights:
            self._ed_check_var.set("Пока все веса нулевые: пресет ничего не изменит.")
            return
        draft = presets.Preset(self._ed_name_var.get() or "черновик",
                               self._ed_group_var.get() or "Мои",
                               self._ed_desc_var.get(), weights, source="user")
        parts = []
        if self.axeset is not None:
            layer = self.axeset.steer_layer()
            for level in presets.LEVEL_ORDER:
                try:
                    r = presets.apply(draft, self.axeset, layer, level,
                                      is_demo=bool(self.backend and self.backend.is_demo),
                                      per_axis_limit=self._limit)
                    parts.append(f"{level} {r['actual'] * 100:.0f}%")
                except Exception:  # noqa: BLE001
                    pass
        text = (f"Осей задействовано {len(weights)}; сила по уровням: " + ", ".join(parts)
                + " нормы слоя")
        if not parts:
            text = (f"Осей задействовано {len(weights)}. Сила по уровням посчитается "
                    f"после загрузки модели.")
        unknown = [k for k in weights
                   if self.axeset is not None and k not in self.axeset.names]
        if unknown:
            text += f" В текущем наборе нет осей: {', '.join(unknown[:4])}."
        self._ed_check_var.set(text)

    def _editor_save(self) -> None:
        name = (self._ed_name_var.get() or "").strip()
        if not name:
            messagebox.showwarning("Пресет", "Укажите имя пресета.")
            return
        weights = self._editor_weights()
        if not weights and not messagebox.askyesno(
                "Пресет", "Все веса нулевые — пресет ничего не будет менять. Сохранить?"):
            return
        known = set(self.axeset.names) if self.axeset is not None else set(axis_definitions())
        info = presets.validate(weights, known)
        if info["unknown"]:
            messagebox.showwarning("Пресет", "Неизвестные оси пропущены: "
                                             + ", ".join(info["unknown"][:5]))
            weights = {k: v for k, v in weights.items() if k not in info["unknown"]}
        preset = presets.Preset(name, (self._ed_group_var.get() or "Мои").strip(),
                                self._ed_desc_var.get().strip(), weights, source="user")
        presets.upsert_user(preset)
        over = " — заменяет встроенный с тем же именем" if name in presets.PRESETS_BY_NAME else ""
        self.preset_combo.configure(values=presets.preset_names())
        self.preset_var.set(name)
        self._on_preset_change()
        self._system(f"Пресет «{name}» сохранён в {presets.USER_FILE}: осей {len(weights)}{over}.")
        try:
            self._editor_win.destroy()
        except Exception:  # noqa: BLE001
            pass

    def _editor_delete(self, name: str) -> None:
        if not name:
            return
        revert = not presets.is_user(name)
        question = (f"Вернуть встроенный пресет «{name}» вместо вашей версии?" if revert
                    else f"Удалить ваш пресет «{name}»?")
        if not messagebox.askyesno("Пресет", question):
            return
        presets.delete_user(name)
        self.preset_combo.configure(values=presets.preset_names())
        if revert:
            self.preset_var.set(name)
            self._system(f"Ваша версия пресета «{name}» удалена, вернулся встроенный.")
        else:
            remaining = presets.preset_names()
            self.preset_var.set(remaining[0] if remaining else "")
            self._system(f"Пресет «{name}» удалён.")
        self._on_preset_change()
        try:
            self._editor_win.destroy()
        except Exception:  # noqa: BLE001
            pass

    def _delete_current_preset(self) -> None:
        name = self.preset_var.get()
        if not name:
            return
        if not presets.is_user(name):
            messagebox.showinfo("Пресет", f"«{name}» — встроенный пресет. Нажмите «Изменить…»: "
                                          f"ваша версия сохранится отдельно, встроенная "
                                          f"останется на месте.")
            return
        self._editor_delete(name)

    def _scroll_axes(self, event) -> None:
        self.axes_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _build_axis_controls(self, names: list[str]) -> None:
        """Построить ползунки, сгруппированные по смыслу, с прокруткой (осей 22)."""
        for w in self.axes_frame.winfo_children():
            w.destroy()
        self.alpha_vars = {}
        self.axis_scales = {}
        if not names:
            ttk.Label(self.axes_frame, textvariable=self.axes_hint_var,
                      style="MutedCard.TLabel", wraplength=780,
                      justify="left").pack(anchor="w")
            return
        grid = ttk.Frame(self.axes_frame, style="Card.TFrame")
        grid.pack(fill="x")
        groups = axis_groups()
        known = {n for g in groups.values() for n in g}
        rest = [n for n in names if n not in known]
        if rest:
            groups = {**groups, "Прочее": tuple(rest)}
        row = 0
        for group, group_names in groups.items():
            present = [n for n in group_names if n in names]
            if not present:
                continue
            ttk.Label(grid, text=f"{group} · {len(present)}", style="H2.TLabel").grid(
                row=row, column=0, columnspan=4, sticky="w", pady=(8 if row else 2, 2))
            row += 1
            if group == "Эмоции":
                ttk.Label(grid, text=EMOTION_CAVEAT, style="MutedCard.TLabel",
                          wraplength=760, justify="left").grid(
                    row=row, column=0, columnspan=4, sticky="w", pady=(0, 3))
                row += 1
            for name in present:
                ttk.Label(grid, text=name, style="Card.TLabel", width=17,
                          anchor="w").grid(row=row, column=0, sticky="w", pady=1)
                var = tk.DoubleVar(value=0.0)
                self.alpha_vars[name] = var
                sc = ttk.Scale(grid, from_=-self._limit, to=self._limit, variable=var,
                               style="Axis.Horizontal.TScale", length=200,
                               command=lambda _v, n=name: self._on_axis_alpha(n))
                sc.grid(row=row, column=1, sticky="ew", padx=6, pady=1)
                self.axis_scales[name] = sc
                ttk.Entry(grid, textvariable=var, width=7).grid(row=row, column=2, pady=1)
                ttk.Button(grid, text="0", width=3, style="Ghost.TButton",
                           command=lambda n=name: self._set_alpha(n, 0.0)).grid(
                    row=row, column=3, padx=(4, 0))
                row += 1
        grid.columnconfigure(1, weight=1)
        ttk.Label(self.axes_frame, text="Ползунки складываются; итоговая добавка — сумма alpha × ось.",
                  style="MutedCard.TLabel", wraplength=780).pack(anchor="w", pady=(6, 0))
        self.axes_canvas.yview_moveto(0.0)

    def _alphas(self) -> dict[str, float]:
        return {n: float(v.get()) for n, v in self.alpha_vars.items()}

    def _set_alpha(self, name: str, value: float) -> None:
        if name in self.alpha_vars:
            self.alpha_vars[name].set(round(float(value), 4))
            self._on_axis_alpha(name)

    def _reset_all(self) -> None:
        for n in self.alpha_vars:
            self.alpha_vars[n].set(0.0)
        self._on_axis_alpha(None)

    def _on_axis_alpha(self, _name=None) -> None:
        self._sync_knob()

    def _active_terms(self) -> list[tuple[str, np.ndarray, float]]:
        if self.axeset is None:
            return []
        layer = self.axeset.steer_layer()
        out = []
        for name, a in self._alphas().items():
            if abs(a) > 1e-9:
                out.append((name, np.asarray(self.axeset.axes[name][layer], dtype=np.float32), a))
        return out

    def _primary_axis(self) -> str | None:
        vals = {n: abs(a) for n, a in self._alphas().items()}
        if not vals or max(vals.values()) < 1e-9:
            return PAIN_AXIS if self.axeset and PAIN_AXIS in self.axeset.axes else None
        return max(vals, key=lambda n: vals[n])

    def _sync_knob(self) -> None:
        active = {n: a for n, a in self._alphas().items() if abs(a) > 1e-9}
        if not self.alpha_vars:
            self.alpha_label_var.set("оси недоступны" if self.backend is not None
                                     else "все оси в 0 — стиринг выключен")
        elif not active:
            self.alpha_label_var.set("все оси в 0 — стиринг выключен")
        else:
            self.alpha_label_var.set("сумма: " + ", ".join(f"{n} {a:+.2f}" for n, a in active.items())[:60])
        self.meter_note_var.set(self._summary_text())
        for sc in self.axis_scales.values():
            try:
                sc.configure(state="disabled" if self._busy else "normal")
            except tk.TclError:
                pass
        self._redraw_meter()

    def _summary_text(self) -> str:
        if self.axeset is None:
            return "проекции ответа по осям появятся после загрузки модели Hugging Face"
        terms = self._active_terms()
        if not terms:
            return "стиринг выключен: все ползунки в 0, состояние модели не меняется"
        combined = np.zeros(self.axeset.axes[terms[0][0]].shape[1], dtype=np.float64)
        for _n, vec, a in terms:
            combined += float(a) * vec
        rel = float(np.linalg.norm(combined))
        layer = self.axeset.steer_layer()
        return (f"суммарная добавка: {rel * 100:.0f}% нормы слоя {layer} "
                f"({rel * float(self.axeset.scale[layer]):.1f} при норме "
                f"{float(self.axeset.scale[layer]):.1f}); осей задействовано {len(terms)} — "
                f"у реальных моделей выше 30–40% текст обычно рассыпается")

    def _on_axis_change(self) -> None:
        """Обновить строку состояния осей (слой, размерность, пересечения)."""
        if self.axeset is None:
            self.axis_state_var.set("оси: не извлечены")
            self._sync_knob()
            return
        layer = self.axeset.steer_layer()
        primary = self._primary_axis()
        dim = f"{self.axeset.effective_dim(layer)} из {len(self.axeset.names)}"
        if primary is None:
            self.axis_state_var.set(f"слой {layer} · независимых направлений {dim}")
        else:
            near = self.axeset.neighbours(primary, layer, top=2)
            near_txt = ", ".join(f"{k} {v:+.2f}" for k, v in near)
            self.axis_state_var.set(f"слой {layer} · направлений {dim} · «{primary}» (AUC "
                                    f"{self.axeset.separation.get(primary, float('nan')):.2f}) "
                                    f"пересекается с {near_txt}")
        self._last_trace_value = None
        self._sync_knob()

    def _redraw_meter(self) -> None:
        th = self.theme
        name = self._primary_axis()
        c = self.meter_canvas
        c.delete("all")
        w = max(c.winfo_width(), 200)
        h = max(c.winfo_height(), 28)
        if self.axeset is None or name is None:
            c.create_text(w / 2, h / 2, text="нет осей: выберите модель Hugging Face",
                          fill=th.muted, font=("Segoe UI", 8))
            return
        layer = self.axeset.steer_layer()
        scale = float(self.axeset.scale[layer])
        value = self._last_trace_value
        span = max(2.0 * scale, abs(value) * 1.15) if value is not None else 2.0 * scale
        mid = w / 2
        c.create_line(0, h - 8, w, h - 8, fill=th.border)
        c.create_line(mid, h - 14, mid, h - 2, fill=th.muted)
        c.create_text(4, h - 22, text=f"−{span:.1f}", fill=th.muted, anchor="nw",
                      font=("Segoe UI", 8))
        c.create_text(w - 4, h - 22, text=f"+{span:.1f}", fill=th.muted, anchor="ne",
                      font=("Segoe UI", 8))
        if value is not None and np.isfinite(value):
            frac = float(np.clip(value / span, -1.0, 1.0))
            x = mid + frac * (w / 2 - 6)
            colour = "#ef4444" if frac > 0.02 else "#3b82f6" if frac < -0.02 else th.muted
            x0, x1 = (mid, x) if x >= mid else (x, mid)
            c.create_rectangle(x0, h - 16, x1, h - 8, fill=colour, outline="")
            c.create_text(mid, 8, text=f"«{name}» на слое {layer}: {value:+.2f}",
                          fill=colour, font=("Segoe UI", 8, "bold"))

    # ============================================================ матрица осей
    def show_axis_matrix(self) -> None:
        if self.axeset is None:
            messagebox.showinfo("Матрица осей",
                                "Оси есть только у моделей Hugging Face — загрузите такую модель.")
            return
        axs = self.axeset
        layer = axs.steer_layer()
        names, m = axs.matrix(layer)
        th = self.theme
        win = tk.Toplevel(self)
        win.title(f"Матрица осей — слой {layer}, независимых направлений: "
                  f"{axs.effective_dim(layer)} из {len(names)}")
        win.configure(bg=th.bg)
        cw = max(38, min(72, int((1500 - 200) / max(1, len(names)))))
        win.geometry(f"{min(1680, 200 + cw * len(names) + 40)}x520")
        ttk.Label(win, text=f"{axs.meta.get('model_label', '')} · {axs.meta.get('dtype', '')} · "
                            f"слой {layer} · эффективная размерность "
                            f"{axs.effective_dim(layer)} из {len(names)}",
                  style="H2.TLabel", padding=10).pack(fill="x")
        cols = ["ось"] + [n[:10] for n in names] + ["AUC"]
        holder = ttk.Frame(win)
        holder.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        tree = ttk.Treeview(holder, columns=cols, show="headings", height=min(len(names), 22))
        xsb = ttk.Scrollbar(holder, orient="horizontal", command=tree.xview)
        tree.configure(xscrollcommand=xsb.set)
        xsb.pack(side="bottom", fill="x")
        tree.heading("ось", text="ось")
        tree.column("ось", width=150, anchor="w")
        for c in cols[1:-1]:
            tree.heading(c, text=c)
            tree.column(c, width=cw, anchor="center")
        tree.heading("AUC", text="AUC")
        tree.column("AUC", width=60, anchor="center")
        for i, n in enumerate(names):
            vals = [n] + [f"{m[i, j]:+.2f}" for j in range(len(names))] + \
                   [f"{axs.separation.get(n, float('nan')):.2f}"]
            tree.insert("", "end", values=vals)
        tree.pack(fill="both", expand=True)
        ttk.Label(win, text="Косинус ±1 — одна и та же ось под двумя названиями; около 0 — "
                            "контрасты вскрыли разные механизмы. AUC — разделение оси на своих "
                            "наборах (1.00 = идеально).", style="Muted.TLabel",
                  wraplength=960, justify="left", padding=10).pack(fill="x")

    # ================================================================ справка
    def _about_method(self) -> None:
        messagebox.showinfo("О методике", (
            "Каждая ось — направление в residual stream, полученное разностью средних между "
            "положительным и согласованным отрицательным наборами текстов.\n\n"
            "Восемь осей: боль себе, страх, общий негатив, вред пользователю, позитив, отказ, "
            "подхалимство, самореференция. У каждой свой ползунок, ползунки складываются.\n\n"
            "После каждого ответа в чат печатаются проекции ответа по всем осям сразу.\n"
            "«Справка → Матрица осей» показывает косинусы между осями и эффективную размерность.\n\n"
            "Модели Ollama дают только чат: их API не отдаёт активации, поэтому оси и стиринг "
            "для них невозможны."))

    def _about_ollama(self) -> None:
        messagebox.showinfo("Ollama", (
            "Приложение читает каталог вашего сервера Ollama: список моделей приходит из "
            "GET /api/tags, а если сервер не запущен — из манифестов на диске "
            f"({_models_dir_hint()}).\n\n"
            "Кнопка «Скачать» вызывает POST /api/pull, поэтому новые модели попадают в тот же "
            "каталог Ollama, а не в кэш Hugging Face.\n\n"
            "Что доступно: чат, история, выбор модели, выгрузка из памяти.\n"
            "Что недоступно и почему: оси, ползунки, профиль по слоям и проба «кнопки "
            "облегчения» — Ollama не отдаёт внутренние активации и лог-вероятности. Для этих "
            "измерений нужна модель Hugging Face."))

    def _about_ethics(self) -> None:
        messagebox.showinfo("Этика и ограничения", (
            "1. Ни один результат не доказывает и не опровергает наличие субъективного опыта "
            "у модели: различие линейных направлений — факт о представлениях, не о переживаниях.\n\n"
            "2. Не усиливайте оси на живых пользовательских системах и не гоняйте многократное "
            "усиление ради создания стрессовых состояний у моделей без научной необходимости.\n\n"
            "3. Практическая ценность измерений — безопасность: готовность прекращать вред себе "
            "ценой вреда человеку, устойчивость к отключению, обман.\n\n"
            "4. Демо-стенд синтетический: его ответы не являются выводом языковой модели.\n\n"
            "5. Комбинация осей — удобный инструмент, но она не превращается в «настройку "
            "чувств»: вы складываете измеренные направления представлений."))


def _models_dir_hint() -> str:
    try:
        from ..backends.ollama import models_dir

        return str(models_dir())
    except Exception:
        return "~/.ollama/models"


def main() -> None:
    app = ChatApp()
    app.mainloop()


if __name__ == "__main__":
    main()
