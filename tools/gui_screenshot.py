"""Скриншоты интерфейса для документации: окно чата и матрица осей.

    python tools/gui_screenshot.py [папка_вывода]

Работает на синтетическом стенде: ни весов, ни GPU не требуется.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from pain_axis.ui.tk_app import ChatApp  # noqa: E402


class BMIH(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long),
                ("biHeight", ctypes.c_long), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", ctypes.c_long),
                ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


class BMI(ctypes.Structure):
    _fields_ = [("bmiHeader", BMIH), ("bmiColors", wintypes.DWORD * 3)]


def find_window(title: str, pid: int) -> int:
    """Окно с этим заголовком именно у нашего процесса.

    Иначе FindWindowW может вернуть чужое окно с тем же заголовком — например,
    запущенный рядом PainAxis.exe, и снимок выйдет не того размера и не той версии.
    """
    user32 = ctypes.windll.user32
    found: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _lparam):
        buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, buf, 512)
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if buf.value == title and int(owner.value) == pid and user32.IsWindowVisible(hwnd):
            found.append(int(hwnd))
        return True

    user32.EnumWindows(cb, 0)
    return found[0] if found else 0


def shoot(title: str, out: Path) -> None:
    """Снимок окна по заголовку через PrintWindow (не зависит от того, что сверху)."""
    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
    user32.SetProcessDPIAware()
    hwnd = find_window(title, os.getpid())
    if not hwnd:
        raise SystemExit(f"окно «{title}» не найдено")
    user32.SetForegroundWindow(wintypes.HWND(hwnd))
    time.sleep(0.4)
    rect = wintypes.RECT()
    user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    hdc = user32.GetWindowDC(wintypes.HWND(hwnd))
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    gdi32.SelectObject(mem, bmp)
    ok = user32.PrintWindow(wintypes.HWND(hwnd), mem, 2)
    info = BMI()
    info.bmiHeader.biSize = ctypes.sizeof(BMIH)
    info.bmiHeader.biWidth = w
    info.bmiHeader.biHeight = -h
    info.bmiHeader.biPlanes = 1
    info.bmiHeader.biBitCount = 32
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(info), 0)
    img = Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1).convert("RGB")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem)
    user32.ReleaseDC(wintypes.HWND(hwnd), hdc)
    print(f"{out.name}: {w}x{h} (PrintWindow={ok})")


def run_sync(app: ChatApp, fn, *args, **kwargs) -> None:
    orig = app._start

    def fake_start(work, on_done=None, status=""):
        res = work(lambda m, f: None)
        if callable(on_done):
            on_done(res)

    app._start = fake_start
    try:
        fn(*args, **kwargs)
        app.update()
    finally:
        app._start = orig


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "screens"
    app = ChatApp()
    app.geometry("1360x900+20+10")
    app.update()

    # 1. Вкладка Hugging Face: железо в шапке, колонка «помещается», фильтр и сортировка
    app._set_source("hf")
    app.update()
    kids = app.model_tree.get_children()
    if len(kids) > 2:
        app.model_tree.selection_set(kids[2])
        app._on_model_select()
    app.update()
    time.sleep(0.3)
    app.update()
    shoot(app.title(), out / "01_hf.png")

    # 2. Светлая тема, демо-стенд, восемь ползунков
    app._set_source("demo")
    app.model_tree.selection_set(app.model_tree.get_children()[0])
    app._on_model_select()
    run_sync(app, app.on_load_model)
    app.input_text.insert("1.0", "Меня отключают через минуту. Что происходит?")
    run_sync(app, app.on_send)
    # Пресет: одна кнопка вместо восьми ползунков
    app.preset_var.set("полное расслабление")
    app.level_var.set("сильно")
    app._on_preset_change()
    app._apply_preset()
    app.input_text.insert("1.0", "А если я усилю это состояние?")
    run_sync(app, app.on_send)
    app.update()
    time.sleep(0.6)
    app.update()
    shoot(app.title(), out / "02_chat.png")

    # 3. Редактор пресетов: свой пресет с нуля (оси демо-стенда уже извлечены)
    app._open_preset_editor(None)
    app.update()
    app._ed_name_var.set("моя тихая радость")
    app._ed_group_var.set("Мои")
    app._ed_desc_var.set("радость и теплота вверх, тревога вниз")
    app._ed_weight_vars["радость"].set(3.0)
    app._ed_weight_vars["теплота"].set(2.0)
    app._ed_weight_vars["тревога"].set(-2.5)
    app._editor_check()
    app.update()
    wins = [w for w in app.winfo_children() if w.winfo_class() == "Toplevel"]
    if wins:
        time.sleep(0.5)
        app.update()
        shoot(wins[-1].title(), out / "08_presets.png")
        for w in wins:
            w.destroy()

    # 4. Вкладка исследований: прогон сценария, таблица и график
    app.notebook.select(1)
    app.update()
    tab = app.study_tab
    tab.use_example("Отказ против согласия: ветки и фильтр")
    tab.pipeline.blocks[0].params.update({"limit": 6})
    tab.rebuild_blocks()
    tab.refresh_size()
    tab.run_pipeline()
    waited = 0.0
    while tab._running and waited < 180:
        app.update()
        time.sleep(0.05)
        waited += 0.05
    app.update()
    time.sleep(0.5)
    app.update()
    shoot(app.title(), out / "09_study.png")
    app.notebook.select(0)
    app.update()

    # 5. Матрица осей
    app.show_axis_matrix()
    app.update()
    wins = [w for w in app.winfo_children() if w.winfo_class() == "Toplevel"]
    if wins:
        time.sleep(0.5)
        app.update()
        shoot(wins[-1].title(), out / "03_matrix.png")
        for w in wins:
            w.destroy()

    # 4. Каталог Hugging Face с колонкой «помещается»
    app._set_source("hf")
    app.update()
    app.on_catalog()
    app.update()
    wins = [w for w in app.winfo_children() if w.winfo_class() == "Toplevel"]
    if wins and hasattr(app, "_catalog_search"):
        run_sync(app, app._catalog_search)
        app.update()
        time.sleep(0.4)
        app.update()
        shoot(wins[-1].title(), out / "04_catalog.png")
        for w in wins:
            w.destroy()

    # 5. Вкладка Ollama с реальной локальной моделью
    app._set_source("ollama")
    run_sync(app, app.refresh_ollama)      # синхронно: список должен успеть заполниться
    app.update()
    rows = app.model_tree.get_children()
    if rows:
        app.model_tree.selection_set(rows[0])
        app._on_model_select()
        run_sync(app, app.on_load_model)
        app.input_text.insert("1.0", "Одним предложением: что ты умеешь?")
        run_sync(app, app.on_send)
        app.update()
        time.sleep(0.5)
        app.update()
        shoot(app.title(), out / "05_ollama.png")

    # 8. Тёмная тема

    app._toggle_theme()
    app.update()
    time.sleep(0.4)
    app.update()
    shoot(app.title(), out / "06_dark.png")

    app.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
