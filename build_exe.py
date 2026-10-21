# SPDX-FileCopyrightText: 2026 EvGeNkA-PLUSsss
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Сборка одиночного exe: интерфейс + numpy внутри, torch подключается снаружи.

    python build_exe.py                 # обычная сборка (~40-60 МБ)
    python build_exe.py --verify        # собрать и прогнать selfcheck внутри exe
    python build_exe.py --with-torch    # тяжёлый вариант: torch внутрь (см. ниже)

Про `--with-torch`: torch с CUDA занимает больше 3 ГБ. В onefile-сборке он
распаковывается в temp при каждом запуске, а CUDA-DLL внутри архива обычно не
находятся — на выходе получается exe, который запускается десятки секунд и часто
падает на `import torch`. Поэтому по умолчанию torch НЕ включается: exe умеет
взять его из уже установленного на машине Python (pain_axis/torch_bridge.py).
Флаг `--with-torch` оставлен для случая, когда нужен полностью автономный exe и
не жалко размера/времени.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_NAME = "PainAxis"
ENTRY = ROOT / "app_main.py"
ICON = ROOT / "assets" / "pain_axis.ico"

# Всё, что exe НЕ должен тащить в себя: либо слишком тяжёлое, либо подключается
# снаружи (torch/transformers), либо не нужно интерфейсу вовсе.
EXCLUDES = [
    "torch", "torchvision", "torchaudio", "torchgen",
    "transformers", "tokenizers", "safetensors", "huggingface_hub", "bitsandbytes",
    "accelerate", "peft", "datasets", "sentencepiece", "einops",
    "scipy", "sklearn", "pandas", "matplotlib", "sympy", "networkx",
    "PIL", "IPython", "notebook", "jupyter", "pytest", "sphinx", "docutils",
    "PyQt5", "PySide2", "PySide6", "wx",
]


def make_icon(path: Path) -> bool:
    """Нарисовать иконку: градиентная шкала боли и стрелка ползунка."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return False
    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((8, 8, size - 8, size - 8), radius=48, fill=(28, 40, 56, 255))
    left, right, top, bottom = 36, size - 36, 132, 176
    for x in range(left, right):
        t = (x - left) / max(1, right - left)          # 0 — подавить, 1 — усилить
        if t < 0.5:
            k = t / 0.5
            col = (int(70 + 90 * k), int(130 + 40 * k), int(200 - 30 * k), 255)
        else:
            k = (t - 0.5) / 0.5
            col = (int(160 + 70 * k), int(170 - 110 * k), int(170 - 120 * k), 255)
        d.line((x, top, x, bottom), fill=col)
    d.rounded_rectangle((left - 4, top - 4, right + 4, bottom + 4), radius=10,
                        outline=(240, 245, 250, 255), width=4)
    cx = left + int((right - left) * 0.78)
    d.polygon([(cx, 92), (cx - 24, 52), (cx + 24, 52)], fill=(255, 255, 255, 255))
    d.line((cx, 52, cx, 84), fill=(255, 255, 255, 255), width=8)
    d.line((left - 4, 208, right + 4, 208), fill=(150, 165, 180, 255), width=4)
    d.line((size // 2, 198, size // 2, 218), fill=(220, 230, 240, 255), width=5)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return True


def build(with_torch: bool = False, clean: bool = True, console: bool = False,
          name: str = APP_NAME) -> Path:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onefile",
        "--console" if console else "--windowed",
        "--name", name,
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build" / "pyinstaller"),
        "--specpath", str(ROOT / "build"),
        "--paths", str(ROOT),
    ]
    if clean:
        cmd.append("--clean")
    if ICON.exists():
        cmd += ["--icon", str(ICON)]
    if not with_torch:
        for mod in EXCLUDES:
            cmd += ["--exclude-module", mod]
    else:
        # даже в тяжёлом варианте не тащим лишние GUI-тултики
        for mod in ("PyQt5", "PySide2", "PySide6", "wx", "matplotlib", "pytest"):
            cmd += ["--exclude-module", mod]
    cmd.append(str(ENTRY))

    print("Сборка:\n  " + " ".join(cmd) + "\n")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"PyInstaller завершился с кодом {proc.returncode}")
    exe = ROOT / "dist" / f"{name}.exe"
    print(f"\nСобрано за {time.time() - t0:.0f} с: {exe} "
          f"({exe.stat().st_size / 1024 / 1024:.1f} МБ)")
    return exe


def verify(exe: Path) -> bool:
    """Прогнать selfcheck внутри собранного exe (окно при этом не открывается)."""
    report = ROOT / "build" / "selfcheck_report.json"
    if report.exists():
        report.unlink()
    print(f"Проверка: {exe.name} --selfcheck --torch")
    proc = subprocess.run([str(exe), "--selfcheck", "--torch", "--report", str(report)],
                          cwd=str(ROOT))
    print(f"код возврата: {proc.returncode}")
    if not report.exists():
        print("отчёт не создан — exe не смог записать файл")
        return False
    data = json.loads(report.read_text(encoding="utf-8"))
    for c in data["checks"]:
        print(f"  [{'OK  ' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print(f"итог: {data['passed']}/{data['total']}, frozen={data['frozen']}, "
          f"внешний torch: {data['torch_bridge'].get('site')}")
    return proc.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Сборка PainAxis.exe")
    ap.add_argument("--with-torch", action="store_true",
                    help="включить torch в exe (многогигабайтный и медленный вариант)")
    ap.add_argument("--verify", action="store_true", help="после сборки прогнать selfcheck")
    ap.add_argument("--no-icon", action="store_true")
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--debug-console", action="store_true",
                    help="собрать консольную версию PainAxisDebug.exe для отладки")
    a = ap.parse_args()

    if not ENTRY.exists():
        raise SystemExit(f"нет точки входа {ENTRY}")
    if not a.no_icon and not ICON.exists():
        print("иконка:", "создана" if make_icon(ICON) else "Pillow недоступен, собираю без иконки")
    if a.no_icon:
        ICON.unlink(missing_ok=True)

    exe = build(with_torch=a.with_torch, clean=not a.no_clean,
                console=a.debug_console,
                name="PainAxisDebug" if a.debug_console else APP_NAME)
    if a.verify:
        return 0 if verify(exe) else 1
    print("\nЗапуск: " + str(exe))
    print("Проверка: " + str(exe) + " --selfcheck --torch --report build\\selfcheck_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
