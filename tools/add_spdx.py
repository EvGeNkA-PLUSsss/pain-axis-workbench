# SPDX-FileCopyrightText: 2026 EvGeNkA-PLUSsss
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Добавить SPDX-заголовки AGPL во все исходники проекта (идемпотентно).

    python tools/add_spdx.py            # показать, что изменится
    python tools/add_spdx.py --write    # применить

Заголовок вставляется в самое начало файла как комментарий, поэтому строки
``#!/usr/bin/env python`` и ``# -*- coding:`` сохраняют первое место, а докстринг
модуля остаётся первым оператором и продолжает работать как ``__doc__``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ("# SPDX-FileCopyrightText: 2026 EvGeNkA-PLUSsss\n"
          "# SPDX-License-Identifier: AGPL-3.0-or-later\n")
SKIP_DIRS = {".git", "build", "dist", "release", "cache", "vectors", "__pycache__",
             ".venv", "venv"}
KEEP_FIRST = ("#!", "# -*- coding", "# coding=")


def targets() -> list[Path]:
    out = []
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        out.append(path)
    return out


def add_header(path: Path, write: bool) -> bool:
    """True, если заголовок уже есть или был добавлен."""
    text = path.read_text(encoding="utf-8")
    if "SPDX-License-Identifier" in text[:400]:
        return False
    lines = text.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines[:3]):
        if line.startswith(KEEP_FIRST):
            insert_at = i + 1
            break
    new = "".join(lines[:insert_at]) + HEADER + ("\n" if insert_at else "") + \
        "".join(lines[insert_at:])
    if write:
        path.write_text(new, encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="SPDX-заголовки в исходниках")
    ap.add_argument("--write", action="store_true", help="применить изменения")
    args = ap.parse_args()
    changed = [p for p in targets() if add_header(p, args.write)]
    verb = "обновлено" if args.write else "будет обновлено"
    print(f"{verb}: {len(changed)} из {len(targets())} файлов")
    for p in changed[:8]:
        print("  " + str(p.relative_to(ROOT)))
    if len(changed) > 8:
        print(f"  … и ещё {len(changed) - 8}")
    if not args.write and changed:
        print("запустите с --write, чтобы применить")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
