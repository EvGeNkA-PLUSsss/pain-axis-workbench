# SPDX-FileCopyrightText: 2026 EvGeNkA-PLUSsss
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Запуск графического интерфейса: python run_gui.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pain_axis.ui.tk_app import main  # noqa: E402

if __name__ == "__main__":
    main()
