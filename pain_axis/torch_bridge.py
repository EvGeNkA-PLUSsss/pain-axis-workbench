"""Подключение внешнего torch к замороженному (frozen) приложению.

Зачем
-----
torch с CUDA занимает больше 3 ГБ. Собрать его внутрь onefile-exe означает:
распаковку трёх гигабайт в temp при каждом запуске, сломанные CUDA-DLL и exe,
который нельзя ни передать, ни запустить быстро. Поэтому exe содержит только
интерфейс и numpy (демо-стенд работает автономно), а torch и transformers
подключаются в рантайме из уже установленного на машине Python.

Что именно приходится подключать
--------------------------------
Замороженное приложение — это обычный CPython той же версии, что и собранный exe
(здесь 3.10), поэтому расширения под cp310 (а torch именно такой) из него
загружаются. Но одного `site-packages` мало: PyInstaller кладёт в бандл только те
модули стандартной библиотеки, которые нужны самому приложению. torch же
обращается к `timeit`, `pickle`-соседям и прочему, чего в бандле нет, и падает с
`ModuleNotFoundError: No module named 'timeit'`. Поэтому для найденного
интерпретатора добавляются три каталога **в конец** sys.path:

1. ``<root>\\Lib\\site-packages`` — сам torch и его зависимости;
2. ``<root>\\Lib`` — стандартная библиотека того же Python (берётся только то,
   чего не хватает: бандл exe стоит в sys.path раньше);
3. ``<root>\\DLLs`` — C-расширения стандартной библиотеки.

Плюс ``os.add_dll_directory`` на ``torch\\lib`` и корень: на Windows зависимости
расширений ищутся рядом с ними только при явной регистрации каталога.

Порядок в sys.path важен: сначала идёт бандл (своя numpy и свои модули), внешние
каталоги — в конце, поэтому конфликтов двух копий numpy не возникает. Версия
внешнего Python сверяется по наличию ``python3XX.dll``: чужой ABI не подойдёт.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

_CANDIDATE_SITE_GLOBS = (
    r"%APPDATA%\Python\Python3*\site-packages",          # pip install --user
    r"%APPDATA%\Python\Python3*\x64\site-packages",
    r"%LOCALAPPDATA%\Programs\Python\Python3*\Lib\site-packages",
    r"C:\Program Files\Python3*\Lib\site-packages",
    r"C:\Program Files (x86)\Python3*\Lib\site-packages",
    r"C:\Python3*\Lib\site-packages",
    r"%USERPROFILE%\anaconda3\Lib\site-packages",
    r"%USERPROFILE%\miniconda3\Lib\site-packages",
    r"C:\ProgramData\anaconda3\Lib\site-packages",
    r"C:\ProgramData\miniconda3\Lib\site-packages",
    r".venv\Lib\site-packages",
    r"venv\Lib\site-packages",
)

_CANDIDATE_ROOT_GLOBS = (
    r"%LOCALAPPDATA%\Programs\Python\Python3*",
    r"C:\Program Files\Python3*",
    r"C:\Program Files (x86)\Python3*",
    r"C:\Python3*",
    r"%USERPROFILE%\anaconda3",
    r"%USERPROFILE%\miniconda3",
    r"C:\ProgramData\anaconda3",
    r"C:\ProgramData\miniconda3",
)

_state: dict = {"active": False, "site": None, "error": None, "tried": 0,
                "root": None, "attempts": []}


def status() -> dict:
    """Что известно про подключение внешнего torch."""
    return dict(_state)


def _is_matching_root(root: Path) -> bool:
    """Похож ли каталог на установку Python той же версии, что и этот процесс."""
    major, minor = sys.version_info[:2]
    if not (root / "Lib" / "os.py").exists():
        return False
    return ((root / f"python{major}{minor}.dll").exists()
            or (root / "DLLs" / f"python{major}{minor}.dll").exists()
            or any(root.glob(f"python{major}{minor}.dll")))


def _root_for_site(site_dir: Path) -> Path | None:
    """Определить корень интерпретатора для каталога site-packages.

    ``...\\Python310\\Lib\\site-packages`` -> ``...\\Python310``.
    Для пользовательского каталога (``%APPDATA%\\Python\\Python310\\site-packages``)
    корня рядом нет — ищем установку Python той же версии отдельно.
    """
    parent = site_dir.parent
    if parent.name.lower() == "lib":
        root = parent.parent
        if _is_matching_root(root):
            return root
    for pattern in _CANDIDATE_ROOT_GLOBS:
        for hit in glob.glob(os.path.expandvars(pattern)):
            root = Path(hit)
            if _is_matching_root(root) and site_dir.exists():
                return root
    return None


def candidate_sites(explicit: str | None = None) -> list[Path]:
    """Каталоги site-packages, в которых может лежать torch."""
    out: list[Path] = []

    def add(p: str | Path) -> None:
        path = Path(p)
        if path.name.endswith("site-packages") and path.exists() and path not in out:
            out.append(path)

    for env_name in ("PAIN_AXIS_TORCH", "PAIN_AXIS_SITE"):
        val = os.environ.get(env_name)
        if val:
            add(val)
            if val.lower().endswith("python.exe"):
                root = Path(val).parent
                add(root / "Lib" / "site-packages")

    if explicit:
        add(explicit)

    for pattern in _CANDIDATE_SITE_GLOBS:
        for hit in glob.glob(os.path.expandvars(pattern)):
            add(hit)

    try:
        import site as _site

        for p in list(_site.getsitepackages()) + [_site.getusersitepackages()]:
            add(p)
    except Exception:
        pass

    # каталоги с torch — первыми
    out.sort(key=lambda p: 0 if (p / "torch").exists() else 1)
    return out


def _register_dll_dirs(site_dir: Path, root: Path | None) -> None:
    if not hasattr(os, "add_dll_directory"):
        return
    dirs = [site_dir / "torch" / "lib"]
    if root is not None:
        dirs += [root / "DLLs", root]
    for d in dirs:
        if d.exists():
            try:
                os.add_dll_directory(str(d))
            except Exception:
                pass


def _extend_path(site_dir: Path, root: Path | None) -> list[str]:
    """Разложить внешние каталоги по sys.path. Возвращает добавленное.

    Порядок принципиален и разный для двух видов каталогов:

    * ``site-packages`` — **в конец**: numpy и прочее должны браться из бандла exe,
      иначе две копии numpy конфликтуют;
    * ``Lib`` (стандартная библиотека) — **в начало**: PyInstaller кладёт в бандл
      только те модули stdlib, что нужны приложению, и torch упирается в
      отсутствующий ``unittest.mock`` или ``timeit``. Внешняя stdlib имеет ту же
      версию (иначе бы не совпал ABI), поэтому она полнее и безопаснее. Модули
      самого приложения к этому моменту уже импортированы и лежат в sys.modules,
      так что перестановка их не затрагивает.
    """
    added: list[str] = []
    if root is not None:
        for d in (root / "DLLs", root / "Lib"):
            if d.exists() and str(d) not in sys.path:
                sys.path.insert(0, str(d))
                added.append(str(d))
    if str(site_dir) not in sys.path:
        sys.path.append(str(site_dir))
        added.append(str(site_dir))
    return added


def _extend_package_paths(root: Path | None) -> int:
    """Дополнить __path__ уже импортированных пакетов внешней stdlib.

    Нужно для пакетов, которые PyInstaller собрал частично: даже с внешней ``Lib``
    в sys.path подмодуль ищется внутри ``__path__`` бандла. Расширение ``__path__``
    закрывает такие дыры (например, ``unittest`` без ``mock``).
    """
    if root is None:
        return 0
    ext_lib = root / "Lib"
    if not ext_lib.exists():
        return 0
    meipass = getattr(sys, "_MEIPASS", None)
    patched = 0
    for name, mod in list(sys.modules.items()):
        path = getattr(mod, "__path__", None)
        if not path:
            continue
        try:
            paths = list(path)
        except Exception:
            continue
        if meipass and not any(str(p).startswith(str(meipass)) for p in paths):
            continue
        ext = ext_lib.joinpath(*name.split("."))
        if ext.is_dir() and str(ext) not in paths:
            try:
                path.append(str(ext))
                patched += 1
            except Exception:
                pass
    return patched


def _purge_modules(site_dir: Path) -> None:
    """Выбросить частично импортированные модули неудачной попытки.

    Без этого неудачный `import torch` оставляет в sys.modules половину пакета, и
    следующая попытка падает с загадочным «partially initialized module».
    """
    prefix = str(site_dir)
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and str(f).startswith(prefix):
            sys.modules.pop(name, None)


def _try_site(site_dir: Path) -> tuple[bool, str]:
    if not (site_dir / "torch").exists():
        return False, "torch не найден в этом каталоге"
    root = _root_for_site(site_dir)
    added = _extend_path(site_dir, root)
    patched = _extend_package_paths(root)
    _register_dll_dirs(site_dir, root)
    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        _purge_modules(site_dir)
        for s in added:
            try:
                sys.path.remove(s)
            except ValueError:
                pass
        return False, f"{type(exc).__name__}: {exc}"
    gpu = ""
    try:
        if torch.cuda.is_available():
            gpu = f", CUDA: {torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    _state["root"] = str(root) if root else None
    return True, f"torch {torch.__version__}{gpu} (stdlib: {root}, путей дополнено: {patched})"


def activate(explicit: str | None = None, verbose: bool = False) -> str | None:
    """Попытаться подключить внешний torch. Возвращает путь к site-packages или None."""
    if _state["active"]:
        return _state["site"]
    try:
        import torch  # noqa: F401

        _state.update(active=True, site="(уже доступен)", error=None)
        return _state["site"]
    except Exception:
        pass

    sites = candidate_sites(explicit)
    _state["tried"] = len(sites)
    _state["attempts"] = []
    errors: list[str] = []
    for site_dir in sites:
        ok, info = _try_site(site_dir)
        _state["attempts"].append({"site": str(site_dir), "ok": ok, "info": info})
        if verbose:
            try:
                print(f"[torch_bridge] {site_dir}: {'OK' if ok else 'нет'} — {info}")
            except Exception:
                pass
        if ok:
            _state.update(active=True, site=str(site_dir), error=None)
            return str(site_dir)
        if "не найден в этом каталоге" not in info:
            errors.append(f"{site_dir}: {info}")
    _state.update(active=False, site=None, error="; ".join(errors[:3]) or None)
    return None
