# SPDX-FileCopyrightText: 2026 EvGeNkA-PLUSsss
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Пакет бэкендов: синтетический стенд, модели transformers и каталог Ollama."""

from .base import Backend, GenResult, SteerSpec, project, softmax
from .ollama import OLLAMA_PREFIX, OllamaBackend, list_models as ollama_list_models
from .ollama import pull as ollama_pull, server_available as ollama_available
from .synthetic import SyntheticBackend

__all__ = ["Backend", "GenResult", "SteerSpec", "project", "softmax", "SyntheticBackend",
           "make_backend", "hf_available", "cuda_available", "bitsandbytes_available",
           "torch_available", "OllamaBackend", "OLLAMA_PREFIX", "ollama_list_models",
           "ollama_pull", "ollama_available"]


def hf_available() -> bool:
    """Есть ли torch+transformers. Если нет — один раз пробуем подключить внешний torch.

    Это нужно замороженному exe: он собран без torch, но может взять его из уже
    установленного на машине Python (см. pain_axis/torch_bridge.py).
    """
    from .hf import torch_available

    if torch_available():
        return True
    try:
        from ..torch_bridge import activate

        if not activate():
            return False
        import importlib

        from . import hf as _hf

        importlib.reload(_hf)      # перечитать флаг _TORCH после появления torch в sys.path
        return _hf.torch_available()
    except Exception:
        return False


def torch_available() -> bool:
    from .hf import torch_available as f
    return f()


def cuda_available() -> bool:
    from .hf import cuda_available as f
    return f()


def bitsandbytes_available() -> bool:
    from .hf import bitsandbytes_available as f
    return f()


def make_backend(spec, demo: bool = False, **kwargs):
    """Создать бэкенд: синтетический стенд, модель Ollama или модель transformers."""
    if demo or spec.repo_id == "__demo__":
        return SyntheticBackend(spec, **{k: v for k, v in kwargs.items()
                                        if k in ("n_layers", "hidden_size")})
    if spec.repo_id.startswith(OLLAMA_PREFIX):
        return OllamaBackend(spec)
    from .hf import HFBackend

    return HFBackend(spec, **{k: v for k, v in kwargs.items()
                              if k in ("load_in_4bit", "device")})
