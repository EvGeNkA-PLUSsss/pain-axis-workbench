# SPDX-FileCopyrightText: 2026 EvGeNkA-PLUSsss
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Определение железа и оценка того, какая модель на нём поместится.

Зачем: список моделей бесполезен, если не видно, что влезает в вашу видеокарту.
Модуль читает реальные параметры машины (GPU, объём VRAM, свободная VRAM, ОЗУ,
поддержка bf16, наличие bitsandbytes) и считает для каждой модели оценку памяти в
двух режимах — bf16 и 4 бита, — а затем относит её к одной из категорий:

* ``bf16``   — влезает целиком в bf16;
* ``4bit``   — целиком не влезает, но влезает в 4 битах;
* ``4bit-nobnb`` — влезла бы в 4 битах, но bitsandbytes не установлен;
* ``cpu``    — не влезает в VRAM даже в 4 битах.

Оценки намеренно консервативные: помимо весов учитываются контекст, активации и
контекст CUDA (около 1.2 ГБ сверху). Калибровка по замерам на RTX 4060 8 ГБ:
Qwen2.5 1.5B в 4 битах занял 1100 МБ (оценка 1.5 ГБ), Qwen2.5 7B — 5302 МБ
(оценка 5.6 ГБ), то есть оценка скорее завышает, чем занижает.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GB = 1024 ** 3

# overhead: контекст CUDA, буферы активаций, KV-кэш под короткий контекст
_BF16_BYTES_PER_PARAM = 2.05
_4BIT_BYTES_PER_PARAM = 0.58
_OVERHEAD_GB = 1.2
_FIT_MARGIN = 0.9          # не занимаем больше 90% VRAM: остальное нужно системе


@dataclass
class Hardware:
    gpu: str = ""
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    ram_total_gb: float = 0.0
    cuda: bool = False
    bf16: bool = False
    torch: str = ""
    bitsandbytes: bool = False
    note: str = ""

    def summary(self) -> str:
        if not self.cuda:
            return (f"GPU нет — считаем на CPU · ОЗУ {self.ram_total_gb:.0f} ГБ"
                    + ("" if self.torch else " · torch не найден"))
        return (f"{self.gpu} · {self.vram_total_gb:.1f} ГБ VRAM "
                f"(свободно {self.vram_free_gb:.1f}) · ОЗУ {self.ram_total_gb:.0f} ГБ"
                + ("" if self.bf16 else " · bf16 не поддерживается")
                + ("" if self.bitsandbytes else " · bitsandbytes нет: 4 бита недоступны"))

    def json(self) -> dict:
        return {"gpu": self.gpu, "vram_total_gb": round(self.vram_total_gb, 2),
                "vram_free_gb": round(self.vram_free_gb, 2),
                "ram_total_gb": round(self.ram_total_gb, 1), "cuda": self.cuda,
                "bf16": self.bf16, "torch": self.torch,
                "bitsandbytes": self.bitsandbytes, "note": self.note}


def detect() -> Hardware:
    """Собрать параметры машины. Никогда не падает: при отсутствии torch сообщает об этом."""
    hw = Hardware()
    try:
        import importlib.util

        hw.bitsandbytes = importlib.util.find_spec("bitsandbytes") is not None
    except Exception:
        pass
    try:
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = _MemStatus()
        st.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            hw.ram_total_gb = st.ullTotalPhys / GB
    except Exception:
        pass
    try:
        import torch

        hw.torch = torch.__version__
        hw.cuda = bool(torch.cuda.is_available())
        if hw.cuda:
            props = torch.cuda.get_device_properties(0)
            hw.gpu = props.name
            hw.vram_total_gb = props.total_memory / GB
            free, _total = torch.cuda.mem_get_info()
            hw.vram_free_gb = free / GB
            try:
                hw.bf16 = bool(torch.cuda.is_bf16_supported())
            except Exception:
                hw.bf16 = False
    except Exception as exc:  # noqa: BLE001
        hw.note = f"torch недоступен: {type(exc).__name__}"
    return hw


# ---------------------------------------------------------------------------
# Оценки памяти и категории
# ---------------------------------------------------------------------------
def estimate_gb(params_b: float, mode: str) -> float:
    """Оценка занимаемой памяти в ГБ для модели с ``params_b`` миллиардами параметров."""
    if not params_b or params_b <= 0:
        return 0.0
    per_param = _BF16_BYTES_PER_PARAM if mode == "bf16" else _4BIT_BYTES_PER_PARAM
    return params_b * per_param + _OVERHEAD_GB


def fit_of(params_b: float, hw: Hardware, vram_4bit_hint: float = 0.0,
           vram_bf16_hint: float = 0.0) -> tuple[str, float, str]:
    """Категория модели для этой машины: ('bf16'|'4bit'|'4bit-nobnb'|'cpu', нужный ГБ, пояснение)."""
    bf16_gb = vram_bf16_hint or estimate_gb(params_b, "bf16")
    bit4_gb = vram_4bit_hint or estimate_gb(params_b, "4bit")
    if not hw.cuda:
        return "cpu", max(bf16_gb, bit4_gb), "GPU нет: только процессор"
    limit = hw.vram_total_gb * _FIT_MARGIN
    if bf16_gb <= limit:
        return "bf16", bf16_gb, "влезает целиком в bf16"
    if bit4_gb <= limit:
        if not hw.bitsandbytes:
            return "4bit-nobnb", bit4_gb, "нужны 4 бита: установите bitsandbytes"
        return "4bit", bit4_gb, "влезает только в 4 битах"
    if bit4_gb <= hw.ram_total_gb * 0.8:
        return "cpu", bit4_gb, "в VRAM не влезает: пойдёт на CPU (медленно)"
    return "cpu", bit4_gb, "не влезает ни в VRAM, ни в разумную часть ОЗУ"


FIT_LABELS = {"bf16": "bf16", "4bit": "4 бита", "4bit-nobnb": "4 бита (нужен bnb)",
              "cpu": "CPU"}
FIT_ORDER = {"bf16": 0, "4bit": 1, "4bit-nobnb": 2, "cpu": 3}
