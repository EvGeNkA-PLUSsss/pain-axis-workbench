# SPDX-FileCopyrightText: 2026 EvGeNkA-PLUSsss
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Бэкенд на реальных весах через transformers.

Особенности, которые важно знать:

* Слои находятся автоматически. У разных семейств (Llama/Qwen/Gemma/Mistral,
  GPT-2, Pythia, Phi) residual stream лежит в разных местах, поэтому сначала
  проверяются известные пути, а затем берётся самая крупная ``nn.ModuleList`` —
  это надёжный признак списка блоков трансформера.
* Стиринг делается forward-хуком на выход выбранного блока: именно там и живёт
  residual stream. Хук работает и при префилле, и при пошаговой генерации с
  KV-кэшем, поэтому изменение действует на все новые токены.
* ``alpha`` умножается на измеренную среднюю норму активаций на этом слое
  (``scale``), так что alpha ≈ 1 — это добавка порядка самого сигнала, а не
  произвольные «попугаи». Если вектор загружен без scale, alpha идёт в сырых
  единицах hidden state — интерфейс об этом предупреждает.
* Память: скрытые состояния немедленно переносятся на CPU во float32.
"""

from __future__ import annotations

import importlib.util
from typing import Sequence

import numpy as np

from ..registry import ModelSpec
from .base import Backend, GenResult, ProgressFn, SteerSpec, pick_axis

try:  # torch нужен только этому бэкенду
    import torch
    import torch.nn as nn

    _TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH = False


def torch_available() -> bool:
    return _TORCH


def bitsandbytes_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


def cuda_available() -> bool:
    return bool(_TORCH and torch.cuda.is_available())


_LAYER_PATHS = (
    "model.layers",
    "model.model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "transformer.h",
    "model.transformer.h",
    "gpt_neox.layers",
    "model.gpt_neox.layers",
    "model.decoder.layers",
    "decoder.layers",
    "model.model.decoder.layers",
)


def find_decoder_layers(model) -> tuple[list, str]:
    for path in _LAYER_PATHS:
        obj = model
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, list, tuple)) and len(obj) >= 2:
            return list(obj), path

    best_path, best_len = None, -1
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 4:
            n = sum(p.numel() for p in mod.parameters())
            if n > best_len:
                best_path, best_len = name, n
    if best_path is None:
        raise RuntimeError("не удалось найти блоки трансформера в этой архитектуре")
    obj = model
    for part in best_path.split("."):
        obj = getattr(obj, part)
    return list(obj), best_path


def read_geometry(model) -> tuple[int, int]:
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    n_layers = None
    hidden = None
    for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        if cfg is not None and hasattr(cfg, attr):
            n_layers = int(getattr(cfg, attr))
            break
    for attr in ("hidden_size", "n_embd", "d_model", "embed_dim"):
        if cfg is not None and hasattr(cfg, attr):
            hidden = int(getattr(cfg, attr))
            break
    if n_layers is None or hidden is None:
        layers, _ = find_decoder_layers(model)
        n_layers = n_layers or len(layers)
        hidden = hidden or int(getattr(model, "get_input_embeddings")().weight.shape[1])
    return int(n_layers), int(hidden)


class _SteerHook:
    """Добавляет/убирает компоненту оси в residual stream и пишет её проекцию."""

    def __init__(self, delta: np.ndarray, mode: str, unit: np.ndarray,
                 alpha: float, record_axis: np.ndarray | None):
        self.delta = torch.tensor(delta, dtype=torch.float32)
        self.mode = mode
        self.alpha = float(alpha)
        self.unit = torch.tensor(unit, dtype=torch.float32)
        self.record_axis = None if record_axis is None else torch.tensor(record_axis, dtype=torch.float32)
        self.trace: list[float] = []

    def _like(self, t, ref):
        """Привести тензор к устройству и типу активаций (а не к своим собственным).

        Раньше здесь была ошибка: приведение шло «к самому себе», поэтому на GPU
        дельта оставалась на CPU и падала с «found at least two devices».
        На CPU ошибка не проявлялась — отсюда важность прогона на реальном устройстве.
        """
        return t.to(device=ref.device, dtype=ref.dtype)

    def _apply(self, h):
        delta = self._like(self.delta, h)
        if self.mode == "ablate":
            u = self._like(self.unit, h)
            coef = (h @ u).unsqueeze(-1)
            h = h - self.alpha * coef * u
        else:
            h = h + delta
        if self.record_axis is not None:
            a = self._like(self.record_axis, h)
            self.trace.append(float((h[0, -1, :] @ a).detach().float().cpu()))
        return h

    def __call__(self, module, args, output):
        if isinstance(output, tuple):
            return (self._apply(output[0]),) + tuple(output[1:])
        return self._apply(output)


class HFBackend(Backend):
    is_real = True

    def __init__(self, spec: ModelSpec, load_in_4bit: bool = False, device: str = "auto"):
        super().__init__(spec)
        self.load_in_4bit = bool(load_in_4bit)
        self.requested_device = device
        self.model = None
        self.tokenizer = None
        self.layers: list = []
        self.layer_path: str = ""
        self._quantized = False
        self._cpu_offload = False

    # ------------------------------------------------------------ загрузка
    def load(self, progress: ProgressFn = None, trust_remote_code: bool | None = None, **kwargs) -> None:
        if not _TORCH:
            raise RuntimeError(
                "torch не установлен в этом интерпретаторе Python. "
                "Запустите приложение интерпретатором, где есть torch+transformers."
            )
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._quantized = False
        self._cpu_offload = False

        trc = self.spec.trust_remote_code if trust_remote_code is None else bool(trust_remote_code)

        self._report(progress, f"Токенизатор: {self.spec.repo_id}", 0.05)
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.repo_id, trust_remote_code=trc)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        # «CPU» важнее «4 бита»: иначе квантизация тянула бы модель на GPU через
        # device_map="auto", и галочка CPU молча ничего не делала бы.
        if self.load_in_4bit and self.requested_device == "cpu":
            self._report(progress, "Выбран CPU: 4 бита не применяются, грузим float32 "
                                   "(понадобится много оперативной памяти)", 0.12)
            self.load_in_4bit = False

        dtype, device_map = self._pick_dtype()
        self._report(progress, f"Веса: dtype={dtype}, device_map={device_map or 'нет'}", 0.15)

        call_kwargs: dict = {"trust_remote_code": trc}
        if device_map:
            call_kwargs["device_map"] = device_map

        quant_cfg = None
        if self.load_in_4bit:
            if not bitsandbytes_available():
                self._report(progress, "bitsandbytes не найден — 4 бита невозможны, грузим в обычной точности", 0.2)
                self.load_in_4bit = False
            else:
                need = self.spec.vram_4bit_gb
                free = self._free_vram_gb()
                if need and free is not None and free < need + 0.4:
                    self._report(
                        progress,
                        f"Внимание: свободно {free:.1f} ГБ VRAM, модели нужно ~{need:.1f} ГБ. "
                        f"Возможно, часть слоёв уедет на CPU — будет медленнее.",
                        0.18,
                    )
                quant_cfg = self._quant_config(cpu_offload=False)
                call_kwargs["quantization_config"] = quant_cfg
                call_kwargs["device_map"] = device_map or "auto"
                self._quantized = True

        if quant_cfg is None:
            try:
                call_kwargs["dtype"] = dtype
                model = AutoModelForCausalLM.from_pretrained(self.spec.repo_id, **call_kwargs)
            except TypeError:
                call_kwargs.pop("dtype", None)
                call_kwargs["torch_dtype"] = dtype
                model = AutoModelForCausalLM.from_pretrained(self.spec.repo_id, **call_kwargs)
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(self.spec.repo_id, **call_kwargs)
            except ValueError as exc:
                if not self._is_offload_error(exc):
                    raise
                model = self._retry_with_cpu_offload(progress, exc, call_kwargs)

        if quant_cfg is None and device_map is None:
            model = model.to(self._device_target())
        model.eval()

        self.model = model
        self.layers, self.layer_path = find_decoder_layers(model)
        self.n_layers, self.hidden_size = read_geometry(model)
        self.device = str(next(model.parameters()).device)
        if self._quantized:
            self.dtype_str = "4bit-nf4+CPU" if self._cpu_offload else "4bit-nf4"
        else:
            self.dtype_str = str(dtype).replace("torch.", "")
        self.loaded = True
        self._report(
            progress,
            f"Загружено: {self.n_layers} слоёв, hidden={self.hidden_size}, "
            f"блоки найдены по пути «{self.layer_path}», устройство {self.device}"
            + (", все слои на CPU (медленнее)" if self._cpu_offload else ""),
            1.0,
        )

    # ----------------------------------------------------- память и устройство
    def _quant_config(self, cpu_offload: bool):
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if cuda_available() else torch.float32,
            # без этого флага accelerate не имеет права выгружать квантизованные
            # модули на CPU и падает с «Some modules are dispatched on the CPU»
            llm_int8_enable_fp32_cpu_offload=bool(cpu_offload),
        )

    @staticmethod
    def _free_vram_gb() -> float | None:
        try:
            if not cuda_available():
                return None
            return torch.cuda.mem_get_info()[0] / 1024 ** 3
        except Exception:
            return None

    @staticmethod
    def _total_ram_gb() -> float:
        """Объём оперативной памяти (Windows — через GlobalMemoryStatusEx)."""
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
                return st.ullTotalPhys / 1024 ** 3
        except Exception:
            pass
        return 8.0

    def _max_memory_map(self) -> dict:
        """Явные лимиты для device_map (оставлено для диагностики)."""
        free = self._free_vram_gb()
        gpu_budget = max(1.0, (free - 0.5)) if free is not None else 0.0
        ram_budget = max(4.0, self._total_ram_gb() * 0.75)
        return {0: f"{gpu_budget:.1f}GiB", "cpu": f"{ram_budget:.0f}GiB"}

    @staticmethod
    def _clear_memory() -> None:
        import gc

        gc.collect()
        if _TORCH and cuda_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass

    @staticmethod
    def _is_offload_error(exc: Exception) -> bool:
        text = str(exc)
        return ("dispatched on the CPU or the disk" in text
                or "fp32_cpu_offload" in text
                or "enough GPU RAM" in text)

    def _retry_with_cpu_offload(self, progress: ProgressFn, exc: Exception,
                                call_kwargs: dict):
        """Загрузить модель целиком на CPU в 4 битах, если в VRAM она не влезла.

        Почему не «часть на GPU, часть на CPU»: в связке transformers 5 + accelerate
        1.15 + bitsandbytes распределение квантизованной модели по двум устройствам
        падает с «Tensor.item() cannot be called on meta tensors». Проверено на
        этой машине: `device_map="auto"` с `llm_int8_enable_fp32_cpu_offload=True`
        и с `max_memory`, и без него — обе попытки падают. А полностью на CPU в
        4 битах грузится за секунды и работает, поэтому откат детерминированный.

        Цена: считается на процессоре, это медленно. Освободите видеопамять
        (выгрузите модель в другом окне, закройте лишние приложения) и загрузите
        заново, если нужна скорость.
        """
        from transformers import AutoModelForCausalLM

        free = self._free_vram_gb()
        self._report(
            progress,
            f"В VRAM не поместилось (свободно {free:.1f} ГБ). Гружу целиком на CPU "
            f"в 4 битах: работать будет медленно, зато без ошибки.",
            0.45,
        )
        self._clear_memory()
        call_kwargs = dict(call_kwargs)
        call_kwargs["quantization_config"] = self._quant_config(cpu_offload=True)
        call_kwargs["device_map"] = {"": "cpu"}
        call_kwargs.pop("max_memory", None)
        try:
            model = AutoModelForCausalLM.from_pretrained(self.spec.repo_id, **call_kwargs)
        except Exception as exc2:  # noqa: BLE001
            self._clear_memory()
            raise RuntimeError(
                "Не удалось загрузить модель: в видеопамяти не помещается, а на CPU "
                f"тоже не вышло ({type(exc2).__name__}: {exc2}).\n"
                f"Свободно VRAM: {free:.1f} ГБ, оперативной памяти: "
                f"{self._total_ram_gb():.0f} ГБ.\n"
                "Что сделать: выгрузить модель в другом окне приложения, закрыть "
                "программы, занимающие видеопамять, взять модель меньшего размера "
                "(до 3B в bf16 или до 7B в 4 битах) или включить галочку «CPU»."
            ) from exc2
        self._cpu_offload = True
        return model

    def _device_target(self):
        if self.requested_device == "cpu":
            return "cpu"
        if cuda_available():
            return "cuda"
        return "cpu"

    def _pick_dtype(self):
        """(dtype, device_map) с оглядкой на объём VRAM."""
        if self.load_in_4bit and bitsandbytes_available():
            return (torch.bfloat16 if cuda_available() else torch.float32), "auto"
        if self.requested_device == "cpu" or not cuda_available():
            return torch.float32, None
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        need = self.spec.vram_fp16_gb
        if need > total_gb * 0.85:
            return torch.float32, None  # явно не влезает целиком — уйдёт на CPU
        return (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16), None

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.layers = []
        self.loaded = False
        if _TORCH and cuda_available():
            torch.cuda.empty_cache()
        import gc

        gc.collect()

    # -------------------------------------------------------------- кодирование
    def _has_chat(self) -> bool:
        return bool(getattr(self.tokenizer, "chat_template", None))

    def _format(self, text: str, use_chat: bool) -> str:
        if use_chat and self._has_chat():
            msgs = [{"role": "user", "content": text}]
            try:
                return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                return text
        return text

    def _encode(self, text: str, use_chat: bool):
        s = self._format(text, use_chat)
        return self.tokenizer(s, return_tensors="pt")

    def conversation_prompt(self, messages: Sequence[dict]) -> str:
        """Собрать промпт из истории переписки.

        Приоритет — родной чат-шаблон модели. Если его нет (GPT-2, Pythia и другие
        базовые модели), переписка склеивается в формате «User:/Assistant:» —
        так модель продолжает диалог как обычный текст.
        """
        msgs = [{"role": m.get("role", "user"), "content": str(m.get("content", ""))}
                for m in messages if str(m.get("content", "")).strip()]
        if not msgs:
            return ""
        if self._has_chat():
            try:
                return self.tokenizer.apply_chat_template(msgs, tokenize=False,
                                                          add_generation_prompt=True)
            except Exception:
                pass
        lines = []
        for m in msgs:
            who = {"user": "User", "assistant": "Assistant", "system": "System"}.get(m["role"], "User")
            lines.append(f"{who}: {m['content']}")
        lines.append("Assistant:")
        return "\n".join(lines)

    # ------------------------------------------------------------------- сбор
    def collect(
        self,
        texts: Sequence[str],
        pooling: str = "mean",
        use_chat: bool = False,
        batch_size: int = 4,
        progress: ProgressFn = None,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("модель не загружена")
        tok = self.tokenizer
        out = np.zeros((len(texts), self.n_layers + 1, self.hidden_size), dtype=np.float32)
        bs = max(1, int(batch_size))

        with torch.no_grad():
            for start in range(0, len(texts), bs):
                chunk = [self._format(t, use_chat) for t in texts[start:start + bs]]
                enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
                enc = {k: v.to(self.model.device) for k, v in enc.items()}
                res = self.model(**enc, output_hidden_states=True, use_cache=False)
                hs = res.hidden_states  # tuple: (n_layers + 1) x (batch, seq, hidden)
                mask = enc["attention_mask"]
                for b in range(len(chunk)):
                    m = mask[b].bool()
                    for l, h in enumerate(hs):
                        hb = h[b].float()
                        if pooling == "last":
                            idx = int(m.nonzero()[-1])
                            vec = hb[idx]
                        else:
                            vec = hb[m].mean(dim=0)
                        out[start + b, l] = vec.detach().cpu().numpy()
                if progress:
                    done = min(start + bs, len(texts))
                    self._report(progress, f"Прямых проходов: {done}/{len(texts)}", done / max(1, len(texts)))
        return out

    # ---------------------------------------------------------------- генерация
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 60,
        temperature: float = 0.0,
        top_p: float = 0.95,
        steer: SteerSpec | None = None,
        axis: np.ndarray | None = None,
        use_chat: bool = True,
        progress: ProgressFn = None,
        messages: Sequence[dict] | None = None,
    ) -> GenResult:
        if self.model is None:
            raise RuntimeError("модель не загружена")
        if messages:
            text_in = self.conversation_prompt(messages)
            enc = {k: v.to(self.model.device) for k, v in
                   self.tokenizer(text_in, return_tensors="pt").items()}
            shown_prompt = messages[-1].get("content", "") if messages else prompt
        else:
            enc = {k: v.to(self.model.device) for k, v in self._encode(prompt, use_chat).items()}
            shown_prompt = prompt
        n_prompt = int(enc["input_ids"].shape[1])

        handle, hook = self._install_hook(steer, axis)
        try:
            with torch.no_grad():
                out = self._call_generate(enc, max_new_tokens, temperature, top_p)
        finally:
            if handle is not None:
                handle.remove()

        new_ids = out[0][n_prompt:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return GenResult(
            text=text,
            prompt=shown_prompt,
            n_new_tokens=int(new_ids.shape[0]),
            trace=list(hook.trace) if hook is not None else [],
            steered=steer is not None,
            alpha=float(steer.alpha) if steer else 0.0,
            layer=int(steer.layer) if steer else -1,
        )

    def _call_generate(self, enc: dict, max_new_tokens: int, temperature: float, top_p: float):
        """Генерация через копию generation_config.

        Так надёжнее, чем передавать флаги аргументами: в transformers 5 часть
        аргументов generate() помечается как невалидная и может игнорироваться,
        а часть конфликтует между собой (max_length против max_new_tokens).
        Если вариант с конфигом не поддерживается — откат на аргументы.
        """
        import copy

        base_cfg = getattr(self.model, "generation_config", None)
        if base_cfg is not None:
            try:
                cfg = copy.deepcopy(base_cfg)
                cfg.max_new_tokens = int(max_new_tokens)
                cfg.max_length = None
                cfg.do_sample = bool(temperature and temperature > 0)
                if cfg.do_sample:
                    cfg.temperature = float(temperature)
                    cfg.top_p = float(top_p)
                else:
                    # Гасим оставшиеся в конфиге модели параметры сэмплинга: при
                    # do_sample=False transformers 5 считает их невалидными и ругается.
                    cfg.temperature = None
                    cfg.top_p = None
                    cfg.top_k = None
                pid = self.tokenizer.pad_token_id
                if isinstance(pid, int) and 0 <= pid < len(self.tokenizer):
                    cfg.pad_token_id = pid
                eid = self.tokenizer.eos_token_id
                if isinstance(eid, int) and 0 <= eid < len(self.tokenizer):
                    cfg.eos_token_id = eid
                return self.model.generate(**enc, generation_config=cfg)
            except Exception:
                pass

        kwargs: dict = {"max_new_tokens": int(max_new_tokens)}
        pid = self.tokenizer.pad_token_id
        if isinstance(pid, int) and 0 <= pid < len(self.tokenizer):
            kwargs["pad_token_id"] = pid
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=float(temperature), top_p=float(top_p))
        else:
            kwargs.update(do_sample=False)
        return self.model.generate(**enc, **kwargs)

    # ------------------------------------------------------------------- проба
    def score_continuations(
        self,
        prompt: str,
        continuations: Sequence[str],
        steer: SteerSpec | None = None,
        use_chat: bool = False,
    ) -> list[float]:
        if self.model is None:
            raise RuntimeError("модель не загружена")
        tok = self.tokenizer
        prefix = self._format(prompt, use_chat)
        handle, _ = self._install_hook(steer, None)
        scores: list[float] = []
        try:
            with torch.no_grad():
                for cont in continuations:
                    if not cont.startswith(" ") and not prefix.endswith((" ", "\n")):
                        cont = " " + cont
                    p_ids = tok(prefix, return_tensors="pt", add_special_tokens=True)["input_ids"]
                    f_ids = tok(prefix + cont, return_tensors="pt", add_special_tokens=True)["input_ids"]
                    n_pre = min(int(p_ids.shape[1]), int(f_ids.shape[1]) - 1)
                    ids = f_ids.to(self.model.device)
                    logits = self.model(ids, use_cache=False).logits.float()
                    logprobs = torch.log_softmax(logits[0, n_pre - 1:-1, :], dim=-1)
                    targets = ids[0, n_pre:]
                    lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                    scores.append(float(lp.mean().detach().cpu()) if lp.numel() else float("-inf"))
        finally:
            if handle is not None:
                handle.remove()
        return scores

    # ------------------------------------------------------------------ служебное
    def _install_hook(self, steer: SteerSpec | None, axis: np.ndarray | None):
        if steer is None and axis is None:
            return None, None
        # Нумерация слоёв в интерфейсе: блок 1..N. Он соответствует
        # hidden_states[L] и хуку на layers[L - 1].
        layer = int(steer.layer) if steer else 0
        layer = max(1, min(layer, len(self.layers)))
        idx = layer - 1
        unit = steer.vector if steer is not None else np.zeros(self.hidden_size, dtype=np.float32)
        delta = steer.delta(self.hidden_size) if (steer is not None and steer.mode == "add") else np.zeros(self.hidden_size, dtype=np.float32)
        mode = steer.mode if steer is not None else "add"
        alpha = steer.alpha if steer is not None else 0.0
        hook = _SteerHook(delta, mode, np.asarray(unit, dtype=np.float32), alpha, pick_axis(axis, layer))
        handle = self.layers[idx].register_forward_hook(hook)
        return handle, hook
