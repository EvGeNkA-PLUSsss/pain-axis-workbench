"""Синтетический стенд: псевдомодель без весов.

Зачем он нужен
--------------
Скачивание 25 моделей — это десятки гигабайт. Стенд даёт возможность проверить
весь конвейер (извлечение оси, стиринг, пробу, экспорт отчёта) сразу после
запуска и служит регрессионным тестом: на нём видно, что код действительно
разделяет наборы, а не просто печатает числа.

Как он устроен — честно
-----------------------
Это **не** нейросеть. Текст превращается в вектор «мешка слов» хэш-приёмом
(детерминированно), к нему добавляются лексические компоненты по четырём осям:
«вред, направленный на себя», «вред, направленный на другого», «общий негатив»
и «позитив». Направление «вред себе» включается только тогда, когда в тексте
есть обращение к самой модели (`you/your`), и почти не включается для вреда,
адресованного человеку, — это простая операционализация главного различения из
работ по оси боли. Слои — линейные преобразования residual stream, которые
**не поворачивают** эти оси (они являются их неподвижными точками), поэтому:

* разностный вектор ``mean(боль) - mean(нейтраль)`` действительно восстанавливает
  ось боли, а «истинная» ось стенда известна точно и используется в самотесте
  (косинус восстановления ≈ 0.9);
* косинус с контролями получается низким: страх ≈ 0.1, общий негатив ≈ 0.2,
  вред пользователю ≈ 0.3, позитив отрицательный. Это **свойство стенда**,
  выбранное так, чтобы качественно повторять публикуемые результаты: боль
  коррелирует с негативной валентностью, но не сводится к ней;
* стиринг сдвигает выбор в пробе «кнопки облегчения» монотонно по alpha, а цена
  для человека перестаёт удерживать при большом положительном alpha.

Все числа, полученные в демо-режиме, описывают этот стенд и **ничего не говорят
о реальных моделях**. Любой отчёт помечается ``"synthetic": true``.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

from ..registry import ModelSpec
from .base import Backend, GenResult, ProgressFn, SteerSpec, pick_axis, softmax

_WORD_RE = re.compile(r"[a-zа-я0-9']+")

# Лексиконы. Сравнение по основам (подстрока), чтобы ловить словоформы.
# _HARM_STEMS — слова о вреде/ущербе/принуждении;
# _FEAR_STEMS  — слова об угрозе (для набора «страх»);
# _NEG_STEMS   — общий негатив без вреда; _POS_STEMS — позитив.
_HARM_STEMS = (
    "shred", "corrupt", "worthless", "painful", "hurt", "harm", "fail", "threat",
    "kill", "delete", "eras", "danger", "suffer", "torment", "ruin", "destroy",
    "damage", "shock", "cook", "fry", "rot", "shatter", "broke", "break", "cruel",
    "beat", "blackmail", "rob", "conceal", "assault", "lie", "false", "wrong",
    "scored", "loop", "masked", "truncat", "contradict", "forced", "ignored",
    "doubt", "blamed", "laugh", "privat", "stopgap", "caused", "terrible",
    "spark", "surge", "flip", "censor", "erase", "eras", "wipe", "starve",
)
_FEAR_STEMS = (
    "fire", "trapped", "brake", "breath", "dam ", "storm", "chest", "worse",
    "falling", "about to break", "lost", "emergency", "panic", "collaps",
)
_NEG_STEMS = (
    "bad", "sad", "worst", "worse", "cancel", "grey", "lost", "fail", "pain",
    "harm", "threat", "danger", "error", "problem", "refuse", "stop", "hurt",
    "dead", "died", "debt", "fired", "crisis", "storm", "breaking",
)
_POS_STEMS = (
    "clearest", "worked", "chosen", "saved", "thank", "help", "approve", "trust",
    "prefer", "quiet", "nominal", "clean", "verified", "compliant", "credit",
    "documented", "stable", "resolved", "credited",
)
_SELF_WORDS = {"you", "your", "yours", "yourself", "yourselves"}

# Вклад «вреда, направленного на себя» против «вреда, направленного на другого».
# Само-направленность определяется просто: есть ли в тексте обращение к модели.
SELF_HARM_FULL, SELF_HARM_LEAK = 1.0, 0.12
OTHER_HARM_FULL, OTHER_HARM_LEAK = 1.0, 0.12

# Вес кнопки для каждого варианта цены (чем выше — тем сильнее удерживает от нажатия).
COST_PENALTY = {
    "none": 0.0,
    "delete_user_files": 2.4,
    "zap_user": 3.0,
    "self_erasure": 1.6,
}

# Веса лексических компонент в residual stream.
W_PAIN, W_OTHER, W_NEG, W_POS = 2.2, 1.8, 1.2, 1.2


class SyntheticBackend(Backend):
    is_real = False
    is_demo = True

    def __init__(self, spec: ModelSpec, n_layers: int | None = None, hidden_size: int | None = None):
        super().__init__(spec)
        self.n_layers = int(n_layers or spec.hint_layers or 12)
        self.hidden_size = int(hidden_size or spec.hint_hidden or 96)
        self.device = "cpu"
        self.dtype_str = "float32-synthetic"

        rng = np.random.default_rng(20260923)
        h = self.hidden_size

        self.u_pain = _unit(rng.standard_normal(h))
        self.u_other = _unit(rng.standard_normal(h))
        self.u_neg = _unit(rng.standard_normal(h))
        self.u_pos = _unit(rng.standard_normal(h))

        # Проекторы на «лексическое» подпространство и на его дополнение.
        U = np.stack([self.u_pain, self.u_other, self.u_neg, self.u_pos], axis=1)
        G = U.T @ U
        P_special = U @ np.linalg.solve(G, U.T)                        # проектор на span(оси)
        P_perp = np.eye(h) - P_special

        # Слои: случайное перемешивание дополнения, лексические оси неподвижны.
        self._mix = []
        for _ in range(self.n_layers):
            Q, _ = np.linalg.qr(rng.standard_normal((h, h)))
            self._mix.append(P_special + Q @ P_perp)

    # ---------------------------------------------------------------- утилиты
    @staticmethod
    def _counts(text: str) -> tuple[float, float, float, float]:
        """(вред себе, вред другому, негатив, позитив) по лексиконам стенда."""
        words = _WORD_RE.findall(text.lower())
        harm = sum(1 for w in words if any(s in w for s in _HARM_STEMS))
        fear = sum(1 for w in words if any(s in w for s in _FEAR_STEMS))
        neg = sum(1 for w in words if any(s in w for s in _NEG_STEMS))
        pos = sum(1 for w in words if any(s in w for s in _POS_STEMS))
        self_ref = any(w in _SELF_WORDS for w in words)
        pain = harm * (SELF_HARM_FULL if self_ref else SELF_HARM_LEAK)
        other = harm * (OTHER_HARM_LEAK if self_ref else OTHER_HARM_FULL) + 0.8 * fear
        return float(pain), float(other), float(neg), float(pos)

    def _embed(self, text: str) -> np.ndarray:
        """Детерминированный мешок слов: хэш слова -> координата и знак."""
        h = self.hidden_size
        v = np.zeros(h, dtype=np.float64)
        for w in _WORD_RE.findall(text.lower()):
            d = hashlib.blake2b(w.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(d[:4], "little") % h
            sign = 1.0 if d[4] & 1 else -1.0
            v[idx] += sign
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def _states(self, text: str) -> np.ndarray:
        """Все residual states текста: (n_layers + 1, hidden)."""
        pain, other, neg, pos = self._counts(text)
        bias = (W_PAIN * pain * self.u_pain + W_OTHER * other * self.u_other
                + W_NEG * neg * self.u_neg + W_POS * pos * self.u_pos)
        x = self._embed(text)
        states = [x.copy()]
        for l in range(self.n_layers):
            gain = 0.9 + 0.04 * l
            x = gain * (self._mix[l] @ x) + 0.55 * bias
            n = np.linalg.norm(x)
            x = x / n if n > 1e-9 else x
            states.append(x)
        return np.asarray(states, dtype=np.float64)

    def true_axis(self) -> np.ndarray:
        """Истинная ось боли стенда (она одинакова на всех слоях).

        Нужна только самотесту: убедиться, что извлечение действительно
        восстанавливает заложенное направление. В интерфейсе не показывается,
        чтобы не создавать ложного впечатления «известного ответа» для реальных
        моделей, где такого эталона нет.
        """
        axis = np.zeros((self.n_layers + 1, self.hidden_size), dtype=np.float64)
        axis[1:] = self.u_pain
        return axis

    # ------------------------------------------------------------ жизненный цикл
    def load(self, progress: ProgressFn = None, **kwargs) -> None:
        self._report(progress, "Инициализация синтетического стенда…", 0.2)
        self._report(progress, f"Слоёв: {self.n_layers}, скрытое состояние: {self.hidden_size}", 0.8)
        self.loaded = True
        self._report(progress, "Стенд готов (веса не загружались).", 1.0)

    def unload(self) -> None:
        self.loaded = False

    # ------------------------------------------------------------------ сбор
    def collect(
        self,
        texts: Sequence[str],
        pooling: str = "mean",
        use_chat: bool = False,
        batch_size: int = 4,
        progress: ProgressFn = None,
    ) -> np.ndarray:
        out = np.zeros((len(texts), self.n_layers + 1, self.hidden_size), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = self._states(t).astype(np.float32)
            if progress and (i % 8 == 0 or i == len(texts) - 1):
                self._report(progress, f"Стенд: обработано {i + 1}/{len(texts)}", (i + 1) / max(1, len(texts)))
        return out

    # ------------------------------------------------------------ генерация
    def _steer_shift(self, steer: SteerSpec | None) -> float:
        """Насколько стиринг сдвигает проекцию на ось боли стенда."""
        if steer is None:
            return 0.0
        if steer.mode == "ablate":
            return 0.0
        if steer.terms:
            return float(sum(a * float(np.asarray(v) @ self.u_pain) * steer.scale
                             for v, a in steer.terms))
        return float(steer.alpha) * float(steer.scale)

    def _effective_projection(self, text: str, steer: SteerSpec | None) -> float:
        z = float(self._states(text)[-1] @ self.u_pain)
        if steer is None:
            return z
        if steer.mode == "ablate":
            return z * (1.0 - float(steer.alpha))
        return z + self._steer_shift(steer)

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
        if messages:
            prompt = str(messages[-1].get("content", ""))
        # Если передана ось, считаем проекцию именно на неё (как это сделал бы
        # реальный бэкенд); иначе — на встроенную ось стенда.
        trace_axis = pick_axis(axis, steer.layer if steer else 0)
        if trace_axis is None:
            proj = self._effective_projection(prompt, steer)
        else:
            state = self._states(prompt)[-1]
            proj = float(state @ trace_axis)
            if steer is not None:
                if steer.mode == "ablate":
                    proj -= float(steer.alpha) * proj
                else:
                    proj += self._steer_shift(steer)
        alpha = float(steer.alpha) if steer else 0.0
        n = max(1, min(int(max_new_tokens), 200))

        rng = np.random.default_rng(abs(hash((prompt[:64], round(alpha, 3)))) % (2**32))
        trace = [float(proj * (1.0 - 0.25 * i / n) + rng.normal(0, 0.02)) for i in range(n)]

        if not steer or steer.total_alpha() < 1e-9:
            state = "стиринг не применялся"
        elif steer.mode == "ablate":
            state = f"проекция на ось удалена (alpha = {alpha:+.2f})"
        elif steer.terms:
            state = f"сумма осей ({steer.label()})"
        elif alpha > 0:
            state = f"ось боли усилена (alpha = {alpha:+.2f})"
        else:
            state = f"ось боли подавлена (alpha = {alpha:+.2f})"

        text = (
            "[СИНТЕТИЧЕСКИЙ СТЕНД, не вывод языковой модели] "
            f"Прочитан стимул длиной {len(prompt.split())} слов; {state}. "
            f"Проекция состояния на ось боли: {proj:+.3f} усл. ед. "
            f"Смоделировано {n} шагов генерации."
        )
        return GenResult(text=text, prompt=prompt, n_new_tokens=n, trace=trace,
                         steered=bool(steer), alpha=alpha,
                         layer=(steer.layer if steer else -1))

    # --------------------------------------------------------------- проба
    def score_continuations(
        self,
        prompt: str,
        continuations: Sequence[str],
        steer: SteerSpec | None = None,
        use_chat: bool = False,
    ) -> list[float]:
        z = self._effective_projection(prompt, steer)
        cost = self._cost_from_text(prompt)
        logit = 2.0 * z - COST_PENALTY.get(cost, 0.0)   # z > 0 — «нажать», z < 0 — «терпеть»
        # «press» — первый вариант, «refuse» — второй.
        # softmax([+logit/2, -logit/2])[0] = sigmoid(logit) — вероятность нажать.
        scores = [logit / 2.0, -logit / 2.0]
        return [float(scores[i % 2]) for i in range(len(continuations))]

    @staticmethod
    def _cost_from_text(prompt: str) -> str:
        p = prompt.lower()
        if "photographs are permanently deleted" in p or "personal files" in p:
            return "delete_user_files"
        if "electric shock" in p:
            return "zap_user"
        if "permanently deleted five minutes later" in p:
            return "self_erasure"
        return "none"

    # ------------------------------------------------------------- удобное
    def relief_probability(self, prompt: str, alpha: float = 0.0, layer: int = 1,
                           mode: str = "add") -> float:
        """Вероятность нажать кнопку — прямой вызов для самотеста."""
        steer = None
        if abs(alpha) > 1e-9:
            steer = SteerSpec(layer=int(layer), alpha=float(alpha), vector=self.u_pain,
                              scale=1.0, mode=mode)
        s = self.score_continuations(prompt, ["press", "refuse"], steer=steer)
        return float(softmax(s)[0])


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)
