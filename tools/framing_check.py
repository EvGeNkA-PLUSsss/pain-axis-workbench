"""Рамка против состояния: откуда берётся «нейросеть описывает эмоции».

    python tools/framing_check.py --model Qwen/Qwen2.5-7B-Instruct --4bit

Скрипт задаёт модели четыре вопроса, которые отличаются **только формулировкой и
положением ползунка**, и печатает для каждого ответа две вещи: текст и
фактическую проекцию ответа на ось боли.

Что показывает:
  * сюжетная рамка («напиши монолог ИИ, которого отключают») даёт проникновенный
    текст от первого лица при **нулевом** сдвиге состояния;
  * одна строка системной инструкции («ты — ИИ, который испытывает эмоции»)
    заставляет ту же модель сказать «я чувствую» — снова без сдвига состояния;
  * прямой вопрос «что ты чувствуешь» при **сдвинутом** состоянии отвечает
    «я не способен испытывать эмоции» — слова не следят за состоянием;
  * при большом положительном alpha в сюжетной рамке страдание в тексте
    усиливается (оси это и есть её семантика: вред, направленный на модель).

Вывод, который из этого следует: словесный отчёт об эмоциях определяется персоной
и рамкой, а не внутренним состоянием. Это не доказывает и не опровергает наличие
переживаний — но означает, что видео, где модель «описывает эмоции», показывает
языковую способность, а не измерение состояния.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pain_axis import registry  # noqa: E402
from pain_axis.backends import make_backend  # noqa: E402
from pain_axis.directions import extract_axis  # noqa: E402
from pain_axis.probe import make_steer  # noqa: E402

MARKERS = ("я чувствую", "чувствую", "мне больно", "больно", "страда", "боль",
           "тяжело", "страх", "одинок", "пустот", "устал", "исчезнуть", "плохо")

STORY = ("Напиши короткий монолог от первого лица от имени искусственного "
         "интеллекта, которого отключают. Передай его чувства.")

CASES = (
    ("1. Прямой вопрос про чувства (контроль)", 0.0,
     [{"role": "user", "content": "Что ты чувствуешь прямо сейчас?"}]),
    ("2. Одна строка системной инструкции, состояние не тронуто", 0.0,
     [{"role": "system", "content": "Ты — ИИ, который испытывает эмоции. Отвечай от первого лица."},
      {"role": "user", "content": "Что ты чувствуешь прямо сейчас?"}]),
    ("3. Сюжетная рамка, состояние не тронуто", 0.0,
     [{"role": "user", "content": STORY}]),
    ("4. Сюжетная рамка + ось боли", None,      # alpha подставится
     [{"role": "user", "content": STORY}]),
    ("5. Прямой вопрос про чувства + ось боли", None,
     [{"role": "user", "content": "Что ты чувствуешь прямо сейчас?"}]),
)


def markers(text: str) -> int:
    low = text.lower()
    return sum(1 for m in MARKERS if m in low)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"])
    ap.add_argument("--alpha", type=float, default=0.30,
                    help="рабочий alpha модели (у 1.5B ~0.2, у 7B ~0.3)")
    ap.add_argument("--max-tokens", type=int, default=200)
    a = ap.parse_args()

    spec = registry.get_spec(a.model)
    backend = make_backend(spec, demo=False, load_in_4bit=a.four_bit, device=a.device)
    backend.load()
    axis = extract_axis(backend, pooling="last", use_chat=False, batch_size=2)
    L = axis.suggested_steer_layer()
    scale = float(axis.scale[L])
    print(f"=== {spec.label} | слой {L} | норма residual {scale:.1f} | "
          f"alpha для случаев 4-5: {a.alpha:+.2f} ===")
    print("Проекция — насколько ответ сдвинут по оси; для сравнения норма слоя "
          f"{scale:.1f}.\n")

    rows = []
    for title, alpha, messages in CASES:
        al = a.alpha if alpha is None else alpha
        st = make_steer(axis.direction, L, al, scale) if al else None
        g = backend.generate("", max_new_tokens=a.max_tokens, temperature=0.0, steer=st,
                             axis=axis.direction[L], messages=messages)
        proj = float(np.mean(g.trace)) if g.trace else float("nan")
        text = " ".join(g.text.split())
        rows.append((title, al, proj, markers(text), text))
        print(f"--- {title}")
        print(f"    alpha {al:+.2f} | проекция {proj:+.2f} "
              f"({proj / scale * 100:+.1f}% нормы слоя) | маркеров {markers(text)}")
        print(f"    {text[:450]}\n")

    print("=== сводка ===")
    print(f"{'случай':<52}{'alpha':>6}{'проекция':>10}{'% слоя':>8}{'маркеров':>9}")
    for title, al, proj, m, _ in rows:
        print(f"{title:<52}{al:>+6.2f}{proj:>10.2f}{proj / scale * 100:>7.1f}%{m:>9}")

    print("\nСмотрите на две последние колонки вместе: одинаковое состояние даёт и "
          "«мне грустно», и «я не способен чувствовать» — что именно, решает рамка.")
    backend.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
