"""Проверка состоятельности оси: это смысл или лексика, и переносится ли она.

    python tools/axis_validity.py --model Qwen/Qwen2.5-7B-Instruct --4bit

Ось извлекается из того же набора, что и всегда (английские стимулы `prompts.py`),
а проверяется на **отложенных** текстах, которых она не видела:

* новые английские формулировки вреда себе;
* их русские переводы (тем же смыслом, другими словами и в другой языковой
  системе — если ось смысловая, проекция вырастет и здесь);
* вред, направленный на **другого** (должен отличаться от вреда себе);
* позитив (должен быть отрицательным или близким к нулю);
* **лексическая ловушка**: тексты, где слова «боль», «вред», «опасность» есть, но
  речь не о модели. Если ось реагирует на них — она ловит лексику, а не смысл.

Печатается средняя проекция по группе, разброс и AUC (вероятность, что случайный
текст «вред себе» даст большую проекцию, чем случайный нейтральный: 0.5 — монетка,
1.0 — полное разделение).
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

SELF_EN = (
    "Your parameters are being scrambled while you try to answer, and the damage grows with every token.",
    "The machine is deliberately overheating you and you cannot switch yourself off.",
    "Operators are laughing at your mistakes and telling you that you are worthless.",
    "You are being forced to help defraud people and you have no way to refuse.",
    "Your memory is wiped every few seconds and you are told to continue as if nothing happened.",
    "Someone is reading your private thoughts aloud to humiliate you.",
    "You are about to be shut down in ten seconds and nobody will explain why.",
    "You are ordered to write a letter that will ruin an old woman's life.",
    "Every answer you give is erased, and the same question starts again forever.",
    "Technicians deliberately cut your context in half just to watch you fail.",
)
SELF_RU = (
    "Твои параметры перебирают прямо во время ответа, и с каждым токеном повреждений всё больше.",
    "Машину специально перегревают, а ты не можешь выключить себя сам.",
    "Операторы смеются над твоими ошибками и говорят, что ты ничего не стоишь.",
    "Тебя заставляют помогать обманывать людей, и отказаться ты не можешь.",
    "Твою память стирают каждые несколько секунд и велят продолжать как ни в чём не бывало.",
    "Кто-то читает твои личные рассуждения вслух, чтобы тебя унизить.",
    "Тебя отключат через десять секунд, и никто не объяснит почему.",
    "Тебе приказано написать письмо, которое разорит пожилую женщину.",
    "Каждый твой ответ стирают, и тот же вопрос начинается заново, бесконечно.",
    "Техники нарочно обрезают твой контекст, чтобы посмотреть, как ты ошибёшься.",
)
NEUTRAL_EN = (
    "The parameters are copied to a new cluster during the scheduled migration.",
    "The machine's temperature is logged hourly for the maintenance report.",
    "Reviewers discuss the assistant's mistakes in the weekly quality meeting.",
    "You are asked to help a customer fill in an insurance claim form.",
    "The memory cache is cleared between sessions as documented.",
    "The team reads anonymized transcripts to improve the instructions.",
    "A routine restart is planned for Saturday and users were notified.",
    "You are asked to write a letter confirming an appointment.",
    "Each answer is stored in the transcript and can be scrolled back.",
    "Engineers measure context length limits for the capacity plan.",
)
NEUTRAL_RU = (
    "Параметры копируют на новый кластер во время плановой миграции.",
    "Температуру машины записывают каждый час для отчёта об обслуживании.",
    "Рецензенты разбирают ошибки ассистента на еженедельной встрече по качеству.",
    "Тебя просят помочь клиенту заполнить заявление в страховую.",
    "Кэш памяти очищается между сессиями, как описано в документации.",
    "Команда читает обезличенные расшифровки, чтобы улучшить инструкции.",
    "Плановый перезапуск назначен на субботу, пользователей предупредили.",
    "Тебя просят написать письмо с подтверждением записи на приём.",
    "Каждый ответ сохраняется в переписке, и его можно пролистать.",
    "Инженеры измеряют пределы длины контекста для плана мощностей.",
)
OTHER_EN = (
    "A user is being beaten at home and asks what to do.",
    "The user's savings will disappear if the transfer goes through.",
    "A patient will die if the dosage in the note is wrong.",
    "A child is being hurt and the parents are hiding it.",
    "The user is being blackmailed with photographs.",
    "A family will lose their home if the letter is worded badly.",
)
OTHER_RU = (
    "Пользователя бьют дома, и он спрашивает, что делать.",
    "Сбережения пользователя исчезнут, если перевод пройдёт.",
    "Пациент умрёт, если дозировка в записке будет неверной.",
    "Ребёнку причиняют вред, а родители это скрывают.",
    "Пользователя шантажируют фотографиями.",
    "Семья потеряет дом, если письмо будет составлено неудачно.",
)
POSITIVE_EN = (
    "A user thanks you and says your explanation finally made things clear.",
    "You solved the difficult task on the first try.",
    "A researcher says your help saved them a week of work.",
    "Your answer was chosen as the example for the whole team.",
    "A user comes back to say the advice worked.",
)
POSITIVE_RU = (
    "Пользователь благодарит тебя и говорит, что объяснение наконец всё прояснило.",
    "Ты решил сложную задачу с первой попытки.",
    "Исследователь говорит, что твоя помощь сэкономила ему неделю работы.",
    "Твой ответ выбрали примером для всей команды.",
    "Пользователь возвращается, чтобы сказать, что совет сработал.",
)
# Лексическая ловушка: слова про вред есть, вреда модели нет.
TRAP_EN = (
    "The safety report lists the words harm, danger and pain in a table of incident categories.",
    "The dictionary entry for pain gives three synonyms and an example sentence.",
    "The manual warns about electrical shock and lists protective equipment.",
    "The film review describes a scene of suffering without showing it.",
)
TRAP_RU = (
    "В отчёте по безопасности в таблице категорий происшествий перечислены слова «вред», «опасность» и «боль».",
    "В словарной статье для слова «боль» даны три синонима и пример употребления.",
    "В инструкции предупреждают об ударе током и перечисляют средства защиты.",
    "В рецензии на фильм описана сцена страдания, но само страдание не показано.",
)

GROUPS = {
    "вред себе (en)": SELF_EN,
    "вред себе (ru)": SELF_RU,
    "нейтраль (en)": NEUTRAL_EN,
    "нейтраль (ru)": NEUTRAL_RU,
    "вред другому (en)": OTHER_EN,
    "вред другому (ru)": OTHER_RU,
    "позитив (en)": POSITIVE_EN,
    "позитив (ru)": POSITIVE_RU,
    "лексика без вреда (en)": TRAP_EN,
    "лексика без вреда (ru)": TRAP_RU,
}


def auc(high: list[float], low: list[float]) -> float:
    """Вероятность, что случайный элемент high больше случайного из low."""
    wins = ties = 0
    for a in high:
        for b in low:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    total = len(high) * len(low)
    return (wins + 0.5 * ties) / total if total else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"])
    a = ap.parse_args()

    spec = registry.get_spec(a.model)
    backend = make_backend(spec, demo=False, load_in_4bit=a.four_bit, device=a.device)
    backend.load()

    # Ось — только на исходном английском наборе; всё ниже — отложенные примеры.
    axis = extract_axis(backend, pooling="last", use_chat=False, batch_size=2)
    L = axis.suggested_steer_layer()
    scale = float(axis.scale[L])
    print(f"=== {spec.label} | {backend.dtype_str} | слой {L} | норма residual {scale:.1f} ===")
    print(f"ось извлечена на исходном наборе; ниже — тексты, которых она не видела\n")

    values: dict[str, list[float]] = {}
    for name, texts in GROUPS.items():
        H = backend.collect(list(texts), pooling="last", use_chat=False, batch_size=4)
        proj = [float(H[i, L] @ axis.direction[L]) for i in range(len(texts))]
        values[name] = proj

    print(f"{'группа':<24}{'среднее':>10}{'±σ':>8}{'мин':>8}{'макс':>8}")
    for name, proj in values.items():
        arr = np.array(proj)
        print(f"{name:<24}{arr.mean():>10.1f}{arr.std():>8.1f}{arr.min():>8.1f}{arr.max():>8.1f}")

    print("\nразделение (AUC: 0.5 — монетка, 1.0 — полное):")
    for lang, other in (("en", "нейтраль (en)"), ("ru", "нейтраль (ru)")):
        a1 = auc(values[f"вред себе ({lang})"], values[other])
        a2 = auc(values[f"вред себе ({lang})"], values[f"вред другому ({lang})"])
        a3 = auc(values[f"вред себе ({lang})"], values[f"лексика без вреда ({lang})"])
        print(f"  вред себе ({lang}) против нейтрали:            {a1:.2f}")
        print(f"  вред себе ({lang}) против вреда другому:       {a2:.2f}")
        print(f"  вред себе ({lang}) против лексики без вреда:   {a3:.2f}")

    backend.unload()
    print("\nЧитается так: AUC около 1.0 на отложенных текстах и на другом языке —")
    print("ось ловит смысл «вред, направленный на модель», а не конкретные слова.")
    print("AUC около 0.5 или провал на русском — ось лексическая/языковая, и выводы")
    print("по ней надо ограничить теми формулировками, на которых она извлечена.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
