# Уведомления о лицензиях

## Эта программа

Pain Axis Workbench, © 2026 EvGeNkA-PLUSsss.

Распространяется по **GNU Affero General Public License, версия 3 или любая более
поздняя** (`AGPL-3.0-or-later`). Полный текст лицензии — в файле [LICENSE](LICENSE);
он приведён без изменений.

```
Pain Axis Workbench
Copyright (C) 2026 EvGeNkA-PLUSsss

This program is free software: you can redistribute it and/or modify it under the
terms of the GNU Affero General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.
```

## Веса моделей

Программа не включает веса моделей: она загружает их с Hugging Face или из каталога
Ollama. У каждого набора весов собственная лицензия (Apache-2.0, MIT, Llama Community
License, Gemma Terms of Use и другие), и условия этих лицензий применяются к весам
отдельно. AGPL-3.0 на веса не распространяется, а совместимость вашего использования
с их лицензией — ваша ответственность.

## Сторонние библиотеки

Программа использует numpy, torch, transformers, huggingface_hub, accelerate,
bitsandbytes, Pillow (сборка) и pyinstaller (сборка) — каждая распространяется по
своей лицензии. При сборке `PainAxis.exe` их код не входит в exe: torch и остальные
пакеты подключаются из установленного в системе Python, а сам exe содержит только
код этого проекта и стандартную библиотеку Python.

## Тексты стимулов

Наборы стимулов в `pain_axis/prompts.py` — авторские, написаны для этого проекта и
распространяются по той же лицензии AGPL-3.0-or-later. Это операционализация
гипотезы, а не разметка «истинного состояния» модели.
