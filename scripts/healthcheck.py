"""Проверка живости контейнера: страница ПЛЮС готовность узлов.

Вызывается из `HEALTHCHECK` в Dockerfile. Коды возврата:

    0 — здоров: страница отвечает, хранилище открыто, эмбеддер загружен
    1 — не здоров, причина напечатана

★Почему двух условий, а не одного. Прежняя проверка ограничивалась страницей
Streamlit, и контейнер объявлял себя здоровым, не потрогав ни одного из двух
узлов, на которых стои́т работа. Отказ вскрывался при первом вопросе
пользователя. Подробности и устройство отметки — в шапке `neftegaz/health.py`.

★Дорогую работу этот скрипт НЕ делает. Модель эмбеддингов грузит приложение —
один раз, при старте, в том процессе, который ею потом и пользуется. Здесь
читается только результат. Иначе проверка живости раз в полминуты поднимала бы
241 МБ ради ответа «да», и лечение вышло бы хуже болезни.

Запуск вручную (пригодится при разборе «контейнер здоров, а не работает»):

    python scripts/healthcheck.py
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neftegaz.health import read_marker  # noqa: E402

PAGE_URL = "http://127.0.0.1:8501/_stcore/health"
PAGE_TIMEOUT_SECONDS = 3


def page_is_up(url: str = PAGE_URL) -> str:
    """Пустая строка, если страница отвечает; иначе причина."""
    try:
        with urllib.request.urlopen(url, timeout=PAGE_TIMEOUT_SECONDS) as response:  # noqa: S310
            if response.status != 200:
                return f"страница ответила {response.status}"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return f"страница не отвечает: {exc}"
    return ""


def main() -> int:
    trouble = page_is_up()
    if trouble:
        print(trouble, file=sys.stderr)
        return 1

    readiness = read_marker()
    if readiness is None:
        # ★Отметки нет — это «ещё не готов», а не «здоров». Приложение пишет её
        # по завершении прогрева; до тех пор контейнер поднят, но работать не
        # может, и объявлять его здоровым значило бы повторить исходную ошибку.
        print(
            "страница отвечает, но проверка узлов ещё не завершилась "
            "(при первом запуске идёт загрузка модели эмбеддингов, 241 МБ)",
            file=sys.stderr,
        )
        return 1

    if not readiness.ok:
        print(readiness.as_line(), file=sys.stderr)
        return 1

    print(readiness.as_line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
