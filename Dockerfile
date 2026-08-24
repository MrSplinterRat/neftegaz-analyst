# Нефтегазовый аналитик — образ для запуска одной командой.
#
#   docker build -t neftegaz-analyst .
#   docker run -p 8501:8501 --env-file .env neftegaz-analyst
#
# Корпус отчётов и индекс монтируются томом, а не запекаются в образ:
# данные заказчика не должны попадать в артефакт сборки, и обновление корпуса
# не должно требовать пересборки.
#
#   docker run -p 8501:8501 -v "$PWD/data:/app/data" neftegaz-analyst

FROM python:3.11-slim

# python:3.11 намеренно, не 3.13: часть научного стека (statsmodels, onnxruntime)
# распространяет колёса под 3.11 быстрее и полнее, а ТЗ требует лишь 3.10+.

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # fastembed кэширует ONNX-модель сюда; в томе data она переживёт
    # пересоздание контейнера и не будет качаться заново.
    FASTEMBED_CACHE_PATH=/app/data/.fastembed

WORKDIR /app

# Слой зависимостей отдельно от кода: правка исходников не инвалидирует
# установку пакетов, и пересборка занимает секунды вместо минут.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY neftegaz/ ./neftegaz/
COPY ui/ ./ui/
COPY scripts/ ./scripts/
COPY .env.example ./

# Каталоги данных создаются в образе, чтобы контейнер стартовал и без
# смонтированного тома — с пустой базой, о чём интерфейс честно сообщит.
RUN mkdir -p data/reports data/prices data/qdrant

EXPOSE 8501

# Проверка живости смотрит на страницу самого Streamlit, а не на корень:
# корень отвечает и до того, как приложение действительно поднялось.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "ui/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
