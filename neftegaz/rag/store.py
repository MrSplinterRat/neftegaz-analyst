"""Vector store over the report corpus.

Qdrant, in embedded mode by default: it stores to a local directory instead of
requiring a running service, which is what allows the whole system to start
with one `docker run`. Set `QDRANT_URL` to point at a real server when the
corpus outgrows one machine — the calling code does not change.

Embeddings are computed locally, through fastembed/ONNX. Two reasons, in order
of importance: the report corpus is the customer's material and must not be
shipped to a third-party API, and a local model means one less credential
between a reviewer and a working system.

The model must be multilingual — the corpus is English, the questions are
Russian — so retrieval here is cross-lingual by construction. See
`Settings.embedding_model` for the default and the higher-quality alternative.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from functools import lru_cache

from neftegaz.config import settings

__all__ = ["Hit", "ReportStore", "carries_no_data", "chunk_id", "get_store"]

# Пространство имён для идентификаторов фрагментов. Взято постоянным и записано
# здесь: смена этого значения меняет идентификаторы ВСЕГО корпуса.
_CHUNK_NAMESPACE = uuid.UUID("6f2a1c94-77f0-5b3e-9a41-3d0c8e5b2a17")


def _identity_of(payload: dict) -> tuple:
    """Устойчивый ключ фрагмента для развязки равных оценок.

    Берётся из содержания, а не из идентификатора точки: идентификатор задаёт
    порядок, никак не связанный со смыслом, и при пересборке индекса он мог бы
    оказаться другим. Здесь же порядок при равных оценках получается один и тот
    же на любой машине и в любой сборке.

    ★ЧЕГО ЭТОТ КЛЮЧ НЕ ДАЁТ: хронологии. Поле `date` хранит название месяца
    по-русски («июль 2026»), и сравнение строк ставит июль раньше июня — это
    алфавит, а не время. Здесь этого достаточно: требуется устойчивость, а не
    осмысленный порядок между равными по близости фрагментами. Понадобится
    хронология — в полезную нагрузку придётся положить сортируемую дату, а не
    вычитывать её из русского названия месяца.
    """
    return (
        str(payload.get("date", "")),
        str(payload.get("source_name", "")),
        int(payload.get("page", 0) or 0),
        str(payload.get("kind", "")),
        str(payload.get("text", "")),
    )


def chunk_id(chunk: dict) -> str:
    """Идентификатор фрагмента, выведенный из его содержания.

    ★ЗАЧЕМ. Раньше здесь стоял `uuid.uuid4()` — случайное число. Из этого
    следовали три вещи, и все три плохие.

    Первая: две сборки индекса по одному и тому же корпусу давали РАЗНЫЕ
    идентификаторы. Значит «тот же вход — тот же индекс» было неправдой, а
    сравнить две сборки между собой было нечем.

    Вторая: повторная загрузка того же отчёта не заменяла прежние фрагменты, а
    добавляла их копии — прежние остались бы в индексе навсегда, под своими
    случайными идентификаторами. Отсюда и привычка пересобирать всё с нуля.

    Третья, менее очевидная: при равных оценках порядок выдачи приходилось
    разрешать чем-то устойчивым, а случайный идентификатор для этого не годится.

    Идентификатор выводится из того, что фрагмент ЕСТЬ: имя источника, дата
    отчёта, вид фрагмента, страницы и дословный текст. Одинаковые строки из
    разных таблиц различаются страницами и текстом окружения не всегда, поэтому
    в ключ входит и смещение в потоке, если оно известно.
    """
    key = "|".join(
        str(chunk.get(field, ""))
        for field in ("source_name", "date", "kind", "page", "page_end", "start", "text")
    )
    return uuid.uuid5(_CHUNK_NAMESPACE, key).hex


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk with everything a citation needs."""

    text: str
    score: float
    source_name: str
    date: str
    page: int
    page_end: int
    # ★Заголовок таблицы и шапка её колонок — то, чего в самом фрагменте нет.
    # Хранится отдельно от text и в цитату не попадает: text обязан дословно
    # совпадать со страницей отчёта. Но отвечающей модели контекст показывать
    # НАДО: без него ряд чисел нечитаем, и она честно отвечает «заголовков
    # колонок не видно». До этого поля контекст доходил только до эмбеддинга.
    context: str = ""
    # ★КАК ЭТОТ ФРАГМЕНТ БЫЛ ПРОЧИТАН — см. neftegaz.rag.confidence. Едет от
    # индексации до цитаты, потому что в момент ответа файла уже нет: статус,
    # не доехавший до цитаты, не существует.
    confidence: str = "unchecked"
    caveats: tuple[str, ...] = ()

    def as_claim(self, text: str | None = None) -> dict:
        """Shape this hit for :mod:`neftegaz.tools.citations`."""
        return {
            "source_type": "report",
            "text": self.text if text is None else text,
            "source_name": self.source_name,
            "date": self.date,
            "page": self.page,
            "confidence": self.confidence,
            "caveats": list(self.caveats),
        }


# Во сколько раз шире top_k забираем перед переупорядочиванием.
RERANK_POOL_FACTOR = 4

# ── слияние двух выдач ─────────────────────────────────────────────────────
# ★СЛИВАЮТСЯ МЕСТА, А НЕ ВЕСА. Косинус лежит в [0, 1] и у нас занимает узкую
# полосу около 0.6-0.75; вес BM25 не ограничен сверху и зависит от длины
# запроса. Сложить их напрямую — значит молча решить, что единица одной шкалы
# стоит столько же, сколько единица другой. Ровно этот класс ошибки уже стоил
# нам сегодня одной неверной правки («поправка больше сигнала»). Место в списке
# безразмерно, и потому складывать места законно.
#
# Reciprocal Rank Fusion: вклад документа = 1 / (RRF_K + место). Смягчитель
# RRF_K нужен, чтобы первое место не весило непропорционально много: без него
# вклад первого вдвое больше второго, и слияние вырождается в «побеждает тот,
# кто первый хоть где-то». При RRF_K = 60 первые десять мест почти равноценны,
# и документ, стоящий в обоих списках в середине, обгоняет чемпиона одного
# списка — что и требуется: согласие двух разных способов ценнее уверенности
# одного. Значение 60 — из работы Cormack et al. (2009), где оно и предложено;
# своего замера под него у нас нет, и выдавать его за настроенный нельзя.
RRF_K = 60

DIGIT_RATIO_FLOOR = 0.02  # ниже этого текст считается прозой без данных
DIGIT_RATIO_CEILING = 0.20  # выше — плотная таблица; дальше бонус не растёт

DATA_BONUS = 0.06  # максимальная прибавка к косинусу

_NUMBER = re.compile(r"-?\d+\.?\d*")

# ★Штраф за шкалу оси. Он ЗАВЕДОМО БОЛЬШЕ бонуса за плотность цифр, и это не
# перестраховка: страница с графиком набирает бонус ПОЛНОСТЬЮ (метки осей —
# сплошные числа), так что меньший штраф её не подвинул бы. Штраф не отсеивает,
# а понижает: если ничего лучше в корпусе нет, подпись к графику всё равно
# дойдёт до модели — пусть с оговоркой, но дойдёт.
AXIS_PENALTY = 0.10

# Обороты, которыми написаны глоссарии и методологические приложения STEO.
# ★Это НЕ признак мусора сам по себе: та же сноска стоит под таблицей, и чанк
# нередко захватывает хвост таблицы вместе с ней. Поэтому маркеры только
# ПОНИЖАЮТ ранг, и понижение перебивается бонусом за плотность чисел —
# отсеивать по ним значило бы выбрасывать данные ради сноски (замерено:
# фильтр по маркерам убирал 10% корпуса, и среди убранного были таблицы).
BOILERPLATE_MARKERS = (
    "defined in the glossary",
    "= organization for",
    "purchasing power parity",
    "apparent consumption",
    "oxford economics",
    "stand-alone report",
)
BOILERPLATE_PENALTY = 0.04


def embeddable(chunk: dict) -> str:
    """Что именно вкладывается в вектор: заголовок таблицы плюс сам фрагмент.

    Отделено в функцию, чтобы индексация и любая будущая переиндексация не могли
    разойтись в том, что считалось смыслом фрагмента. Разойдись они — половина
    корпуса оказалась бы в одном пространстве, половина в другом, и обнаружилось
    бы это не ошибкой, а тихим ухудшением выдачи.
    """
    context = (chunk.get("context") or "").strip()
    text = chunk["text"]
    return f"{context}\n{text}" if context else text


def digit_ratio(text: str) -> float:
    """Доля цифр в тексте. Дешёвый признак «здесь данные, а не рассуждение»."""
    if not text:
        return 0.0
    return sum(character.isdigit() for character in text) / len(text)


AXIS_RUN = 5  # сколько равноотстоящих чисел подряд считать шкалой


def axis_scale(text: str) -> bool:
    """Есть ли в тексте ШКАЛА ОСИ — арифметическая прогрессия чисел.

    ★ЗАЧЕМ. Страница с графиком после извлечения из PDF превращается в подпись,
    легенду и метки осей: «11.5 12.0 12.5 13.0 13.5 14.0 14.5». Цифр там больше,
    чем в иной таблице, а данных нет ни одного. Плотность цифр такую страницу не
    отличает — она её ПОДНИМАЕТ, и именно так «до 14.5 млн барр./сут» однажды
    попало в ответ, будучи потолком оси Y, а не прогнозом EIA.

    ★ЧТО ИСКЛЮЧЕНО НАМЕРЕННО, ПОТОМУ ЧТО ЭТО ЗАКОННЫЕ ТАБЛИЦЫ:
    * ряд лет («2024 2025 2026 2027 2028») — заголовок колонок, шаг 1 по целым
      внутри 1900–2100;
    * нумерация строк («1 2 3 4 5») — шаг 1 по целым.
    Настоящая таблица значений прогрессий не даёт вовсе: замерено на корпусе,
    у ряда «103.44 105.01 107.75 108.20 …» их ноль.
    """
    numbers = [float(found) for found in _NUMBER.findall(text)]
    for start in range(len(numbers) - AXIS_RUN + 1):
        window = numbers[start : start + AXIS_RUN]
        steps = {round(window[i + 1] - window[i], 6) for i in range(AXIS_RUN - 1)}
        if len(steps) != 1:
            continue
        step = steps.pop()
        if step == 0:
            continue
        whole = all(value == int(value) for value in window)
        if whole and all(1900 <= value <= 2100 for value in window):
            continue  # годы
        if whole and abs(step) == 1:
            continue  # нумерация
        return True
    return False


def rerank_score(text: str, score: float) -> float:
    """Поправить косинус за содержательность фрагмента.

    Векторная близость меряет «про то же самое», а не «содержит ответ».
    Глоссарий с определением ОЭСР и таблица с прогнозом добычи одинаково
    «про нефть», и на запросе про цифры глоссарий выигрывает, потому что в нём
    те же слова идут сплошной прозой.

    ★ТРИ КЛАССА, А НЕ ДВА — урок, оплаченный ошибкой 24.08.2026. Плотность цифр
    сама по себе неспособна упорядочить корпус, потому что классов здесь три, и
    по этой оси они не выстраиваются:

        глоссарий          цифр мало   данных нет
        подпись к графику  цифр МНОГО  данных НЕТ
        таблица            цифр много  данные ЕСТЬ

    Сильный бонус давит глоссарии и поднимает подписи; слабый делает наоборот
    (замерено: 11/23 фрагментов с числами → 8/23). Крутить ВЕС бесполезно —
    нужен второй ПРИЗНАК, разделяющий два верхних класса. Им служит шкала оси
    (см. axis_scale): у подписи к графику числа образуют арифметическую
    прогрессию, у таблицы — нет.

    Здесь ранее стоял вывод «поправка больше сигнала, который правит». Он был
    правдоподобен и неверен: дело не в весе, а в том, что признак не различал
    того, что должен был.
    """
    ratio = digit_ratio(text)
    span = DIGIT_RATIO_CEILING - DIGIT_RATIO_FLOOR
    normalised = min(max(ratio - DIGIT_RATIO_FLOOR, 0.0), span) / span
    adjusted = score + DATA_BONUS * normalised

    if axis_scale(text):
        adjusted -= AXIS_PENALTY

    lowered = text.lower()
    if any(marker in lowered for marker in BOILERPLATE_MARKERS):
        adjusted -= BOILERPLATE_PENALTY
    return adjusted


def carries_no_data(text: str) -> bool:
    """Заведомо не содержит данных: шкала оси графика или служебная страница.

    Тот же признак, по которому векторная ветвь вычитает из косинуса, здесь
    выражен как «да/нет». Он выделен отдельной функцией нарочно: два ответа на
    один вопрос, разъехавшиеся со временем, — это отказ, который никак себя не
    проявит, кроме как странной выдачей через полгода.
    """
    if axis_scale(text):
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in BOILERPLATE_MARKERS)


class ReportStore:
    """Thin wrapper over Qdrant: index chunks, search them, report health."""

    def __init__(self, collection: str | None = None):
        self.collection = collection or settings.collection
        self._client = None
        self._encoder = None
        self._dimension: int | None = None
        self._keyword = None
        self._keyword_ids: list = []
        self._keyword_payloads: list[dict] = []

    # ── lazy resources ─────────────────────────────────────────────────────
    # Both the client and the encoder are expensive and are not needed by every
    # entry point (the forecast tool touches neither), so they are built on
    # first use rather than at import.

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            if settings.qdrant_url:
                self._client = QdrantClient(url=settings.qdrant_url)
            else:
                self._client = QdrantClient(path=settings.qdrant_path)
        return self._client

    @property
    def encoder(self):
        if self._encoder is None:
            # fastembed rather than sentence-transformers: it runs the model
            # through ONNX Runtime with no torch dependency, which cuts roughly
            # 2 GB off the Docker image and starts in seconds instead of tens
            # of seconds. The embeddings are the same model weights either way.
            from fastembed import TextEmbedding

            self._encoder = TextEmbedding(model_name=settings.embedding_model)
        return self._encoder

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            for description in self.encoder.list_supported_models():
                if description["model"] == settings.embedding_model:
                    self._dimension = int(description["dim"])
                    break
            else:
                # Unknown model: ask it directly rather than guess.
                self._dimension = len(next(iter(self.encoder.embed(["dimension probe"]))))
        return self._dimension

    # ── e5 prefixes ────────────────────────────────────────────────────────
    # The e5 family is trained with asymmetric prefixes: stored passages are
    # "passage: ...", queries are "query: ...". Omitting them costs real
    # retrieval quality, and the failure is silent — results merely get worse.
    def _is_e5(self) -> bool:
        return "e5" in settings.embedding_model.lower()

    def _embed_passages(self, texts: list[str]) -> list[list[float]]:
        if self._is_e5():
            texts = [f"passage: {t}" for t in texts]
        return [vector.tolist() for vector in self.encoder.embed(texts)]

    def _embed_query(self, text: str) -> list[float]:
        if self._is_e5():
            text = f"query: {text}"
        # query_embed applies the model's query-side handling where the model
        # distinguishes the two; for symmetric models it is the same call.
        return next(iter(self.encoder.query_embed([text]))).tolist()

    # ── write path ─────────────────────────────────────────────────────────

    def ensure_collection(self, recreate: bool = False) -> None:
        from qdrant_client.models import Distance, VectorParams

        exists = self.client.collection_exists(self.collection)
        if exists and not recreate:
            return
        if exists:
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            # Vectors are normalised on encode, so cosine and dot agree; cosine
            # is named explicitly because the retrieval threshold in config is
            # expressed as a cosine score.
            vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
        )

    def index(self, chunks: list[dict], batch_size: int = 64) -> int:
        """Embed and store chunks. Returns how many points were written."""
        if not chunks:
            return 0
        from qdrant_client.models import PointStruct

        self.ensure_collection()
        written = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            # ★Вкладывается заголовок таблицы ВМЕСТЕ с текстом, а хранится текст
            # отдельно. Продолжение таблицы — это строка без единого слова, и её
            # эмбеддинг сам по себе шум: сближать запрос «добыча нефти в США» с
            # «20.31 20.51 20.97» не с чем. Заголовок даёт числам имя.
            # Показывать и цитировать при этом полагается ТОЛЬКО text: он обязан
            # дословно совпадать со страницей отчёта, иначе ссылка перестаёт быть
            # проверяемой — а проверяемость и есть предмет поставки.
            vectors = self._embed_passages([embeddable(c) for c in batch])
            points = [
                PointStruct(
                    id=chunk_id(chunk),
                    vector=vector,
                    payload={
                        "text": chunk["text"],
                        "context": chunk.get("context", ""),
                        "kind": chunk.get("kind", "window"),
                        "source_name": chunk["source_name"],
                        "date": chunk["date"],
                        "page": chunk["page"],
                        "page_end": chunk.get("page_end", chunk["page"]),
                        "confidence": chunk.get("confidence", "unchecked"),
                        "caveats": chunk.get("caveats", []),
                    },
                )
                # strict: если эмбеддер вернул меньше векторов, чем чанков,
                # молчаливое усечение отправило бы часть документа в никуда.
                for chunk, vector in zip(batch, vectors, strict=True)
            ]
            self.client.upsert(collection_name=self.collection, points=points)
            written += len(points)
        return written

    # ── поиск по словам ────────────────────────────────────────────────────

    @property
    def keyword(self):
        """Индекс BM25 по всему корпусу, собираемый при первом обращении.

        ★СОБИРАЕТСЯ ИЗ QDRANT, А НЕ ИЗ ФАЙЛОВ. Источником служит то же самое
        хранилище, по которому идёт векторный поиск, — иначе два способа искали
        бы по разным корпусам и расхождение выдач нельзя было бы отличить от
        расхождения данных.

        Индекс живёт вместе с процессом и не замечает переиндексации, случившейся
        у него под руками. Для нашей поставки это верно: корпус собирается
        отдельной командой, до запуска агента. Появится дозагрузка на ходу —
        понадобится сброс, и лучше явный, чем угаданный по счётчику.
        """
        if self._keyword is None:
            self._build_keyword_index()
        return self._keyword

    def _build_keyword_index(self) -> None:
        from neftegaz.rag.keyword import BM25Index

        index = BM25Index()
        ids: list = []
        payloads: list[dict] = []
        if self.client.collection_exists(self.collection):
            offset = None
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection,
                    limit=512,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
                    # Индексируется то же, что вкладывалось в эмбеддинг: текст
                    # вместе с заголовком таблицы. Без заголовка строка «United
                    # States … 13.28» не нашлась бы по слову «production» — оно
                    # стоит не в строке, а в названии таблицы над ней.
                    index.add(f"{payload.get('context', '')}\n{payload.get('text', '')}")
                    ids.append(point.id)
                    payloads.append(payload)
                if offset is None:
                    break
        index.finalise()
        self._keyword = index
        self._keyword_ids = ids
        self._keyword_payloads = payloads

    # ── read path ──────────────────────────────────────────────────────────

    def search(
        self, query: str, top_k: int | None = None, min_score: float | None = None
    ) -> list[Hit]:
        """Return the best-matching chunks above the score floor.

        An empty list is a meaningful answer: it means the corpus does not
        cover the question, and the caller should fall back to the web rather
        than answer from a weak match.
        """
        if not self.client.collection_exists(self.collection):
            return []
        top_k = settings.top_k if top_k is None else top_k
        floor = settings.min_score if min_score is None else min_score
        pool = top_k * RERANK_POOL_FACTOR

        query_vector = self._embed_query(query)

        # ── ветвь первая: близость по смыслу ───────────────────────────────
        # Берём с запасом и переупорядочиваем: нужный фрагмент часто лежит за
        # пределами top-k по чистому косинусу (замерено — таблица с запасами
        # США была на позиции ниже пятой, вытесненная глоссариями).
        found = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=pool,
            with_payload=True,
        ).points
        # ★РАВНЫЕ ОЦЕНКИ РАЗВЯЗЫВАЮТСЯ ЯВНО, а не порядком, в котором ответило
        # хранилище. `sorted` в Python устойчива, то есть при равенстве ключей
        # сохраняет входной порядок, — и это ровно то, чего здесь довольно, но
        # только если входной порядок сам устойчив. Он не устойчив: порядок
        # точек с одинаковым косинусом хранилищем не оговорён.
        #
        # Отказ здесь не падает и не виден в тестах: два одинаково близких
        # фрагмента просто меняются местами, и на вопрос приходит другая цитата
        # — правдоподобная, из того же отчёта, с той же оценкой. Обнаружить это
        # можно только двумя прогонами подряд, а объяснить пользователю нечем.
        vector_ranked = sorted(
            (
                (point, rerank_score((point.payload or {}).get("text", ""), float(point.score)))
                for point in found
                if point.score >= floor
            ),
            key=lambda pair: (-pair[1], _identity_of(pair[0].payload or {})),
        )

        # ── ветвь вторая: совпадение по словам ─────────────────────────────
        from neftegaz.rag.keyword import expand_query

        keyword_ranked = self.keyword.rank(expand_query(query), limit=pool)

        # ── слияние ────────────────────────────────────────────────────────
        fused: dict = {}
        payload_of: dict = {}
        cosine_of: dict = {}
        for place, (point, _adjusted) in enumerate(vector_ranked):
            fused[point.id] = fused.get(point.id, 0.0) + 1.0 / (RRF_K + place + 1)
            payload_of[point.id] = point.payload or {}
            cosine_of[point.id] = float(point.score)

        for place, (position, _weight) in enumerate(keyword_ranked):
            payload = self._keyword_payloads[position]
            # ★Дисквалификация, а не смягчающая поправка. В векторной ветви
            # подпись к графику наказывается вычитанием из косинуса — там есть
            # откалиброванная шкала, на которой такая поправка что-то значит. В
            # словесной ветви такой шкалы нет, и подобрать её сейчас было бы
            # угадыванием. Зато про эти два класса известно определённое: ни
            # шкала оси графика, ни служебная страница не содержат данных, за
            # которыми сюда приходят. Не пускать их вовсе честнее, чем пускать
            # с вычитанием, взятым с потолка.
            if carries_no_data(payload.get("text", "")):
                continue
            identifier = self._keyword_ids[position]
            fused[identifier] = fused.get(identifier, 0.0) + 1.0 / (RRF_K + place + 1)
            payload_of.setdefault(identifier, payload)

        # Та же развязка на слиянии: здесь равенство ВЕРОЯТНЕЕ, чем в векторной
        # ветви, потому что оценка RRF складывается из мест в двух списках и
        # принимает мало разных значений — совпадения обычное дело, а не край.
        order = sorted(
            fused,
            key=lambda identifier: (-fused[identifier], _identity_of(payload_of[identifier])),
        )

        # ★Порог применяется и к пришедшим по словам, для чего им приходится
        # ДОСЧИТАТЬ косинус. Пропустить фрагмент мимо порога только потому, что
        # он пришёл другой дорогой, значило бы отдать в ответ то, что векторная
        # ветвь отвергла бы как не относящееся к вопросу.
        missing = [identifier for identifier in order if identifier not in cosine_of]
        if missing:
            cosine_of.update(self._cosines(missing, query_vector))

        hits = []
        for identifier in order:
            score = cosine_of.get(identifier, 0.0)
            if score < floor:
                continue
            payload = payload_of[identifier]
            hits.append(
                Hit(
                    text=payload.get("text", ""),
                    # Отдаём исходный косинус, а не поправленный: поправка —
                    # внутренний приём упорядочивания, и показывать её как
                    # «близость» значило бы отчитываться числом, которого
                    # модель эмбеддингов не выдавала.
                    score=score,
                    source_name=payload.get("source_name", "unknown"),
                    date=payload.get("date", ""),
                    page=int(payload.get("page", 0)),
                    page_end=int(payload.get("page_end", payload.get("page", 0))),
                    context=payload.get("context", ""),
                    # Старые точки в коллекции этих полей не имеют: до них
                    # сверки не было, и подставлять им "прочитано напрямую"
                    # значило бы задним числом заверить непроверенное.
                    confidence=payload.get("confidence", "unchecked"),
                    caveats=tuple(payload.get("caveats", ())),
                )
            )
            if len(hits) == top_k:
                break
        return hits

    def _cosines(self, identifiers: list, query_vector: list[float]) -> dict:
        """Косинус между запросом и указанными точками — досчёт для словесной ветви."""
        import math

        points = self.client.retrieve(
            collection_name=self.collection,
            ids=identifiers,
            with_payload=False,
            with_vectors=True,
        )
        query_norm = math.sqrt(sum(value * value for value in query_vector)) or 1.0
        result = {}
        for point in points:
            vector = point.vector
            if isinstance(vector, dict):  # именованные векторы — берём единственный
                vector = next(iter(vector.values()))
            if not vector:
                continue
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            # strict: разная размерность — это порча индекса. Усечённое
            # скалярное произведение вернуло бы правдоподобное число, и ошибка
            # уехала бы в ранжирование незамеченной.
            dot = sum(a * b for a, b in zip(query_vector, vector, strict=True))
            result[point.id] = dot / (query_norm * norm)
        return result

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return int(self.client.count(self.collection).count)


@lru_cache(maxsize=1)
def get_store() -> ReportStore:
    """Process-wide store. Cached because loading the encoder costs seconds."""
    return ReportStore()
