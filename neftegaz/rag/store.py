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

import uuid
from dataclasses import dataclass
from functools import lru_cache

from neftegaz.config import settings

__all__ = ["Hit", "ReportStore", "get_store"]


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk with everything a citation needs."""

    text: str
    score: float
    source_name: str
    date: str
    page: int
    page_end: int

    def as_claim(self, text: str | None = None) -> dict:
        """Shape this hit for :mod:`neftegaz.tools.citations`."""
        return {
            "source_type": "report",
            "text": self.text if text is None else text,
            "source_name": self.source_name,
            "date": self.date,
            "page": self.page,
        }


class ReportStore:
    """Thin wrapper over Qdrant: index chunks, search them, report health."""

    def __init__(self, collection: str | None = None):
        self.collection = collection or settings.collection
        self._client = None
        self._encoder = None
        self._dimension: int | None = None

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
            vectors = self._embed_passages([c["text"] for c in batch])
            points = [
                PointStruct(
                    id=uuid.uuid4().hex,
                    vector=vector,
                    payload={
                        "text": chunk["text"],
                        "source_name": chunk["source_name"],
                        "date": chunk["date"],
                        "page": chunk["page"],
                        "page_end": chunk.get("page_end", chunk["page"]),
                    },
                )
                for chunk, vector in zip(batch, vectors)
            ]
            self.client.upsert(collection_name=self.collection, points=points)
            written += len(points)
        return written

    # ── read path ──────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int | None = None, min_score: float | None = None) -> list[Hit]:
        """Return the best-matching chunks above the score floor.

        An empty list is a meaningful answer: it means the corpus does not
        cover the question, and the caller should fall back to the web rather
        than answer from a weak match.
        """
        if not self.client.collection_exists(self.collection):
            return []
        top_k = settings.top_k if top_k is None else top_k
        floor = settings.min_score if min_score is None else min_score

        found = self.client.query_points(
            collection_name=self.collection,
            query=self._embed_query(query),
            limit=top_k,
            with_payload=True,
        ).points

        hits = []
        for point in found:
            if point.score < floor:
                continue
            payload = point.payload or {}
            hits.append(
                Hit(
                    text=payload.get("text", ""),
                    score=float(point.score),
                    source_name=payload.get("source_name", "unknown"),
                    date=payload.get("date", ""),
                    page=int(payload.get("page", 0)),
                    page_end=int(payload.get("page_end", payload.get("page", 0))),
                )
            )
        return hits

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return int(self.client.count(self.collection).count)


@lru_cache(maxsize=1)
def get_store() -> ReportStore:
    """Process-wide store. Cached because loading the encoder costs seconds."""
    return ReportStore()
