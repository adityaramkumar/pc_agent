"""Hybrid retrieval: vector + FTS5 + recency, fused with reciprocal rank.

The fused list of `RetrievedChunk` is what gets stuffed into the LLM
prompt. We intentionally keep the formula simple (RRF + linear recency
boost) because there's no offline eval harness, so cleverness here is hard
to validate.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.llm import GeminiClient, get_gemini_client
from app.store import RetrievedChunk, Store, get_store

logger = logging.getLogger(__name__)


VECTOR_K = 20
FTS_K = 20
FUSED_K = 8
RRF_C = 60.0
RECENCY_BOOST = 0.2
RECENCY_HALFLIFE_DAYS = 14.0


@dataclass(slots=True)
class RetrievalConfig:
    vector_k: int = VECTOR_K
    fts_k: int = FTS_K
    fused_k: int = FUSED_K
    rrf_c: float = RRF_C
    recency_boost: float = RECENCY_BOOST
    halflife_days: float = RECENCY_HALFLIFE_DAYS


def reciprocal_rank_fusion(
    rankings: dict[str, list[int]],
    *,
    c: float = RRF_C,
) -> dict[int, tuple[float, tuple[str, ...]]]:
    """Standard RRF: score(d) = sum over rankings of 1 / (c + rank(d))."""
    scores: dict[int, float] = {}
    sources: dict[int, list[str]] = {}
    for source, ids in rankings.items():
        for rank, doc_id in enumerate(ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (c + rank + 1)
            sources.setdefault(doc_id, []).append(source)
    return {doc_id: (s, tuple(sources[doc_id])) for doc_id, s in scores.items()}


def recency_factor(ts_ms: int, *, now_ms: int, halflife_days: float) -> float:
    """Exponential decay: 1.0 at ts==now, 0.5 at ts == now - halflife."""
    age_days = max(0.0, (now_ms - ts_ms) / 1000.0 / 86_400.0)
    if halflife_days <= 0:
        return 0.0
    return float(0.5 ** (age_days / halflife_days))


class Retriever:
    def __init__(
        self,
        store: Store | None = None,
        client: GeminiClient | None = None,
        settings: Settings | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._store = store or get_store()
        self._client = client or get_gemini_client()
        self._settings = settings or get_settings()
        self._config = config or RetrievalConfig()

    async def search(self, query: str) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        rankings: dict[str, list[int]] = {}

        # Vector branch: async because it may hit the network.
        try:
            vector = await self._client.embed_query(query)
            vec_results = self._store.vector_search(vector, k=self._config.vector_k)
            rankings["vec"] = [chunk_id for chunk_id, _ in vec_results]
        except Exception as exc:
            logger.warning("vector search failed, falling back to FTS only: %s", exc)

        # FTS5 branch: pure SQLite, no failure modes worth degrading for.
        fts_results = self._store.fts_search(query, k=self._config.fts_k)
        rankings["fts"] = [chunk_id for chunk_id, _ in fts_results]

        if not rankings or all(not v for v in rankings.values()):
            return []

        fused = reciprocal_rank_fusion(rankings, c=self._config.rrf_c)
        if not fused:
            return []

        chunk_ids = list(fused.keys())
        chunks = self._store.fetch_chunks(chunk_ids)

        # Apply recency multiplier on top of the RRF score.
        now_ms = int(time.time() * 1000)
        boosted: list[RetrievedChunk] = []
        for chunk in chunks:
            base, sources = fused.get(chunk.chunk_id, (0.0, ()))
            recency = recency_factor(
                chunk.ts, now_ms=now_ms, halflife_days=self._config.halflife_days
            )
            score = base * (1.0 + self._config.recency_boost * recency)
            boosted.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    event_id=chunk.event_id,
                    text=chunk.text,
                    ts=chunk.ts,
                    url=chunk.url,
                    title=chunk.title,
                    score=score,
                    sources=sources,
                )
            )

        boosted.sort(key=lambda c: c.score, reverse=True)
        return boosted[: self._config.fused_k]


_cached: Retriever | None = None


def get_retriever() -> Retriever:
    global _cached
    if _cached is None:
        _cached = Retriever()
    return _cached


def reset_retriever_for_tests() -> None:
    global _cached
    _cached = None
