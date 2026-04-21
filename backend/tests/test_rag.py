"""Tests for the deterministic pieces of the retriever (RRF, recency)."""

from __future__ import annotations

from app.rag import recency_factor, reciprocal_rank_fusion


def test_rrf_basic_fusion() -> None:
    rankings = {
        "vec": [10, 11, 12, 13],
        "fts": [12, 14, 10, 15],
    }
    fused = reciprocal_rank_fusion(rankings, c=60)
    # 10 appears at rank 1 in vec and rank 3 in fts -> highest score
    assert max(fused, key=lambda d: fused[d][0]) in (10, 12)
    # 12 appears at rank 1 in fts and rank 3 in vec -> high score
    # Both 10 and 12 should outrank 13 and 15 which appear once.
    score10 = fused[10][0]
    score13 = fused[13][0]
    score15 = fused[15][0]
    assert score10 > score13
    assert score10 > score15


def test_rrf_reports_sources() -> None:
    rankings = {"vec": [1, 2], "fts": [2, 3]}
    fused = reciprocal_rank_fusion(rankings)
    assert fused[1][1] == ("vec",)
    assert set(fused[2][1]) == {"vec", "fts"}
    assert fused[3][1] == ("fts",)


def test_recency_factor_decays() -> None:
    now_ms = 1_000_000_000_000
    halflife = 7.0
    fresh = recency_factor(now_ms, now_ms=now_ms, halflife_days=halflife)
    one_halflife_old = recency_factor(
        now_ms - int(halflife * 86_400 * 1000), now_ms=now_ms, halflife_days=halflife
    )
    way_old = recency_factor(
        now_ms - int(halflife * 86_400 * 1000 * 10), now_ms=now_ms, halflife_days=halflife
    )
    assert fresh == 1.0
    assert abs(one_halflife_old - 0.5) < 1e-6
    assert way_old < 1e-2
