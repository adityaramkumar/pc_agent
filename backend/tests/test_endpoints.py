"""HTTP endpoint tests for /ingest and /memories family."""

from __future__ import annotations

from fastapi.testclient import TestClient


def _make_event(url: str = "https://example.com", ts: int = 1, **overrides: object) -> dict:
    base = {
        "type": "page_visit",
        "url": url,
        "title": "Example",
        "text": "extracted main content",
        "ts": ts,
        "meta": {"source": "test"},
    }
    base.update(overrides)  # type: ignore[arg-type]
    return base


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_ingest_then_list_round_trip(client: TestClient) -> None:
    payload = {"events": [_make_event(ts=10), _make_event(ts=20)]}
    response = client.post("/ingest", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["ingested"] == 2
    assert len(body["ids"]) == 2

    listing = client.get("/memories").json()
    assert listing["total"] == 2
    assert [e["ts"] for e in listing["events"]] == [20, 10]


def test_ingest_empty_batch(client: TestClient) -> None:
    response = client.post("/ingest", json={"events": []})
    assert response.status_code == 200
    assert response.json() == {"ingested": 0, "ids": []}


def test_get_memory_404(client: TestClient) -> None:
    response = client.get("/memories/9999")
    assert response.status_code == 404


def test_delete_memory(client: TestClient) -> None:
    [event_id] = client.post("/ingest", json={"events": [_make_event()]}).json()["ids"]

    delete_resp = client.delete(f"/memories/{event_id}")
    assert delete_resp.status_code == 204

    follow_up = client.get(f"/memories/{event_id}")
    assert follow_up.status_code == 404


def test_memories_pagination_bounds(client: TestClient) -> None:
    for limit in (0, 1001):
        response = client.get(f"/memories?limit={limit}")
        assert response.status_code == 400
    response = client.get("/memories?offset=-1")
    assert response.status_code == 400
