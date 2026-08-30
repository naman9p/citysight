"""Tests for Step 15 watchlist + deduplicated alerts (repository + API + dashboard).
No network/Docker/models/external services."""

import threading

import httpx
import pytest

from phase1_anpr.api import create_server
from phase1_anpr.observation.observation_builder import PlateObservation
from phase1_anpr.persistence import (
    SQLiteObservationRepository,
    SQLiteWatchlistRepository,
    WatchlistError,
)


def obs(event_id, plate="MH12AB1234", camera_id="cam_01",
        ts="2026-08-31T10:00:00", status="accepted", conf=0.9):
    abstained = status == "abstained"
    return PlateObservation(
        event_id=event_id, camera_id=camera_id, track_id=1, timestamp=ts,
        plate_raw=None if abstained else plate,
        plate_normalized=None if abstained else plate,
        confidence=conf, status=status,
        detector_confidence=None if abstained else 0.8,
        ocr_confidence=None if abstained else 0.85,
        quality_score=None if abstained else 0.7,
        best_frame_number=None if abstained else 5,
        plate_image_path=None, model_version="phase1-anpr-0.1.0",
    )


@pytest.fixture
def wl():
    r = SQLiteWatchlistRepository(":memory:", now=lambda: "2026-08-31T00:00:00+00:00")
    yield r
    r.close()


# --- repository: watchlist ----------------------------------------------------

def test_add_watchlist_valid_and_normalizes(wl):
    entry = wl.add("mh-12-ab-1234", label="stolen")
    assert entry["normalized_plate"] == "MH12AB1234"
    assert entry["enabled"] == 1 and entry["label"] == "stolen"
    assert wl.list()[0]["watchlist_id"] == entry["watchlist_id"]


def test_add_invalid_plate_rejected(wl):
    with pytest.raises(WatchlistError):
        wl.add("")
    with pytest.raises(WatchlistError):
        wl.add("---")
    with pytest.raises(WatchlistError):
        wl.add(None)


# --- repository: alerts -------------------------------------------------------

def test_accepted_exact_match_creates_alert(wl):
    wl.add("MH12AB1234", label="watch")
    created = wl.process_observation(obs("e1"))
    assert len(created) == 1
    a = created[0]
    assert a["event_id"] == "e1" and a["camera_id"] == "cam_01"
    assert a["normalized_plate"] == "MH12AB1234" and a["status"] == "accepted"
    assert len(wl.list_alerts()) == 1


@pytest.mark.parametrize("status", ["review", "abstained"])
def test_review_abstained_no_alert(wl, status):
    wl.add("MH12AB1234")
    assert wl.process_observation(obs("e1", status=status)) == []
    assert wl.list_alerts() == []


def test_duplicate_processing_one_alert(wl):
    wl.add("MH12AB1234")
    assert len(wl.process_observation(obs("e1"))) == 1
    # Reprocessing the same observation is idempotent.
    assert wl.process_observation(obs("e1")) == []
    assert len(wl.list_alerts()) == 1


def test_disabled_watchlist_does_not_alert(wl):
    e = wl.add("MH12AB1234")
    assert wl.disable(e["watchlist_id"]) is True
    assert wl.process_observation(obs("e1")) == []


def test_no_match_no_alert(wl):
    wl.add("KA05MN4321")
    assert wl.process_observation(obs("e1", plate="MH12AB1234")) == []


# --- API ----------------------------------------------------------------------

@pytest.fixture
def client():
    obs_repo = SQLiteObservationRepository(":memory:")
    wl_repo = SQLiteWatchlistRepository(":memory:")
    obs_repo.save(obs("e1"))
    server = create_server(obs_repo, port=0, watchlist_repo=wl_repo)
    host, port = server.server_address
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with httpx.Client(base_url=f"http://{host}:{port}") as c:
        c.wl_repo = wl_repo
        c.obs_repo = obs_repo
        yield c
    server.shutdown()
    obs_repo.close()
    wl_repo.close()


def test_api_watchlist_crud_and_alerts(client):
    assert client.get("/watchlist").json() == []
    r = client.post("/watchlist", json={"plate": "mh12ab1234", "label": "x"})
    assert r.status_code == 201
    wid = r.json()["watchlist_id"]
    assert r.json()["normalized_plate"] == "MH12AB1234"
    assert len(client.get("/watchlist").json()) == 1

    # Generate an alert via the repo, then read it through the API.
    client.wl_repo.process_observation(client.obs_repo.get("e1"))
    alerts = client.get("/alerts").json()
    assert len(alerts) == 1 and alerts[0]["event_id"] == "e1"

    # Disable via API.
    assert client.delete(f"/watchlist/{wid}").status_code == 200
    assert client.get("/watchlist").json()[0]["enabled"] == 0


def test_api_invalid_plate_4xx(client):
    assert client.post("/watchlist", json={"plate": "---"}).status_code == 400
    assert client.post("/watchlist", json={}).status_code == 400


def test_api_delete_unknown_404(client):
    assert client.delete("/watchlist/does-not-exist").status_code == 404


def test_dashboard_still_serves_and_has_sections(client):
    body = client.get("/dashboard").text
    assert client.get("/dashboard").status_code == 200
    assert "Watchlist" in body and "Recent alerts" in body
    assert "/watchlist" in body and "/alerts" in body
    assert "innerHTML" not in body
