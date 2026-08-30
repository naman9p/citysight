"""Step 17 tests: SQLite city registry (cameras + directed links)."""

import pytest

from phase2_city import (
    Camera,
    CameraLink,
    CameraValidationError,
    CameraNotFoundError,
    CameraInUseError,
    SQLiteCityRepository,
)


@pytest.fixture
def repo():
    r = SQLiteCityRepository(":memory:")
    yield r
    r.close()


def cam(cid="CAM_01", **kw):
    base = dict(
        camera_id=cid, name=f"Cam {cid}", latitude=28.6, longitude=77.2,
        road_name="Demo Road", heading_deg=90.0, zone="demo", enabled=True,
    )
    base.update(kw)
    return Camera(**base)


def link(a="CAM_01", b="CAM_02", **kw):
    base = dict(
        from_camera_id=a, to_camera_id=b, distance_m=1000.0,
        road_name="Demo Road", travel_direction="eastbound",
    )
    base.update(kw)
    return CameraLink(**base)


# --- camera model validation -------------------------------------------
def test_invalid_latitude():
    with pytest.raises(CameraValidationError):
        cam(latitude=91.0)


def test_invalid_longitude():
    with pytest.raises(CameraValidationError):
        cam(longitude=-181.0)


def test_invalid_heading():
    with pytest.raises(CameraValidationError):
        cam(heading_deg=360.0)
    with pytest.raises(CameraValidationError):
        cam(heading_deg=-1.0)


def test_empty_camera_id_rejected():
    with pytest.raises(CameraValidationError):
        cam(cid="")


def test_empty_name_rejected():
    with pytest.raises(CameraValidationError):
        cam(name="  ")


# --- camera CRUD --------------------------------------------------------
def test_save_and_get_camera(repo):
    repo.save_camera(cam())
    got = repo.get_camera("CAM_01")
    assert got is not None
    assert got.camera_id == "CAM_01"
    assert got.enabled is True


def test_get_missing_camera_returns_none(repo):
    assert repo.get_camera("NOPE") is None


def test_upsert_camera(repo):
    repo.save_camera(cam(name="Old"))
    repo.save_camera(cam(name="New", enabled=False))
    got = repo.get_camera("CAM_01")
    assert got.name == "New"
    assert got.enabled is False
    assert len(repo.list_cameras()) == 1


def test_list_cameras(repo):
    repo.save_camera(cam("CAM_02"))
    repo.save_camera(cam("CAM_01"))
    ids = [c.camera_id for c in repo.list_cameras()]
    assert ids == ["CAM_01", "CAM_02"]


def test_delete_unused_camera(repo):
    repo.save_camera(cam())
    assert repo.delete_camera("CAM_01") is True
    assert repo.get_camera("CAM_01") is None
    assert repo.delete_camera("CAM_01") is False


# --- link model validation ---------------------------------------------
def test_reject_self_link():
    with pytest.raises(CameraValidationError):
        link(a="CAM_01", b="CAM_01")


def test_reject_nonpositive_distance():
    with pytest.raises(CameraValidationError):
        link(distance_m=0)
    with pytest.raises(CameraValidationError):
        link(distance_m=-5)


# --- link CRUD + referential integrity ---------------------------------
def test_create_and_get_link(repo):
    repo.save_camera(cam("CAM_01"))
    repo.save_camera(cam("CAM_02"))
    repo.save_link(link())
    got = repo.get_link("CAM_01", "CAM_02")
    assert got is not None
    assert got.distance_m == 1000.0


def test_list_links(repo):
    repo.save_camera(cam("CAM_01"))
    repo.save_camera(cam("CAM_02"))
    repo.save_link(link())
    assert len(repo.list_links()) == 1


def test_upsert_link(repo):
    repo.save_camera(cam("CAM_01"))
    repo.save_camera(cam("CAM_02"))
    repo.save_link(link(distance_m=1000.0))
    repo.save_link(link(distance_m=2500.0, travel_direction="westbound"))
    got = repo.get_link("CAM_01", "CAM_02")
    assert got.distance_m == 2500.0
    assert got.travel_direction == "westbound"
    assert len(repo.list_links()) == 1


def test_delete_link(repo):
    repo.save_camera(cam("CAM_01"))
    repo.save_camera(cam("CAM_02"))
    repo.save_link(link())
    assert repo.delete_link("CAM_01", "CAM_02") is True
    assert repo.get_link("CAM_01", "CAM_02") is None


def test_reject_missing_source_camera(repo):
    repo.save_camera(cam("CAM_02"))
    with pytest.raises(CameraNotFoundError):
        repo.save_link(link(a="CAM_01", b="CAM_02"))


def test_reject_missing_destination_camera(repo):
    repo.save_camera(cam("CAM_01"))
    with pytest.raises(CameraNotFoundError):
        repo.save_link(link(a="CAM_01", b="CAM_99"))


def test_directed_link_not_symmetric(repo):
    repo.save_camera(cam("CAM_01"))
    repo.save_camera(cam("CAM_02"))
    repo.save_link(link(a="CAM_01", b="CAM_02"))
    assert repo.get_link("CAM_01", "CAM_02") is not None
    assert repo.get_link("CAM_02", "CAM_01") is None


def test_prevent_deleting_referenced_camera(repo):
    repo.save_camera(cam("CAM_01"))
    repo.save_camera(cam("CAM_02"))
    repo.save_link(link(a="CAM_01", b="CAM_02"))
    with pytest.raises(CameraInUseError):
        repo.delete_camera("CAM_01")
    with pytest.raises(CameraInUseError):
        repo.delete_camera("CAM_02")
    # Once the link is gone, deletion is allowed.
    repo.delete_link("CAM_01", "CAM_02")
    assert repo.delete_camera("CAM_01") is True


# --- sqlite specifics ---------------------------------------------------
def test_schema_idempotent_memory():
    r1 = SQLiteCityRepository(":memory:")
    r1.save_camera(cam())
    r1._migrate()  # safe to call again
    assert r1.get_camera("CAM_01") is not None
    r1.close()


def test_cameras_and_links_coexist(repo):
    repo.save_camera(cam("CAM_01"))
    repo.save_camera(cam("CAM_02"))
    repo.save_link(link())
    assert len(repo.list_cameras()) == 2
    assert len(repo.list_links()) == 1
