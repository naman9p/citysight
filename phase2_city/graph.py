"""In-memory city camera topology graph (Step 17).

``CityCameraGraph`` is a thin read-only view over cameras and directed links.
It is NOT a routing engine: no shortest-path, no route reconstruction, no
external graph library. It only exposes the direct topology later steps need.
Direction is preserved: an A -> B link appears as outgoing for A and incoming
for B, never the reverse.
"""

from typing import Optional

from phase2_city.models import Camera, CameraLink


class CityCameraGraph:
    """Directed topology of cameras and links, built from repository data."""

    def __init__(self, cameras, links):
        self._cameras = {c.camera_id: c for c in cameras}
        self._links = {}
        self._outgoing = {}
        self._incoming = {}
        for link in links:
            self._links[link.key] = link
            self._outgoing.setdefault(link.from_camera_id, []).append(link)
            self._incoming.setdefault(link.to_camera_id, []).append(link)

    @classmethod
    def from_repository(cls, repo) -> "CityCameraGraph":
        """Build a graph snapshot from a CityRepository."""
        return cls(repo.list_cameras(), repo.list_links())

    def get_camera(self, camera_id: str) -> Optional[Camera]:
        return self._cameras.get(camera_id)

    def get_outgoing_links(self, camera_id: str) -> list:
        """Directed links leaving ``camera_id`` (empty list if none/unknown)."""
        return list(self._outgoing.get(camera_id, []))

    def get_incoming_links(self, camera_id: str) -> list:
        """Directed links arriving at ``camera_id`` (empty list if none/unknown)."""
        return list(self._incoming.get(camera_id, []))

    def get_link(self, from_camera_id: str, to_camera_id: str) -> Optional[CameraLink]:
        return self._links.get((from_camera_id, to_camera_id))

    def are_directly_connected(self, from_camera_id: str, to_camera_id: str) -> bool:
        """True iff a directed link ``from -> to`` exists (direction matters)."""
        return (from_camera_id, to_camera_id) in self._links

    def list_cameras(self) -> list:
        return list(self._cameras.values())

    def list_links(self) -> list:
        return list(self._links.values())

    @property
    def camera_count(self) -> int:
        return len(self._cameras)

    @property
    def link_count(self) -> int:
        return len(self._links)
