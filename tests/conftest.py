"""
A dummy OpenRouteService for the tests.

If ``BASE_URL`` is not set, the tests run against a dummy instead of a real OpenRouteService server: every route is a
straight line, 1.3 times as long as the crow flies, driven at :data:`DUMMY_SPEED_KMH`. The responses are cached in a
temporary directory, so they never mix with the cache of real responses.

With ``BASE_URL`` set, nothing is patched and the real server is used.
"""

import math
import os
from tempfile import mkdtemp
from typing import Any, Dict, Iterator, Sequence, Tuple

import openrouteservice  # type: ignore
import polyline
import pytest

DUMMY_DETOUR_FACTOR = 1.3
DUMMY_SPEED_KMH = 25.0

USE_DUMMY_ORS = "BASE_URL" not in os.environ

if USE_DUMMY_ORS:
    # Set at import time, as the tests check for BASE_URL when they are collected
    os.environ["BASE_URL"] = "http://dummy-openrouteservice.invalid/ors/"
    os.environ["DEPOT_ROTATION_MATCHING_ORS_CACHE"] = mkdtemp(
        prefix="eflips-dummy-ors-cache-"
    )


def haversine_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """The distance between two ``(lon, lat)`` points in meters."""
    lon1, lat1, lon2, lat2 = map(math.radians, (*a, *b))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371000 * math.asin(math.sqrt(h))


def dummy_request(
    self: openrouteservice.Client,
    url: str,
    get_params: Dict[str, Any] | None = None,
    first_request_time: Any = None,
    retry_counter: int = 0,
    requests_kwargs: Any = None,
    post_json: Dict[str, Any] | None = None,
    dry_run: Any = None,
) -> Dict[str, Any]:
    """An OpenRouteService directions response for a straight line between the two requested points."""
    assert post_json is not None
    coordinates: Sequence[Tuple[float, float]] = post_json["coordinates"]
    start, end = ((float(lon), float(lat)) for lon, lat in coordinates)
    distance = DUMMY_DETOUR_FACTOR * haversine_distance(start, end)
    return {
        "routes": [
            {
                # The encoded polyline is (lat, lon)
                "geometry": polyline.encode([(start[1], start[0]), (end[1], end[0])]),
                "segments": [
                    {
                        "distance": distance,
                        "duration": distance / (DUMMY_SPEED_KMH / 3.6),  # km/h -> m/s
                    }
                ],
            }
        ]
    }


@pytest.fixture(autouse=True, scope="session")
def dummy_openrouteservice() -> Iterator[None]:
    """Route through the dummy instead of a real OpenRouteService server, unless BASE_URL was set."""
    if not USE_DUMMY_ORS:
        yield
        return
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(openrouteservice.Client, "request", dummy_request)
        yield
