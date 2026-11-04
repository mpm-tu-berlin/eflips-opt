"""
Connections to departures from *nearby* stops.

Normally, a trip can only be followed by a trip departing from the stop it arrives at. With this module, it can also be
followed by a trip departing from one of the nearest other stops, if the bus spends its break at the arrival stop and
then drives a short deadhead, arriving just in time for the departure. Such a connection is an edge between the two
(real) trips that carries the deadhead's data, see :class:`DeadheadRoute`.
"""

import logging
import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import openrouteservice  # type: ignore
import shapely.wkt
from eflips.model import Station, Trip
from geoalchemy2.shape import to_shape
from shapely import LineString
from sqlalchemy import func
from sqlalchemy.orm import Session, aliased

from eflips.opt.scheduling.util import crosses_service_day
from eflips.opt.util import LonLat, ors_route

logger = logging.getLogger(__name__)

MIN_DEADHEAD_DURATION = timedelta(seconds=1)
"""The shortest duration of a deadhead trip: a trip must arrive after it departs."""


@dataclass(frozen=True)
class DeadheadRoute:
    """A deadhead between two stops, as stored on the edges of the scheduling graph."""

    distance: float
    """The distance in meters. Always positive."""

    duration: timedelta
    """The driving time, in whole seconds (rounded up). At least :data:`MIN_DEADHEAD_DURATION`."""

    geometry: LineString
    """The route taken, with ``(lon, lat)`` vertices."""

    def to_edge_attributes(self) -> Dict[str, float | int | str]:
        """
        The edge attributes describing this deadhead. They are scalars, so the graph can be stored as GraphML.
        """
        return {
            "deadhead_distance": self.distance,
            "deadhead_duration": int(self.duration.total_seconds()),
            "deadhead_geometry": self.geometry.wkt,
        }

    @staticmethod
    def from_edge_attributes(data: Mapping[str, Any]) -> "DeadheadRoute | None":
        """
        The deadhead of an edge, or None if the edge is a connection at the same stop.

        :param data: The attributes of an edge of the scheduling graph
        """
        if "deadhead_distance" not in data:
            return None
        geometry = shapely.wkt.loads(data["deadhead_geometry"])
        assert isinstance(geometry, LineString)
        return DeadheadRoute(
            distance=float(data["deadhead_distance"]),
            duration=timedelta(seconds=int(data["deadhead_duration"])),
            geometry=geometry,
        )


DeadheadRouter = Callable[[Station, Station], DeadheadRoute | None]
"""Calculates the deadhead from the first to the second stop, or None if there is no usable one."""


@dataclass(frozen=True)
class NearbyDepartureOptions:
    """Options for connecting trips to departures from nearby stops, see :func:`eflips.opt.scheduling.create_graph`."""

    router: DeadheadRouter | None = None
    """Calculates the deadheads. None uses OpenRouteService, configured from the environment
    (see :func:`eflips.opt.util.ors_client_from_env`)."""

    max_nearby_stations: int = 5
    """How many of the nearest other stops to consider."""

    max_straight_line_distance: float = 1000.0
    """The maximum straight-line distance to a nearby stop in meters."""

    extra_wait_tolerance: timedelta = timedelta(0)
    """If a trip can be followed by a departure from its arrival stop, a departure from a nearby stop is only
    considered if it does not wait longer than the first one at the arrival stop plus this tolerance."""


def _lon_lat(station: Station) -> LonLat:
    # The column is typed as Geometry, but loaded as a WKBElement
    point = to_shape(station.geom)  # type: ignore[arg-type]
    return point.x, point.y


def ors_deadhead_router(client: openrouteservice.Client) -> DeadheadRouter:
    """
    A :data:`DeadheadRouter` using OpenRouteService.

    The router returns None if a stop has no location, the route has no length or OpenRouteService cannot route
    between the stops.

    :param client: An OpenRouteService client
    """

    def route(origin: Station, destination: Station) -> DeadheadRoute | None:
        if origin.geom is None or destination.geom is None:
            return None
        try:
            distance, duration, geometry = ors_route(
                _lon_lat(origin), _lon_lat(destination), client
            )
        except openrouteservice.exceptions.ApiError as e:
            logger.warning(
                f"No deadhead route from {origin.name} to {destination.name}: {e}"
            )
            return None
        if distance <= 0:
            return None
        return DeadheadRoute(
            distance=distance,
            duration=max(timedelta(seconds=math.ceil(duration)), MIN_DEADHEAD_DURATION),
            geometry=geometry,
        )

    return route


def nearby_stations(
    session: Session,
    origin_ids: Iterable[int],
    candidate_ids: Iterable[int],
    max_count: int,
    max_distance: float,
) -> Dict[int, List[int]]:
    """
    Find the nearest candidate stops for each origin stop, using one spatial query.

    :param session: An open database session
    :param origin_ids: The IDs of the stops to find nearby stops for
    :param candidate_ids: The IDs of the stops that may be returned
    :param max_count: The maximum number of nearby stops per origin
    :param max_distance: The maximum straight-line distance in meters
    :return: For each origin ID that has nearby stops, the IDs of up to ``max_count`` nearby stops, nearest first.
        An origin is never its own nearby stop. Stops without a location are ignored.
    """
    origin_ids, candidate_ids = list(origin_ids), list(candidate_ids)
    if len(origin_ids) == 0 or len(candidate_ids) == 0 or max_count <= 0:
        return {}

    origin = aliased(Station)
    candidate = aliased(Station)
    # The third argument makes this a distance in meters on both PostGIS (where the geometries are cast to geography)
    # and SpatiaLite (where it is the "use_ellipsoid" flag). The Z coordinate is dropped, as some SpatiaLite versions
    # return NULL for the spheroid distance of 3D points.
    force_2d = (
        func.CastToXY
        if session.get_bind().dialect.name == "sqlite"
        else func.ST_Force2D
    )
    distance = func.ST_Distance(force_2d(origin.geom), force_2d(candidate.geom), True)
    ranked = (
        session.query(
            origin.id.label("origin_id"),
            candidate.id.label("candidate_id"),
            func.row_number()
            .over(partition_by=origin.id, order_by=(distance, candidate.id))
            .label("rank"),
        )
        .filter(
            origin.id.in_(origin_ids),
            candidate.id.in_(candidate_ids),
            origin.id != candidate.id,
            origin.geom.isnot(None),
            candidate.geom.isnot(None),
            distance <= max_distance,
        )
        .subquery()
    )
    rows = (
        session.query(ranked.c.origin_id, ranked.c.candidate_id)
        .filter(ranked.c.rank <= max_count)
        .order_by(ranked.c.origin_id, ranked.c.rank)
        .all()
    )

    result: Dict[int, List[int]] = {}
    for origin_id, candidate_id in rows:
        result.setdefault(origin_id, []).append(candidate_id)
    return result


def nearby_connections(
    trip: Trip,
    candidates: Sequence[Tuple[DeadheadRoute, Sequence[Trip]]],
    baseline_wait: timedelta | None,
    minimum_break_time: timedelta,
    regular_break_time: timedelta,
    maximum_break_time: timedelta,
    extra_wait_tolerance: timedelta,
    do_not_cross_service_day_breaks: bool,
) -> List[Tuple[Trip, DeadheadRoute]]:
    """
    Select the departures from nearby stops that can follow a trip.

    The bus spends its break at the arrival stop of ``trip`` and then drives the deadhead, arriving at the nearby stop
    just in time for the departure. The wait (the time out of passenger service) is the time between the arrival of
    ``trip`` and the departure of the following trip.

    - If ``trip`` can be followed at its arrival stop (``baseline_wait`` is not None), departures from nearby stops
      are selected if they wait at most ``baseline_wait + extra_wait_tolerance`` (and at most
      ``maximum_break_time``).
    - Otherwise, all departures waiting at most ``regular_break_time`` are selected. If there are none, the one
      departing first within ``maximum_break_time`` is selected.

    :param trip: The trip to find following trips for
    :param candidates: For each nearby stop, the deadhead to it and the trips departing there
    :param baseline_wait: The shortest wait for a following trip at the arrival stop, or None if there is none
    :param minimum_break_time: The minimum break at the arrival stop (before the deadhead)
    :param regular_break_time: See :func:`eflips.opt.scheduling.create_graph`
    :param maximum_break_time: See :func:`eflips.opt.scheduling.create_graph`
    :param extra_wait_tolerance: See :class:`NearbyDepartureOptions`
    :param do_not_cross_service_day_breaks: See :func:`eflips.opt.scheduling.create_graph`
    :return: The following trips and the deadhead to each of them
    """
    feasible: List[Tuple[timedelta, Trip, DeadheadRoute]] = []
    for deadhead, departures in candidates:
        for following_trip in departures:
            wait = following_trip.departure_time - trip.arrival_time
            if wait - deadhead.duration < minimum_break_time:
                continue
            if wait > maximum_break_time:
                continue
            if do_not_cross_service_day_breaks and crosses_service_day(
                trip, following_trip
            ):
                continue
            feasible.append((wait, following_trip, deadhead))

    if baseline_wait is not None:
        limit = baseline_wait + extra_wait_tolerance
        return [(t, d) for wait, t, d in feasible if wait <= limit]

    regular = [(t, d) for wait, t, d in feasible if wait <= regular_break_time]
    if len(regular) > 0 or len(feasible) == 0:
        return regular
    _, first_trip, first_deadhead = min(feasible, key=lambda x: (x[0], x[1].id))
    return [(first_trip, first_deadhead)]
