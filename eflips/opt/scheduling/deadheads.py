"""
The deadheads to nearby stops in the database.

A scheduling graph may contain edges to departures from nearby stops (see :mod:`eflips.opt.scheduling.nearby`). Each of
them needs an empty trip ("deadhead") from the arrival stop to the nearby stop. If the graph carries energy
consumptions, the consumption of these deadheads must be known before solving. As it depends on the consumption model of
the caller, it is computed in two steps:

1. :func:`materialize_deadhead_candidates` writes an empty trip for every such edge into a temporary rotation, so the
   caller can compute their consumption like the consumption of any other trip.
2. :func:`assign_delta_socs` puts the consumption of the trips onto the nodes and of the deadheads onto the edges.

:func:`eflips.opt.scheduling.write_back_rotation_plan` then moves the deadheads that are used into their rotations and
deletes the others (see :func:`discard_deadhead_candidates`).

The deadheads always use routes created here (one per pair of stops, named by :func:`deadhead_route_name`), never
other routes between the same stops, which may e.g. be passenger routes with many stops in between.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Mapping, Tuple

import networkx as nx  # type: ignore
import sqlalchemy.orm.session
from eflips.model import (
    AssocRouteStation,
    Rotation,
    Route,
    Station,
    StopTime,
    Trip,
    TripType,
)
from eflips.model.util import geometry_has_z, get_altitudes
from geoalchemy2.shape import from_shape
from sqlalchemy import func

from eflips.opt.scheduling.nearby import DeadheadRoute
from eflips.opt.util import LatLon, line_with_altitudes

DEADHEAD_TRIP_ATTRIBUTE = "deadhead_trip_id"
"""The edge attribute holding the ID of the candidate deadhead trip (see :func:`materialize_deadhead_candidates`)."""

CANDIDATE_ROTATION_ATTRIBUTE = "deadhead_candidate_rotation_id"
"""The graph attribute holding the ID of the rotation of the candidate deadhead trips."""

CANDIDATE_ROTATION_NAME = "Deadhead candidates (temporary)"


def deadhead_route_name(origin: Station, destination: Station) -> str:
    """The name of the route of the deadheads from ``origin`` to ``destination``."""
    return f"Leerfahrt {origin.name} → {destination.name}"


@dataclass(frozen=True)
class _DeadheadEdge:
    """An edge of a scheduling graph that needs a deadhead."""

    trip: Trip
    following_trip: Trip
    deadhead: DeadheadRoute
    data: Dict[str, object]
    """The edge's attributes (the dict of the graph, changes are written to the graph)."""

    @property
    def origin(self) -> Station:
        return self.trip.route.arrival_station

    @property
    def destination(self) -> Station:
        return self.following_trip.route.departure_station


def _deadhead_edges(
    graph: nx.Graph,
    session: sqlalchemy.orm.session.Session,
    edges: Iterable[Tuple[int, int]] | None = None,
) -> List[_DeadheadEdge]:
    """The edges of ``graph`` (or the given ones) that need a deadhead."""
    result = []
    for u, v in graph.edges if edges is None else edges:
        data = graph.edges[u, v]
        deadhead = DeadheadRoute.from_edge_attributes(data)
        if deadhead is not None:
            result.append(
                _DeadheadEdge(
                    session.get_one(Trip, u), session.get_one(Trip, v), deadhead, data
                )
            )
    return result


@dataclass
class _DeadheadRouteFactory:
    """Finds or creates the routes of the deadheads to nearby stops, one per pair of stops."""

    session: sqlalchemy.orm.session.Session
    scenario_id: int
    altitude_by_latlon: Dict[LatLon, float]
    """The altitudes of the vertices of the routes to be created; empty if the geometries have no Z coordinate."""
    _routes: Dict[Tuple[int, int], Route] = field(default_factory=dict)

    @staticmethod
    def create(
        session: sqlalchemy.orm.session.Session,
        scenario_id: int,
        edges: Iterable[_DeadheadEdge],
    ) -> "_DeadheadRouteFactory":
        """A factory for the routes of the given edges. The altitudes of all routes to be created are looked up in one
        batch."""
        factory = _DeadheadRouteFactory(session, scenario_id, altitude_by_latlon={})
        if geometry_has_z():
            latlons: List[LatLon] = []
            for edge in edges:
                if factory._existing_route(edge.origin, edge.destination) is None:
                    latlons.extend(
                        (lat, lon) for lon, lat in edge.deadhead.geometry.coords
                    )
            unique = list(dict.fromkeys(latlons))
            if len(unique) > 0:
                factory.altitude_by_latlon = dict(zip(unique, get_altitudes(unique)))
        return factory

    def _existing_route(self, origin: Station, destination: Station) -> Route | None:
        return (
            self.session.query(Route)
            .filter(
                Route.scenario_id == self.scenario_id,
                Route.departure_station_id == origin.id,
                Route.arrival_station_id == destination.id,
                Route.name == deadhead_route_name(origin, destination),
            )
            .order_by(Route.id)
            .first()
        )

    def route(
        self, origin: Station, destination: Station, deadhead: DeadheadRoute
    ) -> Route:
        """The deadhead route from origin to destination: the one created before, or a new one following the
        deadhead's geometry."""
        key = (origin.id, destination.id)
        if key in self._routes:
            return self._routes[key]
        route = self._existing_route(origin, destination)
        if route is None:
            route = self._new_route(origin, destination, deadhead)
        self._routes[key] = route
        return route

    def _new_route(
        self, origin: Station, destination: Station, deadhead: DeadheadRoute
    ) -> Route:
        # Measure the 2D shape in the database, so the distance matches the route's geometry check constraint. The Z
        # values are altitudes in meters, while X/Y are degrees, so a 3D length would mix units.
        shape_2d = from_shape(deadhead.geometry, srid=4326)
        distance = self.session.query(func.ST_Length(shape_2d, True)).scalar()
        if distance > 0:
            geom = (
                from_shape(
                    line_with_altitudes(deadhead.geometry, self.altitude_by_latlon),
                    srid=4326,
                )
                if geometry_has_z()
                else shape_2d
            )
        else:
            # A degenerate geometry (e.g. a single point). Keep the routed distance, without a geometry.
            distance, geom = deadhead.distance, None

        route = Route(
            scenario_id=self.scenario_id,
            departure_station=origin,
            arrival_station=destination,
            line_id=None,
            name=deadhead_route_name(origin, destination),
            distance=distance,
            geom=geom,
        )
        route.assoc_route_stations = [
            AssocRouteStation(
                scenario_id=self.scenario_id,
                station=origin,
                route=route,
                elapsed_distance=0,
            ),
            AssocRouteStation(
                scenario_id=self.scenario_id,
                station=destination,
                route=route,
                elapsed_distance=distance,
            ),
        ]
        self.session.add(route)
        return route


def _deadhead_times(edge: _DeadheadEdge) -> Tuple[datetime, datetime]:
    """The departure and arrival of the deadhead: it arrives just in time for the departure of the following trip."""
    departure_time = edge.following_trip.departure_time - edge.deadhead.duration
    if edge.origin.id == edge.destination.id or departure_time < edge.trip.arrival_time:
        raise ValueError(
            f"The deadhead between trips {edge.trip.id} and {edge.following_trip.id} does not fit."
        )
    return departure_time, edge.following_trip.departure_time


def _new_deadhead_trip(
    edge: _DeadheadEdge, rotation: Rotation, routes: _DeadheadRouteFactory
) -> Trip:
    """A new empty trip for the deadhead of ``edge``, in ``rotation``."""
    departure_time, arrival_time = _deadhead_times(edge)
    scenario_id = edge.following_trip.scenario_id
    deadhead_trip = Trip(
        scenario_id=scenario_id,
        route=routes.route(edge.origin, edge.destination, edge.deadhead),
        rotation=rotation,
        trip_type=TripType.EMPTY,
        departure_time=departure_time,
        arrival_time=arrival_time,
    )
    deadhead_trip.stop_times = [
        StopTime(
            scenario_id=scenario_id,
            trip=deadhead_trip,
            station=edge.origin,
            arrival_time=departure_time,
            dwell_duration=timedelta(seconds=0),
        ),
        StopTime(
            scenario_id=scenario_id,
            trip=deadhead_trip,
            station=edge.destination,
            arrival_time=arrival_time,
            dwell_duration=timedelta(seconds=0),
        ),
    ]
    routes.session.add(deadhead_trip)
    return deadhead_trip


def materialize_deadhead_candidates(
    graph: nx.Graph, session: sqlalchemy.orm.session.Session
) -> Dict[Tuple[int, int], int]:
    """
    Write an empty trip into the database for every edge of ``graph`` that needs a deadhead to a nearby stop, so their
    energy consumption can be computed like that of any other trip (see :func:`assign_delta_socs`).

    The trips are put into one new rotation (named :data:`CANDIDATE_ROTATION_NAME`) of the vehicle type of the trips of
    the graph. They overlap, so the rotation is not a valid one: it must be removed again by
    :func:`eflips.opt.scheduling.write_back_rotation_plan` (which keeps the trips that are used) or
    :func:`discard_deadhead_candidates`. The routes of the deadheads are kept.

    :param graph: A graph created by :func:`eflips.opt.scheduling.create_graph`. Its edges get the ID of their deadhead
        trip as attribute :data:`DEADHEAD_TRIP_ATTRIBUTE`, the graph the ID of the rotation as
        :data:`CANDIDATE_ROTATION_ATTRIBUTE`.
    :param session: An open database session
    :return: The ID of the deadhead trip of each edge needing one
    """
    if graph.graph.get(CANDIDATE_ROTATION_ATTRIBUTE) is not None:
        raise ValueError("The deadhead candidates of this graph already exist.")
    edges = _deadhead_edges(graph, session)
    if len(edges) == 0:
        return {}

    trips = [session.get_one(Trip, node) for node in graph.nodes]
    vehicle_type_ids = {trip.rotation.vehicle_type_id for trip in trips}
    scenario_ids = {trip.scenario_id for trip in trips}
    if len(vehicle_type_ids) != 1 or len(scenario_ids) != 1:
        raise ValueError(
            "The trips of the graph must belong to one scenario and vehicle type."
        )
    scenario_id = scenario_ids.pop()

    rotation = Rotation(
        scenario_id=scenario_id,
        vehicle_type_id=vehicle_type_ids.pop(),
        allow_opportunity_charging=False,
        name=CANDIDATE_ROTATION_NAME,
    )
    session.add(rotation)
    routes = _DeadheadRouteFactory.create(session, scenario_id, edges)
    deadhead_trips = [_new_deadhead_trip(edge, rotation, routes) for edge in edges]
    session.flush()

    result = {}
    for edge, deadhead_trip in zip(edges, deadhead_trips):
        edge.data[DEADHEAD_TRIP_ATTRIBUTE] = deadhead_trip.id
        result[(edge.trip.id, edge.following_trip.id)] = deadhead_trip.id
    graph.graph[CANDIDATE_ROTATION_ATTRIBUTE] = rotation.id
    return result


def assign_delta_socs(graph: nx.Graph, delta_socs: Mapping[int, float]) -> None:
    """
    Set the energy consumption of the trips (node weights) and of the deadheads to nearby stops (edge ``delta_soc``).

    :param graph: A graph created by :func:`eflips.opt.scheduling.create_graph`, with its deadheads materialized by
        :func:`materialize_deadhead_candidates`
    :param delta_socs: The energy consumption by trip ID (range 0-1, with 1 being a full battery discharge), for the
        trips of the graph and the deadhead trips
    :raises ValueError: If a consumption is missing or a deadhead has not been materialized
    """
    missing = [node for node in graph.nodes if node not in delta_socs]
    if len(missing) > 0:
        raise ValueError(f"No delta_soc for the trips {sorted(missing)[:10]}.")
    for node in graph.nodes:
        _, duration = graph.nodes[node]["weight"]
        graph.nodes[node]["weight"] = (delta_socs[node], duration)

    for u, v, data in graph.edges(data=True):
        if DeadheadRoute.from_edge_attributes(data) is None:
            continue
        deadhead_trip_id = data.get(DEADHEAD_TRIP_ATTRIBUTE)
        if deadhead_trip_id is None:
            raise ValueError(
                f"The deadhead between trips {u} and {v} has no trip. Call materialize_deadhead_candidates() first."
            )
        if deadhead_trip_id not in delta_socs:
            raise ValueError(f"No delta_soc for the deadhead trip {deadhead_trip_id}.")
        data["delta_soc"] = delta_socs[deadhead_trip_id]


def discard_deadhead_candidates(
    graph: nx.Graph, session: sqlalchemy.orm.session.Session
) -> None:
    """
    Delete the rotation of the candidate deadhead trips of ``graph`` (see :func:`materialize_deadhead_candidates`) with
    the trips still in it. Does nothing if the graph has none.
    """
    rotation_id = graph.graph.pop(CANDIDATE_ROTATION_ATTRIBUTE, None)
    if rotation_id is None:
        return
    session.flush()
    for trip in session.query(Trip).filter(Trip.rotation_id == rotation_id).all():
        for stop_time in trip.stop_times:
            session.delete(stop_time)
        session.delete(trip)
    session.flush()
    rotation = session.get(Rotation, rotation_id)
    if rotation is not None:
        session.delete(rotation)
        session.flush()


def add_deadhead_trips(
    rot_graph: nx.Graph, session: sqlalchemy.orm.session.Session
) -> None:
    """
    Put the deadheads of the edges of ``rot_graph`` (a solution) into the rotations of the trips they lead to: the
    candidate trip if it was materialized, else a new trip.
    """
    to_create = []
    for edge in _deadhead_edges(rot_graph, session):
        rotation = edge.following_trip.rotation
        deadhead_trip_id = edge.data.get(DEADHEAD_TRIP_ATTRIBUTE)
        if deadhead_trip_id is None:
            to_create.append(edge)
            continue
        deadhead_trip = session.get_one(Trip, deadhead_trip_id)
        if (
            deadhead_trip.departure_time,
            deadhead_trip.arrival_time,
        ) != _deadhead_times(edge):
            raise ValueError(
                f"The deadhead trip {deadhead_trip_id} does not match the edge {edge.trip.id} -> "
                f"{edge.following_trip.id}."
            )
        deadhead_trip.rotation = rotation
    if len(to_create) > 0:
        routes = _DeadheadRouteFactory.create(
            session, to_create[0].following_trip.scenario_id, to_create
        )
        for edge in to_create:
            _new_deadhead_trip(edge, edge.following_trip.rotation, routes)
