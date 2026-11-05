import lzma
import math
import os
import random
from datetime import datetime, timedelta, timezone
from tempfile import gettempdir
from typing import Tuple, List, Dict
from urllib.parse import urlparse

import networkx as nx
import pytest
from eflips.model import (
    VehicleType,
    Scenario,
    Trip,
    Base,
    Rotation,
    Station,
    TripType,
)
from geoalchemy2.shape import from_shape
from shapely import LineString, Point
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session

from eflips.opt.scheduling import (
    passenger_trips_by_vehicle_type,
    create_graph,
    solve,
    write_back_rotation_plan,
)
from eflips.opt.scheduling.deadheads import (
    CANDIDATE_ROTATION_ATTRIBUTE,
    DEADHEAD_TRIP_ATTRIBUTE,
    assign_delta_socs,
    deadhead_route_name,
    discard_deadhead_candidates,
    materialize_deadhead_candidates,
)
import eflips.opt.scheduling.nearby as nearby_module
from eflips.opt.scheduling.nearby import (
    MIN_DEADHEAD_DURATION,
    STRAIGHT_LINE_SPEED_KMH,
    ors_deadhead_router,
    DeadheadRoute,
    NearbyDepartureOptions,
    nearby_connections,
    nearby_stations,
)


def database_url_components(database_url: str) -> Tuple[str, str, str, str, str, str]:
    """
    Extracts the components of a database URL.
    :param database_url: The URL of the database.
    :return: A tuple with the components of the URL: protocol, user, password, host, port, database name.
    """
    o = urlparse(database_url)
    if o.scheme != "postgresql":
        raise ValueError("Only PostgreSQL databases are supported.")
    if o.port is None:
        port = "5432"
    else:
        port = str(o.port)
    return o.scheme, o.username, o.password, o.hostname, port, o.path[1:]


def clear_db():
    DATABASE_URL = os.environ["DATABASE_URL"]

    path_to_this_file = os.path.dirname(os.path.abspath(__file__))
    path_to_clear_database_sql = os.path.join(
        path_to_this_file, "data", "clear_database.sql"
    )

    _, database_user, database_password, database_host, database_port, database_name = (
        database_url_components(DATABASE_URL)
    )

    # Clear the database
    os.environ["PGPASSWORD"] = database_password
    ret_val = os.system(
        f"psql -h {database_host} -U {database_user} -p {database_port} {database_name} -f {path_to_clear_database_sql}"
    )
    if ret_val != 0:
        raise ValueError("Failed to clear the database.")


def import_db():
    DATABASE_URL = os.environ["DATABASE_URL"]

    path_to_this_file = os.path.dirname(os.path.abspath(__file__))
    path_to_import_eflips_model_sql = os.path.join(
        path_to_this_file, "data", "eflips_one_day.sql.xz"
    )

    _, database_user, database_password, database_host, database_port, database_name = (
        database_url_components(DATABASE_URL)
    )

    # Decompress the sql to a temporary file
    with lzma.open(path_to_import_eflips_model_sql, "rb") as f:
        tempfile = os.path.join(gettempdir(), "eflips_one_day.sql")
        with open(tempfile, "wb") as f_out:
            f_out.write(f.read())

    # Import the eflips-model database
    ret_val = os.system(
        f"psql -h {database_host} -U {database_user} -p {database_port} {database_name} -f {tempfile}"
    )
    if ret_val != 0:
        raise ValueError("Failed to import the eflips-model database.")


def clear_and_import():
    """
    Uses psql system commands to clear the database and import the eflips-model database.
    """
    clear_db()
    import_db()


class TestScheduling:
    @pytest.fixture(scope="class")
    def session(self):
        if os.environ.get("DATABASE_URL") is None:
            raise ValueError("Please set the DATABASE_URL environment variable.")

        clear_and_import()
        engine = create_engine(os.environ["DATABASE_URL"])
        session = Session(engine)

        yield session

        session.close()

        Base.metadata.drop_all(engine)

        engine.dispose()

    @pytest.fixture(scope="class")
    def trip_list(self, session) -> List[Trip]:
        scenario = session.query(Scenario).filter(Scenario.id == 1).one()
        trips_by_vehicle_type = passenger_trips_by_vehicle_type(scenario, session)
        vehicle_type = min(
            trips_by_vehicle_type.keys(), key=lambda x: len(trips_by_vehicle_type[x])
        )
        return trips_by_vehicle_type[vehicle_type]

    def test_create_graph_no_weights(self, session):
        scenario = session.query(Scenario).filter(Scenario.id == 1).one()
        trips_by_vehicle_type = passenger_trips_by_vehicle_type(scenario, session)

        assert isinstance(trips_by_vehicle_type, dict)
        for key, value in trips_by_vehicle_type.items():
            assert isinstance(key, VehicleType)
            assert isinstance(value, list)
            for trip in value:
                assert isinstance(trip, Trip)
                assert trip.rotation.vehicle_type == key

        # Now, create the graph for the smallest of the vehicle types
        vehicle_type = min(
            trips_by_vehicle_type.keys(), key=lambda x: len(trips_by_vehicle_type[x])
        )
        graph = create_graph(trips_by_vehicle_type[vehicle_type])

        assert isinstance(graph, nx.graph.Graph)
        assert nx.is_directed_acyclic_graph(graph)

        # Each node's weight should be a tuple of 2 None values
        assert len(graph.nodes) > 0
        for node in graph.nodes:
            assert len(graph.nodes[node]["weight"]) == 2
            assert graph.nodes[node]["weight"] == (None, None)

        # Each edge's weight should be a value greater than 0
        assert len(graph.edges) > 0
        for edge in graph.edges:
            assert graph.edges[edge]["weight"] >= 0

    def test_create_graph_distance_weight(self, trip_list):
        energy_consumption = 2.0  # kWh/km
        battery_capacity = 400.0  # kWh
        max_distance = battery_capacity / energy_consumption
        consumptions: Dict[int, float] = dict()
        for trip in trip_list:
            consumptions[trip.id] = (trip.route.distance / 1000) / max_distance

        graph = create_graph(trip_list, delta_socs=consumptions)

        # Each node's weight should be a tuple containing
        # 1) a float between 0 and 1
        # 2) 2 None values
        assert len(graph.nodes) > 0
        for node in graph.nodes:
            assert len(graph.nodes[node]["weight"]) == 2
            assert 0 <= graph.nodes[node]["weight"][0] <= 1
            assert graph.nodes[node]["weight"][1] == None

        # Each edge's weight should be a value greater than 0
        assert len(graph.edges) > 0
        for edge in graph.edges:
            assert graph.edges[edge]["weight"] >= 0

    def test_create_graph_time_weight(self, trip_list):
        graph = create_graph(trip_list, maximum_schedule_duration=timedelta(days=1))

        # Each node's weight should be a tuple containing
        # 1) a float between 0 and 1
        # 2) 2 None values
        assert len(graph.nodes) > 0
        for node in graph.nodes:
            assert len(graph.nodes[node]["weight"]) == 2
            assert graph.nodes[node]["weight"][0] is None
            assert 0 <= graph.nodes[node]["weight"][1] <= 86400

        # Each edge's weight should be a value greater than 0
        assert len(graph.edges) > 0
        for edge in graph.edges:
            assert graph.edges[edge]["weight"] >= 0

    def test_solver_no_limit(self, trip_list):
        # Now, create the graph for the smallest of the vehicle types
        graph = create_graph(trip_list, maximum_schedule_duration=None)

        solution = solve(graph, write_to_file=False)

        for connected_set in nx.weakly_connected_components(solution):
            # Make sure that an for each edge in the result, there was an edge in the input graph
            for edge in solution.edges:
                assert edge in graph.edges

    def test_solver_time_limit(self, trip_list, session):
        # Now, create the graph for the smallest of the vehicle types
        maximum_schedule_duration = timedelta(hours=3)

        graph = create_graph(
            trip_list, maximum_schedule_duration=maximum_schedule_duration
        )

        solution = solve(
            graph,
            write_to_file=False,
            maximum_schedule_duration=maximum_schedule_duration,
        )

        write_back_rotation_plan(solution, session)

        for rotation in (
            session.query(Rotation)
            .join(Trip)
            .filter(Trip.id.in_([trip.id for trip in trip_list]))
        ):
            first_trip_start = rotation.trips[0].departure_time
            last_trip_end = rotation.trips[-1].arrival_time
            assert last_trip_end - first_trip_start <= maximum_schedule_duration

    def test_solve_both_limits(self, trip_list):
        # Create a dict of trip_id -> energy consumption
        # COntaining a (repeatable) random value between 0.05 and 0.15 for each trip
        saved_state = random.getstate()
        random.seed(42)
        consumptions: Dict[int, float] = dict()
        for trip in trip_list:
            consumptions[trip.id] = random.uniform(0.05, 0.15)
        random.setstate(saved_state)

        # Now, create the graph for the smallest of the vehicle types
        maximum_schedule_duration = timedelta(hours=3)
        graph = create_graph(
            trip_list,
            maximum_schedule_duration=maximum_schedule_duration,
            delta_socs=consumptions,
            nearby_departures=None,
        )

        solution = solve(
            graph,
            write_to_file=True,
            maximum_schedule_duration=maximum_schedule_duration,
        )

        for connected_set in nx.weakly_connected_components(solution):
            total_duration = sum(
                [graph.nodes[node]["weight"][1] for node in connected_set]
            )
            total_energy = sum(
                [graph.nodes[node]["weight"][0] for node in connected_set]
            )
            assert total_duration <= maximum_schedule_duration.total_seconds()
            assert total_energy <= 1

            # Make sure that an for each edge in the result, there was an edge in the input graph
            for edge in solution.edges:
                assert edge in graph.edges


def _transient_trip(
    trip_id: int, departure: datetime, arrival: datetime, rotation_start: datetime
) -> Trip:
    """A trip that is not in the database, in a rotation whose first trip departs at ``rotation_start``."""
    first_trip = Trip(departure_time=rotation_start, arrival_time=rotation_start)
    trip = Trip(id=trip_id, departure_time=departure, arrival_time=arrival)
    Rotation(trips=[first_trip, trip])
    return trip


class TestNearbyConnections:
    """The selection of departures from nearby stops (no database needed)."""

    START = datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc)
    DEADHEAD = DeadheadRoute(
        distance=500.0,
        duration=timedelta(minutes=3),
        geometry=LineString([(13.0, 52.0), (13.005, 52.0)]),
    )

    def trip_waiting(self, trip_id: int, minutes: float, rotation_day: int = 0) -> Trip:
        """A trip departing ``minutes`` after the arrival of :meth:`arriving_trip`."""
        departure = self.START + timedelta(minutes=minutes)
        return _transient_trip(
            trip_id,
            departure,
            departure + timedelta(minutes=20),
            self.START + timedelta(days=rotation_day) - timedelta(hours=1),
        )

    def arriving_trip(self) -> Trip:
        """The trip whose followers are selected; it arrives at START."""
        return _transient_trip(
            1,
            self.START - timedelta(minutes=30),
            self.START,
            self.START - timedelta(hours=1),
        )

    def select(self, waits: List[float], **kwargs) -> List[int]:
        """The waits (in minutes) of the departures selected out of departures waiting ``waits`` minutes."""
        arguments = dict(
            baseline_wait=None,
            minimum_break_time=timedelta(0),
            regular_break_time=timedelta(minutes=30),
            maximum_break_time=timedelta(minutes=60),
            extra_wait_tolerance=timedelta(0),
            do_not_cross_service_day_breaks=False,
        )
        arguments.update(kwargs)
        departures = [self.trip_waiting(100 + i, wait) for i, wait in enumerate(waits)]
        result = nearby_connections(
            self.arriving_trip(), [(self.DEADHEAD, departures)], **arguments
        )
        assert all(deadhead == self.DEADHEAD for _, deadhead in result)
        return sorted(
            round((t.departure_time - self.START).total_seconds() / 60)
            for t, _ in result
        )

    def test_not_longer_than_baseline(self):
        """Nearby departures may not wait longer than the first departure at the arrival stop."""
        assert self.select([5, 10, 12], baseline_wait=timedelta(minutes=10)) == [5, 10]

    def test_tolerance(self):
        """The tolerance allows nearby departures waiting a bit longer than at the arrival stop."""
        assert self.select(
            [5, 10, 12, 13],
            baseline_wait=timedelta(minutes=10),
            extra_wait_tolerance=timedelta(minutes=2),
        ) == [5, 10, 12]

    def test_tolerance_capped_at_maximum_break(self):
        """The tolerance never allows waiting longer than the maximum break."""
        assert self.select(
            [58, 62],
            baseline_wait=timedelta(minutes=55),
            extra_wait_tolerance=timedelta(minutes=10),
        ) == [58]

    def test_break_before_deadhead(self):
        # The deadhead takes 3 minutes, so a 2 minute break only fits with a wait of at least 5 minutes
        """The minimum break is taken before the deadhead."""
        assert self.select(
            [4, 5],
            baseline_wait=timedelta(minutes=10),
            minimum_break_time=timedelta(minutes=2),
        ) == [5]

    def test_no_baseline_regular_window(self):
        """Without a departure at the arrival stop, all nearby departures within the regular break qualify."""
        assert self.select([20, 25, 40]) == [20, 25]

    def test_no_baseline_fallback_to_first_within_maximum(self):
        """Without a regular one, only the first nearby departure within the maximum break qualifies."""
        assert self.select([50, 40, 70]) == [40]

    def test_no_baseline_nothing_within_maximum(self):
        """Departures after the maximum break never qualify."""
        assert self.select([70]) == []

    def test_service_day_break(self):
        """Connections across a service day break are not offered when they are forbidden."""
        arriving = self.arriving_trip()
        same_day = self.trip_waiting(100, 10)
        next_day = self.trip_waiting(101, 12, rotation_day=1)
        result = nearby_connections(
            arriving,
            [(self.DEADHEAD, [same_day, next_day])],
            baseline_wait=None,
            minimum_break_time=timedelta(0),
            regular_break_time=timedelta(minutes=30),
            maximum_break_time=timedelta(minutes=60),
            extra_wait_tolerance=timedelta(0),
            do_not_cross_service_day_breaks=True,
        )
        assert [t.id for t, _ in result] == [100]

    def test_edge_attribute_round_trip(self):
        """A deadhead survives being stored as (scalar) edge attributes."""
        attributes = self.DEADHEAD.to_edge_attributes()
        assert all(
            isinstance(value, (int, float, str)) for value in attributes.values()
        )
        assert DeadheadRoute.from_edge_attributes(attributes) == self.DEADHEAD
        assert DeadheadRoute.from_edge_attributes({"weight": 60}) is None


class TestEdgeDeltaSoc:
    """Edges consuming energy themselves (no database needed)."""

    @staticmethod
    def chain(second_edge_delta_soc: float | None) -> nx.DiGraph:
        """Three trips of 0.3 each in a chain; the second connection optionally consumes energy itself."""
        graph = nx.DiGraph()
        for node in (1, 2, 3):
            graph.add_node(node, weight=(0.3, None))
        graph.add_edge(1, 2, weight=60)
        graph.add_edge(2, 3, weight=60)
        if second_edge_delta_soc is not None:
            graph.edges[2, 3]["delta_soc"] = second_edge_delta_soc
        return graph

    def test_without_edge_delta_soc(self):
        """Without edge energy, the chain is one block."""
        assert sorted(solve(self.chain(None)).edges) == [(1, 2), (2, 3)]

    def test_edge_delta_soc_splits_rotation(self):
        """With edge energy, the chain exceeds the battery and is split before the expensive connection."""
        assert sorted(solve(self.chain(0.5)).edges) == [(1, 2)]

    def test_edge_delta_soc_validated(self):
        """Edge energy above a full battery is rejected."""
        with pytest.raises(ValueError):
            solve(self.chain(1.5))


class TestNearbyDepartures:
    """Graphs with departures from nearby stops, on the one-day database (deadheads from the dummy ORS of conftest.py,
    unless BASE_URL is set)."""

    @pytest.fixture(scope="class")
    def session(self):
        """A session on a fresh import of the one-day database."""
        if os.environ.get("DATABASE_URL") is None:
            raise ValueError("Please set the DATABASE_URL environment variable.")

        clear_and_import()
        engine = create_engine(os.environ["DATABASE_URL"])
        session = Session(engine)

        yield session

        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()

    @pytest.fixture(scope="class")
    def trip_list(self, session) -> List[Trip]:
        """The passenger trips of the smallest vehicle type."""
        scenario = session.query(Scenario).filter(Scenario.id == 1).one()
        trips_by_vehicle_type = passenger_trips_by_vehicle_type(scenario, session)
        vehicle_type = min(
            trips_by_vehicle_type.keys(), key=lambda x: len(trips_by_vehicle_type[x])
        )
        return trips_by_vehicle_type[vehicle_type]

    @staticmethod
    def blue_edges(graph: nx.DiGraph) -> List[Tuple[int, int]]:
        """The edges to departures from nearby stops."""
        return [e for e in graph.edges if graph.edges[e]["color"] == "blue"]

    def test_disabled(self, trip_list):
        """Without nearby departures, no edge carries a deadhead."""
        graph = create_graph(trip_list, nearby_departures=None)
        assert self.blue_edges(graph) == []
        for edge in graph.edges:
            assert DeadheadRoute.from_edge_attributes(graph.edges[edge]) is None

    def test_adds_nearby_edges(self, session, trip_list):
        """Nearby edges are added on top of the same-stop ones, and each of them is feasible and worthwhile."""
        options = NearbyDepartureOptions()
        minimum_break_time = timedelta(minutes=2)
        same_stop = create_graph(
            trip_list, minimum_break_time=minimum_break_time, nearby_departures=None
        )
        graph = create_graph(
            trip_list, minimum_break_time=minimum_break_time, nearby_departures=options
        )
        assert nx.is_directed_acyclic_graph(graph)

        # All same-stop connections are kept
        for edge in same_stop.edges:
            assert graph.edges[edge] == same_stop.edges[edge]

        blue = self.blue_edges(graph)
        assert len(blue) > 0
        assert len(graph.edges) == len(same_stop.edges) + len(blue)

        trips = {trip.id: trip for trip in trip_list}
        for u, v in blue:
            trip, following_trip = trips[u], trips[v]
            deadhead = DeadheadRoute.from_edge_attributes(graph.edges[u, v])
            assert deadhead is not None
            origin = trip.route.arrival_station
            destination = following_trip.route.departure_station
            assert origin != destination
            wait = following_trip.departure_time - trip.arrival_time
            assert graph.edges[u, v]["weight"] == int(wait.total_seconds())
            assert wait - deadhead.duration >= minimum_break_time
            assert "delta_soc" not in graph.edges[u, v]

            # The nearby stop is one of the nearest, and close enough
            distance = session.query(
                func.ST_Distance(origin.geom, destination.geom, True)
            ).scalar()
            assert distance <= options.max_straight_line_distance

            # Not longer than the shortest wait at the arrival stop, if there is one
            same_stop_waits = [
                same_stop.edges[e]["weight"] for e in same_stop.out_edges(u)
            ]
            if len(same_stop_waits) > 0:
                assert graph.edges[u, v]["weight"] <= min(same_stop_waits)

    def test_nearby_stations(self, session, trip_list):
        """The spatial query returns the nearest other stops within the radius, nearest first."""
        origin_ids = {trip.route.arrival_station_id for trip in trip_list}
        candidate_ids = {trip.route.departure_station_id for trip in trip_list}
        result = nearby_stations(session, origin_ids, candidate_ids, 3, 1000.0)
        assert len(result) > 0
        for origin_id, nearby_ids in result.items():
            assert origin_id in origin_ids
            assert 1 <= len(nearby_ids) <= 3
            assert origin_id not in nearby_ids
            assert set(nearby_ids) <= candidate_ids
            origin = session.get(Station, origin_id)
            distances = [
                session.query(
                    func.ST_Distance(origin.geom, session.get(Station, i).geom, True)
                ).scalar()
                for i in nearby_ids
            ]
            assert distances == sorted(distances)
            assert all(d <= 1000.0 for d in distances)

    def test_unpriced_deadheads(self, trip_list):
        """Solving a graph with trip energy but unpriced deadheads is refused."""
        consumptions = {trip.id: 0.01 for trip in trip_list}
        graph = create_graph(trip_list, delta_socs=consumptions)
        assert len(self.blue_edges(graph)) > 0
        assert all("delta_soc" not in graph.edges[e] for e in graph.edges)
        with pytest.raises(ValueError, match="deadhead"):
            solve(graph)

    def test_priced_deadheads(self, session, trip_list):
        """Materialized and priced deadheads limit the blocks, and write-back keeps only the used ones."""
        graph = create_graph(trip_list)
        blue = self.blue_edges(graph)
        assert len(blue) > 0

        candidates = materialize_deadhead_candidates(graph, session)
        assert set(candidates) == set(blue)
        assert len(set(candidates.values())) == len(blue)
        rotation_id = graph.graph[CANDIDATE_ROTATION_ATTRIBUTE]
        trips = {trip.id: trip for trip in trip_list}
        for (u, v), deadhead_trip_id in candidates.items():
            deadhead_trip = session.get(Trip, deadhead_trip_id)
            assert graph.edges[u, v][DEADHEAD_TRIP_ATTRIBUTE] == deadhead_trip_id
            assert deadhead_trip.rotation_id == rotation_id
            assert deadhead_trip.trip_type == TripType.EMPTY
            assert deadhead_trip.arrival_time == trips[v].departure_time
            assert deadhead_trip.departure_time >= trips[u].arrival_time
            # Only the routes created for deadheads are used
            assert deadhead_trip.route.name == deadhead_route_name(
                trips[u].route.arrival_station, trips[v].route.departure_station
            )
            assert len(deadhead_trip.route.assoc_route_stations) == 2
        with pytest.raises(ValueError):
            materialize_deadhead_candidates(graph, session)

        # Made-up consumptions: the trips by distance, the deadheads expensive
        delta_socs = {trip.id: trip.route.distance / 1000 * 0.002 for trip in trip_list}
        delta_socs.update({trip_id: 0.05 for trip_id in candidates.values()})
        with pytest.raises(ValueError):
            assign_delta_socs(
                graph, {k: v for k, v in delta_socs.items() if k in trips}
            )
        assign_delta_socs(graph, delta_socs)
        for node in graph.nodes:
            assert graph.nodes[node]["weight"][0] == delta_socs[node]
        for edge in blue:
            assert graph.edges[edge]["delta_soc"] == 0.05

        solution = solve(graph)
        for rotation_nodes in nx.weakly_connected_components(solution):
            subgraph = solution.subgraph(rotation_nodes)
            total = sum(delta_socs[n] for n in rotation_nodes) + sum(
                subgraph.edges[e].get("delta_soc", 0) for e in subgraph.edges
            )
            assert total <= 1 + 1e-6

        used = {e for e in solution.edges if e in candidates}
        assert len(used) > 0
        write_back_rotation_plan(solution, session)
        session.flush()

        # The used candidates are in the rotations of their following trips, the others are gone
        for edge, deadhead_trip_id in candidates.items():
            deadhead_trip = session.get(Trip, deadhead_trip_id)
            if edge in used:
                assert deadhead_trip is not None
                assert (
                    deadhead_trip.rotation_id == session.get(Trip, edge[1]).rotation_id
                )
            else:
                assert deadhead_trip is None
        assert session.get(Rotation, rotation_id) is None
        assert CANDIDATE_ROTATION_ATTRIBUTE not in solution.graph
        session.rollback()

    def test_discard_candidates(self, session, trip_list):
        """Discarding removes the candidate trips and their rotation."""
        graph = create_graph(trip_list)
        candidates = materialize_deadhead_candidates(graph, session)
        rotation_id = graph.graph[CANDIDATE_ROTATION_ATTRIBUTE]
        discard_deadhead_candidates(graph, session)
        assert session.get(Rotation, rotation_id) is None
        assert all(session.get(Trip, i) is None for i in candidates.values())
        discard_deadhead_candidates(graph, session)  # Nothing left to do
        session.rollback()

    def test_custom_router(self, trip_list):
        """A custom router is used, and asked once per pair of stops."""
        calls: List[Tuple[int, int]] = []

        def no_deadheads(origin: Station, destination: Station) -> None:
            """A router that finds no deadhead, recording its calls."""
            calls.append((origin.id, destination.id))
            return None

        graph = create_graph(
            trip_list, nearby_departures=NearbyDepartureOptions(router=no_deadheads)
        )
        assert self.blue_edges(graph) == []
        assert len(calls) > 0
        # Each pair of stops is routed only once
        assert len(calls) == len(set(calls))

    def test_needs_base_url(self, trip_list, monkeypatch):
        """The default router needs BASE_URL."""
        monkeypatch.delenv("BASE_URL")
        with pytest.raises(ValueError, match="BASE_URL"):
            create_graph(trip_list)

    def test_write_back(self, session, trip_list):
        """Write-back adds deadhead trips on routes that are reused when scheduling again."""
        graph = create_graph(trip_list)
        solution = solve(graph)
        used = [
            (u, v)
            for u, v in solution.edges
            if DeadheadRoute.from_edge_attributes(solution.edges[u, v]) is not None
        ]
        assert len(used) > 0, "The solution should use a deadhead to a nearby stop"
        # solve() keeps the edge attributes
        for edge in solution.edges:
            assert solution.edges[edge] == graph.edges[edge]

        write_back_rotation_plan(solution, session)
        session.flush()
        self.check_deadhead_trips(session, used)
        route_ids = self.deadhead_route_ids(session, used)
        assert len(route_ids) == len(
            {
                (
                    session.get(Trip, u).route.arrival_station_id,
                    session.get(Trip, v).route.departure_station_id,
                )
                for u, v in used
            }
        )

        # Scheduling again reuses the routes and replaces the deadhead trips
        trips = [session.get(Trip, trip.id) for trip in trip_list]
        solution = solve(create_graph(trips))
        write_back_rotation_plan(solution, session)
        session.flush()
        self.check_deadhead_trips(session, used)
        assert self.deadhead_route_ids(session, used) == route_ids
        session.rollback()

    @staticmethod
    def deadhead_route_ids(session, used: List[Tuple[int, int]]) -> set:
        """The routes of the deadheads before the following trips of ``used``."""
        ids = set()
        for u, v in used:
            following_trip = session.get(Trip, v)
            deadhead_trip = (
                session.query(Trip)
                .filter(
                    Trip.rotation_id == following_trip.rotation_id,
                    Trip.trip_type == TripType.EMPTY,
                    Trip.arrival_time == following_trip.departure_time,
                )
                .one()
            )
            ids.add(deadhead_trip.route_id)
        return ids

    @staticmethod
    def check_deadhead_trips(session, used: List[Tuple[int, int]]) -> None:
        """Each used deadhead sits between its trips, from the arrival to the departure stop, just in time."""
        for u, v in used:
            trip, following_trip = session.get(Trip, u), session.get(Trip, v)
            if trip.rotation_id != following_trip.rotation_id:
                continue  # The second solution may not use this connection
            rotation_trips = sorted(
                following_trip.rotation.trips, key=lambda t: t.departure_time
            )
            index = rotation_trips.index(following_trip)
            deadhead_trip = rotation_trips[index - 1]
            assert rotation_trips[index - 2] == trip
            assert deadhead_trip.trip_type == TripType.EMPTY
            assert deadhead_trip.departure_time >= trip.arrival_time
            assert deadhead_trip.arrival_time == following_trip.departure_time
            assert deadhead_trip.route.departure_station == trip.route.arrival_station
            assert (
                deadhead_trip.route.arrival_station
                == following_trip.route.departure_station
            )
            assert [st.station for st in deadhead_trip.stop_times] == [
                trip.route.arrival_station,
                following_trip.route.departure_station,
            ]


class TestOrsDeadheadRouter:
    """The OpenRouteService router's handling of routes without length (no database needed)."""

    @staticmethod
    def station(lon: float, lat: float) -> Station:
        """A stop that is not in the database."""
        return Station(name=f"{lon},{lat}", geom=from_shape(Point(lon, lat), srid=4326))

    @staticmethod
    def router_returning(monkeypatch, distance: float, duration: float):
        """An OpenRouteService router whose (patched) route always has the given distance and duration."""

        def fake_ors_route(start, end, client):
            """The route OpenRouteService would return."""
            line = (
                LineString([start, end]) if distance > 0 else LineString([start, start])
            )
            return distance, duration, line

        monkeypatch.setattr(nearby_module, "ors_route", fake_ors_route)
        return ors_deadhead_router(client=None)

    def test_routed(self, monkeypatch):
        """A routed deadhead keeps its distance; its duration is rounded up to whole seconds."""
        route = self.router_returning(monkeypatch, 120.0, 20.4)(
            self.station(13.0, 52.0), self.station(13.001, 52.0)
        )
        assert route.distance == 120.0
        assert route.duration == timedelta(seconds=21)

    def test_snapped_to_the_same_point(self, monkeypatch):
        # About 30 m apart, but both stops snap to the same point of the road network
        """A route without length between distinct stops becomes a straight line."""
        origin, destination = self.station(13.0, 52.0), self.station(13.0, 52.00027)
        route = self.router_returning(monkeypatch, 0.0, 0.0)(origin, destination)
        assert route is not None
        assert route.distance == pytest.approx(30.0, abs=0.5)
        assert route.duration == timedelta(
            seconds=math.ceil(route.distance / (STRAIGHT_LINE_SPEED_KMH / 3.6))
        )
        assert list(route.geometry.coords) == [(13.0, 52.0), (13.0, 52.00027)]

    def test_very_short(self, monkeypatch):
        """Very short deadheads take at least one second."""
        route = self.router_returning(monkeypatch, 0.4, 0.05)(
            self.station(13.0, 52.0), self.station(13.000005, 52.0)
        )
        assert route.duration == MIN_DEADHEAD_DURATION

    def test_same_place(self, monkeypatch):
        """Stops at the same place get no deadhead."""
        router = self.router_returning(monkeypatch, 0.0, 0.0)
        assert router(self.station(13.0, 52.0), self.station(13.0, 52.0)) is None
