import lzma
import os
import random
from datetime import timedelta
from tempfile import gettempdir
from typing import Tuple, List, Dict
from urllib.parse import urlparse

import networkx as nx
import pytest
from eflips.model import VehicleType, Scenario, Trip, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from eflips.opt.scheduling import (
    passenger_trips_by_vehicle_type,
    create_graph,
    solve,
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
            assert graph.nodes[node]["weight"][0] == None
            assert 0 <= graph.nodes[node]["weight"][1] <= 1

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

    def test_solver_time_limit(self, trip_list):
        # Now, create the graph for the smallest of the vehicle types
        maximum_schedule_duration = timedelta(hours=3)

        graph = create_graph(
            trip_list, maximum_schedule_duration=maximum_schedule_duration
        )

        solution = solve(graph, write_to_file=False)

        for connected_set in nx.weakly_connected_components(graph):
            total_duration = sum(
                [graph.nodes[node]["weight"][1] for node in connected_set]
            )
            assert total_duration <= maximum_schedule_duration.total_seconds()

            # Make sure that an for each edge in the result, there was an edge in the input graph
            for edge in solution.edges:
                assert edge in graph.edges

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
        )

        solution = solve(graph, write_to_file=True)

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
