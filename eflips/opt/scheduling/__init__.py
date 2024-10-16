#!/usr/bin/env python3

"""

Understand the electric vehicle scneduling problem as a Minimum Path Cover problem. This way, we will find *a* schedule
that has the minimum number of rotations, and from that hopefully derive a good schedule, that is SoC-Aware.

Electrified stations are taking into account in that a trip leading to an electrified station will not consume any
energy. This is a simplification, as we do not take into account that the vehicle may recharge multiple trips' worth of
energy at the station.

"""


import itertools
import json
import os
from datetime import timedelta
from tempfile import gettempdir
from typing import Dict, List, Tuple

import dash  # type: ignore
import dash_cytoscape as cyto  # type: ignore
import eflips_schedule_rust  # type: ignore
import networkx as nx  # type: ignore
import numpy as np
import sqlalchemy.orm.session
from dash import html
from eflips.model import Rotation, Scenario, Station, Trip, TripType, VehicleType
from networkx.classes import Graph  # type: ignore

from eflips.opt.scheduling.util import _validate_input_graph, _graph_to_json


def passenger_trips_by_vehicle_type(
    scenario: Scenario, session: sqlalchemy.orm.session.Session
) -> Dict[VehicleType, List[Trip]]:
    """
    Loads all trips from a given scenario and groups them by vehicle type. This is the precondition for creating the
    graph of the electric vehicle scheduling problem.

    :param scenario: A scenario object
    :param session: An open database session
    :return: A list of all trips, grouped by vehicle type
    """
    # Load all vehicle types for a given scenario
    vehicle_types = (
        session.query(VehicleType).filter(VehicleType.scenario == scenario).all()
    )
    passenger_trips_by_vehicle_type: Dict[VehicleType, List[Trip]] = {
        vehicle_type: [] for vehicle_type in vehicle_types
    }

    # Load all rotations
    for vehicle_type in vehicle_types:
        all_trips = (
            session.query(Trip)
            .filter(Trip.trip_type == TripType.PASSENGER)
            .join(Rotation)
            .filter(Rotation.vehicle_type == vehicle_type)
            .all()
        )
        passenger_trips_by_vehicle_type[vehicle_type] = all_trips

    # Remove vehicle types that have no trips
    passenger_trips_by_vehicle_type = {
        vehicle_type: trips
        for vehicle_type, trips in passenger_trips_by_vehicle_type.items()
        if len(trips) > 0
    }

    return passenger_trips_by_vehicle_type


def create_graph(
    trips: List[Trip],
    delta_socs: Dict[int, float] | None = None,
    maximum_schedule_duration: timedelta | None = None,
    minimum_break_time: timedelta = timedelta(minutes=0),
    regular_break_time: timedelta = timedelta(minutes=30),
    maximum_break_time: timedelta = timedelta(minutes=60),
    different_line_malus: int = 0,
    do_not_cross_service_day_breaks: bool = False,
) -> nx.Graph:
    """
    Turns a list of trips into a directed acyclic graph. The nodes are the trips, and the edges are the possible
    transitions between the trips. The edges are colored according to the time between the trips.

    :param trips: A list of trips
    :param delta_socs: A dictionary containing the energy consumption of each trip (range 0-1, with 1 being a full battery discharge). Set to None to disable.
    :param maximum_schedule_duration: The maximum duration of the schedule. We will not add any edges that would make the schedule longer than this. Set to None to disable.
    :param minimum_break_time: The minimum break time between two trips.
    :param regular_break_time: The regular break time between two trips. All trips following the trip in the regular
     break time are added as edges.
    :param maximum_break_time: The maximum break time between two trips. If no edge is added with the regular break time,
    the *first* trip before the maximum break time is added.
    :param different_line_malus: If two trips are on different lines, we add a malus to the wait time.
    :param do_not_cross_service_day_breaks: If True, we do not allow connections between trips that cross the service
    day break.

    :return: A directed acyclic graph havong the trips as nodes and the possible connections as edges.
    """
    # Divide the trips into dictionaries by departure station
    trips_by_departure_station: Dict[Station, List[Trip]] = {}

    for trip in trips:
        departure_station = trip.route.departure_station
        if departure_station not in trips_by_departure_station:
            trips_by_departure_station[departure_station] = []
        trips_by_departure_station[departure_station].append(trip)

    # Sort the lists of trips by departure time
    for trips in trips_by_departure_station.values():
        trips.sort(key=lambda trip: trip.departure_time)

    # Create a graph
    # Set up all the tip endpoints as nodes
    graph = nx.DiGraph()
    for trips in trips_by_departure_station.values():
        for trip in trips:
            # Add the energy consumption and time weight
            if delta_socs is not None:
                delta_soc = delta_socs[trip.id]
            else:
                delta_soc = None

            if maximum_schedule_duration is not None:
                duration_fraction = (
                    trip.arrival_time - trip.departure_time
                ) / maximum_schedule_duration
            else:
                duration_fraction = None

            graph.add_node(
                trip.id,
                name=f"{trip.route.departure_station.name} -> {trip.route.arrival_station.name} "
                f"({trip.departure_time.strftime('%H:%M')} - {trip.arrival_time.strftime('%H:%M')})",
                weight=(delta_soc, duration_fraction),
            )

    # For each trip, find all the possible following trips and add (directed) edges to them
    for trips in trips_by_departure_station.values():
        for trip in trips:
            arrival_station = trip.route.arrival_station

            # Identify all the trips that could follow this trip
            # These are the ones departing from the same station and starting after the arrival of the current trip
            # But not too late
            if arrival_station in trips_by_departure_station.keys():
                for following_trip in trips_by_departure_station[arrival_station]:
                    if (
                        following_trip.departure_time
                        >= trip.arrival_time + minimum_break_time
                        and following_trip.departure_time
                        <= trip.arrival_time + regular_break_time
                    ):
                        if do_not_cross_service_day_breaks:
                            # If we are not allowed to cross the service day break, we have to make an additional check:
                            # What is the date of the start of the following trip's rotation?
                            # What is the date of the end of the current trip's rotation?
                            # If they are not the same, we cannot connect the trips
                            current_trip_rotatiom_start = trip.rotation.trips[
                                0
                            ].departure_time.date()
                            following_trip_rotation_start = (
                                following_trip.rotation.trips[0].departure_time.date()
                            )
                            if (
                                current_trip_rotatiom_start
                                != following_trip_rotation_start
                            ):
                                continue

                        graph.add_edge(
                            trip.id,
                            following_trip.id,
                            color="gray",
                            weight=int(
                                (
                                    following_trip.departure_time - trip.arrival_time
                                ).total_seconds()
                            ),
                        )
                # If we have not added any edge, allow one edge up to 60 minutes
                if graph.out_degree(trip.id) == 0:
                    for following_trip in trips_by_departure_station[arrival_station]:
                        if (
                            following_trip.departure_time
                            >= trip.arrival_time + regular_break_time
                            and following_trip.departure_time
                            <= trip.arrival_time + maximum_break_time
                        ):
                            if do_not_cross_service_day_breaks:
                                # If we are not allowed to cross the service day break, we have to make an additional check:
                                # What is the date of the start of the following trip's rotation?
                                # What is the date of the end of the current trip's rotation?
                                # If they are not the same, we cannot connect the trips
                                current_trip_rotatiom_start = trip.rotation.trips[
                                    0
                                ].departure_time.date()
                                following_trip_rotation_start = (
                                    following_trip.rotation.trips[
                                        0
                                    ].departure_time.date()
                                )
                                if (
                                    current_trip_rotatiom_start
                                    != following_trip_rotation_start
                                ):
                                    continue

                            graph.add_edge(
                                trip.id,
                                following_trip.id,
                                color="red",
                                weight=int(
                                    (
                                        following_trip.departure_time
                                        - trip.arrival_time
                                    ).total_seconds()
                                ),
                            )
                            break

    return graph


def solve(graph: nx.Graph, write_to_file: bool = False) -> nx.Graph:
    """
    Solve the vehicle scheduling on a directed acyclinc graph.

    The vehicle scheduling problem needs to be formulated as a Directed Acyclic Graph (DAG) with the following properties:

    - Nodes are trips. Each trip must have an integer ID. Each trip may have a Tuple of up to eight weights. The weights indicate various cost properties of
    the trip and range from 0 to 1 (float). One example would be to have each first weight of the tuple represent the energy
    consumption of the trip, as fraction of the battery capacity. The second weight could represent the time of the trip as
    a fraction of 24 hours. The solver can then create schedules that exceed neither the battery capacity nor the time.
    - Edges are connections between trips. Each edge has a single weight that represents the cost of the connection. This
    should normally be the waiting time between the trips. However, things such as "try to connect on the same line" can
    also be expressed by scaling the weights. The allowed weights are between 0 and 1e6 (integer).
    - Connections that cannot be made should be represented as not existing, not by a very high weight.

    :param graph: A directed acyclic grpah as a networkx Graph
    :param write_to_file: If True, the graph will be written to a file in the temp directory. This is useful for debugging.
    :return: A list of edges in the form os (NodeID, NodeID) tuples.
    """
    graph = _validate_input_graph(graph)
    json_graph = _graph_to_json(graph)

    if write_to_file:
        # The json is not pretty-printed. Reload it and use json.dumps with indent=4 to pretty-print it.
        json_data = json.loads(json_graph)
        json_graph = json.dumps(json_data, indent=4)
        with open(os.path.join(gettempdir(), "graph.json"), "w") as fp:
            fp.write(json_graph)
            print(f"Saved file to {os.path.join(gettempdir(), 'graph.json')}")

    # Call the rust solver
    result: List[Tuple[int, int]] = eflips_schedule_rust.solve(json_graph)

    # The result is a list of edges
    # Create a copy of our graph, delete all edges and add the ones from the result
    result_graph = graph.copy()
    result_graph.remove_edges_from(graph.edges)
    result_graph.add_edges_from(result)
    return result_graph


def write_back_rotation_plan(
    rot_graph: nx.Graph, session: sqlalchemy.orm.session.Session
) -> None:
    """
    Deletes the original rotations and writes back the new rotations to the database. This is useful when the new
    rotations are better than the original rotations.

    :param rot_graph: A directed acyclic graph containing the new rotations.
    :param session: An open database session.
    :return: Nothing. The new rotations are written to the database.
    """
    # Find the original rotations
    rotations = (
        session.query(Rotation).join(Trip).filter(Trip.id.in_(rot_graph.nodes)).all()
    )

    # Delete all empty trips that are part of the rotations
    for rotation in rotations:
        for trip in rotation.trips:
            if trip.trip_type == TripType.EMPTY:
                for stop_time in trip.stop_times:
                    session.delete(stop_time)
                session.delete(trip)
    session.flush()

    with session.no_autoflush:
        for rotation in rotations:
            for trip in rotation.trips:
                trip.rotation = None  # type: ignore
                trip.rotation_id = None  # type: ignore
            session.delete(rotation)

        # Make sure the rotation ids and vehicle type ids are the same
        vehicle_type_id = rotations[0].vehicle_type_id
        assert all(
            rotation.vehicle_type_id == vehicle_type_id for rotation in rotations
        )
        scenario_id = rotations[0].scenario_id
        assert all(rotation.scenario_id == scenario_id for rotation in rotations)

        for set_of_nodes in nx.connected_components(rot_graph.to_undirected()):
            rotation = Rotation(
                scenario_id=scenario_id,
                vehicle_type_id=vehicle_type_id,
                allow_opportunity_charging=True,
                name=None,
            )
            for node in set_of_nodes:
                trip = session.query(Trip).filter(Trip.id == node).one()
                trip.rotation = rotation
            # Sort the rotation's trips by departure time
            rotation.trips = sorted(
                rotation.trips, key=lambda trip: trip.departure_time
            )
        session.flush()


def visualize_with_dash_cytoscape(graph: nx.Graph) -> None:
    """
    Visualize the graph using dash-cytoscape. This method will start a local server and open a browser window.
    It will not return until the server is stopped.

    :param graph: A directed acyclic graph.
    :return: Nothing
    """
    # Visualize the graph using dash-cytoscape
    # Convert the graph to cyjs
    cytoscape_data = nx.cytoscape_data(graph)
    elements = list(itertools.chain(*cytoscape_data["elements"].values()))
    # Make sure all source and target nodes are strings
    for element in elements:
        if "source" in element["data"]:
            element["data"]["source"] = str(element["data"]["source"])
        if "target" in element["data"]:
            element["data"]["target"] = str(element["data"]["target"])
    cytoscape = cyto.Cytoscape(
        id="cytoscape",
        layout={"name": "cose"},
        style={"width": "100%", "height": "800px"},
        elements=elements,
        stylesheet=[
            {
                "selector": "node",
                "style": {
                    "label": "data(name)",
                    "background-color": "#11479e",
                    "color": "data(color)",
                },
            },
            {
                "selector": "edge",
                "style": {
                    "curve-style": "bezier",
                    "target-arrow-shape": "triangle",
                    "line-color": "data(color)",
                    "target-arrow-color": "data(color)",
                },
            },
        ],
    )
    app = dash.Dash(__name__)
    app.layout = html.Div([html.H1(f"Schedule"), cytoscape])
    app.run_server(debug=True)


def efficiency_info(
    new_trips: List[List[int]], session: sqlalchemy.orm.session.Session
) -> None:
    """
    Calculate the efficiency of the original rotations and the new rotations. Efficiency is defined as the time spent
    driving divided by the total time spent in the rotation.

    :param new_trips: A list of lists of trip IDs. Each list is a rotation.
    :param session: An open database session
    :return: Nothing. Efficiency is printed to the console.
    """
    original_efficiencies = []
    new_efficiencies = []
    for new_rotation in new_trips:
        total_duration = (
            session.query(Trip).filter(Trip.id == new_rotation[-1]).one().arrival_time
            - session.query(Trip)
            .filter(Trip.id == new_rotation[0])
            .one()
            .departure_time
        ).total_seconds() / 60
        driving_duration = 0.0
        for trip_id in new_rotation:
            trip = session.query(Trip).filter(Trip.id == trip_id).one()
            driving_duration += (
                trip.arrival_time - trip.departure_time
            ).total_seconds() / 60
        new_efficiencies.append(driving_duration / total_duration)

    # Find all rotations containing one of the new trips
    all_new_trips = set(itertools.chain(*new_trips))
    old_rotations = (
        session.query(Rotation).join(Trip).filter(Trip.id.in_(all_new_trips)).all()
    )

    for rotation in old_rotations:
        trip_list = [
            trip for trip in rotation.trips if trip.trip_type == TripType.PASSENGER
        ]
        total_duration = (
            trip_list[-1].arrival_time - trip_list[0].departure_time
        ).total_seconds() / 60
        if total_duration == 0:
            continue
        driving_duration = 0
        for trip in trip_list:
            driving_duration += (
                trip.arrival_time - trip.departure_time
            ).total_seconds() / 60
        original_efficiencies.append(driving_duration / total_duration)

    print(
        f"Original efficiency: {np.mean(original_efficiencies):.3f} with {len(original_efficiencies)} rotations"
    )
    print(
        f"New efficiency: {np.mean(new_efficiencies):.3f} with {len(new_efficiencies)} rotations"
    )
