#!/usr/bin/env python3

"""

Understand the electric vehicle scneduling problem as a Minimum Path Cover problem. This way, we will find *a* schedule
that has the minimum number of rotations, and from that hopefully derive a good schedule, that is SoC-Aware.

Electrified stations are taking into account in that a trip leading to an electrified station will not consume any
energy. This is a simplification, as we do not take into account that the vehicle may recharge multiple trips' worth of
energy at the station.

"""


import itertools
import logging
import os
from datetime import timedelta
from multiprocessing import Pool
from typing import List, Dict, FrozenSet, Tuple

import dash # type: ignore
import dash_cytoscape as cyto # type: ignore
import networkx as nx # type: ignore
import numpy as np
import sqlalchemy.orm.session
from dash import html
from eflips.model import (
    Scenario,
    VehicleType,
    Trip,
    Station,
    Rotation,
    TripType,
)
from sqlalchemy.orm import Session


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

    return passenger_trips_by_vehicle_type


def create_graph_of_rotations(rotations: List[Rotation]) -> nx.Graph:
    """
    Creates a graph of the original rotations, as in the database. Useful for debugging.
    :param rotations: A list of rotation objects
    :return: a directed acyclic graph having the trips as nodes and the possible connections as edges.
    """
    G = nx.DiGraph()
    for rotation in rotations:
        for i, trip in enumerate(rotation.trips):
            if trip.trip_type == TripType.PASSENGER:
                G.add_node(
                    trip.id,
                    name=f"{trip.route.departure_station.name} -> {trip.route.arrival_station.name} "
                    f"({trip.departure_time.strftime('%H:%M')} - {trip.arrival_time.strftime('%H:%M')})",
                )
                if i > 0 and rotation.trips[i - 1].trip_type == TripType.PASSENGER:
                    G.add_edge(rotation.trips[i - 1].id, trip.id, color="black")
    return G


def create_graph_of_possible_connections(
    trips: List[Trip],
    minimum_break_time: timedelta = timedelta(minutes=0),
    regular_break_time: timedelta = timedelta(minutes=30),
    maximum_break_time: timedelta = timedelta(minutes=120),
) -> nx.Graph:
    """
    Turns a list of trips into a directed acyclic graph. The nodes are the trips, and the edges are the possible
    transitions between the trips. The edges are colored according to the time between the trips.

    :param trips: A list of trips
    :param minimum_break_time: The minimum break time between two trips.
    :param regular_break_time: The regular break time between two trips. All trips following the trip in the regular
     break time are added as edges.
    :param maximum_break_time: The maximum break time between two trips. If no edge is added with the regular break time,
    the *first* trip before the maximum break time is added.

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
            # Calculate the energy consumption of the trip
            if trip.route.arrival_station.is_electrified:
                delta_soc = 0.0  # We can recharge at the arrival station
            else:
                vt = trip.rotation.vehicle_type
                distance = trip.route.distance / 1000  # Convert to km
                energy_consumption = vt.consumption * distance
                delta_soc = energy_consumption / vt.battery_capacity
            graph.add_node(
                trip.id,
                name=f"{trip.route.departure_station.name} -> {trip.route.arrival_station.name} "
                f"({trip.departure_time.strftime('%H:%M')} - {trip.arrival_time.strftime('%H:%M')})",
                delta_soc=delta_soc,
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
                        graph.add_edge(
                            trip.id,
                            following_trip.id,
                            color="gray",
                            wait_time=int(
                                (
                                    following_trip.departure_time - trip.arrival_time
                                ).seconds
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
                            graph.add_edge(
                                trip.id,
                                following_trip.id,
                                color="red",
                                wait_time=int(
                                    (
                                        following_trip.departure_time
                                        - trip.arrival_time
                                    ).seconds
                                ),
                            )
                            break

    return graph


def compare_graphs(orig: nx.Graph, new: nx.Graph) -> None:
    """
    Print out information about the differences between two graphs. Specifically, we print out the edges that are in
    the original graph but not in the new graph, and the edges that are in the new graph but not in the original graph.

    :param orig: A graph
    :param new: A graph
    :return: Nothing
    """

    # Find the edges that differ between the two graphs, and color them specially
    print(f"Number of edges in original graph: {len(orig.edges)}")
    print("Edges in original graph but not in new graph:")
    for edge in orig.edges:
        if edge not in new.edges:
            prev_node = edge[0]
            next_node = edge[1]
            print(f"Edge {prev_node} -> {next_node} not in new graph")
            print(
                f"Trip info: {orig.nodes[prev_node]['name']} -> {orig.nodes[next_node]['name']}"
            )

    print(f"Number of edges in new graph: {len(new.edges)}")
    print("Edges in new graph but not in original graph:")
    for edge in new.edges:
        if edge not in orig.edges:
            prev_node = edge[0]
            next_node = edge[1]
            print(f"Edge {prev_node} -> {next_node} not in original graph")
            print(
                f"Trip info: {new.nodes[prev_node]['name']} -> {new.nodes[next_node]['name']}"
            )


def minimum_path_cover_rotation_plan(graph: nx.Graph) -> nx.Graph:
    """
    Create a minimum path cover of the graph. This is the same as finding the minimum number of rotations that cover all
    the trips in the graph.

    The approach is based on this article https://archive.ph/0xXuL and other implementations of the
    minimum path cover problem.

    :param trip_graph: A directed acyclic graph, containing the trips as nodes and the possible connections as edges.
    :return: A graph containing the minimum path cover of the original graph.
    """
    logger = logging.getLogger(__name__)
    rotations: List[List[int]] = []

    logger.info(
        f"Graph is composed is {len(graph.nodes)} nodes and {len(graph.edges)} edges"
    )
    logger.info(
        f"Graph is composed is {len(list(nx.connected_components(graph.to_undirected())))} connected components"
    )

    # We create a copy of the graph and remove all edges.
    # We will add the edges back in the next step
    graph_copy = graph.copy()
    edges_to_remove = list(graph_copy.edges)
    graph_copy.remove_edges_from(edges_to_remove)

    # Make sure our graph is acyclic
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The graph is not acyclic")

    # Convert from DAG to bipartite graph. The each node is splitted to two, one
    # has only the inedges and other has only the outedges.
    bipartite_graph = nx.DiGraph()
    for node in graph.nodes:
        bipartite_graph.add_node(f"{node}_in", color="blue", bipartite=0)
        bipartite_graph.add_node(f"{node}_out", color="red", bipartite=1)

    for edge in graph.edges:
        bipartite_graph.add_edge(f"{edge[0]}_out", f"{edge[1]}_in", weight=graph.edges[edge]["wait_time"])

    assert nx.is_directed_acyclic_graph(bipartite_graph)
    assert nx.is_bipartite(bipartite_graph)

    top_nodes = {n for n, d in bipartite_graph.nodes(data=True) if d["bipartite"] == 0}
    bottom_nodes = set(bipartite_graph) - top_nodes

    # Make sure it is a bipartite graph
    if not nx.is_bipartite(bipartite_graph):
        raise ValueError("The graph is not bipartite")

    # Find the maximum matching
    # Edges on the maximum matching will be added to the graph
    for sub_nodes in nx.connected_components(bipartite_graph.to_undirected()):
        if len(sub_nodes) == 1:
            continue
        subgraph = bipartite_graph.subgraph(sub_nodes)
        matching = nx.bipartite.hopcroft_karp_matching(subgraph)

        # If a full matching exists, calculate the minimum weight full matching
        try:
            matching = nx.bipartite.minimum_weight_full_matching(subgraph, weight="weight")
        except ValueError:
            pass

        for entry in matching:
            if "_out" in entry:
                start_node = int(entry.split("_")[0])
                end_node = int(matching[entry].split("_")[0])
                graph_copy.add_edge(start_node, end_node, color="black")

    graph = graph_copy

    # Ensure that our graph is acyclic
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The graph is not acyclic")

    logger.info(f"Found {len(rotations)} rotations")

    return graph


def _effects_of_removal(
    rotation: FrozenSet[int], graph: nx.DiGraph, soc_reserve: float
) -> Dict[List[int], Dict[str, float | int]]:
    """
    Private function to calculate the effects of removing a set of nodes from the graph. This is used in the
    soc_aware_rotation_plan function.

    :param rotation: A set of nodes to remove
    :param graph: A directed acyclic graph, containing the trips as nodes and the possible connections as edges.
    :param soc_reserve: The amount of reserve that we want to keep in the battery
    :return: A dictionary containing the effects of removing the nodes
    """

    effects_of_removal: Dict[Tuple[int], Dict[str, float | int]] = {}

    # Get the node list in the correct order
    sub_subgraph = graph.subgraph(rotation).copy()
    sorted_nodes = list(nx.topological_sort(sub_subgraph))

    # Now, how many of trips on this rotation (from the start) until the energy consumption is below the reserve
    energy_consumed = 0
    forward_node_list = []
    for node in sorted_nodes:
        energy_consumed += graph.nodes[node]["delta_soc"]
        if energy_consumed > (1 - soc_reserve):
            break
        forward_node_list.append(node)

    subgraph_forward_nodes_removed = graph.copy()
    subgraph_forward_nodes_removed.remove_nodes_from(forward_node_list)
    forward_rotation_graph = minimum_path_cover_rotation_plan(
        subgraph_forward_nodes_removed
    )
    rotation_count_forward_nodes_removed = len(
        list(nx.connected_components(forward_rotation_graph.to_undirected()))
    )
    max_energy_forward_nodes_removed = max(
        [
            sum(graph.nodes[node]["delta_soc"] for node in rotation)
            for rotation in nx.connected_components(
                forward_rotation_graph.to_undirected()
            )
        ]
    )
    effects_of_removal[tuple(forward_node_list)] = {
        "rotation_count": rotation_count_forward_nodes_removed,
        "max_energy": max_energy_forward_nodes_removed,
    }

    # Now, how many of trips on this rotation (from the end) until the energy consumption is below the reserve
    energy_consumed = 0
    backward_node_list = []
    for node in reversed(sorted_nodes):
        energy_consumed += graph.nodes[node]["delta_soc"]
        if energy_consumed > (1 - soc_reserve):
            break
        backward_node_list.append(node)

    subgraph_backward_nodes_removed = graph.copy()
    subgraph_backward_nodes_removed.remove_nodes_from(backward_node_list)
    backward_rotation_graph = minimum_path_cover_rotation_plan(
        subgraph_backward_nodes_removed
    )
    rotation_count_backward_nodes_removed = len(
        list(nx.connected_components(backward_rotation_graph.to_undirected()))
    )
    max_energy_backward_nodes_removed = max(
        [
            sum(graph.nodes[node]["delta_soc"] for node in rotation)
            for rotation in nx.connected_components(
                backward_rotation_graph.to_undirected()
            )
        ]
    )

    effects_of_removal[tuple(backward_node_list)] = {
        "rotation_count": rotation_count_backward_nodes_removed,
        "max_energy": max_energy_backward_nodes_removed,
    }

    return effects_of_removal


def soc_aware_rotation_plan(
    graph: nx.Graph, soc_reserve: float = 0.2, parallelism: bool = True
) -> nx.Graph:
    """
    Create a minimum path cover of the graph, taking into account the state of charge of the vehicle. This is the same
    as finding the minimum number of rotations that cover all the trips in the graph. However, the length of the
    rotations is limited by the state of charge of the vehicle.

    This is implemented by solving the MPC problem (for each disconnected component), then for each trip that exceeds
    the SOC limit, either cutting of from the beginning or end and solving the MPC problem for the remaining trips.

    :param graph: A directed acyclic graph, containing the trips as nodes and the possible connections as edges.
    :param soc_reserve: The minimum state of charge that we want to keep in the battery **This includes the reserve
                                            needed to get to and from the depot**
    :return: A graph containing the minimum path cover of the original graph.
    """
    logger = logging.getLogger(__name__)

    # Make sure the graph is acyclic
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The graph is not acyclic")

    # Make sure all nodes have the delta_soc attribute
    for node in graph.nodes:
        if "delta_soc" not in graph.nodes[node]:
            raise ValueError("All nodes must have the delta_soc attribute")

    finished_trips: List[List[int]] = []
    for set_of_nodes in nx.connected_components(graph.to_undirected()):
        subgraph = graph.subgraph(set_of_nodes).copy()

        # Find the minimum path cover
        rotation_graph = minimum_path_cover_rotation_plan(subgraph)

        while True:
            # Find the energy consumption of each rotation
            energy_consumption = {}
            for rotation_set_of_nodes in nx.connected_components(
                rotation_graph.to_undirected()
            ):
                rotation_energy = sum(
                    graph.nodes[node]["delta_soc"] for node in rotation_set_of_nodes
                )
                energy_consumption[frozenset(rotation_set_of_nodes)] = rotation_energy

            # Find the rotation with the highest energy consumption
            max_rotation = max(energy_consumption, key=energy_consumption.get)

            if max(energy_consumption.values()) <= (1 - soc_reserve):
                break
            else:
                number_of_excessive_rotations = sum(
                    [
                        1
                        for rotation in energy_consumption.values()
                        if rotation > (1 - soc_reserve)
                    ]
                )
                logger.debug(
                    f"Number of excessive rotations: {number_of_excessive_rotations}"
                )

            # Get all node sets that exceed the SOC limit
            excessive_rotations = [
                rotation
                for rotation in energy_consumption
                if energy_consumption[rotation] > (1 - soc_reserve)
            ]

            effects_of_removal: Dict[List[int], Dict[str, float | int]] = {}

            if parallelism:
                pool_args = []
                for rotation in excessive_rotations:
                    pool_args.append((rotation, subgraph, soc_reserve))
                with Pool() as pool:
                    results = pool.starmap(_effects_of_removal, pool_args)
                for result in results:
                    effects_of_removal.update(result)
            else:
                for rotation in excessive_rotations:
                    effects_of_removal.update(
                        _effects_of_removal(rotation, subgraph, soc_reserve)
                    )

            # Remove all candidates that do not have the minimum number of rotations
            min_rotation_count = min(
                [effect["rotation_count"] for effect in effects_of_removal.values()]
            )
            effects_of_removal = {
                node_list: effect
                for node_list, effect in effects_of_removal.items()
                if effect["rotation_count"] == min_rotation_count
            }

            # Take the one with the lowest energy consumption
            min_energy = min(
                [effect["max_energy"] for effect in effects_of_removal.values()]
            )
            for node_list, effect in effects_of_removal.items():
                if effect["max_energy"] == min_energy:
                    break

            subgraph.remove_nodes_from(node_list)
            rotation_graph = minimum_path_cover_rotation_plan(subgraph)
            finished_trips.append(node_list)

        # Add the remaining rotations to the list of finished trips
        for rotation_set_of_nodes in nx.connected_components(
            rotation_graph.to_undirected()
        ):
            sub_subgraph = rotation_graph.subgraph(rotation_set_of_nodes).copy()
            sorted_nodes = list(nx.topological_sort(sub_subgraph))
            finished_trips.append(sorted_nodes)

    # Remove all edges from the graph
    graph_copy = graph.copy()
    edges_to_remove = list(graph_copy.edges)
    graph_copy.remove_edges_from(edges_to_remove)

    # Re-add the edges for the finished trips
    for trip_list in finished_trips:
        for i in range(len(trip_list) - 1):
            graph_copy.add_edge(trip_list[i], trip_list[i + 1], color="black")

    return graph_copy


def efficiency_info(
    new_trips: List[List[int]], session: sqlalchemy.orm.session.Session
):
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
        ).seconds / 60
        driving_duration = 0
        for trip_id in new_rotation:
            trip = session.query(Trip).filter(Trip.id == trip_id).one()
            driving_duration += (trip.arrival_time - trip.departure_time).seconds / 60
        new_efficiencies.append(driving_duration / total_duration)

    # Find all rotations containing one of the new trips
    all_new_trips = set(itertools.chain(*new_trips))
    old_rotations = (
        session.query(Rotation).join(Trip).filter(Trip.id.in_(all_new_trips)).all()
    )

    for rotation in old_rotations:
        # Assume that the first and last trip are empty trips
        total_duration = (
            rotation.trips[-2].arrival_time - rotation.trips[1].departure_time
        ).seconds / 60
        driving_duration = 0
        for trip in rotation.trips[1:-1]:
            driving_duration += (trip.arrival_time - trip.departure_time).seconds / 60
        original_efficiencies.append(driving_duration / total_duration)

    print(
        f"Original efficiency: {np.mean(original_efficiencies):.3f} with {len(original_efficiencies)} rotations"
    )
    print(
        f"New efficiency: {np.mean(new_efficiencies):.3f} with {len(new_efficiencies)} rotations"
    )

def write_back_rotation_plan(rot_graph: nx.Graph, session: sqlalchemy.orm.session.Session) -> None:
    """
    Deletes the original rotations and writes back the new rotations to the database. This is useful when the new
    rotations are better than the original rotations.

    :param rot_graph: A directed acyclic graph containing the new rotations.
    :param session: An open database session.
    :return: Nothing. The new rotations are written to the database.
    """
    # Find the original rotations
    rotations = session.query(Rotation).join(Trip).filter(Trip.id.in_(rot_graph.nodes)).all()

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
                trip.rotation = None
                trip.rotation_id = None
            session.delete(rotation)

        # Make sure the rotation ids and vehicle type ids are the same
        vehicle_type_id = rotations[0].vehicle_type_id
        assert all(rotation.vehicle_type_id == vehicle_type_id for rotation in rotations)
        scenario_id = rotations[0].scenario_id
        assert all(rotation.scenario_id == scenario_id for rotation in rotations)

        for set_of_nodes in nx.connected_components(rot_graph.to_undirected()):
            rotation = Rotation(scenario_id=scenario_id,
                                vehicle_type_id=vehicle_type_id,
                                allow_opportunity_charging=True,
                                name=None,
                                )
            for node in set_of_nodes:
                trip = session.query(Trip).filter(Trip.id == node).one()
                trip.rotation = rotation
            # Sort the rotation's trips by departure time
            rotation.trips = sorted(rotation.trips, key=lambda trip: trip.departure_time)
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
