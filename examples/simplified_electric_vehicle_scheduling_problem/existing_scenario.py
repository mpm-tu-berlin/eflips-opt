import json
import os

import networkx as nx
import sqlalchemy
from eflips.model import Scenario
from sqlalchemy.orm import Session

from eflips.opt.simplified_electric_vehicle_scheduling import (
    create_graph_of_possible_connections,
    efficiency_info,
    minimum_path_cover_rotation_plan,
    passenger_trips_by_vehicle_type,
    soc_aware_rotation_plan,
    split_for_performance,
    graph_to_json,
)

if __name__ == "__main__":
    if "DATABASE_URL" not in os.environ:
        raise ValueError("Please set the DATABASE_URL environment variable")
    DATABASE_URL = os.environ["DATABASE_URL"]

    SCENARIO_ID = 1
    SOC_RESERVE = 0.2  # The minimum state of charge that we want to keep in the battery

    engine = sqlalchemy.create_engine(DATABASE_URL)
    session = Session(engine)
    scenario = session.query(Scenario).filter(Scenario.id == SCENARIO_ID).one()

    trips_by_vt = passenger_trips_by_vehicle_type(scenario, session)

    for vehicle_type, trips in trips_by_vt.items():
        print(f"Vehicle type: {vehicle_type.name}")

        graph = create_graph_of_possible_connections(trips)
        graph = split_for_performance(graph)

        rotation_graph = minimum_path_cover_rotation_plan(graph, use_rust=False)
        print("Rotation plan (not SOC-aware , Python):")
        trip_lists = []
        for set_of_nodes in nx.connected_components(rotation_graph.to_undirected()):
            topoogical_order = list(
                nx.topological_sort(rotation_graph.subgraph(set_of_nodes))
            )
            trip_lists.append(topoogical_order)
        efficiency_info(trip_lists, session)

        rotation_graph = minimum_path_cover_rotation_plan(graph, use_rust=True)
        print("Rotation plan (not SOC-aware , Rust):")
        trip_lists = []
        for set_of_nodes in nx.connected_components(rotation_graph.to_undirected()):
            topoogical_order = list(
                nx.topological_sort(rotation_graph.subgraph(set_of_nodes))
            )
            trip_lists.append(topoogical_order)
        efficiency_info(trip_lists, session)

        soc_aware_graph = soc_aware_rotation_plan(
            graph, soc_reserve=SOC_RESERVE, use_rust=False
        )
        print("Rotation plan (SOC-aware, Python):")
        trip_lists = []
        for set_of_nodes in nx.connected_components(soc_aware_graph.to_undirected()):
            topoogical_order = list(
                nx.topological_sort(soc_aware_graph.subgraph(set_of_nodes))
            )
            trip_lists.append(topoogical_order)
        efficiency_info(trip_lists, session)

        soc_aware_graph = soc_aware_rotation_plan(
            graph, soc_reserve=SOC_RESERVE, use_rust=True
        )
        print("Rotation plan (SOC-aware, Rust):")
        trip_lists = []
        for set_of_nodes in nx.connected_components(soc_aware_graph.to_undirected()):
            topoogical_order = list(
                nx.topological_sort(soc_aware_graph.subgraph(set_of_nodes))
            )
            trip_lists.append(topoogical_order)
        efficiency_info(trip_lists, session)

        print("Now, you could visualize the rotation plan with the following command:")
        print("visualize_with_dash_cytoscape(graph, trip_lists, session)")

        print("And write back the rotation plan with the following command:")
        print("write_back_rotation_plan(trip_lists, session)")
