import json
import os
from datetime import date

import networkx as nx
import sqlalchemy
from eflips.model import Scenario, Rotation
from sqlalchemy.orm import Session
from tqdm import tqdm

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

    SCENARIO_ID = 3
    SOC_RESERVE = 0.1  # The minimum state of charge that we want to keep in the battery

    engine = sqlalchemy.create_engine(DATABASE_URL)
    session = Session(engine)
    scenario = session.query(Scenario).filter(Scenario.id == SCENARIO_ID).one()

    # Remove all rotaitons not by vehicles of type 16 on day (2023.7.5)
    the_date = date(2023, 7, 5)
    the_vehicle_type_id = 16
    all_rotations = session.query(Rotation).filter(Rotation.scenario_id == scenario.id).options(
        sqlalchemy.orm.joinedload(Rotation.trips)
    ).all()
    to_delete = []
    for rotation in tqdm(all_rotations):
        departure_date = rotation.trips[0].departure_time.date()
        vehicle_type_id = rotation.vehicle_type_id
        if departure_date != the_date or vehicle_type_id != the_vehicle_type_id:
            for trip in rotation.trips:
                for stop_time in trip.stop_times:
                    to_delete.append(stop_time)
                to_delete.append(trip)
            to_delete.append(rotation)

    for obj in tqdm(to_delete):
        session.delete(obj)
    session.commit()




    trips_by_vt = passenger_trips_by_vehicle_type(scenario, session)

    for vehicle_type, trips in trips_by_vt.items():
        if len(trips) == 0:
            continue
        print(f"Vehicle type: {vehicle_type.name}")

        graph = create_graph_of_possible_connections(trips)

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
