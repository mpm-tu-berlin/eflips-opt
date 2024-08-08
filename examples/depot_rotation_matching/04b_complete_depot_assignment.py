#! /usr/bin/env python3


import os
from collections import Counter
from typing import List, Tuple, Dict, Union

import numpy as np
import sqlalchemy.orm.session
from eflips.eval.input.prepare import (
    geographic_trip_plot as prepare_geographic_trip_plot,
)
from eflips.model import *
from eflips.opt.depot_rotation_matching import DepotRotationOptimizer
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


def depots_for_scenarios(
    scenario: Scenario, session: sqlalchemy.orm.session.Session
) -> List[Dict[str, Union[int, Tuple[float, float], List[int]]]]:
    # # Put the new capacities into a variable
    #
    # - "Abstellfläche Mariendorf" will not be equipped with charging infrastructure, therefore it cannot serve as a depot for electrified buses
    # - There will be a new depot "Köpenicker Landstraße" at the coordinates 52.4654085,13.4964867 with a capacity of 200 12m buses
    # - There will be a new depot "Rummelsburger Landstraße" at the coordinates "52.4714167,13.5053889" with a capacity of 60 12m buses
    # - There will be a new depot "Säntisstraße" at the coordinates "52.416735,13.3844563" with a capacity of 230 12m buses
    # - The capacity of the existing depot "Spandau" will be 220 12m buses
    # - The capacity of the existing depot "Indira-Gandhi-Straße" will be 300 12m buses
    # - The capacity of the existing depot "Britz" weill be 140 12m buses
    # - The capacity of the existing depot "Cicerostraße" will be 209 12m buses
    # - The capacity of the existing depot "Müllerstraße" will be 155 12m buses
    # - The capacity of the existing depot "Lichtenberg" will be 120 12m buses
    # - "Alt Friedrichsfelde" will not be equipped with charging infrastructure, therefore it cannot serve as a depot for electrified buses
    #
    # Allowed vehicle types are also specified for each depot. The are following vehicle types in total:
    # - "EN" for 12m electric buses
    # - "GN" for 18m articulated buses
    # - "DD" for 12m double-decker buses
    #
    # And the vehicle types that can be used at each depot are as follows:
    # - "Abstellfläche Mariendorf": None
    # - "Betriebshof Spandau": EN, GN, DD
    # - "Betriebshof Indira-Gandhi-Straße": EN, GN, DD
    # - "Betriebshof Britz": EN, GN, DD
    # - "Betriebshof Cicerostraße": EN, GN, DD
    # - "Betriebshof Müllerstraße": EN, GN, DD
    # - "Betriebshof Lichtenberg": GN
    # - "Betriebshof Köpenicker Landstraße": EN, GN
    # - "Betriebshof Rummelsburger Landstraße": GN
    # - "Betriebshof Säntisstraße": EN, GN
    # - "Betriebshof Alt Friedrichsfelde": None
    #
    # The new capacities should be specified as a dictionary containing the following keys:
    # - "depot_station": Either the ID of the existing station or a (lon, lat) tuple for a depot that does not yet exist in the database
    # - "capacity": The new capacity of the depot, in 12m buses
    # - "vehicle_type": A list of vehicle type ids that can be used at this depot
    # - "name": The name of the depot (only for new depots)
    depot_list: List[Dict[str, Union[int, Tuple[float, float], List[int]]]] = []
    all_vehicle_type_ids = (
        session.query(VehicleType.id).filter(VehicleType.scenario == scenario).all()
    )
    all_vehicle_type_ids = [x[0] for x in all_vehicle_type_ids]

    # "Abstellfläche Mariendorf" will have a capacity of zero
    station_id = (
        session.query(Station.id)
        .filter(Station.name_short == "BF MDA")
        .filter(Station.scenario == scenario)
        .one()[0]
    )

    vehicle_types = ["EN", "GN", "DD"]
    vehicle_type_ids = (
        session.query(VehicleType.id)
        .filter(VehicleType.name_short.in_(vehicle_types))
        .filter(VehicleType.scenario == scenario)
        .all()
    )
    vehicle_type_ids = [x[0] for x in vehicle_type_ids]

    depot_list.append(
        {
            "depot_station": station_id,
            "capacity": 0,
            "vehicle_type": vehicle_type_ids,
        }
    )

    # "Betriebshof Spandau will hava a capacity of 220
    station_id = (
        session.query(Station.id)
        .filter(Station.name_short == "BF S")
        .filter(Station.scenario == scenario)
        .one()[0]
    )
    vehicle_types = ["EN", "GN", "DD"]
    vehicle_type_ids = (
        session.query(VehicleType.id)
        .filter(VehicleType.name_short.in_(vehicle_types))
        .filter(VehicleType.scenario == scenario)
        .all()
    )
    vehicle_type_ids = [x[0] for x in vehicle_type_ids]

    depot_list.append(
        {
            "depot_station": station_id,
            "capacity": 220,
            "vehicle_type": vehicle_type_ids,
        }
    )

    # "Betriebshof Indira-Gandhi-Straße" will have a capacity of 300
    station_id = (
        session.query(Station.id)
        .filter(Station.name_short == "BF I")
        .filter(Station.scenario == scenario)
        .one()[0]
    )

    vehicle_types = ["EN", "GN", "DD"]
    vehicle_type_ids = (
        session.query(VehicleType.id)
        .filter(VehicleType.name_short.in_(vehicle_types))
        .filter(VehicleType.scenario == scenario)
        .all()
    )
    vehicle_type_ids = [x[0] for x in vehicle_type_ids]

    depot_list.append(
        {
            "depot_station": station_id,
            "capacity": 300,
            "vehicle_type": vehicle_type_ids,
        }
    )

    # "Betriebshof Britz" will have a capacity of 140
    station_id = (
        session.query(Station.id)
        .filter(Station.name_short == "BF B")
        .filter(Station.scenario == scenario)
        .one()[0]
    )

    vehicle_types = ["EN", "GN", "DD"]
    vehicle_type_ids = (
        session.query(VehicleType.id)
        .filter(VehicleType.name_short.in_(vehicle_types))
        .filter(VehicleType.scenario == scenario)
        .all()
    )
    vehicle_type_ids = [x[0] for x in vehicle_type_ids]

    depot_list.append(
        {
            "depot_station": station_id,
            "capacity": 140,
            "vehicle_type": vehicle_type_ids,
        }
    )

    # "Betriebshof Cicerostraße" will have a capacity of 209
    station_id = (
        session.query(Station.id)
        .filter(Station.name_short == "BF C")
        .filter(Station.scenario == scenario)
        .one()[0]
    )

    vehicle_types = ["EN", "GN", "DD"]
    vehicle_type_ids = (
        session.query(VehicleType.id)
        .filter(VehicleType.name_short.in_(vehicle_types))
        .filter(VehicleType.scenario == scenario)
        .all()
    )
    vehicle_type_ids = [x[0] for x in vehicle_type_ids]

    depot_list.append(
        {
            "depot_station": station_id,
            "capacity": 209,
            "vehicle_type": vehicle_type_ids,
        }
    )

    # "Betriebshof Müllerstraße" will have a capacity of 155
    station_id = (
        session.query(Station.id)
        .filter(Station.name_short == "BF M")
        .filter(Station.scenario == scenario)
        .one()[0]
    )

    vehicle_types = ["EN", "GN", "DD"]
    vehicle_type_ids = (
        session.query(VehicleType.id)
        .filter(VehicleType.name_short.in_(vehicle_types))
        .filter(VehicleType.scenario == scenario)
        .all()
    )
    vehicle_type_ids = [x[0] for x in vehicle_type_ids]

    depot_list.append(
        {
            "depot_station": station_id,
            "capacity": 155,
            "vehicle_type": vehicle_type_ids,
        }
    )

    # "Betriebshof Lichtenberg" will have a capacity of 120
    station_id = (
        session.query(Station.id)
        .filter(Station.name_short == "BF L")
        .filter(Station.scenario == scenario)
        .one()[0]
    )

    vehicle_types = ["GN"]
    vehicle_type_ids = (
        session.query(VehicleType.id)
        .filter(VehicleType.name_short.in_(vehicle_types))
        .filter(VehicleType.scenario == scenario)
        .all()
    )
    vehicle_type_ids = [x[0] for x in vehicle_type_ids]
    depot_list.append(
        {
            "depot_station": station_id,
            "capacity": 120,
            "vehicle_type": vehicle_type_ids,
        }
    )

    # "Betriebshof Köpenicker Landstraße" will have a capacity of 200

    vehicle_types = ["EN", "GN"]
    vehicle_type_ids = (
        session.query(VehicleType.id)
        .filter(VehicleType.name_short.in_(vehicle_types))
        .filter(VehicleType.scenario == scenario)
        .all()
    )
    vehicle_type_ids = [x[0] for x in vehicle_type_ids]
    depot_list.append(
        {
            "depot_station": (13.4964867, 52.4654085),
            "name": "Betriebshof Köpenicker Landstraße",
            "capacity": 200,
            "vehicle_type": vehicle_type_ids,
        }
    )

    # "Betriebshof Rummelsburger Landstraße" will have a capacity of 60
    vehicle_types = ["GN"]
    vehicle_type_ids = (
        session.query(VehicleType.id)
        .filter(VehicleType.name_short.in_(vehicle_types))
        .filter(VehicleType.scenario == scenario)
        .all()
    )
    vehicle_type_ids = [x[0] for x in vehicle_type_ids]
    depot_list.append(
        {
            "depot_station": (13.5053889, 52.4714167),
            "name": "Betriebshof Rummelsburger Landstraße",
            "capacity": 60,
            "vehicle_type": vehicle_type_ids,
        }
    )

    # "Betriebshof Säntisstraße" will have a capacity of 230
    vehicle_types = ["EN", "GN"]
    vehicle_type_ids = (
        session.query(VehicleType.id)
        .filter(VehicleType.name_short.in_(vehicle_types))
        .filter(VehicleType.scenario == scenario)
        .all()
    )
    vehicle_type_ids = [x[0] for x in vehicle_type_ids]
    depot_list.append(
        {
            "depot_station": (13.3844563, 52.416735),
            "name": "Betriebshof Säntisstraße",
            "capacity": 230,
            "vehicle_type": vehicle_type_ids,
        }
    )

    # "Betriebshof Alt Friedrichsfelde" will have a capacity of 0
    depot_list.append(
        {
            "depot_station": (13.5401389, 52.5123056),
            "name": "Betriebshof Alt Friedrichsfelde",
            "capacity": 0,
            "vehicle_type": all_vehicle_type_ids,
        }
    )

    return depot_list


def optimize_scenario(scenario: Scenario, session: sqlalchemy.orm.session.Session):
    # Save a geographic trip plot
    df = prepare_geographic_trip_plot(scenario.id, session)

    # Also, prepare for a bar chart of how many rotations each depot supports
    # Create a counter of the number of rotations per depot
    pre_opt_counter = Counter()
    dropped = df.drop_duplicates(subset=["rotation_id"])
    grouped = dropped.groupby("originating_depot_name")
    for depot, group in grouped:
        name_and_id = group[["originating_depot_name", "originating_depot_id"]].iloc[0]
        pre_opt_counter[depot] = len(group)

    depot_list = depots_for_scenarios(scenario, session)

    os.environ["BASE_URL"] = "http://mpm-v-ors.mpm.tu-berlin.de:8080/ors/"
    # # Intialize the Optimizer
    optimizer = DepotRotationOptimizer(session, scenario.id)
    original_capacities = [depot["capacity"] for depot in depot_list]
    

    # Using the optimizer iteratively to reach a lower depot capacity until the solution is not feasible
    DEPOT_USAGE = 1.0
    STEP_SIZE = 0.1

    ITER = 5
    while ITER > 0:
        for depot, orig_cap in zip(depot_list, original_capacities):
            depot["capacity"] = int(orig_cap * DEPOT_USAGE)

        optimizer.get_depot_from_input(depot_list)
        optimizer.data_preparation()
        try:
            optimizer.optimize(time_report=True)
        except ValueError as e:
            print("cannot decrease depot capacity any further")
            break

        DEPOT_USAGE -= STEP_SIZE
        ITER -= 1

    optimizer.write_optimization_results(delete_original_data=True)

    assert optimizer.data["result"] is not None
    assert optimizer.data["result"].shape[0] == optimizer.data["rotation"].shape[0]

    fig = optimizer.visualize()
    fig.write_html(f"04b_complete_rotation_plan_{scenario.name_short}.html")
    fig.write_image(f"04b_complete_rotation_plan_{scenario.name_short}.svg")

    # Visualize the deadhead costs
    # optimizer.visualize_deadhead_costs()

    # Save a geographic trip plot
    # We will need to flush and expunge the session in order for the geom to be converted to binary
    # (which is necessary for the plot to be created)
    session.flush()
    session.expunge_all()
    post_df = prepare_geographic_trip_plot(scenario.id, session)

    # Also, prepare for a bar chart of how many rotations each depot supports
    # Create a counter of the number of rotations per depot
    post_opt_counter = Counter()
    dropped = post_df.drop_duplicates(subset=["rotation_id"])
    grouped = dropped.groupby("originating_depot_name")
    for depot, group in grouped:
        name_and_id = group[["originating_depot_name", "originating_depot_id"]].iloc[0]
        print(list(name_and_id))
        post_opt_counter[depot] = len(group)

    # Make sure the pre- and post-optimization counters have the same keys
    all_keys = set(pre_opt_counter.keys()).union(set(post_opt_counter.keys()))
    for key in all_keys:
        if key not in pre_opt_counter:
            pre_opt_counter[key] = 0
        if key not in post_opt_counter:
            post_opt_counter[key] = 0

    # Compare the number of rotations per depot before and after optimization in a barh plot
    fig, ax = plt.subplots()
    ind = np.arange(len(pre_opt_counter))
    width = 0.4

    # Turn the counter dictionaries into lists
    keys = list(pre_opt_counter.keys())
    pre_opt_values = [pre_opt_counter[key] for key in keys]
    post_opt_values = [post_opt_counter[key] for key in keys]

    ax.barh(ind - width / 2, pre_opt_values, width, label="Before Optimization")
    ax.barh(ind + width / 2, post_opt_values, width, label="After Optimization")
    ax.set_xlabel("Number of Rotations")
    ax.set_ylabel("Depot")
    ax.legend()

    # Set the y-ticks to the depot names (keys)
    ax.set_yticks(ind)
    ax.set_yticklabels(keys)

    plt.tight_layout()
    plt.savefig(f"04b_depot_rotation_count_{scenario.name_short}.svg")
    plt.show()


if __name__ == "__main__":
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    # Create a dataframe of the rotation distance and vehicle type short name for each rotaiton
    SCENARIO_NAMES = ["OU", "DEP", "TERM"]

    for scenario_name in SCENARIO_NAMES:
        scenario = (
            session.query(Scenario).filter(Scenario.name_short == scenario_name).first()
        )

        optimize_scenario(scenario, session)
    session.commit()
