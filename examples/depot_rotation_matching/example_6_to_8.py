import os

from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from eflips.opt.depot_rotation_matching import DepotRotationOptimizer

SCENARIO_ID = 10

if __name__ == "__main__":

    if (
        "DATABASE_URL" not in os.environ
        or os.environ["DATABASE_URL"] is None
        or os.environ["DATABASE_URL"] == ""
    ):
        raise ValueError(
            "The database url must be specified either as an argument or as the environment variable DATABASE_URL."
        )
    else:
        DATABASE_URL = os.environ["DATABASE_URL"]

    with Session(create_engine(DATABASE_URL)) as session:
        optimizer = DepotRotationOptimizer(session, SCENARIO_ID)

        user_input_depot = [
            {
                "depot_station": 103281393,
                "capacity": 300,
                "vehicle_type": [84, 86, 87, 90],
            },  # Indira-Gandhi-Str
            {
                "depot_station": 103280619,
                "capacity": 140,
                "vehicle_type": [82, 84, 85, 87],
            },  # Britz
            {
                "depot_station": 103281456,
                "capacity": 120,
                "vehicle_type": [82, 84],
            },  # Lichtenberg
            {
                "depot_station": 103282126,
                "capacity": 209,
                "vehicle_type": [82, 84, 85],
            },  # Cicerostr
            {
                "depot_station": 103282127,
                "capacity": 155,
                "vehicle_type": [82, 84],
            },  # Müllerstr
            {
                "depot_station": 103282128,
                "capacity": 220,
                "vehicle_type": [82, 84],
            },  # Spandau
            {
                "name": "Saentisstr.",
                "depot_station": (13.385661335581752, 52.41678762604055),
                "capacity": 230,
                "vehicle_type": [82, 84, 85, 86, 87, 90],
            },  # Säntisstr.
            {
                "name": "Suedost",
                "depot_station": (13.497371828836501, 52.46541010322369),
                "capacity": 260,
                "vehicle_type": [82, 84, 85, 86, 87, 90],
            },  # Südost
        ]

        original_capacities = [depot["capacity"] for depot in user_input_depot]
        DEPOT_USAGE = 1.0

        ITER = 5
        while ITER > 0:
            for depot, orig_cap in zip(user_input_depot, original_capacities):
                depot["capacity"] = int(orig_cap * DEPOT_USAGE)

            optimizer.get_depot_from_input(user_input_depot)
            optimizer.data_preparation()

            try:
                optimizer.optimize(time_report=True)
            except ValueError as e:
                print("cannot decrease depot capacity any further")
                break

            DEPOT_USAGE -= 0.1
            ITER -= 1

        optimizer.write_optimization_results(delete_original_data=True)
        session.commit()
