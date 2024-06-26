import os

from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from eflips.opt.depot_rotation_optimizer import DepotRotationOptimizer

SCENARIO_ID = 8

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
                "depot_station": 103276448,
                "capacity": 150,
                "vehicle_type": list(range(62, 72)),
            },  # Cicerostraße
            {
                "depot_station": 103263800,
                "capacity": 150,
                "vehicle_type": list(range(62, 72)),
            },  # Gorkistr./Ziekowstr.
        ]

        optimizer.get_depot_from_input(user_input_depot)
        optimizer.data_preparation()

        optimizer.optimize(time_report=True)
        optimizer.visualize()
        optimizer.write_optimization_results(delete_original_data=True)
        session.commit()
