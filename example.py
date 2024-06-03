import os

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from eflips.opt.depot_rotation_optimizer import DepotRotationOptimizer

SCENARIO_ID = 11

if __name__ == "__main__":

    with Session(create_engine(os.environ.get("DATABASE_URL"))) as session:
        optimizer = DepotRotationOptimizer(session, SCENARIO_ID)

        user_input_depot = [
            {
                "depot_station": 103281393,
                "capacity": 400,
                "vehicle_type": [84, 86, 87, 91],
            },  # Indira-Gandhi-Str
            {
                "depot_station": 103280619,
                "capacity": 225,
                "vehicle_type": [82, 84, 85, 87],
            },  # Britz
            {
                "depot_station": 103281456,
                "capacity": 170,
                "vehicle_type": [82, 83, 87],
            },  # Lichtenberg
            {
                "depot_station": 103282126,
                "capacity": 250,
                "vehicle_type": [82, 84, 85, 87, 90],
            },  # Cicerostr
            {
                "depot_station": 103282127,
                "capacity": 240,
                "vehicle_type": [85, 87, 82, 84],
            },  # Müllerstr
            {
                "depot_station": 103282128,
                "capacity": 290,
                "vehicle_type": [82, 84, 88],
            },  # Spandau
        ]

        optimizer.get_depot_from_input(user_input_depot)
        optimizer.data_preparation()

        optimizer.optimize(time_report=True)
        optimizer.data["result"].to_csv("results.csv")
        optimizer.visualize()
        optimizer.write_optimization_results(delete_original_data=True)
        session.rollback()
