import os

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from eflips.opt.depot_rotation_optimizer import DepotRotationOptimizer

SCENARIO_ID = 10
DATA_FROM_FILE = False
if __name__ == "__main__":

    with Session(create_engine(os.environ.get("DATABASE_URL"))) as session:
        optimizer = DepotRotationOptimizer(session, SCENARIO_ID)

        if DATA_FROM_FILE is False:
            user_input_depot = [
                {
                    "depot_station": 103281393,
                    "capacity": 400,
                    "vehicle_type": [84, 86, 87, 90],
                },  # Indira-Gandhi-Str
                {
                    "depot_station": 103280619,
                    "capacity": 225,
                    "vehicle_type": [82, 84, 85, 87],
                },  # Britz
                {
                    "depot_station": 103281456,
                    "capacity": 170,
                    "vehicle_type": [82, 84],
                },  # Lichtenberg
                {
                    "depot_station": 103282126,
                    "capacity": 250,
                    "vehicle_type": [82, 84, 85],
                },  # Cicerostr
                {
                    "depot_station": 103282127,
                    "capacity": 240,
                    "vehicle_type": [82, 84],
                },  # Müllerstr
                {
                    "depot_station": 103282128,
                    "capacity": 290,
                    "vehicle_type": [82, 84],
                },  # Spandau
            ]

            optimizer.get_depot_from_input(user_input_depot)
            optimizer.data_preparation()
            for k, v in optimizer.data.items():
                v.to_csv(f"{k}.csv", index=False)
        else:
            optimizer.data = {
                "depot": pd.read_csv("depot.csv"),
                "vehicletype_depot": pd.read_csv("vehicletype_depot.csv"),
                "rotation": pd.read_csv("rotation.csv"),
                "occupancy": pd.read_csv("occupancy.csv"),
                "cost": pd.read_csv("cost.csv"),
            }
        # TODO only for debugging

        # optimizer.optimize()
