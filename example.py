import os

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from eflips.opt.depot_rotation_optimizer import DepotRotationOptimizer

SCENARIO_ID = 11
DATA_FROM_FILE = False

if __name__ == "__main__":

    with Session(create_engine(os.environ.get("DATABASE_URL"))) as session:
        optimizer = DepotRotationOptimizer(session, SCENARIO_ID)

        if DATA_FROM_FILE is False:
            user_input_depot = [
                {
                    "depot_station": 103281393,
                    "capacity": 400,
                    "vehicle_type": [84, 86, 87, 91],
                },  # Indira-Gandhi-Str
                {
                    "depot_station": 103280619,
                    "capacity": 225,
                    "vehicle_type": [83, 84, 85, 87],
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
                    "vehicle_type": [82, 84, 86],
                },  # Müllerstr
                {
                    "depot_station": 103282128,
                    "capacity": 290,
                    "vehicle_type": [82, 84, 88],
                },  # Spandau
            ]

            optimizer.get_depot_from_input(user_input_depot)
            optimizer.data_preparation()
            for k, v in optimizer.data.items():
                v.to_csv(f"{k}.csv", index=False)
        else:
            optimizer.data = {
                "depot": pd.read_csv("depot.csv"),
                "vehicletype_depot": pd.read_csv("vehicletype_depot_df.csv"),
                "rotation": pd.read_csv("rotation.csv"),
                "occupancy": pd.read_csv("occupancy.csv"),
                "cost": pd.read_csv("cost.csv"),
                "vehicle_type": pd.read_csv("vehicle_type.csv"),
                "assignment": pd.read_csv("assignment.csv"),
            }


            optimizer.data["occupancy"].set_index("rotation_id", inplace=True)
            optimizer.data["vehicletype_depot"].set_index("vehicle_type_id", inplace=True)
            tmp_dict = optimizer.data["vehicletype_depot"].to_dict()
            a = {}
            for k, v in tmp_dict.items():
                a[int(k)] = v

            optimizer.data["vehicletype_depot"] = a


        # TODO only for debugging

        optimizer.optimize(time_report=True)
        optimizer.data["result"].to_csv("results.csv")
        optimizer.visualize()
