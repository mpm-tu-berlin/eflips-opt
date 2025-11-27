import datetime
import os
from typing import Dict, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import create_engine

import pandas as pd
from eflips.model import (
    Rotation,
    Trip,
    TripType,
    Scenario,
    Station,
    Event,
    EventType,
    VehicleType,
    Area,
)

from eflips.model import create_engine as create_engine_sqlite


from eflips.tco import init_tco_parameters


SCENARIO_ID = 1
DATABASE_URL_SQLITE = os.getenv("DATABASE_URL_SQLITE")
DATABASE_URL = os.getenv("DATABASE_URL")


if __name__ == "__main__":

    # with Session(create_engine_sqlite(DATABASE_URL_SQLITE)) as session:
    with Session(create_engine(DATABASE_URL)) as session:

        scenario = session.query(Scenario).filter(Scenario.id == SCENARIO_ID).one()

        # getting blocks

        blocks = (
            session.query(Rotation)
            .filter(Rotation.scenario_id == SCENARIO_ID)
            .order_by(Rotation.id)
            .all()
        )
        # blocks = blocks[: 4000]
        block_index = sorted([b.id for b in blocks] + [0])

        # getting block costs
        block_cost: Dict[Tuple[int, int], float] = {}


        for bi in blocks:
            block_cost[(0, bi.id)] = 200.0
            block_cost[(bi.id, 0)] = 200.0

            for bj in blocks:
                if bi.id != bj.id:
                    if (
                        bi.trips[-1].route.arrival_station_id
                        == bj.trips[0].route.departure_station_id
                        and bj.trips[0].departure_time - bi.trips[-1].arrival_time
                        >= datetime.timedelta(minutes=15)
                        and bj.vehicle_type_id == bi.vehicle_type_id
                    ):
                        block_cost[(bi.id, bj.id)] = (
                            bj.trips[0].departure_time - bi.trips[-1].arrival_time
                        ).total_seconds() / 3600.0
        # setting up model



        import pyomo.environ as pyo

        LARGE_NUMBER = 1e8

        assert LARGE_NUMBER > max(block_cost.values()) * len(block_index) * 2

        model = pyo.ConcreteModel()

        model.BLOCKS = pyo.Set(initialize=block_index)
        model.x = pyo.Var(model.BLOCKS, model.BLOCKS, within=pyo.Binary)

        # def obj_expression(m):
        #     return sum(
        #         block_cost.get((i, j), 1e6) * m.x[i, j]
        #         for i in model.BLOCKS
        #         for j in model.BLOCKS
        #         if i != j
        #     )
        #
        # model.OBJ = pyo.Objective(rule=obj_expression, sense=pyo.minimize)

        def num_vehicles_rule(m):
            return sum(m.x[0, j] for j in m.BLOCKS if j != 0)

        model.NumVehicles = pyo.Objective(rule=num_vehicles_rule, sense=pyo.minimize)

        def total_cost(m):
            return (
                sum(
                    block_cost.get((i, j), LARGE_NUMBER) * m.x[i, j]
                    for i in model.BLOCKS
                    for j in model.BLOCKS
                    if i != j
                )
                <= LARGE_NUMBER
            )

        model.OBJ = pyo.Constraint(rule=total_cost)

        def one_path_rule(m, j):
            if j == 0:
                return pyo.Constraint.Skip

            return sum(m.x[i, j] for i in m.BLOCKS if (i != j)) == 1

        model.PATH_CONS = pyo.Constraint(model.BLOCKS, rule=one_path_rule)

        def flow_conservation_rule(m, j):
            if j == 0:
                return pyo.Constraint.Skip
            return sum(m.x[i, j] for i in m.BLOCKS if (i != j)) == sum(
                m.x[j, k] for k in m.BLOCKS if (j != k)
            )

        model.FLOW_CONS = pyo.Constraint(model.BLOCKS, rule=flow_conservation_rule)

        # fix some variables to 0

        for i in model.BLOCKS:
            for j in model.BLOCKS:
                if (i, j) not in block_cost:
                    model.x[i, j].fix(0)

        # solving model
        solver = pyo.SolverFactory("gurobi")
        results = solver.solve(model, tee=True)

        # extracting solution
        # solution = []
        # for i in model.BLOCKS:
        #     for j in model.BLOCKS:
        #         if i != j and pyo.value(model.x[i, j]) > 0.5:
        #             solution.append((i, j))
        #
        #
        # num_vehicles = sum(1 for i, j in solution if i == 0)

        print("number of vehicles needed:", pyo.value(model.NumVehicles))
