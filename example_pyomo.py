import os

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import create_engine


from eflips.opt.data_preperation import rotation_data, depot_data, depot_capacity, rotation_vehicle_assign, \
    cost_rotation_depot, \
    vehicletype_data, get_rand_rotation, get_deport_rot_assign


import pyomo.environ as pyo
from pyomo.common.timing import report_timing

from matplotlib import pyplot as plt

if __name__ == "__main__":
    engine = create_engine(os.environ.get("DATABASE_URL"))
    session = Session(engine)

    rotidx = get_rand_rotation(session, 10, 400)

    orig_assign = get_deport_rot_assign(session, 10, rotidx)

    # Data preparation
    # Dummy cost list between depot and rotation

    rotation_df = rotation_data(session, rotidx)
    depot_df = depot_data(session, 10)

    capacities = depot_capacity(session, 10)

    vt_df = vehicletype_data(session, 10)

    assignment = rotation_vehicle_assign(session, 10, rotidx)

    cost = cost_rotation_depot(rotation_df, depot_df)
    cost.reset_index()

    orig_assign["cost"] = orig_assign.apply(
        lambda x: cost.loc[x["rotation_id"], x["depot_id"]], axis=1)

    # Building model in pyomo
    # i for rotations
    I = rotidx
    # j for depots
    J = depot_df["depot_id"].tolist()
    # t for vehicle types
    T = vt_df.index.tolist()

    # n_jt: depot-vehicle type capacity
    n = capacities.to_dict()["capacity"]

    # v_it: rotation-type assignment
    v = assignment.set_index(["rotation_id", "vehicle_type_id"]).to_dict()["assignment"]

    # c_ij: rotation-depot cost
    c = cost.to_dict()["cost"]

    print("data acquired")

    # Set up pyomo model
    report_timing()

    model = pyo.ConcreteModel(name="depot_rot_problem")

    model.x = pyo.Var(I, J, domain=pyo.Binary)


    # Objective function
    @model.Objective()
    def obj(m):
        return sum(c[i, j] * model.x[i, j] for i in I for j in J)


    # Constraints
    # Each rotation is assigned to exactly one depot
    @model.Constraint(I)
    def one_depot_per_rot(m, i):
        return sum(model.x[i, j] for j in J) == 1


    # Depot capacity constraint
    @model.Constraint(J, T)
    def depot_capacity_constraint(m, j, t):
        return sum(v[i, t] * model.x[i, j] for i in I) <= n[j, t]


    # Solve

    result = pyo.SolverFactory("gurobi").solve(model, tee=True)
    # model.pprint()

    # Extract results
    data = result.Problem._list

    new_assign = pd.DataFrame({"rotation_id": [i[0] for i in model.x if model.x[i].value == 1.0],
                               "depot_id": [i[1] for i in model.x if model.x[i].value == 1.0],
                               "assignment": [model.x[i].value for i in model.x if model.x[i].value == 1.0]})

    new_assign["cost"] = new_assign.apply(lambda x: cost.loc[x["rotation_id"], x["depot_id"]], axis=1)

    # Plotting
    orig_cost = orig_assign["cost"]
    new_cost = new_assign["cost"]
    orig_cost_mean = orig_cost.mean()
    new_cost_mean = new_cost.mean()

    plt.hist(orig_cost, alpha=0.5)
    plt.hist(new_cost, alpha=0.5)
    plt.legend(["Original", "New"])

    min_ylim, max_ylim = plt.ylim()
    plt.axvline(orig_cost_mean, color='k', linestyle='dashed', linewidth=1)
    plt.text(orig_cost_mean * 1.1, max_ylim * 0.9, 'Old_Mean: {:.2f}'.format(orig_cost_mean))
    plt.axvline(new_assign["cost"].mean(), color='b', linestyle='dashed', linewidth=1)
    plt.text(new_cost_mean * 1.1, max_ylim * 0.1, 'New_Mean: {:.2f}'.format(new_cost_mean))

    plt.show()

