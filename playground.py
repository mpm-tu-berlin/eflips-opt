import os
import random
import sys

from random import randint
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from random import randint

from eflips.opt import rotation_data, depot_data, depot_capacity, rotation_vehicle_assign, cost_rotation_depot, \
    vehicletype_data, get_rand_rotation, get_deport_rot_assign

from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum, Sense

from matplotlib import pyplot as plt

if __name__ == "__main__":
    engine = create_engine(os.environ.get("DATABASE_URL"))
    session = Session(engine)

    rotidx = get_rand_rotation(session, 10, 330)

    orig_assign = get_deport_rot_assign(session, 10, rotidx)

    # Data preparation
    # Dummy cost list between depot and rotation


    rotation_df = rotation_data(session, 10, rotidx)
    depot_df = depot_data(session, 10)



    capacities = depot_capacity(session, 10)


    vt_df = vehicletype_data(session, 10)


    assignment = rotation_vehicle_assign(session, 10, rotidx)

    cost = cost_rotation_depot(rotation_df, depot_df)
    cost.reset_index()

    orig_assign["cost"] = orig_assign.apply(
        lambda x: cost.loc[x["rotation_id"], x["depot_id"]], axis=1)



    # Set up gampspy container
    container = Container()

    # Set up sets
    # i: rotations
    i = Set(container=container, name="i", description="Rotations", records=rotation_df["rotation_id"])
    # print("i")
    print(i.isValid())

    # j: depots
    j = Set(container=container, name="j", description="Depots", records=depot_df["depot_id"])
    # print("j")
    print(j.isValid())

    # t: vehicle types
    t = Set(container=container, name="t", description="Vehicle types", records=vt_df.index)
    # print("t")
    print(t.isValid())

    # Set up parameters

    # n_jt: depot-vehicle type capacity
    n = Parameter(container=container, name="n", domain=[j, t], description="Depot-vehicle type capacity",
                  records=capacities.reset_index())
    # print(n.records)
    print("n shape:")
    print(n.records.shape)
    print(n.isValid())

    # v_it: vehicle type-rotation assignment

    assigment = assignment.reset_index()


    # v must contain all the t and i values
    v = Parameter(container=container, name="v", domain=[i, t], description="Vehicle type-rotation assignment",
                  records=assignment)
    v.isValid()
    print(v.records)

    # c_ij: cost between depot and rotation
    c = Parameter(container=container, name="c", domain=[i, j], description="Cost between depot and rotation",
                  records=cost.reset_index())
    print(c.records)
    print(c.isValid())

    # Set up variables
    # x_ij: decision variable for depot-rotation assignment
    x = Variable(container=container, name="x", domain=[i, j], type="binary", description="Depot-rotation assignment")

    # Set up equations

    # Equation 1: Each rotation is assigned to exactly one depot
    rot_depot_constraint = Equation(container=container, name="rot_depot_constraint",
                                    domain=[i], description="Each rotation is assigned to exactly one depot",
                                    definition=Sum(j, x[i, j]) == 1)

    # rot_depot_constraint[i] = Sum(j, x[i, j]) == 1

    # Equation 2: Depot capacity constraint
    depot_capacity_constraint = Equation(container=container, name="depot_capacity_constraint",
                                         domain=[j, t], description="Depot capacity constraint",
                                         definition=Sum(i, v[i, t] * x[i, j]) <= n[j, t])

    # depot_capacity_constraint[j, t] = Sum(i, v[i, t] * x[i, j]) <= n[j, t]

    # Equation 3: Objective function

    obj = Sum((i, j), c[i, j] * x[i, j])
    #
    # Model
    depot_rot_problem = Model(container=container, name="depot_rot_problem", equations=container.getEquations(),
                              problem="MIP", sense=Sense.MIN, objective=obj)

    # Solve

    depot_rot_problem.solve(output=sys.stdout)

    x.records.set_index(["i", "j"])


    print(x.records)
    new_assign = x.records.loc[x.records['level'] == 1.0]

    # Plot
    plt.hist(orig_assign["cost"])

    plt.hist(new_assign["marginal"])
    plt.legend(["Original", "New"])
    plt.show()




