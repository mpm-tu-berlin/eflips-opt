import os
import sqlalchemy
import random
from math import sin, cos, acos
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy import func

from geoalchemy2.shape import to_shape

import pandas as pd
from shapely import Point

from eflips.model import Depot, Rotation, Trip, Route, Station, VehicleType, Area


def distance(p1: Point, p2: Point):
    return 6371.01 * acos(sin(p1.y) * sin(p2.y) + cos(p1.y) * cos(p2.y) * cos(p1.x - p2.x))


def get_rand_rotation(session, scenario_id, n):
    rotations = session.query(Rotation.id).filter(Rotation.scenario_id == scenario_id).order_by(func.random()).limit(
        n).all()
    randidx = random.sample(range(0, len(rotations)), n)
    rotidx = [rotations[r][0] for r in randidx]
    return rotidx


def get_deport_rot_assign(session, scenario_id, rotidx):
    data = []
    for rid in rotidx:
        trips = session.query(Trip.id).filter(Trip.rotation_id == rid).order_by(Trip.departure_time).all()

        depot_id = session.query(Depot.id).join(Station, Station.id == Depot.station_id).join(Route,
                                                                                              Station.id == Route.departure_station_id).join(
            Trip, Trip.route_id == Route.id).filter(Trip.id == trips[0][0]).one()

        data.append([rid, depot_id[0]])

    return pd.DataFrame(data, columns=["rotation_id", "depot_id"])


def rotation_data(session, scenario_id, randidx) -> pd.DataFrame:
    """


    :param session:
    :param scenario_id:
    :return:
    """
    # get non depot start and end station for each rotation

    rot_start_end = []
    rotations = []

    # rotations = session.query(Rotation.id).filter(Rotation.scenario_id == scenario_id).all()
    for rotation in randidx:
        trips = session.query(Trip.id).filter(Trip.rotation_id == rotation).order_by(Trip.departure_time).all()

        # Find the first and last non-depot station for each rotation
        # TODO potential optimization?
        first_non_depot_station = session.query(Station.id, Station.geom).join(Route,
                                                                               Station.id == Route.arrival_station_id).join(
            Trip, Trip.route_id == Route.id).filter(Trip.id == trips[0][0]).one()

        last_non_depot_station = session.query(Station.id, Station.geom).join(Route,
                                                                              Station.id == Route.departure_station_id).join(
            Trip, Trip.route_id == Route.id).filter(Trip.id == trips[-1][0]).one()

        rot_start_end.append(
            [rotation, first_non_depot_station[0], to_shape(first_non_depot_station[1]), last_non_depot_station[0],
             to_shape(last_non_depot_station[1])])

    rotation_df = pd.DataFrame(rot_start_end,
                               columns=["rotation_id", "first_non_depot_station_id", "first_non_depot_station_coord",
                                        "last_non_depot_station_id", "last_non_depot_station_coord"])
    return rotation_df


def depot_data(session, scenario_id):
    """

    :param session:
    :param scenario_id:
    :return:
    """

    depots = session.query(Depot.id, Station.geom).join(Station, Depot.station_id == Station.id).filter(
        Depot.scenario_id == scenario_id).all()

    depot_df = pd.DataFrame([(depot[0], to_shape(depot[1])) for depot in depots], columns=["depot_id", "depot_coord"])

    return depot_df


def depot_capacity(session, scenario_id):
    """

    :param session:
    :param scenario_id:
    :return:
    """

    vehicle_types = session.query(VehicleType.id).filter(VehicleType.scenario_id == scenario_id).all()
    depots = session.query(Depot.id).filter(Depot.scenario_id == scenario_id).all()

    capacities = []
    for depot in depots:
        for vehicle_type in vehicle_types:
            area_capacity = session.query(func.max(Area.capacity)).filter(Area.depot_id == depot[0],
                                                                          Area.vehicle_type_id == vehicle_type[0],
                                                                          Area.scenario_id == scenario_id).group_by(
                Area.depot_id, Area.vehicle_type_id).all()
            capacities.append(
                [depot[0], vehicle_type[0], 0 if len(area_capacity) == 0 else area_capacity[0][0]])

    return pd.DataFrame(capacities, columns=["depot_id", "vehicle_type_id", "capacity"]).set_index(
        ["depot_id", "vehicle_type_id"])


def vehicletype_data(session, scenario_id):
    """

    :param session:
    :param scenario_id:
    :return:
    """

    vehicle_types = session.query(VehicleType.id).filter(VehicleType.scenario_id == scenario_id).all()
    vt_df = pd.DataFrame([v[0] for v in vehicle_types], columns=["vehicle_type_id"]).set_index("vehicle_type_id")

    return vt_df


def rotation_vehicle_assign(session, scenario_id, rotidx):
    """

    :param session:
    :param scenario_id:
    :return:
    """

    # rotations = session.query(Rotation.id, Rotation.vehicle_type_id).filter(Rotation.scenario_id == scenario_id).all()
    vehicle_types = session.query(VehicleType.id).filter(VehicleType.scenario_id == scenario_id).all()

    assignment = []

    for rotation in rotidx:
        for vehicle_type in vehicle_types:
            r_vid = session.query(Rotation.vehicle_type_id).filter(Rotation.id == rotation).one()[0]
            assignment.append([rotation, vehicle_type[0], 1 if r_vid == vehicle_type[0] else 0])

    return pd.DataFrame(assignment, columns=["rotation_id", "vehicle_type_id", "assignment"])


def cost_rotation_depot(rotation_data: pd.DataFrame, depot_data: pd.DataFrame):
    """

    :param rotation_data:
    :param depot_data:
    :return:
    """

    rotation_data = rotation_data.merge(depot_data, how="cross")

    rotation_data["cost"] = rotation_data.apply(
        lambda x: distance(x["first_non_depot_station_coord"], x["depot_coord"]) + distance(
            x["last_non_depot_station_coord"], x["depot_coord"]), axis=1)

    cost = rotation_data[["rotation_id", "depot_id", "cost"]].set_index(["rotation_id", "depot_id"])
    return cost
