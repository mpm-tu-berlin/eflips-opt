import openrouteservice
import os
import math

from datetime import timedelta

import random

from sqlalchemy.orm import Session
from sqlalchemy import func, select

from geoalchemy2.shape import to_shape

import pandas as pd
import numpy as np
from shapely import Point

from sqlalchemy.orm.exc import NoResultFound

from eflips.model import (
    Depot,
    Rotation,
    Trip,
    Route,
    Station,
    VehicleType,
    Area,
    AssocRouteStation,
    StopTime,
    Line,
    TripType,
)


def deadhead_cost(
    p1: Point,
    p2: Point,
    cost="distance",
    profile="driving-car",
    service="directions",
    data_format="geojson",
):
    """
    Calculate the cost between two points using the openrouteservice API

    :param p1: Point 1
    :param p2: Point 2
    :param cost: Cost metric to use, default is distance
    :param profile: Profile to use, default is driving-car
    :param service: Service to use, default is directions
    :param data_format: Data format to use, default is geojson
    """

    base_url = os.environ["BASE_URL"]
    new_url = os.path.join("v2", service, profile)
    if base_url is None:
        raise ValueError("BASE_URL is not set")

    client = openrouteservice.Client(base_url=base_url)
    coords = ((p1.x, p1.y), (p2.x, p2.y))

    routes = client.request(
        url=new_url, post_json={"coordinates": coords, "format": data_format}
    )

    return routes["routes"][0]["segments"][0][
        cost
    ]  # Using segments instead of summary for 0 distance cases


def get_rand_rotation(session, scenario_id, n):
    """
    Get n random rotations from the scenario. It is used to select different set of rotations in testing cases.
    :param session:
    :param scenario_id:
    :param n:
    :return:
    """
    rotidx = session.scalars(
        select(Rotation.id)
        .filter(Rotation.scenario_id == scenario_id)
        .order_by(func.random())
        .limit(n)
    ).all()

    return rotidx


def get_deport_rot_assign(session, scenario_id, rotidx):
    data = []
    for rid in rotidx:
        trips = (
            session.query(Trip.id)
            .filter(Trip.rotation_id == rid)
            .order_by(Trip.departure_time)
            .all()
        )

        depot_id = (
            session.query(Depot.id)
            .join(Station, Station.id == Depot.station_id)
            .join(Route, Station.id == Route.departure_station_id)
            .join(Trip, Trip.route_id == Route.id)
            .filter(Trip.id == trips[0][0])
            .one()
        )

        data.append([rid, depot_id[0]])

    return pd.DataFrame(data, columns=["rotation_id", "depot_id"])


def rotation_data(session, randidx) -> pd.DataFrame:
    """


    :param session:
    :param scenario_id:
    :return:
    """
    # get non depot start and end station for each rotation

    rot_start_end = []

    # rotations = session.query(Rotation.id).filter(Rotation.scenario_id == scenario_id).all()
    for rotation in randidx:
        trips = (
            session.query(Trip.id)
            .filter(Trip.rotation_id == rotation)
            .order_by(Trip.departure_time)
            .all()
        )

        # Find the first and last non-depot station for each rotation
        # TODO potential optimization?
        first_non_depot_station = (
            session.query(Station.id, Station.geom)
            .join(Route, Station.id == Route.arrival_station_id)
            .join(Trip, Trip.route_id == Route.id)
            .filter(Trip.id == trips[0][0])
            .one()
        )

        last_non_depot_station = (
            session.query(Station.id, Station.geom)
            .join(Route, Station.id == Route.departure_station_id)
            .join(Trip, Trip.route_id == Route.id)
            .filter(Trip.id == trips[-1][0])
            .one()
        )

        rot_start_end.append(
            [
                rotation,
                first_non_depot_station[0],
                to_shape(first_non_depot_station[1]),
                last_non_depot_station[0],
                to_shape(last_non_depot_station[1]),
            ]
        )

    rotation_df = pd.DataFrame(
        rot_start_end,
        columns=[
            "rotation_id",
            "first_non_depot_station_id",
            "first_non_depot_station_coord",
            "last_non_depot_station_id",
            "last_non_depot_station_coord",
        ],
    )
    return rotation_df


def depot_data(session, scenario_id):
    """

    :param session:
    :param scenario_id:
    :return:
    """

    depots = (
        session.query(Depot.id, Station.geom)
        .join(Station, Depot.station_id == Station.id)
        .filter(Depot.scenario_id == scenario_id)
        .all()
    )

    depot_df = pd.DataFrame(
        [(depot[0], to_shape(depot[1])) for depot in depots],
        columns=["depot_id", "depot_coord"],
    )

    return depot_df


def depot_capacity(session, scenario_id):
    """

    :param session:
    :param scenario_id:
    :return:
    """

    vehicle_types = (
        session.query(VehicleType.id)
        .filter(VehicleType.scenario_id == scenario_id)
        .all()
    )
    depots = session.query(Depot.id).filter(Depot.scenario_id == scenario_id).all()

    capacities = []
    for depot in depots:
        for vehicle_type in vehicle_types:
            area_capacity = (
                session.query(func.max(Area.capacity))
                .filter(
                    Area.depot_id == depot[0],
                    Area.vehicle_type_id == vehicle_type[0],
                    Area.scenario_id == scenario_id,
                )
                .group_by(Area.depot_id, Area.vehicle_type_id)
                .all()
            )
            capacities.append(
                [
                    depot[0],
                    vehicle_type[0],
                    0 if len(area_capacity) == 0 else area_capacity[0][0],
                ]
            )

    return pd.DataFrame(
        capacities, columns=["depot_id", "vehicle_type_id", "capacity"]
    ).set_index(["depot_id", "vehicle_type_id"])


def vehicletype_data(session, scenario_id):
    """

    :param session:
    :param scenario_id:
    :return:
    """

    vehicle_types = (
        session.query(VehicleType.id)
        .filter(VehicleType.scenario_id == scenario_id)
        .all()
    )
    vt_df = pd.DataFrame(
        [v[0] for v in vehicle_types], columns=["vehicle_type_id"]
    ).set_index("vehicle_type_id")

    return vt_df


def rotation_vehicle_assign(session, scenario_id, rotidx):
    """

    :param session:
    :param scenario_id:
    :return:
    """

    # rotations = session.query(Rotation.id, Rotation.vehicle_type_id).filter(Rotation.scenario_id == scenario_id).all()
    vehicle_types = (
        session.query(VehicleType.id)
        .filter(VehicleType.scenario_id == scenario_id)
        .all()
    )

    assignment = []

    for rotation in rotidx:
        for vehicle_type in vehicle_types:
            r_vid = (
                session.query(Rotation.vehicle_type_id)
                .filter(Rotation.id == rotation)
                .one()[0]
            )
            assignment.append(
                [rotation, vehicle_type[0], 1 if r_vid == vehicle_type[0] else 0]
            )

    return pd.DataFrame(
        assignment, columns=["rotation_id", "vehicle_type_id", "assignment"]
    )


def cost_rotation_depot(rotation_data: pd.DataFrame, depot_data: pd.DataFrame):
    # TODO: make cost a callable
    """

    :param rotation_data:
    :param depot_data:
    :return:
    """

    rotation_data = rotation_data.merge(depot_data, how="cross")

    rotation_data["distance"] = rotation_data.apply(
        lambda x: deadhead_cost(
            x["first_non_depot_station_coord"], x["depot_coord"], cost="distance"
        )
        + deadhead_cost(
            x["last_non_depot_station_coord"], x["depot_coord"], cost="distance"
        ),
        axis=1,
    )

    rotation_data["duration"] = rotation_data.apply(
        lambda x: deadhead_cost(
            x["first_non_depot_station_coord"], x["depot_coord"], cost="duration"
        )
        + deadhead_cost(
            x["last_non_depot_station_coord"], x["depot_coord"], cost="duration"
        ),
        axis=1,
    )

    cost = rotation_data[["rotation_id", "depot_id", "distance", "duration"]].set_index(
        ["rotation_id", "depot_id"]
    )
    return cost


def get_occupancy(
    session: Session,
    scenario_id: int,
    rotation_id: list[int],
    time_window=timedelta(minutes=40),
):
    """
    Get the occupancy of the rotation
    :param session:
    :param scenario_id:
    :param rotation_id:
    :return:
    """
    start_and_end_time = (
        session.query(func.min(Trip.departure_time), func.max(Trip.arrival_time))
        .filter(Trip.scenario_id == scenario_id, Trip.rotation_id.in_(rotation_id))
        .one()
    )
    start_time = start_and_end_time[0].timestamp()
    end_time = start_and_end_time[1].timestamp()

    rotations = session.query(Rotation).filter(Rotation.id.in_(rotation_id)).all()
    sampled_time_stamp = np.arange(
        start_time, end_time, time_window.total_seconds(), dtype=int
    )
    occupancy = np.zeros((len(rotations), len(sampled_time_stamp)), dtype=int)

    for idx, rotation in enumerate(rotations):

        rotation_start = rotation.trips[0].departure_time.timestamp()
        rotation_end = rotation.trips[-1].arrival_time.timestamp()
        occupancy[idx] = np.interp(
            sampled_time_stamp,
            [rotation_start, rotation_end],
            [1, 1],
            left=0,
            right=0,
        )
    occupancy = pd.DataFrame(occupancy, columns=[sampled_time_stamp], index=rotation_id)
    return occupancy


def update_deadhead_trip(session: Session, new_assign: pd.DataFrame):
    """
    Update the deadhead trip in the database according to result of the optimization
    :param session:
    :param new_assign:
    :return:
    """

    for idx, row in new_assign.iterrows():
        # Create new route with the new depot

        # Create new trip with the new route and time

        # Update the rotation.

        depot_id = row["depot_id"]

        depot_station_id = (
            session.query(Depot.station_id).filter(Depot.id == depot_id).one()[0]
        )
        trip = (
            session.query(Trip)
            .filter(Trip.rotation_id == int(row["rotation_id"]))
            .order_by(Trip.departure_time)
            .all()
        )

        # Update the first trip
        ferry_route = trip[0].route
        if ferry_route.departure_station_id == depot_station_id:
            # The assignment is the same.
            continue
        else:

            # Create new data entries
            scenario_id = (
                session.query(Depot.scenario_id).filter(Depot.id == depot_id).one()[0]
            )
            first_station = ferry_route.arrival_station
            last_station = trip[-1].route.departure_station
            depot_station = (
                session.query(Station)
                .join(Depot, Station.id == Depot.station_id)
                .filter(Depot.id == depot_id)
                .one()
            )
            # Get distance from depot to first station
            ferry_distance = deadhead_cost(
                to_shape(depot_station.geom),
                to_shape(first_station.geom),
                cost="distance",
            )
            ferry_duration = math.ceil(
                deadhead_cost(
                    to_shape(depot_station.geom),
                    to_shape(first_station.geom),
                    cost="duration",
                )
            )

            return_distance = deadhead_cost(
                to_shape(last_station.geom),
                to_shape(depot_station.geom),
                cost="distance",
            )
            return_duration = math.ceil(
                deadhead_cost(
                    to_shape(last_station.geom),
                    to_shape(depot_station.geom),
                    cost="duration",
                )
            )
            # Ferry trip

            ferry_route = Route(
                scenario_id=scenario_id,
                name="RE" + str(depot_station.id) + "_" + str(first_station.id),
                departure_station_id=depot_station.id,
                arrival_station_id=first_station.id,
                distance=ferry_distance,
            )

            return_route = Route(
                scenario_id=scenario_id,
                name="RA" + str(first_station.id) + "_" + str(depot_station.id),
                departure_station_id=last_station.id,
                arrival_station_id=depot_station.id,
                distance=return_distance,
            )

            assoc_ferry_station = [
                AssocRouteStation(
                    scenario_id=scenario_id,
                    station_id=depot_station.id,
                    route=ferry_route,
                    elapsed_distance=0,
                ),
                AssocRouteStation(
                    scenario_id=scenario_id,
                    station_id=first_station.id,
                    route=ferry_route,
                    elapsed_distance=ferry_distance,
                ),
            ]

            ferry_route.assoc_route_stations = assoc_ferry_station

            assoc_return_station = [
                AssocRouteStation(
                    scenario_id=scenario_id,
                    station_id=last_station.id,
                    route=return_route,
                    elapsed_distance=0,
                ),
                AssocRouteStation(
                    scenario_id=scenario_id,
                    station_id=depot_station.id,
                    route=return_route,
                    elapsed_distance=return_distance,
                ),
            ]

            return_route.assoc_route_stations = assoc_return_station
            session.add(ferry_route)
            session.add(return_route)

            # Create new trip
            new_ferry_trip = Trip(
                scenario_id=scenario_id,
                route=ferry_route,
                trip_type=TripType.EMPTY,
                departure_time=trip[0].arrival_time - timedelta(seconds=ferry_duration),
                arrival_time=trip[0].arrival_time,
                rotation_id=row["rotation_id"],
            )
            session.add(new_ferry_trip)

            new_return_trip = Trip(
                scenario_id=scenario_id,
                route=return_route,
                trip_type=TripType.EMPTY,
                departure_time=trip[-1].departure_time,
                arrival_time=trip[-1].departure_time
                + timedelta(seconds=return_duration),
                rotation_id=row["rotation_id"],
            )

            session.add(new_return_trip)

            session.flush()

            # Assign new trip to the stop times
            session.query(StopTime).filter(StopTime.trip_id == trip[0].id).update(
                {StopTime.trip_id: new_ferry_trip.id}
            )
            session.query(StopTime).filter(StopTime.trip_id == trip[-1].id).update(
                {StopTime.trip_id: new_return_trip.id}
            )
            # Delete the old trip
            session.query(Trip).filter(Trip.id == trip[0].id).delete()
            session.query(Trip).filter(Trip.id == trip[-1].id).delete()

            session.flush()
