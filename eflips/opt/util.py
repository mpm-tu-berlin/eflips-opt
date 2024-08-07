import asyncio
import pickle
import urllib.parse
import posixpath
from tempfile import gettempdir
from typing import Tuple, List, Dict, Coroutine, Any, Awaitable

import os

import openrouteservice  # type: ignore
import sqlalchemy.orm.session
from sqlalchemy.orm import Session
from sqlalchemy import func

from geoalchemy2.shape import to_shape

import pandas as pd
import numpy as np

from eflips.model import (
    Depot,
    Rotation,
    Trip,
    Route,
    Station,
    VehicleType,
)


async def deadhead_cost(
    point_start: Tuple[float, float],
    point_end: Tuple[float, float],
    point_depot: Tuple[float, float],
    client: openrouteservice.Client,
    profile: str = "driving-car",
    service: str = "directions",
    data_format: str = "geojson",
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate the cost between two points using the openrouteservice API

    :param client:
    :param point_start: Point start station after depot
    :param point_end: Point end station before depot
    :param point_depot: Point depot
    :param cost: Cost metric to use, default is distance
    :param profile: Profile to use, default is driving-car
    :param service: Service to use, default is directions
    :param data_format: Data format to use, default is geojson

    :return: A dictionary with the distance and duration between the two points
    """

    base_url = os.environ["BASE_URL"]
    relative_url = posixpath.join("v2", service, profile)
    new_url = urllib.parse.urljoin("v2", relative_url)
    if base_url is None:
        raise ValueError("BASE_URL is not set")

    coords = (point_start, point_end, point_depot)

    temporary_directory = os.path.join(gettempdir(), f"eflips-ors-cache-{os.getuid()}")
    os.makedirs(temporary_directory, exist_ok=True)

    file_name = f"{coords}.pkl"
    file_path = os.path.join(temporary_directory, file_name)

    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            routes = pickle.load(file)
    else:
        routes_ferry = client.request(
            url=new_url,
            post_json={
                "coordinates": (point_depot, point_start),
                "format": data_format,
            },
        )
        routes_return = client.request(
            url=new_url,
            post_json={"coordinates": (point_end, point_depot), "format": data_format},
        )

        routes = (routes_ferry, routes_return)

        with open(file_path, "wb") as file:
            pickle.dump(routes, file)

    return {
        "distance": (
            routes[0]["routes"][0]["segments"][0]["distance"],
            routes[1]["routes"][0]["segments"][0]["distance"],
        ),
        "duration": (
            round(routes[0]["routes"][0]["segments"][0]["duration"]),
            round(routes[1]["routes"][0]["segments"][0]["duration"]),
        ),
    }  # Using segments instead of summary for 0 distance cases


async def calculate_deadhead_costs(
    df: pd.DataFrame, client: openrouteservice.Client
) -> List[Coroutine[Awaitable[Dict[str, Tuple[float, float]]], Any, Any]]:
    # Asynchronously compute deadhead cost
    deadhead_costs: List[
        Coroutine[Awaitable[Dict[str, Tuple[float, float]]], Any, Any]
    ] = []
    for row in df.itertuples():
        # Make mypy happy
        assert (
            isinstance(row.start_station_coord, tuple)
            and len(row.start_station_coord) == 2
            and all([isinstance(x, float) for x in row.start_station_coord])
        )
        assert (
            isinstance(row.end_station_coord, tuple)
            and len(row.end_station_coord) == 2
            and all([isinstance(x, float) for x in row.end_station_coord])
        )
        assert (
            isinstance(row.depot_station, tuple)
            and len(row.depot_station) == 2
            and all([isinstance(x, float) for x in row.depot_station])
        )

        cost_promise = deadhead_cost(
            row.start_station_coord, row.end_station_coord, row.depot_station, client
        )
        deadhead_costs.append(cost_promise)

    # Now the list is filled with promises/coroutines. We need to await them
    deadhead_costs = await asyncio.gather(*deadhead_costs)
    return deadhead_costs


def get_depot_rot_assign(
    session: sqlalchemy.orm.session.Session, scenario_id: int
) -> pd.DataFrame:
    data = []

    rotation_ids = (
        session.query(Rotation.id).filter(Rotation.scenario_id == scenario_id).all()
    )
    for rid in rotation_ids:
        trips = (
            session.query(Trip.id)
            .filter(Trip.rotation_id == rid[0])
            .order_by(Trip.departure_time)
            .all()
        )

        depot_station_id = (
            session.query(Station.id)
            .join(Route, Station.id == Route.departure_station_id)
            .join(Trip, Trip.route_id == Route.id)
            .filter(Trip.id == trips[0][0])
            .one()
        )

        data.append([rid[0], depot_station_id[0]])

    return pd.DataFrame(data, columns=["rotation_id", "orig_depot_station"])


def get_rotation(session: Session, scenario_id: int) -> pd.DataFrame:
    """
    This function takes a :class:'sqlalchemy.orm.Session' object and scenario_id and returns the rotation data in a
    :class:'pandas.DataFrame' object

    :param session: a :class:'sqlalchemy.orm.Session' object connected to the database
    :param scenario_id: The scenario id of the current scenario
    :return: a :class:'pandas.DataFrame' object with the rotation data, which includes the following columns:
        - rotation_id: The id of the rotation
        - start_station_coord: The coordinates of the first non-depot station of the rotation
        - end_station_coord: The coordinates of the last non-depot station of the rotation
        - vehicle_type_id: The id of the vehicle type of the rotation
    """
    # get non depot start and end station for each rotation

    rot_info_for_df = []

    rotations = session.scalars(
        session.query(Rotation).filter(Rotation.scenario_id == scenario_id)
    ).all()
    for rotation in rotations:
        trips = (
            session.query(Trip.id)
            .filter(Trip.rotation_id == rotation.id)
            .order_by(Trip.departure_time)
            .all()
        )

        # Find the first and last non-depot station for each rotation
        start_station = (
            session.query(Station.geom)
            .join(Route, Station.id == Route.arrival_station_id)
            .join(Trip, Trip.route_id == Route.id)
            .filter(Trip.id == trips[0][0])
            .one()[0]
        )

        end_station = (
            session.query(Station.geom)
            .join(Route, Station.id == Route.departure_station_id)
            .join(Trip, Trip.route_id == Route.id)
            .filter(Trip.id == trips[-1][0])
            .one()[0]
        )
        start_station_point = to_shape(start_station)
        end_station_point = to_shape(end_station)

        rot_info_for_df.append(
            [
                rotation.id,
                (start_station_point.x, start_station_point.y),
                (end_station_point.x, end_station_point.y),
            ]
        )

    rotation_df = pd.DataFrame(
        rot_info_for_df,
        columns=[
            "rotation_id",
            "start_station_coord",
            "end_station_coord",
        ],
    )
    return rotation_df


def depot_data(
    session: sqlalchemy.orm.session.Session, scenario_id: int
) -> pd.DataFrame:
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


def get_vehicletype(
    session: sqlalchemy.orm.session.Session,
    scenario_id: int,
    standard_bus_length: float = 12.0,
) -> pd.DataFrame:
    """
    This function takes the session and scenario_id and returns the vehicle types and there size factors compared to a
    normal 12-meter bus
    :param standard_bus_length: The capacity of the depot is calculated based on the size of a standard bus. Default is 12.0.
    :param session: A :class:'sqlalchemy.orm.Session' object connected to the database
    :param scenario_id: The scenario id of the current scenario
    :return: A :class:'pandas.DataFrame' object including the following columns:
        - vehicle_type_id: The id of the vehicle type
        - size_factor: The size factor of the vehicle type compared to a standard 12-meter bus
    """

    distinct_vehicle_type_ids = (
        session.query(Rotation.vehicle_type_id)
        .distinct(Rotation.vehicle_type_id)
        .filter(Rotation.scenario_id == scenario_id)
        .all()
    )
    distinct_vehicle_type_ids = [vid[0] for vid in distinct_vehicle_type_ids]

    vehicle_types = (
        session.query(VehicleType)
        .filter(VehicleType.id.in_(distinct_vehicle_type_ids))
        .all()
    )
    vehicle_types_size = [
        v.length / standard_bus_length if (v.length is not None) else 1.0
        for v in vehicle_types
    ]

    vt_df = pd.DataFrame()
    vt_df["vehicle_type_id"] = distinct_vehicle_type_ids
    vt_df["size_factor"] = vehicle_types_size
    return vt_df


def get_rotation_vehicle_assign(
    session: sqlalchemy.orm.session.Session, scenario_id: int
) -> pd.DataFrame:
    """

    :param session:
    :param scenario_id:
    :return:
    """

    rotations = (
        session.query(Rotation.id, Rotation.vehicle_type_id)
        .filter(Rotation.scenario_id == scenario_id)
        .all()
    )
    vehicle_types = (
        session.query(VehicleType.id)
        .filter(VehicleType.scenario_id == scenario_id)
        .all()
    )

    assignment = []

    for rotation in rotations:
        for vehicle_type in vehicle_types:
            r_vid = rotation[1]
            assignment.append(
                [rotation[0], vehicle_type[0], 1 if r_vid == vehicle_type[0] else 0]
            )

    return pd.DataFrame(
        assignment, columns=["rotation_id", "vehicle_type_id", "assignment"]
    )


def get_occupancy(
    session: Session,
    scenario_id: int,
) -> pd.DataFrame:
    """
    Evaluate the occupancy over time of all the rotations with the time resolution given in :param time_window:. This
    helps to evaluate the minimum fleet size for serving the amount of rotations.

    :param session: a :class:'sqlalchemy.orm.Session' object connected to the database
    :param scenario_id: The scenario id of the current scenario


    :return: a :class:'pandas.DataFrame' object with the occupancy data, which includes the following columns: -
    rotation_id: The id of the rotation - time slots in the unit of seconds after the simulation start: The time
    stamp of the occupancy, with 1 meaning this time slot is occupied and 0 vice versa.

    """

    # Get the shortest rotation duration as time window

    rotations = (
        session.query(Rotation).filter(Rotation.scenario_id == scenario_id).all()
    )
    rotation_ids = [r.id for r in rotations]

    min_duration = 0
    for rotation in rotations:
        first_trip = rotation.trips[0]
        last_trip = rotation.trips[-1]
        duration = last_trip.arrival_time - first_trip.departure_time
        if min_duration == 0 or duration.seconds < min_duration:
            min_duration = duration.seconds

    time_window = min_duration

    start_and_end_time = (
        session.query(func.min(Trip.departure_time), func.max(Trip.arrival_time))
        .filter(Trip.scenario_id == scenario_id, Trip.rotation_id.in_(rotation_ids))
        .one()
    )
    start_time = start_and_end_time[0].timestamp()
    end_time = start_and_end_time[1].timestamp()

    sampled_time_stamp = np.arange(start_time, end_time, time_window, dtype=int)
    occupancy = np.zeros((len(rotations), len(sampled_time_stamp)), dtype=int)

    for idx, rotation_id in enumerate(rotation_ids):
        rotation_start = (
            session.query(
                func.min(Trip.departure_time).filter(Trip.rotation_id == rotation_id)
            )
            .one()[0]
            .timestamp()
        )
        rotation_end = (
            session.query(
                func.max(Trip.arrival_time).filter(Trip.rotation_id == rotation_id)
            )
            .one()[0]
            .timestamp()
        )

        occupancy[idx] = np.interp(
            sampled_time_stamp,
            [rotation_start, rotation_end],
            [1, 1],
            left=0,
            right=0,
        )

        assert sum(occupancy[idx]) != 0, (
            f"Rotation {rotation_id} has no occupancy over this time period. Please "
            f"check if the time window is too large."
        )

    occupancy_df = pd.DataFrame(
        occupancy, columns=sampled_time_stamp, index=rotation_ids
    )
    return occupancy_df
