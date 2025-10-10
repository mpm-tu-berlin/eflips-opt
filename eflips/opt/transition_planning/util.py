from typing import List, Any
import pandas as pd
from eflips.model import (
    Scenario,
    VehicleType,
    Vehicle,
    Station,
    Trip,
    Rotation,
    Event,
    EventType,
    Route,
)
from sqlalchemy import func, and_, or_, desc
from sqlalchemy.orm import Session
from typing import Dict, Tuple

#


def vehicle_relevant_assignments(session: Session, scenario: Scenario):
    """

    Get vehicle relevant assignments for a given scenario.
    :param session:
    :param scenario:
    :return:
    """
    vehicles = (
        session.query(Vehicle)
        .filter(Vehicle.scenario_id == scenario.id)
        .order_by(Vehicle.id)
        .all()
    )
    vehicle_indices = [v.id for v in vehicles]
    block_indices = []

    block_vehicle_assignments: Dict[Tuple[int, int], int] = {}
    for vehicle in vehicles:
        for rotation in vehicle.rotations:
            block_indices.append(rotation.id)
            block_vehicle_assignments[(rotation.id, vehicle.id)] = 1

    station_vehicle_assignments: Dict[Tuple[int, int], int] = {}
    station_indices = []
    for vehicle in vehicles:
        charging_stations = (
            session.query(Event.station_id)
            .filter(
                Event.vehicle_id == vehicle.id,
                Event.event_type == EventType.CHARGING_OPPORTUNITY,
            )
            .distinct()
            .all()
        )
        for station in charging_stations:
            station_indices.append(station[0])
            station_vehicle_assignments[(station[0], vehicle.id)] = 1

    station_indices = list(set(station_indices))

    vehicle_type_assignments: Dict[Tuple[int, int], int] = {}
    for vehicle in vehicles:
        vehicle_type_assignments[(vehicle.vehicle_type_id, vehicle.id)] = 1

    vehicle_driving_times: Dict[int, float] = {}

    driving_time = (
        session.query(
            func.sum(
                func.extract("epoch", Event.time_end)
                - func.extract("epoch", Event.time_start)
            ).label("driving_time"),
            Event.vehicle_id,
        )
        .filter(
            Event.scenario_id == scenario.id,
            or_(
                Event.trip_id.isnot(None),
                and_(Event.station_id.isnot(None), Event.area_id.is_(None)),
            ),
        )
        .group_by(Event.vehicle_id)
        .order_by(Event.vehicle_id)
    ).all()

    for dt in driving_time:
        vehicle_driving_times[dt.vehicle_id] = (
            dt.driving_time / 3600.0
        )  # Convert seconds to hours

    return (
        block_vehicle_assignments,
        station_vehicle_assignments,
        vehicle_type_assignments,
        vehicle_driving_times,
        vehicle_indices,
        block_indices,
        station_indices,
    )


def block_info(session: Session, scenario: Scenario):

    block_mileages: Dict[int, float] = {}
    block_durations: Dict[int, float] = {}

    for mileage, rotation_id in (
        session.query(func.sum(Route.distance).label("total_mileage"), Rotation.id)
        .join(Trip, Trip.route_id == Route.id)
        .join(Rotation, Trip.rotation_id == Rotation.id)
        .filter(Route.scenario_id == scenario.id)
        .group_by(Rotation.id)
        .all()
    ):
        block_mileages[rotation_id] = mileage / 1000.0  # Convert meters to kilometers

    for block_time, rotation_id in (
        session.query(
            func.sum(
                func.extract("epoch", Trip.arrival_time)
                - func.extract("epoch", Trip.departure_time)
            ).label("total_time"),
            Rotation.id,
        )
        .join(Rotation, Trip.rotation_id == Rotation.id)
        .filter(
            Trip.scenario_id == scenario.id,
        )
        .group_by(Rotation.id)
        .all()
    ):
        block_durations[rotation_id] = block_time / 3600.0
    return block_mileages, block_durations


def npv_with_escalation(cost, escalation_rate, discount_rate, year):
    """
    Calculate the net present value (NPV) of a cost with escalation over a given number of years.
    :param cost: Initial cost
    :param escalation_rate: Annual escalation rate (as a decimal)
    :param discount_rate: Annual discount rate (as a decimal)
    :param year: Year for which to calculate the NPV
    :return: NPV of the cost in the given year
    """
    escalated_cost = cost * (1 + escalation_rate) ** year
    npv = escalated_cost / ((1 + discount_rate) ** year)
    return npv


def item_procurement_annuity(item: Any, scenario, project_duration):
    """
    Get vehicle procurement costs per year for a given scenario and project duration.
    :param session:
    :param scenario:
    :param project_duration:
    :return:
    """

    # TODO this is a general function to get procurement cost npv. Should be working for VehicleType,
    #  BatteryType (somehow with capacity), Station with charging points, Depot, Depot charger. Should decide calculated with annuity or not.

    pass
