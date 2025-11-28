import logging
import warnings
import datetime
from enum import Enum
from typing import Dict, Tuple, Optional, Callable

import pandas as pd
import matplotlib.pyplot as plt

import pyomo.environ as pyo  # type: ignore
from eflips.model import (
    VehicleType,
    BatteryType,
    Station,
    ChargingPointType,
    Area,
    Trip,
    Vehicle,
    Rotation,
    Event,
    EventType,
    Route,
    Depot,
)
from sqlalchemy import func, and_, or_, distinct
from eflips.opt.transition_planning.util import npv_with_escalation

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


from typing import Any, Dict, List, Tuple
from sqlalchemy.orm import Session


class ParameterRegistry:
    def __init__(self, session: Session, scenario: Any):
        """
        Shared context/data source wrapper for setting up MILP models.
        Initializes project duration, vehicle parameters, station assignments, block metrics, and all NPV-related values.


        """
        self.session = session
        self.scenario = scenario
        self.project_duration = scenario.tco_parameters.get("project_duration")
        self.annual_budget_limit = scenario.tco_parameters.get("annual_budget_limit")

        self.vehicles = self._fetch_vehicles()
        self.vehicle_indices = [v.id for v in self.vehicles]

        self.vehicle_type_indices, self.vehicle_types = self._fetch_vehicle_types()

        self.vehicle_electricity_consumption, self.vehicle_diesel_consumption = (
            self._fetch_vehicle_type_operational_params()
        )

        self.station_indices, self.station_vehicle_assignments = (
            self._fetch_station_assignments()
        )
        self.block_indices, self.block_vehicle_assignments = (
            self._fetch_block_assignments()
        )
        self.vehicle_type_assignments = {
            (v.vehicle_type_id, v.id): 1 for v in self.vehicles
        }
        self.block_vehicle_type_assignments = (
            self._fetch_block_vehicle_type_assignments()
        )

        self.vehicle_driving_times = self._fetch_vehicle_driving_times()
        self.block_mileage = self._fetch_block_mileage()
        self.block_durations = self._fetch_block_durations()
        self.block_cost = self._fetch_block_cost()

        self.station_occupancy = self._fetch_station_occupancy()

        self.average_mileage_vehicle_type = (
            self._calculate_vehicle_type_average_mileage()
        )
        self.time_scaling_factor_to_year = self._calculate_time_scaling_factor()

        self._initialize_npv_parameters()

    def _fetch_vehicles(self) -> List[Any]:
        return (
            self.session.query(Vehicle)
            .filter(Vehicle.scenario_id == self.scenario.id)
            .order_by(Vehicle.id)
            .all()
        )

    def _fetch_vehicle_types(self) -> Tuple[List[int], List[Any]]:
        vehicle_types = (
            self.session.query(VehicleType)
            .filter(VehicleType.scenario_id == self.scenario.id)
            .all()
        )
        vehicle_type_indices = [vt.id for vt in vehicle_types]
        return vehicle_type_indices, vehicle_types

    def _fetch_vehicle_type_operational_params(
        self,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        electricity_consumption = {}
        diesel_consumption = {}
        for vt in self.vehicle_types:
            electricity_consumption[vt.id] = vt.tco_parameters.get(
                "average_electricity_consumption"
            )
            diesel_consumption[vt.id] = vt.tco_parameters.get(
                "average_diesel_consumption"
            )
        return electricity_consumption, diesel_consumption

    def _fetch_station_assignments(
        self,
    ) -> Tuple[List[int], Dict[Tuple[int, int], int]]:
        assignments = {}
        station_ids = set()
        for v in self.vehicles:
            stations = (
                self.session.query(Event.station_id)
                .filter(
                    Event.vehicle_id == v.id,
                    Event.event_type == EventType.CHARGING_OPPORTUNITY,
                )
                .distinct()
                .all()
            )
            for st in stations:
                assignments[(st[0], v.id)] = 1
                station_ids.add(st[0])
        return sorted(station_ids), assignments

    def _fetch_block_assignments(self) -> Tuple[List[int], Dict[Tuple[int, int], int]]:
        assignments = {}
        block_ids = set()
        for v in self.vehicles:
            for rot in v.rotations:
                assignments[(rot.id, v.id)] = 1
                block_ids.add(rot.id)
        return sorted(block_ids), assignments

    def _fetch_block_vehicle_type_assignments(self) -> Dict[Tuple[int, int], int]:
        assignments = {}
        block_ids = set()
        for v in self.vehicles:
            for rot in v.rotations:
                assignments[(rot.id, v.vehicle_type_id)] = 1
                block_ids.add(rot.id)
        return assignments

    def _fetch_vehicle_driving_times(self) -> Dict[int, float]:
        results = (
            self.session.query(
                func.sum(
                    func.extract("epoch", Event.time_end)
                    - func.extract("epoch", Event.time_start)
                ).label("driving_time"),
                Event.vehicle_id,
            )
            .filter(
                Event.scenario_id == self.scenario.id,
                or_(
                    Event.trip_id.isnot(None),
                    and_(Event.station_id.isnot(None), Event.area_id.is_(None)),
                ),
            )
            .group_by(Event.vehicle_id)
            .all()
        )
        return {dt.vehicle_id: float(dt.driving_time) / 3600.0 for dt in results}

    def _fetch_block_mileage(self) -> Dict[int, float]:
        results = (
            self.session.query(
                func.sum(Route.distance).label("total_mileage"), Rotation.id
            )
            .join(Trip, Trip.route_id == Route.id)
            .join(Rotation, Trip.rotation_id == Rotation.id)
            .filter(Route.scenario_id == self.scenario.id)
            .group_by(Rotation.id)
            .all()
        )
        return {rotation_id: mileage / 1000.0 for mileage, rotation_id in results}

    def _fetch_block_durations(self) -> Dict[int, float]:
        results = (
            self.session.query(
                func.sum(
                    func.extract("epoch", Trip.arrival_time)
                    - func.extract("epoch", Trip.departure_time)
                ).label("total_time"),
                Rotation.id,
            )
            .join(Rotation, Trip.rotation_id == Rotation.id)
            .filter(Trip.scenario_id == self.scenario.id)
            .group_by(Rotation.id)
            .all()
        )
        return {
            rotation_id: float(block_time) / 3600.0
            for block_time, rotation_id in results
        }

    def _fetch_block_cost(self) -> Dict[Tuple[int, int], float]:

        # getting blocks

        blocks = (
            self.session.query(Rotation)
            .filter(Rotation.scenario_id == self.scenario.id)
            .order_by(Rotation.id)
            .all()
        )

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
                        and bj.trips[0].route.departure_station_id
                        == bi.trips[0].route.departure_station_id
                    ):
                        block_cost[(bi.id, bj.id)] = (
                            bj.trips[0].departure_time - bi.trips[-1].arrival_time
                        ).total_seconds() / 3600.0

        return block_cost

    def _fetch_station_occupancy(self) -> Dict[int, int]:
        from eflips.eval.output.prepare import power_and_occupancy

        occupancy = {}
        for station_id in self.station_indices:
            occupancy_df = power_and_occupancy(
                area_id=None,
                station_id=station_id,
                session=self.session,
            )["occupancy_charging"]
            occupancy[station_id] = int(occupancy_df.max())
        return occupancy

    def _calculate_vehicle_type_average_mileage(self) -> Dict[int, float]:

        # TODO redo it in diesel scenario. Now it is actually ebus scenario mileage per vehicle type
        vt_mileage = {}
        vt_count_mileage = (
            self.session.query(
                func.sum(Route.distance),
                Rotation.vehicle_type_id,
                func.count(distinct(Vehicle.id)),
            )
            .join(Trip, Trip.route_id == Route.id)
            .join(Rotation, Trip.rotation_id == Rotation.id)
            .join(Vehicle, Rotation.vehicle_id == Vehicle.id)
            .filter(Route.scenario_id == self.scenario.id)
            .group_by(Rotation.vehicle_type_id)
            .all()
        )

        for total_mileage, vt_id, vehicle_count in vt_count_mileage:
            if vehicle_count > 0:
                vt_mileage[vt_id] = (total_mileage / 1000.0) / vehicle_count
            else:
                vt_mileage[vt_id] = 0.0
        return vt_mileage

    def _calculate_time_scaling_factor(self) -> float:
        all_trips = (
            self.session.query(Trip)
            .filter(Trip.scenario_id == self.scenario.id)
            .order_by(Trip.departure_time)
            .all()
        )
        elapsed_time = all_trips[-1].departure_time - all_trips[0].departure_time
        return (365 * 24 * 3600) / elapsed_time.total_seconds()

    def _initialize_npv_parameters(self) -> None:
        self.npv_electric_vehicle = {}
        self.useful_life_electric_vehicle = {}
        self.npv_battery = {}
        self.useful_life_battery = {}
        self.npv_station_with_chargers = {}
        self.npv_depot_charger = {}
        self.npv_electricity = {}
        self.npv_diesel = {}
        self.npv_maintenance_dv = {}
        self.npv_maintenance_ev = {}
        self.npv_staff = {}
        self.npv_maintenance_infra = {}
        self.npv_diesel_bus = {}
        for year in range(1, self.project_duration + 1):
            self._yearly_npv_entries(year)

    def _yearly_npv_entries(self, year: int) -> None:
        # Vehicle types and batteries
        for vt in self.vehicle_types:
            self.npv_electric_vehicle[(vt.id, year)] = npv_with_escalation(
                vt.tco_parameters["procurement_cost"],
                vt.tco_parameters["cost_escalation"],
                self.scenario.tco_parameters["inflation_rate"],
                year,
            )
            self.useful_life_electric_vehicle[vt.id] = vt.tco_parameters["useful_life"]
            self.npv_diesel_bus[(vt.id, year)] = npv_with_escalation(
                vt.tco_parameters["procurement_cost_diesel_equivalent"],
                vt.tco_parameters["cost_escalation_diesel_equivalent"],
                self.scenario.tco_parameters["inflation_rate"],
                year,
            )
            bt = (
                self.session.query(BatteryType)
                .filter(
                    BatteryType.id == vt.battery_type_id,
                    BatteryType.scenario_id == self.scenario.id,
                )
                .one()
            )
            self.useful_life_battery[vt.id] = bt.tco_parameters["useful_life"]
            self.npv_battery[(vt.id, year)] = npv_with_escalation(
                bt.tco_parameters["procurement_cost"] * vt.battery_capacity,
                bt.tco_parameters["cost_escalation"],
                self.scenario.tco_parameters["inflation_rate"],
                year,
            )
        for st_id in self.station_indices:
            station = self.session.query(Station).filter(Station.id == st_id).one()
            charger = station.charging_point_type

            self.npv_station_with_chargers[(st_id, year)] = npv_with_escalation(
                station.tco_parameters["procurement_cost"],
                station.tco_parameters["cost_escalation"],
                self.scenario.tco_parameters["inflation_rate"],
                year,
            ) + npv_with_escalation(
                charger.tco_parameters["procurement_cost"]
                * self.station_occupancy[station.id],
                charger.tco_parameters["cost_escalation"],
                self.scenario.tco_parameters["inflation_rate"],
                year,
            )

        # TODO for now both chargers are assumed to be of same type across all depots/stations.
        depot_charger_id = (
            self.session.query(Area.charging_point_type_id)
            .filter(
                Area.scenario_id == self.scenario.id,
                Area.charging_point_type_id.isnot(None),
            )
            .distinct()
            .one()[0]
        )
        depot_charger = (
            self.session.query(ChargingPointType)
            .filter(
                ChargingPointType.scenario_id == self.scenario.id,
                ChargingPointType.id == depot_charger_id,
            )
            .one()
        )

        self.useful_life_depot_charger = depot_charger.tco_parameters["useful_life"]

        station_charger_id = (
            self.session.query(Station.charging_point_type_id)
            .filter(
                Station.scenario_id == self.scenario.id,
                Station.charging_point_type_id.isnot(None),
            )
            .distinct()
            .one()[0]
        )

        station_charger = (
            self.session.query(ChargingPointType)
            .filter(
                ChargingPointType.scenario_id == self.scenario.id,
                ChargingPointType.id == station_charger_id,
            )
            .one()
        )

        self.useful_life_station_charger = station_charger.tco_parameters["useful_life"]

        self.npv_depot_charger[year] = npv_with_escalation(
            depot_charger.tco_parameters["procurement_cost"],
            depot_charger.tco_parameters["cost_escalation"],
            self.scenario.tco_parameters["inflation_rate"],
            year,
        )
        self.npv_electricity[year] = npv_with_escalation(
            self.scenario.tco_parameters["fuel_cost"]["electricity"],
            self.scenario.tco_parameters["cost_escalation_rate"]["electricity"],
            self.scenario.tco_parameters["inflation_rate"],
            year,
        )
        self.npv_diesel[year] = npv_with_escalation(
            self.scenario.tco_parameters["fuel_cost"]["diesel"],
            self.scenario.tco_parameters["cost_escalation_rate"]["diesel"],
            self.scenario.tco_parameters["inflation_rate"],
            year,
        )
        self.npv_maintenance_ev[year] = npv_with_escalation(
            self.scenario.tco_parameters["vehicle_maint_cost"]["electricity"],
            self.scenario.tco_parameters["cost_escalation_rate"]["general"],
            self.scenario.tco_parameters["inflation_rate"],
            year,
        )
        self.npv_maintenance_dv[year] = npv_with_escalation(
            self.scenario.tco_parameters["vehicle_maint_cost"]["diesel"],
            self.scenario.tco_parameters["cost_escalation_rate"]["general"],
            self.scenario.tco_parameters["inflation_rate"],
            year,
        )
        self.npv_staff[year] = npv_with_escalation(
            self.scenario.tco_parameters["staff_cost"],
            self.scenario.tco_parameters["cost_escalation_rate"]["staff"],
            self.scenario.tco_parameters["inflation_rate"],
            year,
        )
        self.npv_maintenance_infra[year] = npv_with_escalation(
            self.scenario.tco_parameters["infra_maint_cost"],
            self.scenario.tco_parameters["cost_escalation_rate"]["general"],
            self.scenario.tco_parameters["inflation_rate"],
            year,
        )


class ConstraintRegistry:
    """ """

    def __init__(self, params: ParameterRegistry):
        self.params = params
        self.constraints: Dict[str, Callable] = {}
        self.constraint_sets: Dict[str, Any] = {}

        self._register_constraints()
        self.large_M = 1e8

    def _register_constraints(self) -> None:
        def full_electrification_rule(m, v):
            return sum(m.X_vehicle_year[v, i] for i in m.I) == 1

        self.constraints["FullElectrificationConstraint"] = full_electrification_rule
        self.constraint_sets["FullElectrificationConstraint"] = ["V"]

        def no_duplicate_vehicle_electrification_rule(m, v):
            return sum(m.X_vehicle_year[v, i] for i in m.I) <= 1

        self.constraints["NoDuplicatedVehicleElectrificationConstraint"] = no_duplicate_vehicle_electrification_rule
        self.constraint_sets["NoDuplicatedVehicleElectrificationConstraint"] = ["V"]

        def initial_electric_vehicle_rule(m, v):

            return m.X_vehicle_year[v, 0] == 0

        self.constraints["InitialElectricVehicleConstraint"] = (
            initial_electric_vehicle_rule
        )
        self.constraint_sets["InitialElectricVehicleConstraint"] = ["V"]

        # TODO currently I assume no electrified stations in year 0. If there are, need to add parameter and change here.
        #  Possibly during some data containing "initial electric vehicles"

        # Actually adding non-zero initial states could be interesting
        def initial_electrified_station_rule(m, s):
            return m.Z_station_year[s, 0] == 0

        self.constraints["InitialElectrifiedStationConstraint"] = (
            initial_electrified_station_rule
        )
        self.constraint_sets["InitialElectrifiedStationConstraint"] = ["S"]

        def no_station_uninstallation_rule(m, s, i):
            if i == 0:
                return pyo.Constraint.Skip
            return m.Z_station_year[s, i] >= m.Z_station_year[s, i - 1]

        self.constraints["NoStationUninstallationConstraint"] = (
            no_station_uninstallation_rule
        )
        self.constraint_sets["NoStationUninstallationConstraint"] = ["S", "I"]

        # Station must be built before vehicle assignment
        def station_before_vehicle_rule(m, s, v, i):
            if self.params.station_vehicle_assignments.get((s, v)) == 0:
                return pyo.Constraint.Skip
            return m.Z_station_year[s, i] >= m.X_vehicle_year[v, i]

        self.constraints["StationBeforeVehicleConstraint"] = station_before_vehicle_rule
        self.constraint_sets["StationBeforeVehicleConstraint"] = ["S", "V", "I"]

        def no_early_station_building_rule(m, s, i):
            assigned_vehicles = [
                v
                for v in m.V
                if self.params.station_vehicle_assignments.get((s, v)) == 1
            ]
            if not assigned_vehicles:
                return pyo.Constraint.Skip
            return m.Z_station_year[s, i] <= sum(
                m.X_vehicle_year[v, years_before]
                for v in assigned_vehicles
                for years_before in m.I
                if years_before <= i
            )

        self.constraints["NoEarlyStationBuildingConstraint"] = (
            no_early_station_building_rule
        )
        self.constraint_sets["NoEarlyStationBuildingConstraint"] = ["S", "I"]

        def assignment_block_year(m, b, i):

            return m.Z_block_year[b, i] == sum(
                self.params.block_vehicle_assignments.get((b, v), 0)
                * sum(m.X_vehicle_year[v, i_t] for i_t in m.I if i_t <= i)
                for v in m.V
            )

        self.constraints["AssignmentBlockYearConstraint"] = assignment_block_year
        self.constraint_sets["AssignmentBlockYearConstraint"] = ["B", "I"]

        def budget_constraint_rule(m, i):
            annual_cost = (
                m.AnnualEbusProcurement[i]
                + m.AnnualBatteryProcurement[i]
                + m.AnnualStationWithChargerProcurement[i]
                + m.AnnualDepotChargerProcurement[i]
            )
            # Example budget limit, can be adjusted or made a parameter
            budget_limit = self.params.annual_budget_limit
            return annual_cost <= budget_limit

        self.constraints["BudgetConstraint"] = budget_constraint_rule
        self.constraint_sets["BudgetConstraint"] = ["I"]

        def block_schedule_one_path_rule(m, b, i):
            if b == 0:
                return pyo.Constraint.Skip
            return (
                sum(
                    m.U_diesel_block_schedule_year[b_t, b, i]
                    for b_t in m.B
                    if (b_t != b)
                )
                == 1 - m.Z_block_year[b, i]
            )

        self.constraints["BlockScheduleOnePathConstraint"] = (
            block_schedule_one_path_rule
        )
        self.constraint_sets["BlockScheduleOnePathConstraint"] = ["B", "I"]

        def block_schedule_flow_conservation_rule(m, b, i):
            if b == 0:
                return pyo.Constraint.Skip
            return sum(
                m.U_diesel_block_schedule_year[b_t, b, i] for b_t in m.B if (b_t != b)
            ) == sum(
                m.U_diesel_block_schedule_year[b, b_q, i] for b_q in m.B if (b != b_q)
            )

        self.constraints["BlockScheduleFlowConservationConstraint"] = (
            block_schedule_flow_conservation_rule
        )
        self.constraint_sets["BlockScheduleFlowConservationConstraint"] = ["B", "I"]


        def block_schedule_cost_constraint_rule(m, i):
            return (
                sum(
                    m.U_diesel_block_schedule_year[b_t, b_q, i]
                    * self.params.block_cost.get((b_t, b_q), self.large_M)
                    for b_t in m.B
                    for b_q in m.B
                    if b_t != b_q
                )
                <= self.large_M
            )

        self.constraints["BlockScheduleCostConstraint"] = (
            block_schedule_cost_constraint_rule
        )
        self.constraint_sets["BlockScheduleCostConstraint"] = ["I"]

        def no_scheduling_electrified_block_rule(m, b_t, b_q, i):

            if b_t == b_q or b_q == 0:
                return pyo.Constraint.Skip

            return (
                m.U_diesel_block_schedule_year[b_t, b_q, i]
                <= 1 - m.Z_block_year[b_q, i]
            )

        self.constraints["NoSchedulingElectrifiedBlockConstraint"] = (
            no_scheduling_electrified_block_rule
        )
        self.constraint_sets["NoSchedulingElectrifiedBlockConstraint"] = ["B", "B", "I"]

        def no_scheduling_electrified_block_2_rule(m, b_t, b_q, i):
            if b_t == b_q or b_t == 0:
                return pyo.Constraint.Skip

            return (
                m.U_diesel_block_schedule_year[b_t, b_q, i]
                <= 1 - m.Z_block_year[b_t, i]
            )

        self.constraints["NoSchedulingElectrifiedBlock2Constraint"] = (
            no_scheduling_electrified_block_2_rule
        )
        self.constraint_sets["NoSchedulingElectrifiedBlock2Constraint"] = [
            "B",
            "B",
            "I",
        ]


class ExpressionRegistry:
    def __init__(self, params: ParameterRegistry):
        self.params = params
        self.expressions: Dict[str, Callable] = {}
        self.expression_sets: Dict[str, Any] = {}

        self._register_expressions()

    def _register_expressions(self) -> None:

        def newly_built_station_rule(m, s, i):
            if i == 0:

                return m.Z_station_year[s, i]
            return m.Z_station_year[s, i] - m.Z_station_year[s, i - 1]

        self.expressions["NewlyBuiltStation"] = newly_built_station_rule
        self.expression_sets["NewlyBuiltStation"] = ["S", "I"]

        def annual_electric_bus_procurement_rule(m, i):

            return sum(
                self.params.npv_electric_vehicle.get((vt, i), 0)
                * sum(
                    self.params.vehicle_type_assignments.get((vt, v), 0)
                    * m.X_vehicle_year[v, i]
                    for v in m.V
                )
                for vt in m.VT
            )

        self.expressions["AnnualEbusProcurement"] = annual_electric_bus_procurement_rule
        self.expression_sets["AnnualEbusProcurement"] = ["I"]

        def annual_vehicle_replacement_cost(m, i):
            annual_vehicle_replacement = 0

            for vt in m.VT:
                useful_life = self.params.useful_life_electric_vehicle.get(vt)
                cycles = self.params.project_duration // useful_life

                for cycle in range(1, cycles + 1):

                    initial_procurement_year = i - cycle * useful_life
                    if initial_procurement_year < 0:
                        continue
                    else:

                        annual_vehicle_replacement += (
                            self.params.npv_electric_vehicle.get((vt, i), 0)
                            * sum(
                                self.params.vehicle_type_assignments.get((vt, v), 0)
                                * m.X_vehicle_year[v, initial_procurement_year]
                                for v in m.V
                            )
                        )

            return annual_vehicle_replacement

        self.expressions["AnnualVehicleReplacement"] = annual_vehicle_replacement_cost
        self.expression_sets["AnnualVehicleReplacement"] = ["I"]

        def electric_bus_depreciation_rule(m, i):

            total_vehicle_depreciation = 0

            for vt in m.VT:

                useful_life = self.params.useful_life_electric_vehicle.get(vt)
                cycles = self.params.project_duration // useful_life
                for i_t in m.I:
                    for cycle in range(0, cycles + 1):

                        replacement_year = i_t + cycle * useful_life
                        if replacement_year > self.params.project_duration:
                            continue
                        if 0 <= i - replacement_year < useful_life:
                            total_vehicle_depreciation += (
                                self.params.npv_electric_vehicle.get(
                                    (vt, replacement_year), 0
                                )
                                * sum(
                                    self.params.vehicle_type_assignments.get((vt, v), 0)
                                    * m.X_vehicle_year[v, i_t]
                                    for v in m.V
                                )
                                / useful_life
                            )
            return total_vehicle_depreciation

        self.expressions["ElectricBusDepreciation"] = electric_bus_depreciation_rule
        self.expression_sets["ElectricBusDepreciation"] = ["I"]

        def diesel_bus_depreciation_rule(m, i):
            # TODO this is equivalent to replacing 1/useful_life of the diesel fleet each year
            total_diesel_annuity = 0
            for vt in m.VT:

                # total_diesel_annuity += sum(
                #     m.U_diesel_block_schedule_year[0, b_q, i]
                #     * self.params.npv_diesel_bus.get((vt, i), 0)
                #     * self.params.block_vehicle_type_assignments.get((b_q, vt), 0)
                #     for b_q in m.B
                #     if (b_q != 0)
                # ) / self.params.useful_life_electric_vehicle.get(vt)

                # alternative
                total_diesel_annuity += sum(
                    (1 - sum(m.X_vehicle_year[v, i_t] for i_t in m.I if i_t <= i))
                    * self.params.vehicle_type_assignments.get((vt, v), 0)
                    * self.params.npv_diesel_bus.get((vt, i), 0)
                    for v in m.V
                ) / self.params.useful_life_electric_vehicle.get(vt)

            return total_diesel_annuity

        self.expressions["DieselBusDepreciation"] = diesel_bus_depreciation_rule
        self.expression_sets["DieselBusDepreciation"] = ["I"]

        def annual_battery_procurement_rule(m, i):
            return sum(
                self.params.npv_battery.get((vt, i), 0)
                * sum(
                    self.params.vehicle_type_assignments.get((vt, v), 0)
                    * m.X_vehicle_year[v, i]
                    for v in m.V
                )
                for vt in m.VT
            )

        self.expressions["AnnualBatteryProcurement"] = annual_battery_procurement_rule
        self.expression_sets["AnnualBatteryProcurement"] = ["I"]

        def annual_battery_replacement_cost(m, i):
            annual_battery_replacement = 0

            for vt in m.VT:
                useful_life = self.params.useful_life_battery.get(vt)
                cycles = self.params.project_duration // useful_life

                for cycle in range(1, cycles + 1):

                    initial_procurement_year = i - cycle * useful_life
                    if initial_procurement_year < 0:
                        continue
                    else:

                        annual_battery_replacement += self.params.npv_battery.get(
                            (vt, i), 0
                        ) * sum(
                            self.params.vehicle_type_assignments.get((vt, v), 0)
                            * m.X_vehicle_year[v, initial_procurement_year]
                            for v in m.V
                        )

            return annual_battery_replacement

        self.expressions["AnnualBatteryReplacement"] = annual_battery_replacement_cost
        self.expression_sets["AnnualBatteryReplacement"] = ["I"]

        def battery_depreciation_rule(m, i):
            total_battery_replacement = 0

            for vt in m.VT:

                useful_life = self.params.useful_life_battery.get(vt)
                cycles = self.params.project_duration // useful_life
                for i_t in m.I:
                    for cycle in range(0, cycles + 1):

                        replacement_year = i_t + cycle * useful_life
                        if replacement_year > self.params.project_duration:
                            continue
                        if 0 <= i - replacement_year < useful_life:
                            total_battery_replacement += (
                                self.params.npv_battery.get((vt, replacement_year), 0)
                                * sum(
                                    self.params.vehicle_type_assignments.get(
                                        (vt, v), 0
                                    )  # TODO this 0 means if vt is not assigned to v, but it is not clear from code
                                    * m.X_vehicle_year[v, i_t]
                                    for v in m.V
                                )
                                / useful_life
                            )
            return total_battery_replacement

        self.expressions["BatteryDepreciation"] = battery_depreciation_rule
        self.expression_sets["BatteryDepreciation"] = ["I"]

        def annual_station_with_charger_procurement_rule(m, i):

            return sum(
                self.params.npv_station_with_chargers.get((s, i), 0)
                * m.NewlyBuiltStation[s, i]
                for s in m.S
            )

        self.expressions["AnnualStationWithChargerProcurement"] = (
            annual_station_with_charger_procurement_rule
        )
        self.expression_sets["AnnualStationWithChargerProcurement"] = ["I"]

        def station_charger_depreciation_rule(m, i):

            useful_life = self.params.useful_life_depot_charger
            station_charger_depreciation = 0

            for i_t in m.I:
                if i_t <= i:
                    station_charger_depreciation += (
                        sum(
                            self.params.npv_station_with_chargers.get((s, i_t), 0)
                            * m.NewlyBuiltStation[s, i_t]
                            for s in m.S
                        )
                        / useful_life
                    )
            return station_charger_depreciation

        self.expressions["StationChargerDepreciation"] = (
            station_charger_depreciation_rule
        )
        self.expression_sets["StationChargerDepreciation"] = ["I"]

        def annual_depot_charger_procurement_rule(m, i):
            # TODO considering adding a rate for charger/vehicle ratio at depot
            return self.params.npv_depot_charger.get(i, 0) * sum(
                m.X_vehicle_year[v, i] for v in m.V
            )

        self.expressions["AnnualDepotChargerProcurement"] = (
            annual_depot_charger_procurement_rule
        )
        self.expression_sets["AnnualDepotChargerProcurement"] = ["I"]

        def depot_charger_depreciation_rule(m, i):
            useful_life = self.params.useful_life_station_charger
            depot_charger_depreciation = 0

            for i_t in m.I:
                if i_t <= i:
                    depot_charger_depreciation += (
                        self.params.npv_depot_charger.get(i_t, 0)
                        * sum(m.X_vehicle_year[v, i_t] for v in m.V)
                        / useful_life
                    )
            return depot_charger_depreciation

        self.expressions["DepotChargerDepreciation"] = depot_charger_depreciation_rule
        self.expression_sets["DepotChargerDepreciation"] = ["I"]

        def electricity_cost_rule(m, i):
            return (
                sum(
                    self.params.npv_electricity.get(i, 0)
                    * self.params.block_mileage.get(b)
                    * self.params.vehicle_electricity_consumption.get(vt)
                    * m.Z_block_year[b, i]
                    for v in m.V
                    for vt in m.VT
                    for b in m.B
                    if self.params.block_vehicle_assignments.get((b, v), 0) == 1
                    and self.params.vehicle_type_assignments.get((vt, v), 0) == 1
                )
                * self.params.time_scaling_factor_to_year
            )

        self.expressions["ElectricityCost"] = electricity_cost_rule
        self.expression_sets["ElectricityCost"] = ["I"]

        def diesel_cost_rule(m, i):
            return (
                sum(
                    self.params.npv_diesel.get(i, 0)
                    * self.params.block_mileage.get(b)
                    * self.params.vehicle_diesel_consumption.get(vt)
                    * (1 - m.Z_block_year[b, i])
                    for v in m.V
                    for vt in m.VT
                    for b in m.B
                    if self.params.block_vehicle_assignments.get((b, v), 0) == 1
                    and self.params.vehicle_type_assignments.get((vt, v), 0) == 1
                )
                * self.params.time_scaling_factor_to_year
            )

        self.expressions["DieselCost"] = diesel_cost_rule
        self.expression_sets["DieselCost"] = ["I"]

        def maintenance_diesel_cost_rule(m, i):
            return (
                sum(
                    self.params.npv_maintenance_dv.get(i, 0)
                    * self.params.block_mileage.get(b)
                    * (1 - m.Z_block_year[b, i])
                    for v in m.V
                    for b in m.B
                    if self.params.block_vehicle_assignments.get((b, v), 0) == 1
                )
                * self.params.time_scaling_factor_to_year
            )

        self.expressions["MaintenanceDieselCost"] = maintenance_diesel_cost_rule
        self.expression_sets["MaintenanceDieselCost"] = ["I"]

        def maintenance_electric_cost_rule(m, i):
            return (
                sum(
                    self.params.npv_maintenance_ev.get(i, 0)
                    * self.params.block_mileage.get(b)
                    * m.Z_block_year[b, i]
                    for v in m.V
                    for b in m.B
                    if self.params.block_vehicle_assignments.get((b, v), 0) == 1
                )
                * self.params.time_scaling_factor_to_year
            )

        self.expressions["MaintenanceElectricCost"] = maintenance_electric_cost_rule
        self.expression_sets["MaintenanceElectricCost"] = ["I"]

        def staff_cost_ebus_rule(m, i):
            return (
                sum(
                    self.params.vehicle_driving_times.get(v)
                    * sum(m.X_vehicle_year[v, i_t] for i_t in m.I if i_t <= i)
                    for v in m.V
                )
                * self.params.time_scaling_factor_to_year
                * self.params.npv_staff.get(i, 0)
            )

        self.expressions["StaffCostEbus"] = staff_cost_ebus_rule
        self.expression_sets["StaffCostEbus"] = ["I"]

        def staff_cost_diesel_rule(m, i):
            return (
                sum(
                    self.params.block_durations.get(b, 0) * (1 - m.Z_block_year[b, i])
                    for b in m.B
                )
                * self.params.time_scaling_factor_to_year
                * self.params.npv_staff.get(i, 0)
            )

        self.expressions["StaffCostDiesel"] = staff_cost_diesel_rule
        self.expression_sets["StaffCostDiesel"] = ["I"]

        def maintenance_infra_cost_rule(m, i):

            num_station_chargers = sum(
                self.params.station_occupancy.get(s) * m.Z_station_year[s, i]
                for s in m.S
            )

            num_depot_chargers = sum(
                sum(m.X_vehicle_year[v, i_t] for i_t in m.I if i_t <= i) for v in m.V
            )

            return self.params.npv_maintenance_infra.get(i, 0) * (
                num_station_chargers + num_depot_chargers
            )

        self.expressions["MaintenanceInfraCost"] = maintenance_infra_cost_rule
        self.expression_sets["MaintenanceInfraCost"] = ["I"]

        def diesel_mileage_rule(m, i, vt):
            return sum(
                self.params.block_mileage.get(b) * (1 - m.Z_block_year[b, i])
                for v in m.V
                for b in m.B
                if self.params.block_vehicle_assignments.get((b, v), 0) == 1
                and self.params.vehicle_type_assignments.get((vt, v), 0) == 1
            )

        self.expressions["DieselMileage"] = diesel_mileage_rule
        self.expression_sets["DieselMileage"] = ["I", "VT"]


class TransitionPlannerModel:
    def __init__(
        self,
        params: ParameterRegistry,
        constraint_registry: ConstraintRegistry,
        expression_registry: ExpressionRegistry,
        constraints: List[str],
        expressions: List[str],
        objective_components: List[str],
        objective_sense: str = "minimize",
        name: str = "Transition_Planning_Model",
    ):
        self.params = params
        self.constraint_registry = constraint_registry
        self.expression_registry = expression_registry
        self.model = pyo.ConcreteModel(name=name)

        self.expressions = expressions
        self.constraints = constraints
        self.objective_components = objective_components
        self.objective_sense = objective_sense

        self._define_sets_and_variables()
        self._register_expressions()
        self._register_constraints()
        self._build_objective()

    def solve(self, solver="gurobi"):
        """

        :return:
        """

        solver = pyo.SolverFactory(solver)

        # TODO expose those?

        solver.options["Threads"] = 4  # fewer threads = less memory per thread
        solver.options["NodefileStart"] = (
            0.5  # start writing nodes to disk at 0.5 GB usage
        )
        solver.options["NodefileDir"] = "/tmp"  # or another fast local disk
        solver.options["Presolve"] = 2
        solver.options["ConcurrentMIP"] = 1

        solver.options.update(
            {
                "Presolve": 2,
                "Cuts": 2,
                "Heuristics": 0.2,
            }
        )

        result = solver.solve(
            self.model, tee=True, keepfiles=False, symbolic_solver_labels=False
        )

        if result.solver.termination_condition == pyo.TerminationCondition.infeasible:
            raise ValueError(
                "No feasible solution found. Please check your constraints."
            )

        logger.info("Optimization complete.")

        self.result = result

    def visualize(self, optional_visualization_targets: Optional[List[str]] = None):

        vehicle_assignment = pd.DataFrame(
            [
                {
                    "vehicle_id": v,
                    "year": i,
                    "is_electric": pyo.value(self.model.X_vehicle_year[v, i]),
                }
                for v in self.model.V
                for i in self.model.I if i > 0
            ]
        )

        yearly_vehicle_assignment = vehicle_assignment.groupby("year").sum()[
            "is_electric"
        ]

        yearly_vehicle_assignment.plot(
            kind="bar",
            stacked=True,
            figsize=(12, 8),
        )

        plt.title("Yearly Vehicle Electrification Assignment")
        plt.ylabel("Number of Vehicles Electrified")
        plt.xlabel("Year")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(self.model.name + "_yearly_vehicle_electrification.png")
        plt.show()

        # plot cumulative vehicle type

        vehicle_type_per_year = pd.DataFrame(
            [
                {
                    "year": i,
                    "vehicle_type": vt,
                    "num_vehicles": sum(
                        pyo.value(self.model.X_vehicle_year[v, j])
                        * self.params.vehicle_type_assignments.get((vt, v), 0)
                        for v in self.model.V
                        for j in self.model.I
                        if j <= i
                    ),
                    # "num_diesel_vehicles": sum(
                    #     pyo.value(self.model.U_diesel_block_schedule_year[0, b_q, i])
                    #     * self.params.block_vehicle_type_assignments.get((b_q, vt), 0)
                    #     for b_q in self.model.B
                    #     if (b_q != 0)
                    # ),

                    "num_diesel_vehicles": sum(
                        (1 - sum(pyo.value(self.model.X_vehicle_year[v, j]) for j in self.model.I if j <= i))
                        * self.params.vehicle_type_assignments.get((vt, v), 0)
                        for v in self.model.V
                    ),
                }
                for vt in self.model.VT
                for i in self.model.I if i > 0
            ]
        )

        # plot num_vehicles and num_diesel_vehicles side by side for each vehicle type and year

        vehicle_type_pivot = vehicle_type_per_year.pivot(
            index="year",
            columns="vehicle_type",
            values=["num_vehicles", "num_diesel_vehicles"],
        )

        vehicle_type_pivot.plot(
            kind="bar",
            stacked=True,
            figsize=(12, 8),
        )
        plt.title("Cumulative Vehicle Type Electrification Over Years")
        plt.ylabel("Number of Vehicles Electrified")
        plt.xlabel("Year")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(self.model.name + "_cumulative_vehicle_type_electrification.png")
        plt.show()

        dict_cost_breakdown = {
            str(exp_name): [
                pyo.value(getattr(self.model, exp_name)[i]) for i in self.model.I if i > 0
            ]
            for exp_name in self.objective_components
        }

        yearly_cost_breakdown = pd.DataFrame(
            dict_cost_breakdown, index=[i for i in self.model.I if i > 0]
        )

        yearly_cost_breakdown.plot(
            kind="bar", stacked=True, figsize=(12, 8), colormap="tab20"
        )

        plt.title("Yearly Cost Breakdown")
        plt.ylabel("Cost")
        plt.xlabel("Year")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        plt.savefig(self.model.name + "_yearly_cost_breakdown.png")
        plt.show()
        # save dataframes to csv
        yearly_vehicle_assignment.to_csv(
            self.model.name + "_vehicle_assignment.csv", index=False
        )
        yearly_cost_breakdown.to_csv(self.model.name + "_yearly_cost.csv", index=False)

        if optional_visualization_targets is not None:
            dict_optional_cost_breakdown = {
                str(exp_name): [
                    pyo.value(getattr(self.model, exp_name)[i]) for i in self.model.I if i > 0
                ]
                for exp_name in optional_visualization_targets
            }

            optional_cost_breakdown = pd.DataFrame(
                dict_optional_cost_breakdown, index=[i for i in self.model.I if i > 0]
            )

            optional_cost_breakdown.plot(
                kind="bar", stacked=True, figsize=(12, 8), colormap="tab20"
            )

            plt.title("Optional Cost Breakdown")
            plt.ylabel("Cost")
            plt.xlabel("Year")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(self.model.name + "_optional_cost_breakdown.png")
            plt.show()

    def visualize_diesel_baseline(self):

        raise NotImplementedError("To be implemented")

    def _define_sets_and_variables(self):
        model = self.model

        # sets
        # Vehicle
        model.V = pyo.Set(initialize=self.params.vehicle_indices, doc="Vehicle indices")
        # Vehicle Types
        model.VT = pyo.Set(
            initialize=self.params.vehicle_type_indices, doc="Vehicle type indices"
        )
        # Blocks
        model.B = pyo.Set(
            initialize=self.params.block_indices + [0], doc="Block indices"
        )
        # Stations
        model.S = pyo.Set(initialize=self.params.station_indices, doc="Station indices")
        # Years. Year 0 is the initial scenario. It is possible that already vehicles are electric in year 0.
        model.I = pyo.Set(
            initialize=list(range(0, self.params.project_duration + 1)),
            doc="Project years",
        )

        # Variables
        model.X_vehicle_year = pyo.Var(
            model.V, model.I, within=pyo.Binary, doc="Electric vehicle deployed in year"
        )

        # Auxiliary variable ? (If remove the upper constraint: each station cannot be built earlier than the first
        # acquisition of a vehicle assigned to this sand self.block_vehicle_assignments.get((b, v), 0) == 1tation, this will because "true" variable)
        model.Z_station_year = pyo.Var(
            model.S,
            model.I,
            within=pyo.Binary,
            doc="Station with chargers existing by the year",
        )

        model.Z_block_year = pyo.Var(
            model.B,
            model.I,
            within=pyo.Binary,
            doc="Block electrified by the year",
        )

        model.U_diesel_block_schedule_year = pyo.Var(
            model.B,
            model.B,
            model.I,
            within=pyo.Binary,
            doc="Block b scheduled after block b2 in year i",
        )

        # Fix some variables in block scheduling to reduce computational burden

        for b_t in model.B:
            for b_q in model.B:
                if (b_t, b_q) not in self.params.block_cost:
                    for i in model.I:
                        model.U_diesel_block_schedule_year[b_t, b_q, i].fix(0)

    def _register_constraints(self):

        for name in self.constraints:
            rule = self.constraint_registry.constraints[name]

            sets = [
                getattr(self.model, s)
                for s in self.constraint_registry.constraint_sets[name]
            ]
            setattr(
                self.model,
                name,
                pyo.Constraint(*sets, rule=rule),
            )

    def _register_expressions(self):
        for name in self.expressions:
            rule = self.expression_registry.expressions[name]
            sets = [
                getattr(self.model, s)
                for s in self.expression_registry.expression_sets[name]
            ]
            setattr(
                self.model,
                name,
                pyo.Expression(*sets, rule=rule),
            )

    def _build_objective(self):

        def objective_rule(m):
            return sum(
                sum(getattr(m, expr_name)[i] for expr_name in self.objective_components)
                for i in m.I if i > 0
            )

        self.model.TotalCostObjective = pyo.Objective(
            rule=objective_rule,
            sense=pyo.minimize if self.objective_sense == "minimize" else pyo.maximize,
        )

    def get_model(self):
        return self.model
