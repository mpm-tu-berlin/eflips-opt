import logging
from enum import Enum
from typing import Dict, Tuple, Optional, Callable

import pandas as pd

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

        self.vehicle_driving_times = self._fetch_vehicle_driving_times()
        self.block_mileage = self._fetch_block_mileage()
        self.block_durations = self._fetch_block_durations()

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

        station_charger = self.session.query(ChargingPointType).filter(
            ChargingPointType.scenario_id == self.scenario.id,
            ChargingPointType.id == station_charger_id,
        ).one()

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

    def _register_constraints(self) -> None:
        def vehicle_electrification_rule(m, v):
            return sum(m.X_vehicle_year[v, i] for i in m.I) == 1

        self.constraints["VehicleElectrificationConstraint"] = (
            vehicle_electrification_rule
        )
        self.constraint_sets["VehicleElectrificationConstraint"] = ["V"]

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

        # Infrastructure
        def newly_built_station_rule(m, s, i):
            if i == 0:
                # TODO not sure if it is correct in case of initial stations
                return m.Y_station_year[s, i] == m.Z_station_year[s, i]
            return (
                m.Y_station_year[s, i]
                == m.Z_station_year[s, i] - m.Z_station_year[s, i - 1]
            )

        self.constraints["NewlyBuiltStationConstraint"] = newly_built_station_rule
        self.constraint_sets["NewlyBuiltStationConstraint"] = ["S", "I"]

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


class ExpressionRegistry:
    def __init__(self, params: ParameterRegistry):
        self.params = params
        self.expressions: Dict[str, Callable] = {}
        self.expression_sets: Dict[str, Any] = {}

        self._register_expressions()

    def _register_expressions(self) -> None:
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
                        if (
                                0 <= i - replacement_year < useful_life
                        ):
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

                if self.params.average_mileage_vehicle_type.get(vt, 0) == 0:
                    continue
                mileage = sum(
                    self.params.block_mileage.get(b) * (1 - m.Z_block_year[b, i])
                    for v in m.V
                    for b in m.B
                    if self.params.block_vehicle_assignments.get((b, v), 0) == 1
                    and self.params.vehicle_type_assignments.get((vt, v), 0) == 1
                )
                total_diesel_annuity += (
                    self.params.npv_diesel_bus.get((vt, i), 0)
                    * mileage
                    / (self.params.average_mileage_vehicle_type.get(vt) * self.params.useful_life_electric_vehicle.get(vt))
                )
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
                        if (
                                0 <= i - replacement_year < useful_life
                        ):
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

            # TODO making station with charger dependent on station. Now it's just average cost
            return sum(
                self.params.npv_station_with_chargers.get(i, 0) * m.Y_station_year[s, i]
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
                            self.params.npv_station_with_chargers.get(i_t, 0)
                            * m.Y_station_year[s, i_t]
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
                    self.params.block_durations.get(b) * (1 - m.Z_block_year[b, i])
                    for b in m.B
                )
                * self.params.time_scaling_factor_to_year
                * self.params.npv_staff.get(i, 0)
            )

        self.expressions["StaffCostDiesel"] = staff_cost_diesel_rule
        self.expression_sets["StaffCostDiesel"] = ["I"]

        def maintenance_infra_cost_rule(m, i):

            num_station_chargers = sum(
                self.params.station_occupancy.get(s)
                * m.Z_station_year[s, i] for s in m.S)


            num_depot_chargers = sum(
                    sum(m.X_vehicle_year[v, i_t] for i_t in m.I if i_t <= i)
                    for v in m.V
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
    def __init__(self, params: ParameterRegistry,
                 constraint_registry: ConstraintRegistry,
                 expression_registry: ExpressionRegistry,
                 constraints: List[str],
                 expressions: List[str],
                 objective_components: List[str],
                 objective_sense: str = "minimize",
                 name: str = "Transition_Planning_Model"):
        self.params = params
        self.constraint_registry = constraint_registry
        self.expression_registry = expression_registry
        self.model = pyo.ConcreteModel(name=name)

        self.constraints = constraints
        self.expressions = expressions
        self.objective_components = objective_components
        self.objective_sense = objective_sense

        self._define_sets_and_variables()
        self._register_constraints()
        self._register_expressions()
        self._build_objective()


    def solve(self, solver="gurobi"):

        """

        :return:
        """

        solver = pyo.SolverFactory(solver)

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
        model.B = pyo.Set(initialize=self.params.block_indices, doc="Block indices")
        # Stations
        model.S = pyo.Set(initialize=self.params.station_indices, doc="Station indices")
        # Years. Year 0 is the initial scenario. It is possible that already vehicles are electric in year 0.
        model.I = pyo.Set(
            initialize=list(range(0, self.params.project_duration + 1)), doc="Project years"
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

        # Auxiliary variable
        model.Y_station_year = pyo.Var(
            model.S,
            model.I,
            within=pyo.Binary,
            doc="Station with chargers built in the year",
        )

        model.Z_block_year = pyo.Var(
            model.B,
            model.I,
            within=pyo.Binary,
            doc="Block electrified by the year",
        )

    def _register_constraints(self):

        for name in self.constraints:
            rule = self.constraint_registry.constraints[name]

            sets = [getattr(self.model, s) for s in self.constraint_registry.constraint_sets[name]]
            setattr(
                self.model,
                name,
                pyo.Constraint(*sets, rule=rule),
            )

    def _register_expressions(self):
        for name in self.expressions:
            rule = self.expression_registry.expressions[name]
            sets = [getattr(self.model, s) for s in self.expression_registry.expression_sets[name]]
            setattr(
                self.model,
                name,
                pyo.Expression(*sets, rule=rule),
            )

    def _build_objective(self):

        def objective_rule(m):
            return sum(
                sum(getattr(m, expr_name)[i] for expr_name in self.objective_components)
                for i in m.I
            )

        self.model.TotalCostObjective = pyo.Objective(rule=objective_rule, sense=pyo.minimize if self.objective_sense == "minimize" else pyo.maximize)

    def get_model(self):
        return self.model



class TransitionPlanner:

    def __init__(
        self,
        session,
        scenario,
        project_duration,
        charger_per_station: Optional[int] = 3,
    ):
        self.session = session
        self.scenario = scenario
        self.project_duration = project_duration
        self.charger_per_station = charger_per_station

        self._prepare_data()
        self._economic_params_prep()
        logger.debug("Data preparation complete.")
        self._model_setup()

        from pyomo.util.model_size import build_model_size_report

        report = build_model_size_report(self.model)
        print(report)

        logger.debug("Model setup complete.")

    def _prepare_data(self):
        session, scenario = self.session, self.scenario
        vehicles = (
            session.query(Vehicle)
            .filter(Vehicle.scenario_id == scenario.id)
            .order_by(Vehicle.id)
            .all()
        )
        self.vehicle_indices = [v.id for v in vehicles]
        self.block_vehicle_assignments = {
            (rotation.id, vehicle.id): 1
            for vehicle in vehicles
            for rotation in vehicle.rotations
        }
        self.station_vehicle_assignments = {
            (station[0], vehicle.id): 1
            for vehicle in vehicles
            for station in session.query(Event.station_id)
            .filter(
                Event.vehicle_id == vehicle.id,
                Event.event_type == EventType.CHARGING_OPPORTUNITY,
            )
            .distinct()
            .all()
        }
        self.vehicle_type_assignments = {
            (vehicle.vehicle_type_id, vehicle.id): 1 for vehicle in vehicles
        }
        self.block_indices = list(
            {rotation.id for vehicle in vehicles for rotation in vehicle.rotations}
        )
        self.station_indices = list(
            {
                station_vehicle_pair[0]
                for station_vehicle_pair in self.station_vehicle_assignments
            }
        )
        self.vehicle_driving_times = {
            dt.vehicle_id: float(dt.driving_time) / 3600.0
            for dt in session.query(
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
            .all()
        }
        self.block_mileage = {
            rotation_id: mileage / 1000.0
            for mileage, rotation_id in session.query(
                func.sum(Route.distance).label("total_mileage"), Rotation.id
            )
            .join(Trip, Trip.route_id == Route.id)
            .join(Rotation, Trip.rotation_id == Rotation.id)
            .filter(Route.scenario_id == scenario.id)
            .group_by(Rotation.id)
            .all()
        }
        self.block_durations = {
            rotation_id: float(block_time) / 3600.0
            for block_time, rotation_id in session.query(
                func.sum(
                    func.extract("epoch", Trip.arrival_time)
                    - func.extract("epoch", Trip.departure_time)
                ).label("total_time"),
                Rotation.id,
            )
            .join(Rotation, Trip.rotation_id == Rotation.id)
            .filter(Trip.scenario_id == scenario.id)
            .group_by(Rotation.id)
            .all()
        }

    def _economic_params_prep(self):
        """
        Prepares economic parameters with escalation and inflation for each year of the project.
        """
        # Initialize NPV dictionaries
        self.npv_vehicle_types: Dict[Tuple[int, int], float] = {}
        self.useful_life_vehicle_types: Dict[int, int] = {}
        self.npv_batteries: Dict[Tuple[int, int], float] = {}
        self.useful_life_batteries: Dict[int, int] = {}
        self.npv_station_with_chargers: Dict[int, float] = {}
        self.depot_charger_npv: Dict[int, float] = {}
        self.electricity_price_npv: Dict[int, float] = {}
        self.diesel_price_npv: Dict[int, float] = {}
        self.maintenance_diesel_npv: Dict[int, float] = {}
        self.maintenance_electric_cost_npv: Dict[int, float] = {}
        self.staff_cost_npv: Dict[int, float] = {}
        self.charging_infra_maintenance_npv: Dict[int, float] = {}

        self.npv_annuity_diesel_bus: Dict[Tuple[int, int], float] = {}

        def npv_param_opex(param, escalation, year):
            return npv_with_escalation(
                self.scenario.tco_parameters[param],
                self.scenario.tco_parameters[escalation],
                self.scenario.tco_parameters["inflation_rate"],
                year,
            )

        for year in range(1, self.project_duration + 1):
            # Vehicle types and batteries
            vehicle_types = (
                self.session.query(VehicleType)
                .filter(VehicleType.scenario_id == self.scenario.id)
                .all()
            )
            self.vehicle_type_indices = [vt.id for vt in vehicle_types]
            for vt in vehicle_types:
                self.npv_vehicle_types[(vt.id, year)] = npv_with_escalation(
                    vt.tco_parameters["procurement_cost"],
                    vt.tco_parameters["cost_escalation"],
                    self.scenario.tco_parameters["inflation_rate"],
                    year,
                )
                self.useful_life_vehicle_types[vt.id] = vt.tco_parameters["useful_life"]
                bt = (
                    self.session.query(BatteryType)
                    .filter(
                        BatteryType.id == vt.battery_type_id,
                        BatteryType.scenario_id == self.scenario.id,
                    )
                    .one()
                )

                # TODO here I just use vehicle type indices for batteries to save from establishing a battery-vehicle type data.
                #  If battery types are far fewer than vehicle types, should change this.

                self.useful_life_batteries[vt.id] = bt.tco_parameters["useful_life"]
                self.npv_batteries[(vt.id, year)] = npv_with_escalation(
                    bt.tco_parameters["procurement_cost"] * vt.battery_capacity,
                    bt.tco_parameters["cost_escalation"],
                    self.scenario.tco_parameters["inflation_rate"],
                    year,
                )

                self.npv_annuity_diesel_bus[vt.id, year] = npv_with_escalation(
                    self.scenario.tco_parameters["npv_annuity_diesel_bus"][str(vt.id)],
                    0,
                    self.scenario.tco_parameters["inflation_rate"],
                    year,
                )

            # Station with chargers
            electrified_station = (
                self.session.query(Station)
                .filter(
                    Station.scenario_id == self.scenario.id,
                    Station.tco_parameters.isnot(None),
                    Station.is_electrified.is_(True),
                    ~self.session.query(Depot)
                    .filter(Depot.station_id == Station.id)
                    .exists(),
                )
                .first()
            )
            station_charger = (
                self.session.query(ChargingPointType)
                .filter(
                    ChargingPointType.scenario_id == self.scenario.id,
                    ChargingPointType.id == electrified_station.charging_point_type_id,
                )
                .first()
            )

            # TODO add number of chargers each station to this
            self.npv_station_with_chargers[year] = npv_with_escalation(
                electrified_station.tco_parameters["procurement_cost"],
                electrified_station.tco_parameters["cost_escalation"],
                self.scenario.tco_parameters["inflation_rate"],
                year,
            ) + npv_with_escalation(
                station_charger.tco_parameters["procurement_cost"]
                * self.charger_per_station,
                station_charger.tco_parameters["cost_escalation"],
                self.scenario.tco_parameters["inflation_rate"],
                year,
            )

            # Depot charger
            depot_charger_id = (
                self.session.query((Area.charging_point_type_id))
                .filter(
                    Area.scenario_id == self.scenario.id,
                    Area.charging_point_type_id.isnot(None),
                )
                .distinct()
                .all()[0][0]
            )
            depot_charger = (
                self.session.query(ChargingPointType)
                .filter(
                    ChargingPointType.scenario_id == self.scenario.id,
                    ChargingPointType.id == depot_charger_id,
                )
                .one()
            )
            self.depot_charger_npv[year] = npv_with_escalation(
                depot_charger.tco_parameters["procurement_cost"],
                depot_charger.tco_parameters["cost_escalation"],
                self.scenario.tco_parameters["inflation_rate"],
                year,
            )

            # Opex parameters
            self.electricity_price_npv[year] = npv_param_opex(
                "electricity_cost", "pef_electricity", year
            )
            self.diesel_price_npv[year] = npv_param_opex(
                "diesel_cost", "pef_diesel", year
            )
            self.maintenance_diesel_npv[year] = npv_param_opex(
                "maint_cost_diesel", "pef_general", year
            )
            self.maintenance_electric_cost_npv[year] = npv_param_opex(
                "maint_cost_electric", "pef_general", year
            )
            self.staff_cost_npv[year] = npv_param_opex("staff_cost", "pef_wages", year)
            self.charging_infra_maintenance_npv[year] = npv_param_opex(
                "maint_infr_cost", "pef_general", year
            )

        self.electricity_consumption_kwh_per_km = {
            int(k): v
            for k, v in self.scenario.tco_parameters[
                "const_electricity_consumption"
            ].items()
        }

        self.diesel_consumption_l_per_km = {
            int(k): v
            for k, v in self.scenario.tco_parameters["const_diesel_consumption"].items()
        }

        self.mean_mileage_total_schedule = {
            int(k): v
            for k, v in self.scenario.tco_parameters[
                "mean_mileage_total_schedule"
            ].items()
        }

        all_trips = (
            self.session.query(Trip)
            .filter(Trip.scenario_id == self.scenario.id)
            .order_by(Trip.departure_time)
            .all()
        )
        elapsed_time = all_trips[-1].departure_time - all_trips[0].departure_time
        self.time_scaling_factor_to_year = (
            365 * 24 * 3600
        ) / elapsed_time.total_seconds()

    def _model_setup(self):
        """

        :return:
        """

        logging.info("Setting up the optimization model.")
        model = pyo.ConcreteModel(name="fleet_transition_planning")

        # sets
        # Vehicle
        model.V = pyo.Set(initialize=self.vehicle_indices, doc="Vehicle indices")
        # Vehicle Types
        model.VT = pyo.Set(
            initialize=self.vehicle_type_indices, doc="Vehicle type indices"
        )
        # Blocks
        model.B = pyo.Set(initialize=self.block_indices, doc="Block indices")
        # Stations
        model.S = pyo.Set(initialize=self.station_indices, doc="Station indices")
        # Years. Year 0 is the initial scenario. It is possible that already vehicles are electric in year 0.
        model.I = pyo.Set(
            initialize=list(range(0, self.project_duration + 1)), doc="Project years"
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

        # Auxiliary variable
        model.Y_station_year = pyo.Var(
            model.S,
            model.I,
            within=pyo.Binary,
            doc="Station with chargers built in the year",
        )

        model.Z_block_year = pyo.Var(
            model.B,
            model.I,
            within=pyo.Binary,
            doc="Block electrified by the year",
        )

        # Auxiliary variable
        model.Z_vehicletype_block_year = pyo.Var(
            model.VT,
            model.B,
            model.I,
            within=pyo.Binary,
            doc="Block served by vehicle type by the year",
        )

        # Constraints
        # Vehicle acquisition
        def vehicle_electrification_rule(m, v):
            return sum(m.X_vehicle_year[v, i] for i in m.I) == 1

        model.VehicleElectrificationConstraint = pyo.Constraint(
            model.V,
            rule=vehicle_electrification_rule,
            doc="Each vehicle must be deployed and the project end.",
        )

        # Initial deployed electric vehicles
        # TODO currently I assume no electric vehicles in year 0. If there are, need to add parameter and change here.
        #  Possibly during some data containing "initial electric vehicles"
        def initial_electric_vehicle_rule(m, v):

            return m.X_vehicle_year[v, 0] == 0

        model.InitialElectricVehicleConstraint = pyo.Constraint(
            model.V,
            rule=initial_electric_vehicle_rule,
            doc="No initial electric vehicles in year 0.",
        )

        # TODO currently I assume no electrified stations in year 0. If there are, need to add parameter and change here.
        #  Possibly during some data containing "initial electric vehicles"

        # Actually adding non-zero initial states could be interesting
        def initial_electrified_station_rule(m, s):
            return m.Z_station_year[s, 0] == 0

        model.InitialElectrifiedStationConstraint = pyo.Constraint(
            model.S,
            rule=initial_electrified_station_rule,
            doc="No initial electrified stations in year 0.",
        )

        # Infrastructure
        def newly_built_station_rule(m, s, i):
            if i == 0:
                # TODO not sure if it is correct in case of initial stations
                return m.Y_station_year[s, i] == m.Z_station_year[s, i]
            return (
                m.Y_station_year[s, i]
                == m.Z_station_year[s, i] - m.Z_station_year[s, i - 1]
            )

        model.NewlyBuiltStationConstraint = pyo.Constraint(
            model.S,
            model.I,
            rule=newly_built_station_rule,
            doc="Newly built station in year i.",
        )

        def no_station_uninstallation_rule(m, s, i):
            if i == 0:
                return pyo.Constraint.Skip
            return m.Z_station_year[s, i] >= m.Z_station_year[s, i - 1]

        model.NoStationUninstallationConstraint = pyo.Constraint(
            model.S,
            model.I,
            rule=no_station_uninstallation_rule,
            doc="No station uninstallation.",
        )

        # Station must be built before vehicle assignment
        def station_before_vehicle_rule(m, s, v, i):
            if self.station_vehicle_assignments.get((s, v)) == 0:
                return pyo.Constraint.Skip
            return m.Z_station_year[s, i] >= m.X_vehicle_year[v, i]

        model.StationBeforeVehicleConstraint = pyo.Constraint(
            model.S,
            model.V,
            model.I,
            rule=station_before_vehicle_rule,
            doc="Station must be built before vehicle assignment.",
        )

        # Optional: no early station building
        # def no_early_station_building_rule(m, s, i):
        #     assigned_vehicles = [
        #         v for v in m.V if self.station_vehicle_assignments.get((s, v)) == 1
        #     ]
        #     if not assigned_vehicles:
        #         return pyo.Constraint.Skip
        #     return m.Z_station_year[s, i] <= sum(
        #         m.X_vehicle_year[v, years_before]
        #         for v in assigned_vehicles
        #         for years_before in m.I
        #         if years_before <= i
        #     )
        #
        # model.NoEarlyStationBuildingConstraint = pyo.Constraint(
        #     model.S,
        #     model.I,
        #     rule=no_early_station_building_rule,
        #     doc="No early station building before vehicle assignment.",
        # )

        # Expressions for building costs
        def annual_electric_bus_procurement_rule(m, i):

            return sum(
                self.npv_vehicle_types.get((vt, i), 0)
                * sum(
                    self.vehicle_type_assignments.get((vt, v), 0)
                    * m.X_vehicle_year[v, i]
                    for v in m.V
                )
                for vt in m.VT
            )

        model.AnnualEbusProcurement = pyo.Expression(
            model.I,
            rule=annual_electric_bus_procurement_rule,
            doc="Annual vehicle procurement costs.",
        )

        def electric_bus_depreciation_rule(m, i):

            total_vehicle_depreciation = 0

            for vt in m.VT:

                useful_life = self.useful_life_vehicle_types.get(vt)
                cycles = self.project_duration // useful_life
                for i_t in m.I:
                    for cycle in range(0, cycles + 1):

                        replacement_year = i_t + cycle * useful_life
                        if replacement_year > self.project_duration:
                            continue
                        if (
                            i - replacement_year >= 0
                            and i - replacement_year < useful_life
                        ):
                            total_vehicle_depreciation += (
                                self.npv_vehicle_types.get((vt, replacement_year), 0)
                                * sum(
                                    self.vehicle_type_assignments.get((vt, v), 0)
                                    * m.X_vehicle_year[v, i_t]
                                    for v in m.V
                                )
                                / useful_life
                            )
            return total_vehicle_depreciation

        model.EBusDepreciation = pyo.Expression(
            model.I,
            rule=electric_bus_depreciation_rule,
            doc="Annual vehicle replacement costs.",
        )

        def annual_battery_procurement_rule(m, i):
            return sum(
                self.npv_batteries.get((vt, i), 0)
                * sum(
                    self.vehicle_type_assignments.get((vt, v), 0)
                    * m.X_vehicle_year[v, i]
                    for v in m.V
                )
                for vt in m.VT
            )

        model.AnnualBatteryProcurement = pyo.Expression(
            model.I,
            rule=annual_battery_procurement_rule,
            doc="Annual battery procurement costs.",
        )

        def battery_depreciation_rule(m, i):
            total_battery_replacement = 0

            for vt in m.VT:

                useful_life = self.useful_life_batteries.get(vt)
                cycles = self.project_duration // useful_life
                for i_t in m.I:
                    for cycle in range(0, cycles + 1):

                        replacement_year = i_t + cycle * useful_life
                        if replacement_year > self.project_duration:
                            continue
                        if (
                            i - replacement_year >= 0
                            and i - replacement_year < useful_life
                        ):
                            total_battery_replacement += (
                                self.npv_batteries.get((vt, replacement_year), 0)
                                * sum(
                                    self.vehicle_type_assignments.get(
                                        (vt, v), 0
                                    )  # TODO this 0 means if vt is not assigned to v, but it is not clear from code
                                    * m.X_vehicle_year[v, i_t]
                                    for v in m.V
                                )
                                / useful_life
                            )
            return total_battery_replacement

        model.BatteryDepreciation = pyo.Expression(
            model.I,
            # rule=annual_vehicle_replacement_cost,
            rule=battery_depreciation_rule,
            doc="Annual vehicle replacement costs.",
        )

        def annual_station_with_charger_procurement_rule(m, i):

            # TODO making station with charger dependent on station. Now it's just average cost
            return sum(
                self.npv_station_with_chargers.get(i, 0) * m.Y_station_year[s, i]
                for s in m.S
            )

        model.AnnualStationWithChargerProcurement = pyo.Expression(
            model.I,
            rule=annual_station_with_charger_procurement_rule,
            doc="Annual station with chargers procurement costs.",
        )

        def station_charger_depreciation_rule(m, i):

            useful_life = 20
            station_charger_depreciation = 0

            for i_t in m.I:
                if i_t <= i:
                    station_charger_depreciation += (
                        sum(
                            self.npv_station_with_chargers.get(i_t, 0)
                            * m.Y_station_year[s, i_t]
                            for s in m.S
                        )
                        / useful_life
                    )
            return station_charger_depreciation

        model.StationChargerDepreciation = pyo.Expression(
            model.I,
            rule=station_charger_depreciation_rule,
            doc="Annual station with chargers replacement costs.",
        )

        def annual_depot_charger_procurement_rule(m, i):
            # TODO considering adding a rate for charger/vehicle ratio at depot
            return self.depot_charger_npv.get(i, 0) * sum(
                m.X_vehicle_year[v, i] for v in m.V
            )

        model.AnnualDepotChargerProcurement = pyo.Expression(
            model.I,
            rule=annual_depot_charger_procurement_rule,
            doc="Annual depot charger procurement costs.",
        )

        def depot_charger_depreciation_rule(m, i):
            useful_life = 20
            depot_charger_depreciation = 0

            for i_t in m.I:
                if i_t <= i:
                    depot_charger_depreciation += (
                        self.depot_charger_npv.get(i_t, 0)
                        * sum(m.X_vehicle_year[v, i_t] for v in m.V)
                        / useful_life
                    )
            return depot_charger_depreciation

        model.DepotChargerDepreciation = pyo.Expression(
            model.I,
            rule=depot_charger_depreciation_rule,
            doc="Annual depot charger replacement costs.",
        )

        # Opex

        def assignment_block_year(m, b, i):

            return model.Z_block_year[b, i] == sum(
                self.block_vehicle_assignments.get((b, v), 0)
                * sum(model.X_vehicle_year[v, i_t] for i_t in m.I if i_t <= i)
                for v in m.V
            )

        model.AssignmentBlockYearConstraint = pyo.Constraint(
            model.B,
            model.I,
            rule=assignment_block_year,
            doc="Block served by vehicle by the year.",
        )

        def electricity_cost_rule(m, i):
            return (
                sum(
                    self.electricity_price_npv.get(i, 0)
                    * self.block_mileage.get(b)
                    * self.electricity_consumption_kwh_per_km.get(vt)
                    * model.Z_block_year[b, i]
                    for v in m.V
                    for vt in m.VT
                    for b in m.B
                    if self.block_vehicle_assignments.get((b, v), 0) == 1
                    and self.vehicle_type_assignments.get((vt, v), 0) == 1
                )
                * self.time_scaling_factor_to_year
            )

        model.ElectricityCost = pyo.Expression(
            model.I, rule=electricity_cost_rule, doc="Annual electricity costs."
        )

        def diesel_cost_rule(m, i):
            return (
                sum(
                    self.diesel_price_npv.get(i, 0)
                    * self.block_mileage.get(b)
                    * self.diesel_consumption_l_per_km.get(vt)
                    * (1 - model.Z_block_year[b, i])
                    for v in m.V
                    for vt in m.VT
                    for b in m.B
                    if self.block_vehicle_assignments.get((b, v), 0) == 1
                    and self.vehicle_type_assignments.get((vt, v), 0) == 1
                )
                * self.time_scaling_factor_to_year
            )

        model.DieselCost = pyo.Expression(
            model.I, rule=diesel_cost_rule, doc="Annual diesel costs."
        )

        def maintenance_diesel_cost_rule(m, i):
            return (
                sum(
                    self.maintenance_diesel_npv.get(i, 0)
                    * self.block_mileage.get(b)
                    * (1 - model.Z_block_year[b, i])
                    for v in m.V
                    for b in m.B
                    if self.block_vehicle_assignments.get((b, v), 0) == 1
                )
                * self.time_scaling_factor_to_year
            )

        model.MaintenanceDieselCost = pyo.Expression(
            model.I,
            rule=maintenance_diesel_cost_rule,
            doc="Annual maintenance diesel costs.",
        )

        def maintenance_electric_cost_rule(m, i):
            return (
                sum(
                    self.maintenance_electric_cost_npv.get(i, 0)
                    * self.block_mileage.get(b)
                    * model.Z_block_year[b, i]
                    for v in m.V
                    for b in m.B
                    if self.block_vehicle_assignments.get((b, v), 0) == 1
                )
                * self.time_scaling_factor_to_year
            )

        model.MaintenanceElectricCost = pyo.Expression(
            model.I,
            rule=maintenance_electric_cost_rule,
            doc="Annual maintenance electric costs.",
        )

        def staff_cost_ebus_rule(m, i):
            return (
                sum(
                    self.vehicle_driving_times.get(v)
                    * sum(model.X_vehicle_year[v, i_t] for i_t in m.I if i_t <= i)
                    for v in m.V
                )
                * self.time_scaling_factor_to_year
                * self.staff_cost_npv.get(i, 0)
            )

        model.StaffCostEbus = pyo.Expression(
            model.I,
            rule=staff_cost_ebus_rule,
            doc="Annual staff costs for e-bus driving.",
        )

        def staff_cost_diesel_rule(m, i):
            return (
                sum(
                    self.block_durations.get(b) * (1 - model.Z_block_year[b, i])
                    for b in m.B
                )
                * self.time_scaling_factor_to_year
                * self.staff_cost_npv.get(i, 0)
            )

        model.StaffCostDiesel = pyo.Expression(
            model.I,
            rule=staff_cost_diesel_rule,
            doc="Annual staff costs for diesel bus driving.",
        )

        def charging_infra_maintenance_rule(m, i):
            return self.charging_infra_maintenance_npv.get(i, 0) * (
                self.charger_per_station
                * sum(m.Z_station_year[s, i] for s in m.S)  # Station chargers
                + sum(
                    sum(model.X_vehicle_year[v, i_t] for i_t in m.I if i_t <= i)
                    for v in m.V
                )  # Depot chargers
            )

        model.ChargingInfraMaintenance = pyo.Expression(
            model.I,
            rule=charging_infra_maintenance_rule,
            doc="Annual charging infrastructure maintenance costs.",
        )

        def diesel_bus_depreciation_rule(m, i):
            # TODO this is equivalent to replacing 1/useful_life of the diesel fleet each year
            total_diesel_annuity = 0
            for vt in m.VT:
                mileage = sum(
                    self.block_mileage.get(b) * (1 - model.Z_block_year[b, i])
                    for v in m.V
                    for b in m.B
                    if self.block_vehicle_assignments.get((b, v), 0) == 1
                    and self.vehicle_type_assignments.get((vt, v), 0) == 1
                )
                total_diesel_annuity += (
                    self.npv_annuity_diesel_bus.get((vt, i), 0)
                    * mileage
                    / self.mean_mileage_total_schedule.get(vt)
                )
            return total_diesel_annuity

        model.DieselBusDepreciation = pyo.Expression(
            model.I,
            rule=diesel_bus_depreciation_rule,
            doc="Total diesel bus depreciation.",
        )

        def diesel_mileage_rule(m, i, vt):
            return sum(
                self.block_mileage.get(b) * (1 - model.Z_block_year[b, i])
                for v in m.V
                for b in m.B
                if self.block_vehicle_assignments.get((b, v), 0) == 1
                and self.vehicle_type_assignments.get((vt, v), 0) == 1
            )

        model.DieselMileage = pyo.Expression(
            model.I,
            model.VT,
            rule=diesel_mileage_rule,
            doc="Diesel mileage per vehicle type.",
        )

        def budget_constraint_rule(m, i):
            annual_cost = (
                m.AnnualEbusProcurement[i]
                + m.AnnualBatteryProcurement[i]
                + m.AnnualStationWithChargerProcurement[i]
                + m.AnnualDepotChargerProcurement[i]
            )
            # Example budget limit, can be adjusted or made a parameter
            budget_limit = 1.5e7
            return annual_cost <= budget_limit

        model.BudgetConstraint = pyo.Constraint(
            model.I,
            rule=budget_constraint_rule,
            doc="Annual budget constraint.",
        )

        # Objective
        def total_cost_rule(m):
            return sum(
                m.EBusDepreciation[i]
                + m.BatteryDepreciation[i]
                + m.AnnualStationWithChargerProcurement[i]
                + m.AnnualDepotChargerProcurement[i]
                # + m.StationChargerDepreciation[i]
                # + m.DepotChargerDepreciation[i]
                + m.ElectricityCost[i]
                + m.DieselCost[i]
                + m.MaintenanceDieselCost[i]
                + m.MaintenanceElectricCost[i]
                + m.StaffCostEbus[i]
                + m.StaffCostDiesel[i]
                + m.ChargingInfraMaintenance[i]
                # + m.DieselBusDepreciation[i]
                for i in m.I
            )

        model.TotalCost = pyo.Objective(
            rule=total_cost_rule, sense=pyo.minimize, doc="Minimize total cost."
        )
        self.model = model

    def solve(self):
        """

        :return:
        """

        solver = pyo.SolverFactory("gurobi")

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

    def view_results(self):
        """

        :return:
        """

        import matplotlib.pyplot as plt

        # Extract results
        vehicle_assignment = pd.DataFrame(
            [
                {
                    "vehicle_id": v,
                    "year": i,
                    "is_electric": pyo.value(self.model.X_vehicle_year[v, i]),
                }
                for v in self.model.V
                for i in self.model.I
            ]
        )

        # list of vehicles bought each year

        yearly_vehicle_assigment = vehicle_assignment.groupby("year").sum()
        # plt.bar(yearly_vehicle_assigment.index, yearly_vehicle_assigment["is_electric"])
        # plt.xlabel("Year")
        # plt.ylabel("Number of Electric Vehicles Purchased")
        # plt.title("Yearly Electric Vehicle Purchases")
        # plt.show()

        station_assignment = pd.DataFrame(
            [
                {
                    "station_id": s,
                    "year": i,
                    "is_built": pyo.value(self.model.Z_station_year[s, i]),
                    "is_newly_built": pyo.value(self.model.Y_station_year[s, i]),
                }
                for s in self.model.S
                for i in self.model.I
            ]
        )

        block_assignment = pd.DataFrame(
            [
                {
                    "block_id": b,
                    "year": i,
                    "is_electrified_by_year": pyo.value(self.model.Z_block_year[b, i]),
                }
                for b in self.model.B
                for i in self.model.I
            ]
        )
        diesel_mileage = pd.DataFrame(
            [
                {
                    "year": i,
                    "vehicle_type": vt,
                    "diesel_mileage": pyo.value(self.model.DieselMileage[i, vt]),
                }
                for i in self.model.I
                for vt in self.model.VT
            ]
        )

        yearly_procured_vehicle_by_type = pd.DataFrame(
            [
                {
                    "year": i,
                    "vehicle_type": vt,
                    "num_vehicles": sum(
                        self.vehicle_type_assignments.get((vt, v), 0)
                        * pyo.value(self.model.X_vehicle_year[v, i])
                        for v in self.model.V
                    ),
                }
                for i in self.model.I
                for vt in self.model.VT
            ]
        )

        yearly_cost = pd.DataFrame(
            {
                "year": self.model.I,
                # "annual_ebus_depreciation": [pyo.value(self.model.EBusDepreciation[i]) for i in self.model.I],
                # "annual_battery_depreciation": [pyo.value(self.model.BatteryDepreciation[i]) for i in self.model.I],
                "annual_station_with_charger_procurement": [
                    pyo.value(self.model.AnnualStationWithChargerProcurement[i])
                    for i in self.model.I
                ],
                "annual_depot_charger_procurement": [
                    pyo.value(self.model.AnnualDepotChargerProcurement[i])
                    for i in self.model.I
                ],
                "station_charger_depreciation": [
                    pyo.value(self.model.StationChargerDepreciation[i])
                    for i in self.model.I
                ],
                "depot_charger_depreciation": [
                    pyo.value(self.model.DepotChargerDepreciation[i])
                    for i in self.model.I
                ],
                "electricity_cost": [
                    pyo.value(self.model.ElectricityCost[i]) for i in self.model.I
                ],
                "diesel_cost": [
                    pyo.value(self.model.DieselCost[i]) for i in self.model.I
                ],
                "maintenance_diesel_cost": [
                    pyo.value(self.model.MaintenanceDieselCost[i]) for i in self.model.I
                ],
                "maintenance_electric_cost": [
                    pyo.value(self.model.MaintenanceElectricCost[i])
                    for i in self.model.I
                ],
                "staff_cost_ebus": [
                    pyo.value(self.model.StaffCostEbus[i]) for i in self.model.I
                ],
                "staff_cost_diesel": [
                    pyo.value(self.model.StaffCostDiesel[i]) for i in self.model.I
                ],
                "charging_infra_maintenance": [
                    pyo.value(self.model.ChargingInfraMaintenance[i])
                    for i in self.model.I
                ],
                # "diesel_bus_depreciation": [pyo.value(self.model.DieselBusDepreciation[i]) for i in self.model.I],
                "total_cost": [
                    pyo.value(self.model.TotalCost) if i == 0 else 0
                    for i in self.model.I
                ],
            }
        )
        # plot yearly cost breakdown

        yearly_cost.set_index("year").drop(columns=["total_cost"]).plot(
            kind="bar", stacked=True, figsize=(12, 8), colormap="tab20"
        )
        # change color map

        plt.title("Yearly Cost Breakdown")
        plt.ylabel("Cost")
        plt.xlabel("Year")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()
        plt.savefig("yearly_cost_breakdown.png")
        # save dataframes to csv
        vehicle_assignment.to_csv("vehicle_assignment.csv", index=False)
        yearly_cost.to_csv("yearly_cost.csv", index=False)
