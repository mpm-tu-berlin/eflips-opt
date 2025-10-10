import logging
from typing import Dict, Tuple, Optional

import pandas as pd

import pyomo.environ as pyo  # type: ignore
from eflips.model import (
    VehicleType,
    BatteryType,
    Station,
    ChargingPointType,
    Area,
    Trip,
)

from eflips.opt.transition_planning.util import (
    vehicle_relevant_assignments,
    block_info,
)

from eflips.opt.transition_planning.util import npv_with_escalation

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TransitionPlanner:

    # TODO I write this at first only for scenarios with tco parameters stored in db
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

        self._assignment_data_prep()
        self._economic_params_prep()
        self._model_setup()

        logger.debug("Model setup complete.")

    def solve(self):
        """

        :return:
        """

        if not pyo.SolverFactory("gurobi_direct").available():
            logger.warning("Gurobi is not available. Using GLPK instead.")
            if not pyo.SolverFactory("glpk").available():
                raise ValueError(
                    "GLPK is not available. Install it using your package manager."
                )
            solver = pyo.SolverFactory("glpk")
        else:
            solver = pyo.SolverFactory("gurobi_direct")

        result = solver.solve(self.model, tee=True)
        if result.solver.termination_condition == pyo.TerminationCondition.infeasible:
            raise ValueError(
                "No feasible solution found. Please check your constraints."
            )

        logger.info("Optimization complete.")

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

        # list of vehicles bougt each year
        vehicle_assignment.groupby("year").sum()

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

        return vehicle_assignment

    def _assignment_data_prep(self):
        """

        :return:
        """
        (
            self.block_vehicle_assignments,
            self.station_vehicle_assignments,
            self.vehicle_type_assignments,
            self.vehicle_driving_times,
            self.vehicle_indices,
            self.block_indices,
            self.station_indices,
        ) = vehicle_relevant_assignments(self.session, self.scenario)
        self.block_mileage, self.block_durations = block_info(
            self.session, self.scenario
        )

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

            # Station with chargers
            electrified_station = (
                self.session.query(Station)
                .filter(
                    Station.scenario_id == self.scenario.id,
                    Station.tco_parameters.isnot(None),
                    Station.is_electrified.is_(True),
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

        # TODO currently I assume no electrfied stations in year 0. If there are, need to add parameter and change here.
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
        def no_early_station_building_rule(m, s, i):
            assigned_vehicles = [
                v for v in m.V if self.station_vehicle_assignments.get((s, v)) == 1
            ]
            if not assigned_vehicles:
                return pyo.Constraint.Skip
            return m.Z_station_year[s, i] <= sum(
                m.X_vehicle_year[v, years_before]
                for v in assigned_vehicles
                for years_before in m.I
                if years_before <= i
            )

        model.NoEarlyStationBuildingConstraint = pyo.Constraint(
            model.S,
            model.I,
            rule=no_early_station_building_rule,
            doc="No early station building before vehicle assignment.",
        )

        # Expressions for building costs
        def annual_vehicle_procurement_rule(m, i):
            return sum(
                self.npv_vehicle_types.get((vt, i), 0)
                * sum(
                    self.vehicle_type_assignments.get((vt, v), 0)
                    * m.X_vehicle_year[v, i]
                    for v in m.V
                )
                for vt in m.VT
            )

        model.AnnualVehicleProcurement = pyo.Expression(
            model.I,
            rule=annual_vehicle_procurement_rule,
            doc="Annual vehicle procurement costs.",
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

        def annual_vehicle_replacement_cost(m, i):
            total_vehicle_replacement = 0

            for vt in m.VT:
                useful_life = self.useful_life_vehicle_types.get(vt)
                if i >= useful_life:
                    total_vehicle_replacement += self.npv_vehicle_types.get(
                        (vt, i), 0
                    ) * sum(
                        self.vehicle_type_assignments.get((vt, v), 0)
                        * m.X_vehicle_year[v, i - useful_life]
                        for v in m.V
                    )
            return total_vehicle_replacement

        model.AnnualVehicleReplacement = pyo.Expression(
            model.I,
            rule=annual_vehicle_replacement_cost,
            doc="Annual vehicle replacement costs.",
        )

        def annual_battery_replacement_cost(m, i):
            total_battery_replacement = 0

            for vt in m.VT:
                useful_life = self.useful_life_batteries.get(vt)
                if i >= useful_life:
                    total_battery_replacement += self.npv_batteries.get(
                        (vt, i), 0
                    ) * sum(
                        self.vehicle_type_assignments.get((vt, v), 0)
                        * m.X_vehicle_year[v, i - useful_life]
                        for v in m.V
                    )
            return total_battery_replacement

        model.AnnualBatteryReplacement = pyo.Expression(
            model.I,
            rule=annual_battery_replacement_cost,
            doc="Annual battery replacement costs.",
        )

        def annual_station_with_charger_procurement_rule(m, i):
            # TODO the price at i = 0 is 0 by default, meaning at the beginning no stations with chargers are procured.
            return sum(
                self.npv_station_with_chargers.get(i, 0) * m.Y_station_year[s, i]
                for s in m.S
            )

        model.AnnualStationWithChargerProcurement = pyo.Expression(
            model.I,
            rule=annual_station_with_charger_procurement_rule,
            doc="Annual station with chargers procurement costs.",
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
                    * self.staff_cost_npv.get(i, 0)
                    for v in m.V
                )
                * self.time_scaling_factor_to_year
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
            )

        model.StaffCostDiesel = pyo.Expression(
            model.I,
            rule=staff_cost_diesel_rule,
            doc="Annual staff costs for diesel bus driving.",
        )

        def charging_infra_maintenance_rule(m, i):
            return self.charging_infra_maintenance_npv.get(i, 0) * (
                self.charger_per_station * sum(m.Z_station_year[s, i] for s in m.S) # Station chargers
                + sum(
                    sum(model.X_vehicle_year[v, i_t] for i_t in m.I if i_t <= i)
                    for v in m.V
                ) # Depot chargers
            )

        model.ChargingInfraMaintenance = pyo.Expression(
            model.I,
            rule=charging_infra_maintenance_rule,
            doc="Annual charging infrastructure maintenance costs.",
        )

        # Objective
        def total_cost_rule(m):
            return sum(
                m.AnnualVehicleProcurement[i]
                + m.AnnualBatteryProcurement[i]
                + m.AnnualVehicleReplacement[i]
                + m.AnnualBatteryReplacement[i]
                + m.AnnualStationWithChargerProcurement[i]
                + m.AnnualDepotChargerProcurement[i]
                + m.ElectricityCost[i]
                + m.DieselCost[i]
                + m.MaintenanceDieselCost[i]
                + m.MaintenanceElectricCost[i]
                + m.StaffCostEbus[i]
                + m.StaffCostDiesel[i]
                + m.ChargingInfraMaintenance[i]
                for i in m.I
            )

        model.TotalCost = pyo.Objective(
            rule=total_cost_rule, sense=pyo.minimize, doc="Minimize total cost."
        )
        self.model = model
