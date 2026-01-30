import argparse
import os


from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from eflips.model import (
    Scenario,
    Station,
    VehicleType,
    Area,
)

from eflips.model import create_engine as create_engine_sqlite
from eflips.opt.transition_planning.transition_planning import (
    ParameterRegistry,
    SetVariableRegistry,
    ConstraintRegistry,
    ExpressionRegistry,
    TransitionPlannerModel,
)

from eflips.tco import init_tco_parameters


# SCENARIO_ID = 1
DATABASE_URL_SQLITE = os.getenv("DATABASE_URL_SQLITE")
DATABASE_URL = os.getenv("DATABASE_URL")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario-id", type=int, required=True, help="Scenario ID to use"
    )
    parser.add_argument("--w-sqlite", action="store_true", help="Use SQLite database")
    parser.add_argument(
        "--w-postgres", action="store_true", help="Use Postgres database"
    )
    parser.add_argument("--diesel-vsp", action="store_true", help="Use diesel VSP")
    args = parser.parse_args()

    SCENARIO_ID = args.scenario_id
    engine = None

    if args.w_postgres:
        engine = create_engine(DATABASE_URL)
    if args.w_sqlite:
        engine = create_engine_sqlite(DATABASE_URL_SQLITE)

    with Session(engine) as session:

        scenario = session.query(Scenario).filter(Scenario.id == SCENARIO_ID).one()

        id_en, b_id_en = (
            session.query(VehicleType.id, VehicleType.battery_type_id)
            .filter(
                VehicleType.scenario_id == SCENARIO_ID, VehicleType.name_short == "EN"
            )
            .one()
        )

        id_gn, b_id_gn = (
            session.query(VehicleType.id, VehicleType.battery_type_id)
            .filter(
                VehicleType.scenario_id == SCENARIO_ID, VehicleType.name_short == "GN"
            )
            .one()
        )

        id_dd, b_id_dd = (
            session.query(VehicleType.id, VehicleType.battery_type_id)
            .filter(
                VehicleType.scenario_id == SCENARIO_ID, VehicleType.name_short == "DD"
            )
            .one()
        )

        vehicle_types = [
            {
                "id": id_en,
                "name": "Ebusco 3.0 12 large battery",
                "useful_life": 14,
                "procurement_cost": 340000.0,
                "cost_escalation": -0.02,
                "average_electricity_consumption": 1.48,
                "procurement_cost_diesel_equivalent": 275000.0,
                "cost_escalation_diesel_equivalent": 0.02,
                "average_diesel_consumption": 0.449,
            },
            {
                "id": id_dd,
                "name": "Solaris Urbino 18 large battery",
                "useful_life": 14,
                "procurement_cost": 603000.0,
                "cost_escalation": -0.02,
                "average_electricity_consumption": 2.16,
                "procurement_cost_diesel_equivalent": 330000.0,
                "cost_escalation_diesel_equivalent": 0.02,
                "average_diesel_consumption": 0.589,
            },
            {
                "id": id_gn,
                "name": "Alexander Dennis Enviro500EV large battery",
                "useful_life": 14,
                "procurement_cost": 650000.0,
                "cost_escalation": -0.02,
                "average_electricity_consumption": 2.16,
                "procurement_cost_diesel_equivalent": 510000.0,
                "cost_escalation_diesel_equivalent": 0.02,
                "average_diesel_consumption": 0.589,
            },
        ]
        # We use the battery prices from Wirtschaftlichkeit von Elektromobilität in gewerblichen Anwendungen, April 2015
        # The price is a prognose for 2025 in an optimistic scenario

        battery_types = [
            {
                "id": b_id_en,
                "name": "Ebusco 3.0 12 large battery",
                "procurement_cost": 190,
                "useful_life": 7,
                "cost_escalation": -0.03,
            },
            {
                "id": b_id_gn,
                "name": "Solaris Urbino 18 large battery",
                "procurement_cost": 190,
                "useful_life": 7,
                "cost_escalation": -0.03,
            },
            {
                "id": b_id_dd,
                "name": "Alexander Dennis Enviro500EV large battery",
                "procurement_cost": 190,
                "useful_life": 7,
                "cost_escalation": -0.03,
            },
        ]

        # We use the prices from Jefferies and Göhlich (2020) since the data from bvg internal report is not allowed to be shared.

        depot_charger_id = (
            session.query(Area.charging_point_type_id)
            .filter(
                Area.scenario_id == SCENARIO_ID, Area.charging_point_type_id.isnot(None)
            )
            .first()[0]
        )
        station_charger_id = (
            session.query(Station.charging_point_type_id)
            .filter(
                Station.scenario_id == SCENARIO_ID,
                Station.charging_point_type_id.isnot(None),
            )
            .first()[0]
        )

        charging_point_types = [
            {
                "id": depot_charger_id,
                "type": "depot",
                "name": "Depot Charging Point",
                "procurement_cost": 100000.0,
                "useful_life": 20,
                "cost_escalation": 0,
            },
            {
                "id": station_charger_id,
                "type": "opportunity",
                "name": "Opportunity Charging Point",
                "procurement_cost": 250000.0,
                "useful_life": 20,
                "cost_escalation": 0,
            },
        ]

        # We use the prices from Jefferies and Göhlich (2020) since the data from bvg internal report is not allowed to be shared.

        charging_infrastructure = [
            {
                "type": "depot",
                "name": "Depot Charging Infrastructure",
                "procurement_cost": 2000000.0,  # TODO
                "useful_life": 20,
                "cost_escalation": 0,
            },
            {
                "type": "station",
                "name": "Opportunity Charging Infrastructure",
                "procurement_cost": 500000.0,
                "useful_life": 20,
                "cost_escalation": 0,
            },
        ]
        # Energy consumption from bvg's report and it can be shared

        scenario_tco_parameters = {
            "project_duration": 15,
            "interest_rate": 0.05,
            "inflation_rate": 0.02,
            "staff_cost": 25.0,
            "fuel_cost": {"diesel": 1, "electricity": 0.1794},
            "vehicle_maint_cost": {"diesel": 0.5, "electricity": 0.35},
            "infra_maint_cost": 1000,
            "cost_escalation_rate": {
                "general": 0.02,
                "staff": 0.03,
                "diesel": 0.07,
                "electricity": 0.038,
            },
            "annual_budget_limit": 2.0e7,
            "depot_time_plan": {
                "BF RL": 2032,
                "BF KL": 2028,
                "BF SN": 2027,
                "BF I": 2030,
                "BF S": 2030,
                "BF B": 2030,
                "BF C": 2034,
                "BF M": 2035,
                "BF L": 2030,
            },

            "current_year": 2026,
            "max_station_construction_per_year": 5,
        }

        init_tco_parameters(
            scenario=scenario,
            scenario_tco_parameters=scenario_tco_parameters,
            vehicle_types=vehicle_types,
            battery_types=battery_types,
            charging_point_types=charging_point_types,
            charging_infrastructure=charging_infrastructure,
        )

        transition_planner_parameters = ParameterRegistry(session, scenario)

        set_variable_registry = SetVariableRegistry(transition_planner_parameters)
        constraint_registry = ConstraintRegistry(transition_planner_parameters)
        expression_registry = ExpressionRegistry(transition_planner_parameters)

        name = "long_term_min_cost"

        sets = ["V", "VT", "B", "S", "I", "B_pairs"]
        variables = [
            "X_vehicle_year",
            "Z_station_year",
            # "U_diesel_block_schedule_year",
        ]

        constraints_long_term = [
            "InitialElectricVehicleConstraint",
            "InitialElectrifiedStationConstraint",
            "NoStationUninstallationConstraint",
            "StationBeforeVehicleConstraint",
            "VehicleDeployTimeLimitConstraint",
            "StationConstructionPerYearConstraint",
            # "NoEarlyStationBuildingConstraint",
            "AssignmentBlockYearConstraint",
            "FullElectrificationConstraint",
            "NoDuplicatedVehicleElectrificationConstraint",
            # "BlockScheduleOnePathConstraint",
            # "BlockScheduleFlowConservationConstraint",
            # "BlockScheduleCostConstraint",
            "BudgetConstraint",

        ]

        expressions_long_term = [
            "Z_block_year",
            "NewlyBuiltStation",
            # "ElectricBusDepreciation",
            # "DieselBusDepreciation",
            # "BatteryDepreciation",
            # "StationChargerDepreciation",
            # "DepotChargerDepreciation",
            "AnnualEbusProcurement",
            "AnnualBatteryProcurement",
            "AnnualVehicleReplacement",
            "AnnualBatteryReplacement",
            "AnnualStationWithChargerProcurement",
            "AnnualDepotChargerProcurement",
            # "ElectricityCost",
            # "DieselCost",
            # "MaintenanceDieselCost",
            # "MaintenanceElectricCost",
            "MaintenanceInfraCost",
            "StaffCostEbus",
            "StaffCostDiesel",
            "EbusEnergySaving",
            "EbusMaintenanceSaving",
            "EbusExtraStaffCost",
        ]

        objective_components = [
            # "ElectricBusDepreciation",
            # "DieselBusDepreciation",
            # "BatteryDepreciation",
            # "StationChargerDepreciation",
            # "DepotChargerDepreciation",
            # "ElectricityCost",
            # "DieselCost",
            # "MaintenanceDieselCost",
            # "MaintenanceElectricCost",
            "MaintenanceInfraCost",
            # "StaffCostEbus",
            # "StaffCostDiesel",
            "AnnualEbusProcurement",
            "AnnualBatteryProcurement",
            "AnnualVehicleReplacement",
            "AnnualBatteryReplacement",
            "AnnualStationWithChargerProcurement",
            "AnnualDepotChargerProcurement",
            "EbusEnergySaving",
            "EbusMaintenanceSaving",
            "EbusExtraStaffCost",

        ]

        model_long_term = TransitionPlannerModel(
            params=transition_planner_parameters,
            set_variable_registry=set_variable_registry,
            constraint_registry=constraint_registry,
            expression_registry=expression_registry,
            name=name,
            sets=sets,
            variables=variables,
            constraints=constraints_long_term,
            expressions=expressions_long_term,
            objective_components=objective_components,
        )

        model_long_term.solve()
        optional_visualization_target = [
            "AnnualEbusProcurement",
            "AnnualBatteryProcurement",
            "AnnualVehicleReplacement",
            "AnnualBatteryReplacement",
            "AnnualStationWithChargerProcurement",
            "AnnualDepotChargerProcurement",
            "DieselBusDepreciation",
            "ElectricityCost",
            "DieselCost",
            "MaintenanceDieselCost",
            "MaintenanceElectricCost",
            "MaintenanceInfraCost",
            "StaffCostEbus",
            "StaffCostDiesel",
        ]
        model_long_term.visualize(
            # optional_visualization_targets=optional_visualization_target
        )

        electrified_vehicles = {}
        electrified_blocks = {}

        for year in range(1, scenario.tco_parameters["projection_duration"] + 1):
            electrified_vehicles[year] = model_long_term.get_electrified_vehicles(year=year)
            electrified_blocks[year] = model_long_term.get_electrified_blocks(year=year)


        print("Electrified Vehicles by Year:")
        print(electrified_vehicles)
        print("Electrified Blocks by Year:")
        print(electrified_blocks)

