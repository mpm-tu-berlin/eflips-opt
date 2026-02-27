import argparse
import os
from pathlib import Path

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
from eflips.tco.default_params import get_params_from_file


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

        defaults = get_params_from_file(
            Path(__file__).parent.parent.parent
            / "eflips"
            / "opt"
            / "transition_planning"
            / "berlin_literature.py"
        )

        init_tco_parameters(
            scenario=scenario,
            scenario_params=defaults.SCENARIO_TCO,
            vehicle_type_params=defaults.VEHICLE_TYPES,
            battery_type_params=defaults.BATTERY_TYPES,
            charging_point_type_params=defaults.CHARGING_POINT_TYPES,
            charging_infra_params=defaults.CHARGING_INFRASTRUCTURE,
        )

        transition_planner_parameters = ParameterRegistry(session, scenario)

        set_variable_registry = SetVariableRegistry(transition_planner_parameters)
        constraint_registry = ConstraintRegistry(transition_planner_parameters)
        expression_registry = ExpressionRegistry(transition_planner_parameters)

        name = "long_term_min_cost"

        sets = ["V", "VT", "B", "S", "I", "D", "DE_pairs"]
        variables = [
            "X_vehicle_year",
            "Z_station_year",
            "Charger_count_depot_year",
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
            # "BudgetConstraint",
            "DieselBusReplacementLimitUpperBound",
            "DieselBusReplacementLimitLowerBound",
            "DepotChargerConstructionLimit",
            "DepotChargerNoUninstallation",
            "InitialDepotChargerConstraint",
        ]

        expressions_long_term = [
            "Z_block_year",
            "NewlyBuiltStation",
            "ElectricBusDepreciation",
            "DieselBusDepreciation",
            "BatteryDepreciation",
            # "StationChargerDepreciation",
            # "DepotChargerDepreciation",
            # "AnnualEbusProcurement",
            # "AnnualBatteryProcurement",
            # "AnnualVehicleReplacement",
            # "AnnualBatteryReplacement",
            # "EbusResidualValue",
            # "BatteryResidualValue",

            "AnnualStationWithChargerProcurement",
            "AnnualDepotChargerProcurement",
            "ElectricityCost",
            "DieselCost",
            "MaintenanceDieselCost",
            "MaintenanceElectricCost",
            "MaintenanceInfraCost",
            "StaffCostEbus",
            "StaffCostDiesel",
            # "EbusEnergySaving",
            # "EbusMaintenanceSaving",
            # "EbusExtraStaffCost",
        ]

        objective_components = [
            "ElectricBusDepreciation",
            "DieselBusDepreciation",
            "BatteryDepreciation",
            # "StationChargerDepreciation",
            # "DepotChargerDepreciation",
            "ElectricityCost",
            "DieselCost",
            "MaintenanceDieselCost",
            "MaintenanceElectricCost",
            "MaintenanceInfraCost",
            "StaffCostEbus",
            "StaffCostDiesel",
            # "AnnualEbusProcurement",
            # "AnnualBatteryProcurement",
            # "AnnualVehicleReplacement",
            # "AnnualBatteryReplacement",
            # "EbusResidualValue",
            # "BatteryResidualValue",

            "AnnualStationWithChargerProcurement",
            "AnnualDepotChargerProcurement",
            # "EbusEnergySaving",
            # "EbusMaintenanceSaving",
            # "EbusExtraStaffCost",

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
        model_long_term.get_results(
            save_results=True
        )

        electrified_vehicles = {}
        electrified_blocks = {}

        for year in range(1, scenario.tco_parameters["project_duration"] + 1):
            electrified_vehicles[year] = model_long_term.get_electrified_vehicles(
                year=year
            )
            electrified_blocks[year] = model_long_term.get_electrified_blocks(year=year)

        print("Electrified Vehicles by Year:")
        print(electrified_vehicles)
        # print("Electrified Blocks by Year:")
        # print(electrified_blocks)
