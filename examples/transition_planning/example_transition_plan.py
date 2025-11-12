import os


from sqlalchemy.orm import Session

from eflips.model import (
    Scenario,
    Station,
    VehicleType,
    Area,
)

from eflips.model import create_engine as create_engine_sqlite
from eflips.opt.transition_planning.transition_planning import (
    ParameterRegistry,
    ConstraintRegistry,
    ExpressionRegistry,
    TransitionPlannerModel,
)

from eflips.tco import init_tco_parameters


SCENARIO_ID = 1
DATABASE_URL_SQLITE = os.getenv("DATABASE_URL_SQLITE")
DATABASE_URL = os.getenv("DATABASE_URL")


if __name__ == "__main__":

    with Session(create_engine_sqlite(DATABASE_URL_SQLITE)) as session:
        # with Session(create_engine(DATABASE_URL)) as session:
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
            "project_duration": 20,
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
            "annual_budget_limit": 2.5e7,
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

        constraint_registry = ConstraintRegistry(transition_planner_parameters)
        expression_registry = ExpressionRegistry(transition_planner_parameters)

        name = "Long-term minimum cost"

        constraints_long_term = [
            "InitialElectricVehicleConstraint",
            "InitialElectrifiedStationConstraint",
            "NewlyBuiltStationConstraint",
            "NoStationUninstallationConstraint",
            "StationBeforeVehicleConstraint",
            "NoEarlyStationBuildingConstraint",
            "AssignmentBlockYearConstraint",

        ]

        expressions_long_term = [
            "ElectricBusDepreciation",
            "DieselBusDepreciation",
            "BatteryDepreciation",
            "StationChargerDepreciation",
            "DepotChargerDepreciation",
            "ElectricityCost",
            "DieselCost",
            "MaintenanceDieselCost",
            "MaintenanceElectricCost",
            "MaintenanceInfraCost",
            "StaffCostEbus",
            "StaffCostDiesel",
        ]

        model_long_term = TransitionPlannerModel(
            params=transition_planner_parameters,
            constraint_registry=constraint_registry,
            expression_registry=expression_registry,
            name=name,
            constraints=constraints_long_term,
            expressions=expressions_long_term,
            objective_components=expressions_long_term,
        )

        model_long_term.solve()
        model_long_term.visualize()

        scenario.tco_parameters["project_duration"] = 7
        session.add(scenario)

        parameters_registry_short_term = ParameterRegistry(session, scenario)

        constraints_short_term = [
            "InitialElectricVehicleConstraint",
            "InitialElectrifiedStationConstraint",
            "VehicleElectrificationConstraint",
            "NewlyBuiltStationConstraint",
            "NoStationUninstallationConstraint",
            "StationBeforeVehicleConstraint",
            "NoEarlyStationBuildingConstraint",
            "AssignmentBlockYearConstraint",
            "BudgetConstraint",
        ]
        expressions_short_term = [
            "AnnualEbusProcurement",
            "AnnualBatteryProcurement",
            "AnnualStationWithChargerProcurement",
            "AnnualDepotChargerProcurement",
            "ElectricityCost",
            "DieselCost",
            "MaintenanceDieselCost",
            "MaintenanceElectricCost",
            "MaintenanceInfraCost",
            "StaffCostEbus",
            "StaffCostDiesel",
        ]

        model_short_term = TransitionPlannerModel(
            params=parameters_registry_short_term,
            constraint_registry=constraint_registry,
            expression_registry=expression_registry,
            name="Short-term minimum cost",
            constraints=constraints_short_term,
            expressions=expressions_short_term,
            objective_components=expressions_short_term,
        )
        model_short_term.solve()
        model_short_term.visualize()
        print("Finished transition planning example.")
