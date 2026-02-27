"""Berlin TCO parameter defaults from literature.

Prices taken from Jefferies and Goehlich (2020), with inflation adjustment.
"""

from eflips.tco.tco_parameter_config import (
    VehicleTypeTCOParameter,
    BatteryTypeTCOParameter,
    ChargingPointTypeTCOParameter,
    ChargingInfrastructureTCOParameter,
    ScenarioTCOParameter,
)

# Vehicle type defaults
VEHICLE_TYPES = [
    VehicleTypeTCOParameter(
        name_short="EN",
        name="Ebusco 3.0 12 large battery",
        useful_life=14,
        procurement_cost=340000.0,
        procurement_cost_diesel_equivalent=275000.0,
        cost_escalation=-0.02,
        cost_escalation_diesel_equivalent=0.02,
        average_electricity_consumption=1.48,
        average_diesel_consumption=0.449,
    ),
    VehicleTypeTCOParameter(
        name_short="GN",
        name="Solaris Urbino 18 large battery",
        useful_life=14,
        procurement_cost=650000.0,
        procurement_cost_diesel_equivalent=330000.0,
        cost_escalation=-0.02,
        cost_escalation_diesel_equivalent=0.02,
        average_electricity_consumption=2.16,
        average_diesel_consumption=0.589,
    ),
    VehicleTypeTCOParameter(
        name_short="DD",
        name="Alexander Dennis Enviro500EV large battery",
        useful_life=14,
        procurement_cost=603000.0,
        procurement_cost_diesel_equivalent=510000.0,
        cost_escalation=-0.02,
        cost_escalation_diesel_equivalent=0.02,
        average_electricity_consumption=2.16,
        average_diesel_consumption=0.589,
    ),
]

# Battery type defaults
BATTERY_TYPES = [
    BatteryTypeTCOParameter(
        name="Ebusco 3.0 12 large battery",
        vehicle_name_short="EN",
        procurement_cost=190,
        useful_life=7,
        cost_escalation=-0.03,
    ),
    BatteryTypeTCOParameter(
        name="Solaris Urbino 18 large battery",
        vehicle_name_short="GN",
        procurement_cost=190,
        useful_life=7,
        cost_escalation=-0.03,
    ),
    BatteryTypeTCOParameter(
        name="Alexander Dennis Enviro500EV large battery",
        vehicle_name_short="DD",
        procurement_cost=190,
        useful_life=7,
        cost_escalation=-0.03,
    ),
]

# Charging point defaults
# Prices from Jefferies and Goehlich (2020), with inflation adjustment
CHARGING_POINT_TYPES = [
    ChargingPointTypeTCOParameter(
        type="depot",
        name="Depot Charging Point",
        procurement_cost=119899.50,
        useful_life=20,
        cost_escalation=0.02,
    ),
    ChargingPointTypeTCOParameter(
        type="opportunity",
        name="Opportunity Charging Point",
        procurement_cost=299748.74,
        useful_life=20,
        cost_escalation=0.02,
    ),
]

# Charging infrastructure defaults
CHARGING_INFRASTRUCTURE = [
    ChargingInfrastructureTCOParameter(
        type="depot",
        name="Depot Charging Infrastructure",
        procurement_cost=2397989.95,  # TODO
        useful_life=20,
        cost_escalation=0.02,
    ),
    ChargingInfrastructureTCOParameter(
        type="station",
        name="Opportunity Charging Infrastructure",
        procurement_cost=269773.87,
        useful_life=20,
        cost_escalation=0.02,
    ),
]

# Scenario TCO parameters
SCENARIO_TCO = ScenarioTCOParameter(
    project_duration=15,
    interest_rate=0.04,
    inflation_rate=0.02,
    staff_cost=25.0,  # calculated: 35,000 EUR p.a. per driver / 1600 h p.a. per driver
    fuel_cost={"diesel": 1.5, "electricity": 0.1794},
    vehicle_maint_cost={"diesel": 0.45, "electricity": 0.35},
    infra_maint_cost=1000,  # maintenance cost infrastructure per year and charging slot
    cost_escalation_rate={
        "general": 0.02,
        "staff": 0.025,
        "diesel": 0.0,
        "electricity": 0.038,
        "insurance": 0.02,
    },
    insurance=9693,  # DCO #9703, # EBU
    taxes=278,  # taxes in EUR per year and bus
    annual_budget_limit=2.0e7,
    depot_time_plan={
        "Depot at Betriebshof Rummelsburger Landstraße": 2032,
        "Depot at Betriebshof Köpenicher Landstraße": 2028,
        "Depot at Betriebshof Säntisstraße": 2027,
        "Depot at Betriebshof Indira-Gandhi-Str.": 2030,
        "Depot at Betriebshof Spandau": 2030,
        "Depot at Betriebshof Britz": 2030,
        "Depot at Betriebshof Cicerostr.": 2034,
        "Depot at Betriebshof Müllerstr.": 2035,
        "Depot at Betriebshof Lichtenberg": 2030,
    },
    current_year=2026,
    max_station_construction_per_year=10,
)
