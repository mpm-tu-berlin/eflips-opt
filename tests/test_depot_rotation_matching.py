import os
from datetime import datetime, timedelta, timezone

import eflips.model
import plotly.graph_objects as go
import pytest
from eflips.model import (
    Area,
    AreaType,
    AssocPlanProcess,
    AssocRouteStation,
    Base,
    BatteryType,
    Depot,
    Line,
    Plan,
    Process,
    Rotation,
    Route,
    Scenario,
    Station,
    StopTime,
    Trip,
    TripType,
    VehicleClass,
    VehicleType,
)
from geoalchemy2.shape import from_shape
from shapely import Point
from sqlalchemy import create_engine
from sqlalchemy import func
from sqlalchemy.orm import Session

import pandas as pd
from shapely import LineString

from eflips.opt.depot_rotation_matching import (
    DepotRotationOptimizer,
    _line_with_altitudes,
)


class TestAltitudeBatching:
    """DB-free tests for the altitude pre-pass of write_optimization_results."""

    def test_line_with_altitudes_keys_by_lat_lon(self):
        line = LineString([(13.0, 52.0), (13.1, 52.1)])
        altitudes = {(52.0, 13.0): 30.0, (52.1, 13.1): 35.5}
        result = _line_with_altitudes(line, altitudes)
        assert result.has_z
        assert list(result.coords) == [(13.0, 52.0, 30.0), (13.1, 52.1, 35.5)]

    def test_collect_altitudes_makes_one_deduplicated_call(self, monkeypatch):
        calls = []

        def fake_get_altitudes(latlons):
            calls.append(list(latlons))
            return [float(index) for index in range(len(latlons))]

        monkeypatch.delenv("ELEVATION_DUMMY_MODE", raising=False)
        monkeypatch.setattr(
            "eflips.opt.depot_rotation_matching.get_altitudes", fake_get_altitudes
        )

        ferry = LineString([(13.0, 52.0), (13.1, 52.1)])
        return_ = LineString(
            [(13.1, 52.1), (13.0, 52.0)]
        )  # shares both vertices with ferry
        other = LineString([(13.2, 52.2), (13.3, 52.3)])
        depot_from_user = [
            {
                "name": "New depot",
                "depot_station": (13.5, 52.5),
            },  # (lon, lat) tuple → new Station
            {
                "name": "Existing depot",
                "depot_station": 42,
            },  # station id → nothing to look up
        ]
        new_assign = pd.DataFrame({"rotation_id": [1, 2], "new_depot_id": [0, 0]})
        cost = pd.DataFrame(
            {
                "rotation_id": [1, 2],
                "depot_id": [0, 0],
                "cost": [
                    {
                        "distance": (1, 1),
                        "duration": (1, 1),
                        "geometry": (ferry, return_),
                    },
                    {
                        "distance": (1, 1),
                        "duration": (1, 1),
                        "geometry": (ferry, other),
                    },
                ],
            }
        )

        result = DepotRotationOptimizer._collect_altitudes(
            depot_from_user, new_assign, cost
        )

        assert len(calls) == 1
        expected = [
            (52.5, 13.5),
            (52.0, 13.0),
            (52.1, 13.1),
            (52.2, 13.2),
            (52.3, 13.3),
        ]
        assert calls[0] == expected
        assert result == {latlon: float(index) for index, latlon in enumerate(expected)}
        assert _line_with_altitudes(ferry, result).coords[0] == (13.0, 52.0, 1.0)

    def test_collect_altitudes_dummy_mode(self, monkeypatch):
        monkeypatch.setenv("ELEVATION_DUMMY_MODE", "True")
        new_assign = pd.DataFrame({"rotation_id": [1], "new_depot_id": [0]})
        line = LineString([(13.0, 52.0), (13.1, 52.1)])
        cost = pd.DataFrame(
            {"rotation_id": [1], "depot_id": [0], "cost": [{"geometry": (line, line)}]}
        )
        result = DepotRotationOptimizer._collect_altitudes([], new_assign, cost)
        assert result == {(52.0, 13.0): 9999.0, (52.1, 13.1): 9999.0}


class TestHelpers:
    @pytest.fixture(autouse=True)
    def disable_altitude_lookups(self, monkeypatch) -> None:
        """Bypass network altitude lookups for all tests in this class."""
        monkeypatch.setenv("ELEVATION_DUMMY_MODE", "True")

    @pytest.fixture()
    def scenario(self, session):
        """
        Creates a scenario.

        :param session: An SQLAlchemy Session with the eflips-model schema
        :return: A :class:`Scenario` object
        """
        scenario = Scenario(name="Test Scenario")
        session.add(scenario)
        session.commit()
        return scenario

    @pytest.fixture()
    def full_scenario(self, session):
        """
        Creates a scenario that comes filled with sample content for each type.

        :param session: An SQLAlchemy Session with the eflips-model schema
        :return: A :class:`Scenario` object
        """

        # Add a scenario
        scenario = Scenario(name="Test Scenario")
        session.add(scenario)

        # Add a vehicle type with a battery type
        vehicle_type = VehicleType(
            scenario=scenario,
            name="Test Vehicle Type",
            battery_capacity=100,
            charging_curve=[[0, 150], [1, 150]],
            opportunity_charging_capable=True,
        )
        session.add(vehicle_type)
        battery_type = BatteryType(
            scenario=scenario, specific_mass=100, chemistry="test"
        )
        session.add(battery_type)
        vehicle_type.battery_type = battery_type

        # Add a vehicle type without a battery type
        vehicle_type = VehicleType(
            scenario=scenario,
            name="Test Vehicle Type 2",
            battery_capacity=100,
            charging_curve=[[0, 150], [1, 150]],
            opportunity_charging_capable=True,
        )

        session.add(vehicle_type)

        # Add a VehicleClass
        vehicle_class = VehicleClass(
            scenario=scenario,
            name="Test Vehicle Class",
            vehicle_types=[vehicle_type],
        )
        session.add(vehicle_class)

        line = Line(
            scenario=scenario,
            name="Test Line",
            name_short="TL",
        )
        session.add(line)

        stop_1 = Station(
            scenario=scenario,
            name="Tiergarten",
            name_short="TA",
            geom=from_shape(
                Point(13.335799579256504, 52.514000247127576, 0.0), srid=4326
            ),
            is_electrified=False,
        )
        session.add(stop_1)

        stop_2 = Station(
            scenario=scenario,
            name="Ernst Reuter Platz",
            name_short="ERP",
            geom=from_shape(Point(13.32280013838422, 52.5116502402821, 0.0), srid=4326),
            is_electrified=False,
        )
        session.add(stop_2)

        stop_3 = Station(
            scenario=scenario,
            name="Adenauer Platz",
            name_short="AP",
            geom=from_shape(
                Point(13.308215181759383, 52.49999600735662, 0.0), srid=4326
            ),
            is_electrified=False,
        )

        route_1 = Route(
            scenario=scenario,
            name="Forward Route",
            name_short="FR",
            departure_station=stop_1,
            arrival_station=stop_3,
            line=line,
            distance=3900,
        )
        assocs = [
            AssocRouteStation(
                scenario=scenario, station=stop_1, route=route_1, elapsed_distance=0
            ),
            AssocRouteStation(
                scenario=scenario, station=stop_2, route=route_1, elapsed_distance=1000
            ),
            AssocRouteStation(
                scenario=scenario, station=stop_3, route=route_1, elapsed_distance=3900
            ),
        ]
        route_1.assoc_route_stations = assocs
        session.add(route_1)

        route_2 = Route(
            scenario=scenario,
            name="Backward Route",
            name_short="BR",
            departure_station=stop_3,
            arrival_station=stop_1,
            line=line,
            distance=3900,
        )
        assocs = [
            AssocRouteStation(
                scenario=scenario, station=stop_3, route=route_2, elapsed_distance=0
            ),
            AssocRouteStation(
                scenario=scenario, station=stop_2, route=route_2, elapsed_distance=2900
            ),
            AssocRouteStation(
                scenario=scenario, station=stop_1, route=route_2, elapsed_distance=3900
            ),
        ]
        route_2.assoc_route_stations = assocs
        session.add(route_2)

        # Add the schedule objects
        first_departure = datetime(
            year=2020, month=1, day=1, hour=12, minute=0, second=0, tzinfo=timezone.utc
        )
        interval = timedelta(minutes=10)
        duration = timedelta(minutes=15)

        # Create a number of rotations
        number_of_rotations = 3
        for rotation_id in range(number_of_rotations):
            trips = []

            rotation = Rotation(
                scenario=scenario,
                trips=trips,
                vehicle_type=vehicle_type,
                allow_opportunity_charging=False,
            )
            session.add(rotation)

            # Add first empty trip

            for i in range(0, 15):
                # forward
                trips.append(
                    Trip(
                        scenario=scenario,
                        route=route_1,
                        trip_type=TripType.PASSENGER if i > 0 else TripType.EMPTY,
                        departure_time=first_departure + 2 * i * interval,
                        arrival_time=first_departure + 2 * i * interval + duration,
                        rotation=rotation,
                    )
                )
                stop_times = [
                    StopTime(
                        scenario=scenario,
                        station=stop_1,
                        arrival_time=first_departure + 2 * i * interval,
                    ),
                    StopTime(
                        scenario=scenario,
                        station=stop_2,
                        arrival_time=first_departure
                        + 2 * i * interval
                        + timedelta(minutes=5),
                        dwell_duration=timedelta(minutes=1),
                    ),
                    StopTime(
                        scenario=scenario,
                        station=stop_3,
                        arrival_time=first_departure + 2 * i * interval + duration,
                    ),
                ]
                trips[-1].stop_times = stop_times

                # backward
                trips.append(
                    Trip(
                        scenario=scenario,
                        route=route_2,
                        trip_type=TripType.PASSENGER if i < 14 else TripType.EMPTY,
                        departure_time=first_departure + (2 * i + 1) * interval,
                        arrival_time=first_departure
                        + (2 * i + 1) * interval
                        + duration,
                        rotation=rotation,
                    )
                )
                stop_times = [
                    StopTime(
                        scenario=scenario,
                        station=stop_3,
                        arrival_time=first_departure + (2 * i + 1) * interval,
                    ),
                    StopTime(
                        scenario=scenario,
                        station=stop_2,
                        arrival_time=first_departure
                        + (2 * i + 1) * interval
                        + timedelta(minutes=5),
                    ),
                    StopTime(
                        scenario=scenario,
                        station=stop_1,
                        arrival_time=first_departure
                        + (2 * i + 1) * interval
                        + duration,
                    ),
                ]
                trips[-1].stop_times = stop_times
            session.add_all(trips)

            first_departure += timedelta(minutes=20)

        # Create a simple depot

        depot = Depot(
            scenario=scenario, name="Test Depot", name_short="TD", station=stop_1
        )
        session.add(depot)

        # Create plan

        plan = Plan(scenario=scenario, name="Test Plan")
        session.add(plan)

        depot.default_plan = plan

        # Create area
        arrival_area = Area(
            scenario=scenario,
            name="Arrival",
            depot=depot,
            area_type=AreaType.DIRECT_ONESIDE,
            capacity=number_of_rotations + 2,
        )
        session.add(arrival_area)
        arrival_area.vehicle_type = vehicle_type

        cleaning_area = Area(
            scenario=scenario,
            name="Cleaning Area",
            depot=depot,
            area_type=AreaType.DIRECT_ONESIDE,
            capacity=1,
        )
        session.add(cleaning_area)
        cleaning_area.vehicle_type = vehicle_type

        charging_area = Area(
            scenario=scenario,
            name="Line Charging Area",
            depot=depot,
            area_type=AreaType.LINE,
            capacity=24,
            row_count=4,
        )
        session.add(charging_area)
        charging_area.vehicle_type = vehicle_type

        # Create processes
        standby_arrival = Process(
            name="Standby Arrival",
            scenario=scenario,
            dispatchable=False,
        )

        clean = Process(
            name="Arrival Cleaning",
            scenario=scenario,
            dispatchable=False,
            duration=timedelta(minutes=30),
        )

        charging = Process(
            name="Charging",
            scenario=scenario,
            dispatchable=False,
            electric_power=15,
        )

        standby_departure = Process(
            name="Standby Pre-departure",
            scenario=scenario,
            dispatchable=True,
        )

        session.add(standby_arrival)
        session.add(clean)
        session.add(charging)
        session.add(standby_departure)

        cleaning_area.processes.append(clean)
        arrival_area.processes.append(standby_arrival)
        charging_area.processes.append(charging)
        charging_area.processes.append(standby_departure)

        assocs = [
            AssocPlanProcess(
                scenario=scenario, process=standby_arrival, plan=plan, ordinal=0
            ),
            AssocPlanProcess(scenario=scenario, process=clean, plan=plan, ordinal=1),
            AssocPlanProcess(scenario=scenario, process=charging, plan=plan, ordinal=2),
            AssocPlanProcess(
                scenario=scenario, process=standby_departure, plan=plan, ordinal=3
            ),
        ]
        session.add_all(assocs)

        # We need to set the consumption values for all vehicle types to 1
        for vehicle_type in scenario.vehicle_types:
            vehicle_type.consumption = 1
        session.flush()

        session.commit()
        return scenario

    @pytest.fixture()
    def session(self):
        """
        Creates a session with the eflips-model schema.

        NOTE: THIS DELETE ALL DATA IN THE DATABASE
        :return: an SQLAlchemy Session with the eflips-model schema
        """
        url = os.environ["DATABASE_URL"]
        engine = create_engine(
            url, echo=False
        )  # Change echo to True to see SQL queries
        Base.metadata.drop_all(engine)
        eflips.model.setup_database(engine)
        session = Session(bind=engine)
        yield session
        session.close()

    @pytest.fixture()
    def optimizer(self, session, full_scenario):
        optimizer = DepotRotationOptimizer(session, full_scenario.id)

        return optimizer


class TestDepotRotationOptimizer(TestHelpers):
    def test_delete_original_data(self, session, full_scenario, optimizer):
        optimizer._delete_original_data()
        session.commit()

        assert (
            session.query(Trip)
            .filter(
                Trip.scenario_id == full_scenario.id, Trip.trip_type == TripType.EMPTY
            )
            .count()
            == 0
        )

    def test_get_depot_from_input(self, session, full_scenario, optimizer):
        # Giving station id
        user_input = [
            {"depot_station": 1, "capacity": 10, "vehicle_type": [1, 2]},
            {"depot_station": 100, "capacity": 10, "vehicle_type": [1, 2]},
            {
                "depot_station": (13.323828521189995, 52.517102453684146),
                "capacity": 10,
                "vehicle_type": [1, 2],
            },
            {"depot_station": 1, "capacity": 10, "vehicle_type": []},
            {"depot_station": 1, "capacity": 10, "vehicle_type": [100, 200]},
            {"depot_station": 1, "capacity": 10.5, "vehicle_type": [1, 2]},
        ]
        with pytest.raises(AssertionError):
            optimizer.get_depot_from_input(user_input)

        # Test if all depots fail to provide all demanded vehicle types
        # Adding a new vehicle type
        vehicle_type = VehicleType(
            id=5,
            scenario=full_scenario,
            name="Test Vehicle Type 3",
            battery_capacity=100,
            charging_curve=[[0, 150], [1, 150]],
            opportunity_charging_capable=True,
        )
        session.add(vehicle_type)
        rotation = session.query(Rotation).first()
        rotation.vehicle_type = vehicle_type

        session.flush()

        user_input = [
            {"depot_station": 1, "capacity": 10, "vehicle_type": [1, 2]},
            {
                "depot_station": (13.323828521189995, 52.517102453684146),
                "name": "new depot station",
                "capacity": 10,
                "vehicle_type": [2],
            },
        ]
        with pytest.raises(ValueError):
            optimizer.get_depot_from_input(user_input)

        session.rollback()

    @pytest.mark.skipif(
        "BASE_URL" not in os.environ,
        reason="Needs OpenRouteService API access (set BASE_URL in env variables)",
    )
    def test_data_preparation(self, session, full_scenario, optimizer):
        user_input_depot = [
            {"depot_station": 1, "capacity": 10, "vehicle_type": [1]},
            {
                "depot_station": (13.331493462156047, 52.50356808223075),
                "name": "new depot station",
                "capacity": 10,
                "vehicle_type": [2],
            },
        ]

        optimizer.get_depot_from_input(user_input_depot)
        optimizer.data_preparation()

        assert optimizer.data["depot"] is not None
        assert optimizer.data["depot"].shape[0] == len(user_input_depot)
        assert optimizer.data["vehicletype_depot"] is not None

        # TODO re-write this test
        # assert optimizer.data["vehicletype_depot"].shape == (
        #     session.query(func.count(VehicleType.id)).scalar(),
        #     len(user_input_depot),
        # )

        assert optimizer.data["vehicle_type"] is not None

        # TODO re-write this test
        # assert (
        #     optimizer.data["vehicle_type"].shape[0]
        #     == session.query(func.count(VehicleType.id)).scalar()
        # )
        assert optimizer.data["rotation"] is not None
        assert (
            optimizer.data["rotation"].shape[0]
            == session.query(func.count(Rotation.id)).scalar()
        )
        assert optimizer.data["occupancy"] is not None
        assert (
            optimizer.data["occupancy"].shape[0]
            == session.query(func.count(Rotation.id)).scalar()
        )
        assert optimizer.data["cost"] is not None
        assert optimizer.data["cost"].shape[0] == (
            session.query(func.count(Rotation.id)).scalar() * len(user_input_depot)
        )

    @pytest.mark.skipif(
        "BASE_URL" not in os.environ,
        reason="Needs OpenRouteService API access (set BASE_URL in env variables)",
    )
    def test_optimize(self, session, full_scenario, optimizer):
        user_input_depot = [
            {"depot_station": 1, "capacity": 10, "vehicle_type": [1]},
            {
                "depot_station": (13.332105437227769, 52.50929116968019),
                "name": "Station Hertzallee",
                "capacity": 10,
                "vehicle_type": [2],
            },
        ]

        optimizer.get_depot_from_input(user_input_depot)
        optimizer.data_preparation()

        optimizer.optimize()

        assert optimizer.data["result"] is not None
        assert optimizer.data["result"].shape[0] == optimizer.data["rotation"].shape[0]

        fig = optimizer.visualize()

        assert isinstance(fig, go.Figure)
        optimizer.write_optimization_results(delete_original_data=True)
        session.commit()

    @pytest.mark.skipif(
        "BASE_URL" not in os.environ,
        reason="Needs OpenRouteService API access (set BASE_URL in env variables)",
    )
    def test_optimize_rotations_at_depot_station(
        self, session, full_scenario, optimizer
    ):
        # Once the original empty trips are deleted, every rotation starts and
        # ends at "Adenauer Platz". A depot at that station must not get
        # zero-length ferry or return routes.
        depot_station = (
            session.query(Station)
            .filter(
                Station.scenario_id == full_scenario.id,
                Station.name_short == "AP",
            )
            .one()
        )
        user_input_depot = [
            {"depot_station": depot_station.id, "capacity": 10, "vehicle_type": [1, 2]},
        ]

        optimizer.get_depot_from_input(user_input_depot)
        optimizer.data_preparation()
        optimizer.optimize()
        optimizer.write_optimization_results(delete_original_data=True)
        session.commit()

        assert (
            session.query(Route)
            .filter(Route.departure_station_id == Route.arrival_station_id)
            .count()
            == 0
        )
        assert (
            session.query(Trip)
            .filter(
                Trip.scenario_id == full_scenario.id, Trip.trip_type == TripType.EMPTY
            )
            .count()
            == 0
        )

    @pytest.mark.skipif(
        "BASE_URL" not in os.environ,
        reason="Needs OpenRouteService API access (set BASE_URL in env variables)",
    )
    def test_optimize_with_infeasible_model(self, session, full_scenario, optimizer):
        user_input_depot = [
            {"depot_station": 1, "capacity": 1, "vehicle_type": [1]},
            {
                "depot_station": (13.332105437227769, 52.50929116968019),
                "name": "Station Hertzallee",
                "capacity": 1,
                "vehicle_type": [2],
            },
        ]

        optimizer.get_depot_from_input(user_input_depot)
        optimizer.data_preparation()

        with pytest.raises(ValueError):
            optimizer.optimize()

        session.rollback()
