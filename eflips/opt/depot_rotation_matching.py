import asyncio
import itertools
import os
import warnings
from datetime import timedelta

import openrouteservice  # type: ignore

from typing import Dict, Any, List, Tuple

import sqlalchemy.orm.session
from geoalchemy2.shape import to_shape
import pandas as pd

import pyomo.environ as pyo  # type: ignore
from pyomo.common.timing import report_timing  # type: ignore

import plotly.graph_objects as go  # type: ignore

from eflips.model import (
    Rotation,
    Trip,
    TripType,
    Station,
    Route,
    VehicleType,
    StopTime,
    AssocRouteStation,
    Event,
    Vehicle,
)

from eflips.opt.util import (
    get_vehicletype,
    get_rotation,
    get_occupancy,
    get_rotation_vehicle_assign,
    get_depot_rot_assign,
    calculate_deadhead_costs,
)


class DepotRotationOptimizer:
    def __init__(
        self, session: sqlalchemy.orm.session.Session, scenario_id: int
    ) -> None:
        self.session = session
        self.scenario_id = scenario_id
        self.data: Dict[
            str, List[Dict[str, int | List[int] | Tuple[float, float]]] | pd.DataFrame
        ] = {}

    def _delete_original_data(self) -> None:
        """
        Delete the original deadhead trips from the database, which are determined by the first and the last empty
        trips of each rotation. It is called by :meth:`write_optimization_results` method and must be executed before
        new results are written to the database.

        :return: Nothing. The original data will be deleted from the database.
        """

        # Get the rotations
        rotations = (
            self.session.query(Rotation)
            .filter(Rotation.scenario_id == self.scenario_id)
            .all()
        )

        # Get the first and the last empty trips of each rotation
        trips_to_delete = []
        stoptimes_to_delete = []
        for rotation in rotations:
            first_trip = (
                self.session.query(Trip)
                .filter(Trip.rotation_id == rotation.id)
                .order_by(Trip.departure_time)
                .first()
            )
            last_trip = (
                self.session.query(Trip)
                .filter(Trip.rotation_id == rotation.id)
                .order_by(Trip.arrival_time.desc())
                .first()
            )
            # Delete the trip if:
            # - it is the first/last trip of the rotation
            # - it has the type of TripType.EMPTY
            # - Meanwhile, delete the stoptimes of the trip

            if first_trip is not None and first_trip.trip_type == TripType.EMPTY:
                trips_to_delete.append(first_trip)
                stoptimes_to_delete.extend(first_trip.stop_times)

            if last_trip is not None and last_trip.trip_type == TripType.EMPTY:
                trips_to_delete.append(last_trip)
                stoptimes_to_delete.extend(last_trip.stop_times)

        # Delete those trips and stoptimes
        for stoptime in stoptimes_to_delete:
            self.session.delete(stoptime)
        for trip in trips_to_delete:
            self.session.delete(trip)

        self.session.flush()

    def get_depot_from_input(
        self, user_input_depot: List[Dict[str, int | List[int] | Tuple[float, float]]]
    ) -> None:
        """

        Get the depot data from the user input, validate and store it in the data attribute.

        :param user_input_depot: A dictionary containing the user input for the depot data. It should include the
        following items:
        - station: The station bounded to the depot. It should either be an integer representing station id in the
        database, or a tuple of 2 floats representing the latitude and longitude of the station.
        - capacity: should be a positive integer representing the capacity of the depot.
        - vehicle_type: should be a list of integers representing the vehicle type id in the database.
        - name: should be provided if the station is not in the database.


        :return: Nothing. The data will be stored in the data attribute.
        """

        # Validate
        # - if the station exists when station id is given
        # - if the vehicle type exists when vehicle type id is given
        # - if the capacity is a positive integer
        # - if the vehicle type in the rotations are available in all the depots

        all_vehicle_types = []
        # Get the station
        for depot in user_input_depot:
            station = depot["depot_station"]
            if isinstance(station, int):
                assert (
                    self.session.query(Station).filter(Station.id == station).first()
                    is not None
                ), "Station not found"

            elif isinstance(station, tuple):
                assert len(station) == 2, "Station should be a tuple of 2 floats"
                assert all(
                    isinstance(coord, float) for coord in station
                ), "Station should be a tuple of 2 floats"

                assert "name" in depot, (
                    "Name of the depot should be provided if it's not a station in the "
                    "database"
                )

            else:
                raise ValueError(
                    "Station should be either an integer or a tuple of 2 floats"
                )

            # Get the vehicle type
            vehicle_type = depot["vehicle_type"]
            assert isinstance(
                vehicle_type, list
            ), "Vehicle type should be a list of integers"
            assert len(vehicle_type) > 0, "Vehicle type should not be empty"

            for vt in vehicle_type:
                assert (
                    self.session.query(VehicleType).filter(VehicleType.id == vt).first()
                    is not None
                ), f"Vehicle type {vt} not found"

                all_vehicle_types.append(vt)

            # Get the capacity
            capacity = depot["capacity"]
            assert (
                isinstance(capacity, int) and capacity >= 0
            ), "Capacity should be a non-negative integer"
        # Store the data

        # Check if the vehicle types in the rotations are available in all the depots

        all_vehicle_types = list(set(all_vehicle_types))
        all_vehicle_types.sort()
        all_demanded_types = (
            self.session.query(Rotation.vehicle_type_id)
            .filter(Rotation.scenario_id == self.scenario_id)
            .distinct(Rotation.vehicle_type_id)
            .order_by(Rotation.vehicle_type_id)
            .all()
        )

        for vt in [vt.vehicle_type_id for vt in all_demanded_types]:
            if vt not in all_vehicle_types:
                raise ValueError(
                    "Not all demanded vehicle types are available in all depots"
                )

        self.data["depot_from_user"] = user_input_depot

    def data_preparation(self) -> None:
        """
        Prepare the data for the optimization problem and store them into self.data. All the data are in :class:`pandas.DataFrame` format.
        The data includes:
        - depot: depot id and station coordinates
        - vehicletype_depot: availability of vehicle types in depots
        - vehicle_type: vehicle type size factors
        - orig_assign: original depot rotation assignment
        - rotation: start and end station of each rotation
        - assignment: assignment between vehicle type and rotation
        - occupancy: time-wise occupancy of each rotation
        - cost: cost table of each rotation and depot

        :return: Nothing. The data will be stored in the data attribute.
        """

        # depot
        depot_input = self.data["depot_from_user"]
        assert isinstance(depot_input, list), "Depot input should be a list"
        # station
        station_coords = []
        capacities = []
        names = []

        for depot in depot_input:
            if isinstance(depot["depot_station"], int):
                point = to_shape(
                    self.session.query(Station.geom)
                    .filter(Station.id == depot["depot_station"])
                    .one()[0]
                )

                names.append(
                    self.session.query(Station.name)
                    .filter(Station.id == depot["depot_station"])
                    .one()[0]
                )

                station_coords.append((point.x, point.y))
            else:
                assert (
                    isinstance(depot["depot_station"], tuple)
                    and len(depot["depot_station"]) == 2
                    and all(
                        isinstance(coord, float) for coord in depot["depot_station"]
                    )
                ), "Depot station should be a tuple of 2 floats"
                station_coords.append(depot["depot_station"])
                names.append(depot["name"])

            capacities.append(depot["capacity"])

        depot_df = pd.DataFrame()
        depot_df["depot_id"] = list(range(len(depot_input)))
        depot_df["depot_station"] = station_coords
        depot_df["name"] = names
        depot_df["capacity"] = capacities
        self.data["depot"] = depot_df

        # Get original depot rotation assignment
        orig_assign = get_depot_rot_assign(self.session, self.scenario_id)
        self.data["orig_assign"] = orig_assign

        # VehicleType-Depot availability
        total_vehicle_type = self.session.scalars(
            (
                self.session.query(Rotation.vehicle_type_id)
                .distinct(Rotation.vehicle_type_id)
                .filter(Rotation.scenario_id == self.scenario_id)
            )
        ).all()
        vehicletype_depot_df = pd.DataFrame(
            total_vehicle_type, columns=["vehicle_type_id"]
        )
        for i in range(len(depot_input)):
            vehicle_type_factors = []
            for v in total_vehicle_type:
                assert isinstance(
                    depot_input[i]["vehicle_type"], list
                ), "Vehicle type should be a list of integers"
                assert all(isinstance(vt, int) for vt in depot_input[i]["vehicle_type"]), "Vehicle type should be a list of integers"  # type: ignore
                if v in depot_input[i]["vehicle_type"]:  # type: ignore
                    vehicle_type = (
                        self.session.query(VehicleType)
                        .filter(VehicleType.id == v)
                        .first()
                    )
                    assert vehicle_type is not None, f"Vehicle type {v} not found"
                    if vehicle_type.length is None:
                        vehicle_type_factors.append(1.0)
                    else:
                        vehicle_type_factors.append(vehicle_type.length / 12.0)

                else:
                    vehicle_type_factors.append(depot_df.iloc[i]["capacity"])

            vehicletype_depot_df[i] = vehicle_type_factors

        # TODO where to set index?
        vehicletype_depot_df.set_index("vehicle_type_id", inplace=True)
        self.data["vehicletype_depot"] = vehicletype_depot_df

        # Vehicle type size factors
        # How many vehicle types are there and match them to factors and depots
        vehicle_type_df = get_vehicletype(self.session, self.scenario_id)
        self.data["vehicle_type"] = vehicle_type_df

        # Rotation related data
        # Get the start and end station of each rotation
        rotation_df = get_rotation(self.session, self.scenario_id)
        self.data["rotation"] = rotation_df

        # Get the assignment between vehicle type and rotation
        assignment = get_rotation_vehicle_assign(self.session, self.scenario_id)
        self.data["assignment"] = assignment

        # Get time-wise occupancy of each rotation
        occupancy_df = get_occupancy(self.session, self.scenario_id)
        self.data["occupancy"] = occupancy_df

        # Generate cost table
        cost_df = rotation_df.merge(depot_df, how="cross")

        base_url = os.environ["BASE_URL"]

        if base_url is None:
            raise ValueError("BASE_URL is not set")

        client = openrouteservice.Client(base_url=base_url)

        # Run the async function
        deadhead_costs = asyncio.run(calculate_deadhead_costs(cost_df, client))

        cost_df["cost"] = deadhead_costs
        self.data["cost"] = cost_df

    def optimize(
        self, cost: str = "distance", time_report: bool = False, solver: str = "gurobi"
    ) -> None:
        """
        Optimize the depot rotation assignment problem and store the results in the data attribute.
        :param cost: the cost to be optimized. It can be either "distance" or "duration" for now with the default value of "distance".
        :param time_report: if set to True, the time report of the optimization will be printed.
        :param solver: the solver to be used for the optimization. The default value is "gurobi". In order to use it, a valid license
        should be available.

        :return: Nothing. The results will be stored in the data attribute.
        """
        # Building model in pyomo
        # i for rotations
        I = self.data["rotation"]["rotation_id"].tolist()  # type: ignore
        # j for depots
        J = self.data["depot"]["depot_id"].tolist()  # type: ignore
        # t for vehicle types
        T = self.data["vehicle_type"]["vehicle_type_id"].tolist()  # type: ignore
        # s for time slots
        S = self.data["occupancy"].columns.values.tolist()  # type: ignore
        S = [int(i) for i in S]

        # n_j: depot-vehicle type capacity
        depot = self.data["depot"]
        assert isinstance(depot, pd.DataFrame), "Depot data should be a DataFrame"
        n = depot.set_index("depot_id").to_dict()["capacity"]

        # f_jt: vehicle size factor for each depot. if a vehicle type is unavailable in a depot, the factor will be
        # the same as the depot capacity
        assert isinstance(
            self.data["vehicletype_depot"], pd.DataFrame
        ), "Vehicle type depot data should be a DataFrame"
        f = self.data["vehicletype_depot"].to_dict()

        # v_it: rotation-type
        assert isinstance(
            self.data["assignment"], pd.DataFrame
        ), "Assignment data should be a DataFrame"
        v = (
            self.data["assignment"]
            .set_index(["rotation_id", "vehicle_type_id"])
            .to_dict()["assignment"]
        )

        # c_ij: rotation-depot cost
        assert isinstance(
            self.data["cost"], pd.DataFrame
        ), "Cost data should be a DataFrame"
        c = self.data["cost"].set_index(["rotation_id", "depot_id"]).to_dict()["cost"]

        # o_si: rotation-time slot occupancy
        assert isinstance(
            self.data["occupancy"], pd.DataFrame
        ), "Occupancy data should be a DataFrame"
        o = self.data["occupancy"].to_dict()
        o = {int(k): v for k, v in o.items()}  # type: ignore

        print("data acquired")

        # Set up pyomo model
        if time_report is True:
            report_timing()

        model = pyo.ConcreteModel(name="depot_rot_problem")
        model.x = pyo.Var(I, J, domain=pyo.Binary)

        # Objective function
        @model.Objective()
        def obj(m):  # type: ignore
            return sum(
                (c[i, j][cost][0] + c[i, j][cost][1]) * model.x[i, j]
                for i in I
                for j in J
            )

        # Constraints
        # Each rotation is assigned to exactly one depot
        @model.Constraint(I)
        def one_depot_per_rot(m, i):  # type: ignore
            return sum(model.x[i, j] for j in J) == 1

        # Depot capacity constraint
        @model.Constraint(J, S)
        def depot_capacity_constraint(m, j, s):  # type: ignore
            occupancy_of_depot = 0
            for t in T:
                occupancy_for_type = 0
                for i in I:
                    vehicle_is_present = o[s][i] * v[i, t] > 0
                    if vehicle_is_present:
                        occupancy_for_type += o[s][i] * v[i, t] * model.x[i, j]
                occupancy_of_depot += occupancy_for_type * f[j][t]
            return occupancy_of_depot <= n[j]

        # Solve
        result = pyo.SolverFactory(solver).solve(model, tee=True)
        if result.solver.termination_condition == pyo.TerminationCondition.infeasible:
            raise ValueError(
                "No feasible solution found. Please check your constraints."
            )

        new_assign = pd.DataFrame(
            {
                "rotation_id": [i[0] for i in model.x if model.x[i].value == 1.0],
                "new_depot_id": [i[1] for i in model.x if model.x[i].value == 1.0],
                "assignment": [
                    model.x[i].value for i in model.x if model.x[i].value == 1.0
                ],
            }
        )

        self.data["result"] = new_assign

        # TODO for validation
        new_assign.to_csv("new_assign.csv")

    def write_optimization_results(self, delete_original_data: bool = False) -> None:

        if "result" not in self.data:
            raise ValueError("No feasible solution found")

        if delete_original_data is False:
            raise ValueError(
                "Original data should be deleted in order to write the results to the database."
            )
        else:
            self._delete_original_data()

        # delete all events from the database

        rotation_q = self.session.query(Rotation).filter(
            Rotation.scenario_id == self.scenario_id
        )
        rotation_q.update({"vehicle_id": None})
        self.session.query(Event).filter(Event.scenario_id == self.scenario_id).delete()
        self.session.query(Vehicle).filter(
            Vehicle.scenario_id == self.scenario_id
        ).delete()
        self.session.flush()

        # Write new depot as stations
        depot_from_user = self.data["depot_from_user"]
        assert isinstance(depot_from_user, list), "Depot data should be a list"
        for depot in depot_from_user:
            if isinstance(depot["depot_station"], tuple):
                new_depot_station = Station(
                    name=depot["name"],
                    scenario_id=self.scenario_id,
                    geom=f"POINT({depot['depot_station'][0]} {depot['depot_station'][1]} 0)",
                    is_electrified=False,  # TODO Hardcoded for now
                )
                self.session.add(new_depot_station)
        self.session.flush()

        new_assign = self.data["result"]
        cost = self.data["cost"]

        assert isinstance(new_assign, pd.DataFrame), "Result data should be a DataFrame"
        for row in new_assign.itertuples():

            # Add depot if it is a new depot, else get the depot station id
            if isinstance(depot_from_user[row.new_depot_id]["depot_station"], Tuple):  # type: ignore
                # newly added depot
                depot_name = depot_from_user[row.new_depot_id]["name"]  # type: ignore
                depot_station = (
                    self.session.query(Station)
                    .filter(Station.name == depot_name)
                    .filter(Station.scenario_id == self.scenario_id)
                    .one()
                )
            else:
                depot_station_id = depot_from_user[row.new_depot_id]["depot_station"]  # type: ignore
                depot_station = (
                    self.session.query(Station)
                    .filter(Station.id == depot_station_id)
                    .one()
                )
                assert depot_station is not None, "Depot station not found"
                depot_name = depot_station.name  # type: ignore

            assert isinstance(cost, pd.DataFrame), "Cost data should be a DataFrame"
            route_cost = cost.loc[
                (cost["rotation_id"] == row.rotation_id)
                & (cost["depot_id"] == row.new_depot_id)
            ]["cost"].iloc[0]
            ferry_route_distance = route_cost["distance"][0]
            return_route_distance = route_cost["distance"][1]
            ferry_route_duration = route_cost["duration"][0]
            return_route_duration = route_cost["duration"][1]

            trips = (
                self.session.query(Trip)
                .filter(Trip.rotation_id == row.rotation_id)
                .order_by(Trip.departure_time)
                .all()
            )
            first_trip = trips[0]

            # Ferry-route
            ferry_route = (
                self.session.query(Route)
                .filter(
                    Route.departure_station_id == depot_station.id,
                    Route.arrival_station_id == first_trip.route.departure_station_id,
                )
                .all()
            )
            if len(ferry_route) == 0:
                # There is no such route, create a new one
                new_ferry_route = Route(
                    departure_station=depot_station,
                    arrival_station=first_trip.route.departure_station,
                    line_id=first_trip.route.line_id,
                    scenario_id=self.scenario_id,
                    distance=(
                        ferry_route_distance if ferry_route_distance > 100 else 100
                    ),
                    # assume minimum distance is 100m
                    name="Einsetzfahrt "
                    + str(depot_name)
                    + " "
                    + str(first_trip.route.departure_station.name),
                )

                assoc_ferry_station = [
                    AssocRouteStation(
                        scenario_id=self.scenario_id,
                        station=depot_station,
                        route=new_ferry_route,
                        elapsed_distance=0,
                    ),
                    AssocRouteStation(
                        scenario_id=self.scenario_id,
                        station=first_trip.route.departure_station,
                        route=new_ferry_route,
                        elapsed_distance=(
                            ferry_route_distance if ferry_route_distance > 100 else 100
                        ),
                    ),
                ]
                new_ferry_route.assoc_route_stations = assoc_ferry_station
                self.session.add(new_ferry_route)

            else:
                # There is such a route
                new_ferry_route = ferry_route[0]

            # Add ferry trip
            new_ferry_trip = Trip(
                scenario_id=self.scenario_id,
                route=new_ferry_route,
                rotation_id=row.rotation_id,
                trip_type=TripType.EMPTY,
                departure_time=first_trip.departure_time
                - timedelta(
                    seconds=ferry_route_duration if ferry_route_duration > 60 else 60
                ),  #
                # minimum duration is 60s
                arrival_time=first_trip.departure_time,
            )

            # Add stop times
            ferry_stop_times = [
                StopTime(
                    scenario_id=self.scenario_id,
                    trip=new_ferry_trip,
                    station=depot_station,
                    arrival_time=new_ferry_trip.departure_time,
                    dwell_duration=timedelta(seconds=0),
                ),
                StopTime(
                    scenario_id=self.scenario_id,
                    trip=new_ferry_trip,
                    station=first_trip.route.departure_station,
                    arrival_time=new_ferry_trip.arrival_time,
                    dwell_duration=timedelta(seconds=0),
                ),
            ]
            self.session.add_all(ferry_stop_times)
            new_ferry_trip.stop_times = ferry_stop_times
            self.session.add(new_ferry_trip)

            # Return-route
            last_trip = trips[-1]
            return_route = (
                self.session.query(Route)
                .filter(
                    Route.departure_station_id == last_trip.route.arrival_station_id,
                    Route.arrival_station_id == depot_station.id,
                )
                .all()
            )
            if len(return_route) == 0:
                new_return_route = Route(
                    departure_station=last_trip.route.arrival_station,
                    arrival_station=depot_station,
                    line_id=first_trip.route.line_id,
                    scenario_id=self.scenario_id,
                    distance=(
                        return_route_distance if return_route_distance > 100 else 100
                    ),
                    name="Aussetzfahrt "
                    + str(last_trip.route.arrival_station.name)
                    + " "
                    + str(depot_name),
                )
                assoc_return_station = [
                    AssocRouteStation(
                        scenario_id=self.scenario_id,
                        station=depot_station,
                        route=new_return_route,
                        elapsed_distance=(
                            return_route_distance
                            if return_route_distance > 100
                            else 100
                        ),
                    ),
                    AssocRouteStation(
                        scenario_id=self.scenario_id,
                        station=last_trip.route.arrival_station,
                        route=new_return_route,
                        elapsed_distance=0,
                    ),
                ]
                new_return_route.assoc_route_stations = assoc_return_station
                self.session.add(new_return_route)

            else:
                new_return_route = return_route[0]

            # Add return trip
            new_return_trip = Trip(
                scenario_id=self.scenario_id,
                route=new_return_route,
                rotation_id=row.rotation_id,
                trip_type=TripType.EMPTY,
                departure_time=last_trip.arrival_time,
                arrival_time=last_trip.arrival_time
                + timedelta(
                    seconds=return_route_duration if return_route_duration > 60 else 60
                ),
            )

            # Add stop times
            return_stop_times = [
                StopTime(
                    scenario_id=self.scenario_id,
                    trip=new_return_trip,
                    station=depot_station,
                    arrival_time=new_return_trip.arrival_time,
                    dwell_duration=timedelta(seconds=0),
                ),
                StopTime(
                    scenario_id=self.scenario_id,
                    trip=new_return_trip,
                    station=last_trip.route.arrival_station,
                    arrival_time=new_return_trip.departure_time,
                    dwell_duration=timedelta(seconds=0),
                ),
            ]
            self.session.add_all(return_stop_times)
            new_return_trip.stop_times = return_stop_times
            self.session.add(new_return_trip)

    def visualize(self) -> go.Figure:
        """
        Visualize the changes of the depot-rotation assignment in a Sankey diagram.
        :return: A :class:`plotly.graph_objects.Figure` object.
        """
        if "result" not in self.data:
            raise ValueError("No feasible solution found")

        new_assign = self.data["result"]

        depot_df = self.data["depot"]
        orig_assign = self.data["orig_assign"]

        orig_depot_stations = list(set(orig_assign["orig_depot_station"].tolist()))  # type: ignore
        old_depot_names = []
        for orig_depot_station in orig_depot_stations:
            depot_station_q = (
                self.session.query(Station.name)
                .filter(Station.id == orig_depot_station)
                .first()
            )
            assert depot_station_q is not None, "Depot station not found"
            depot_station_name = depot_station_q[0]
            old_depot_names.append("From " + depot_station_name)

        new_depot_ids = depot_df["depot_id"].tolist()  # type: ignore
        new_depot_names = depot_df["name"].tolist()  # type: ignore

        source = []
        target = []
        value = []

        assert isinstance(
            orig_assign, pd.DataFrame
        ), "Original data should be a DataFrame"
        assert isinstance(new_assign, pd.DataFrame), "Result data should be a DataFrame"
        diff = orig_assign.merge(new_assign, how="outer", on="rotation_id")

        for key in itertools.product(orig_depot_stations, new_depot_ids):
            source.append(orig_depot_stations.index(key[0]))
            target.append(new_depot_ids.index(key[1]) + len(orig_depot_stations))
            value.append(
                diff.loc[
                    (diff["orig_depot_station"] == key[0])
                    & (diff["new_depot_id"] == key[1])
                ].shape[0]
            )

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=old_depot_names + new_depot_names,
                        color="blue",
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=value,
                    ),
                )
            ]
        )

        fig.update_layout(
            title_text="Changes of Assignment of Depot-Rotation", font_size=10
        )
        fig.show()
        return fig
