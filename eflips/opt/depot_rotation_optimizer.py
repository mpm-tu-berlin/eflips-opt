from typing import Dict, Any, List
from geoalchemy2.shape import to_shape
import pandas as pd

import pyomo.environ as pyo
from pyomo.common.timing import report_timing

from matplotlib import pyplot as plt

from eflips.model import Rotation, Trip, TripType, Depot, Station, Route, VehicleType
from eflips.opt.data_preperation import (
    get_vehicletype,
    get_rotation,
    get_occupancy,
    deadhead_cost,
)


class DepotRotationOptimizer:
    def __init__(self, session, scenario_id):
        self.session = session
        self.scenario_id = scenario_id
        self.data = {}

    def delete_original_data(self):
        """
        Delete the original deadhead trips from the database, which are determined by the first and the last empty trips of each rotation

        :return:
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
            # - the station of the depot is the departure/arrival station of the route
            # - Meanwhile, delete the stoptimes of the trip

            if (
                first_trip is not None
                and first_trip.trip_type == TripType.EMPTY
                and self.session.query(Depot.station_id)
                .join(Station, Station.id == Depot.station_id)
                .join(Route, Route.departure_station_id == Station.id)
                .filter(Route.id == first_trip.route_id)
                .first()
                is not None
            ):
                trips_to_delete.append(first_trip)
                stoptimes_to_delete.extend(first_trip.stop_times)

            if (
                last_trip is not None
                and last_trip.trip_type == TripType.EMPTY
                and self.session.query(Depot.station_id)
                .join(Station, Station.id == Depot.station_id)
                .join(Route, Route.arrival_station_id == Station.id)
                .filter(Route.id == last_trip.route_id)
                .first()
            ):
                trips_to_delete.append(last_trip)
                stoptimes_to_delete.extend(last_trip.stop_times)

        # Delete those trips and stoptimes
        for stoptime in stoptimes_to_delete:
            self.session.delete(stoptime)
        for trip in trips_to_delete:
            self.session.delete(trip)

        self.session.flush()

    def get_depot_from_input(self, user_input_depot: List[Dict[str, Any]]):
        """
        #TODO redo this docstring
        Get the depot data from the user input, validate and store it in the data attribute.

        :param user_input_depot: A dictionary containing the user input for the depot data. It should include the
        following items:
        - station: The station bounded to the depot. It should either be an integer representing station id in the
        database, or a tuple of 2 floats representing the latitude and longitude of the station.
        - capacity: should be a positive integer representing the capacity of the depot.
        - vehicle_type: should be a list of integers representing the vehicle type id in the database.


        :return: Nothing. The data will be stored in the data attribute.
        """

        # Validate
        # - if the station exists when station id is given
        # - if the vehicle type exists when vehicle type id is given
        # - if the capacity is a positive integer

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

            else:
                raise ValueError(
                    "Station should be either an integer or a tuple of 2 floats"
                )

            # Get the capacity
            capacity = depot["capacity"]
            assert isinstance(capacity, int), "Capacity should be an integer"

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
                ), ("Vehicle type " "not found")

        # Store the data
        self.data["depot"] = user_input_depot

    def data_preparation(self):
        # depot
        depot_input = self.data["depot"]
        # station
        station_coords = []
        capacities = []

        for depot in depot_input:
            if isinstance(depot["depot_station"], int):
                point = to_shape(
                    self.session.query(Station.geom)
                    .filter(Station.id == depot["depot_station"])
                    .one()[0]
                )

                station_coords.append((point.x, point.y))
            else:
                station_coords.append(depot["depot_station"])

            capacities.append(depot["capacity"])

        depot_df = pd.DataFrame()
        depot_df["depot_id"] = list(range(len(depot_input)))
        depot_df["depot_station"] = station_coords
        depot_df["capacity"] = capacities
        # depot_df.set_index("depot_id", inplace=True)
        self.data["depot"] = depot_df

        # VehicleType-Depot availability
        total_vehicle_type = self.session.scalars(
            (
                self.session.query(VehicleType.id).filter(
                    VehicleType.scenario_id == self.scenario_id
                )
            )
        ).all()
        vehicletype_depot_df = pd.DataFrame(
            total_vehicle_type, columns=["vehicle_type_id"]
        )
        for i in range(len(depot_input)):
            vehicletype_depot_df[i] = [
                int(v in depot_input[i]["vehicle_type"]) for v in total_vehicle_type
            ]

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

        # Get time-wise occupancy of each rotation
        occupancy_df = get_occupancy(self.session, self.scenario_id)
        self.data["occupancy"] = occupancy_df

        # Generate cost table
        cost_df = rotation_df.merge(depot_df, how="cross")
        cost_df["cost"] = cost_df.apply(
            lambda x: deadhead_cost(x["start_station_coord"], x["depot_station"]),
            axis=1,
        )
        self.data["cost"] = cost_df

    def optimize(self, time_report=False):
        # Building model in pyomo
        # i for rotations
        I = self.data["rotation"]["rotation_id"].tolist()
        # j for depots
        J = self.data["depot"]["depot_id"].tolist()
        # t for vehicle types
        T = self.data["vehicle_type"]["vehicle_type_id"].tolist()
        # s for time slots
        S = self.data["occupancy"].columns.values.tolist()
        # S = [i[0] for i in S]

        # n_j: depot-vehicle type capacity
        n = self.data["depot"].set_index("depot_id").to_dict()["capacity"]

        # a_jt: depot-vehicle type availability
        a = self.data["vehicletype_depot"].to_dict()

        # v_it: rotation-type
        def v(i, t):
            return (
                1
                if self.data["rotation"]
                .loc[self.data["rotation"]["rotation_id"] == i, "vehicle_type_id"]
                .values
                == t
                else 0
            )

        # c_ij: rotation-depot cost
        c = self.data["cost"].set_index(["rotation_id", "depot_id"]).to_dict()["cost"]

        # o_si: rotation-time slot occupancy
        o = self.data["occupancy"].to_dict()

        # f_t: vehicle type factor
        f = (
            self.data["vehicle_type"]
            .set_index("vehicle_type_id")
            .to_dict()["size_factor"]
        )

        print("data acquired")

        # Set up pyomo model
        if time_report is True:
            report_timing()

        model = pyo.ConcreteModel(name="depot_rot_problem")
        model.x = pyo.Var(I, J, domain=pyo.Binary)

        # Objective function
        @model.Objective()
        def obj(m):
            return sum(c[i, j]["distance"] * model.x[i, j] for i in I for j in J)

        # Constraints
        # Each rotation is assigned to exactly one depot
        @model.Constraint(I)
        def one_depot_per_rot(m, i):
            return sum(model.x[i, j] for j in J) == 1

        # Depot capacity constraint
        @model.Constraint(J, S)
        def depot_capacity_constraint(m, j, s):
            return (
                sum(
                    sum(o[s][i] * v(i, t) * model.x[i, j] for i in I) * f[t] for t in T
                )  # TODO f[t,j] = 0 if depot j has
                # no place for type t
                <= n[j]
            )

        @model.Constraint(I, J, T)
        def vehicle_type_depot_availability(m, i, j, t):
            return model.x[i, j] * v(i, t) * a[j][t] == 1

        # Solve

        result = pyo.SolverFactory("gurobi").solve(model, tee=True)
        # model.pprint()

        new_assign = pd.DataFrame(
            {
                "rotation_id": [i[0] for i in model.x if model.x[i].value == 1.0],
                "depot_id": [i[1] for i in model.x if model.x[i].value == 1.0],
                "assignment": [
                    model.x[i].value for i in model.x if model.x[i].value == 1.0
                ],
            }
        )

        self.data["result"] = new_assign

    def write_optimization_results(self):
        raise NotImplementedError

    def visualize(self):
        new_assign = self.data["result"]
        cost = self.data["cost"]
        new_assign["cost"] = new_assign.apply(
            lambda x: cost.loc[x["rotation_id"], x["depot_id"]]["cost"]["distance"],
            axis=1,
        )
        # orig_assign["cost"] = orig_assign.apply(
        #     lambda x: cost.loc[x["rotation_id"], x["depot_id"]]["distance"], axis=1
        # )

        # Save results

        # Plotting
        # orig_cost = orig_assign["cost"]
        new_cost = new_assign["cost"]
        # orig_cost_mean = orig_cost.mean()
        new_cost_mean = new_cost.mean()

        # plt.hist(orig_cost, alpha=0.5)
        plt.hist(new_cost, alpha=0.5)
        plt.legend(["Original", "New"])

        min_ylim, max_ylim = plt.ylim()
        # plt.axvline(orig_cost_mean, color="k", linestyle="dashed", linewidth=1)
        plt.text(
            # orig_cost_mean * 1.1, max_ylim * 0.9, "Old_Mean: {:.2f}".format(orig_cost_mean)
        )
        plt.axvline(
            new_assign["cost"].mean(), color="b", linestyle="dashed", linewidth=1
        )
        plt.text(
            new_cost_mean * 1.1,
            max_ylim * 0.1,
            "New_Mean: {:.2f}".format(new_cost_mean),
        )

        plt.show()
        plt.savefig("cost_comparison_distance.png")
