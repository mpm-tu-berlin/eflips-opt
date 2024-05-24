from eflips.opt.base import AbstractOptimizer

from eflips.model import Rotation, Trip, TripType, Depot, Station, Route
from typing import Dict, Any


class DepotRotationOptimizer(AbstractOptimizer):
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

            if (
                last_trip is not None
                and last_trip.trip_type == TripType.EMPTY
                and self.session.query(Depot.station_id)
                .join(Station, Station.id == Depot.station_id)
                .join(Route, Route.arrival_station_id == Station.id)
                .filter(Route.id == first_trip.route_id)
                .first()
            ):
                trips_to_delete.append(last_trip)

        # Delete those trips
        for trip in trips_to_delete:
            self.session.delete(trip)

        # delete stoptime
        # delete route



        self.session.flush()

    def get_user_inputs(self, user_input_depot: Dict[str, Any]):
        """
        Get the new data for the optimization problem from the user. They will not be stored in the database yet.

        :param user_input_depot: A dictionary containing the user input for the depot data. It should include the following items:
        - station: The station bounded to the depot. It should either be an integer representing

        :return:
        """

        pass

    def data_preparation(self):
        pass

    def optimize(self):
        pass

    def write_optimization_results(self):
        pass

    def visualize(self):
        pass
