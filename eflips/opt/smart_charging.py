import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from typing import List, Tuple, Dict

import numpy as np
import numpy.typing as npt
import pyomo.environ as pyo  # type: ignore
import sqlalchemy.orm.session
from eflips.model import Depot
from eflips.model import Event, EventType, Area

TIME_STEP_DURATION = timedelta(minutes=5)  # Maybe change this to 1 minute`TODO
POWER_QUANTIZATION = 10  # kW
ENERGY_PER_PACKET = (
    TIME_STEP_DURATION.total_seconds() / 3600
) * POWER_QUANTIZATION  # kWh


def max_charging_power_for_event(event: Event) -> float:
    """
    Find the maximum charging power for an event.

    This is the minimum of the vehicle's maximum charging power and the
    maximum power draw at the depot
    :param event: An event
    :return: A float representing the maximum charging power in kW
    """

    assert event.event_type == EventType.CHARGING_DEPOT

    # For now, we do not support charging curves
    charging_curve = event.vehicle_type.charging_curve
    # The second entry of each tuple in the charging curve must be the same
    all_powers = [p[1] for p in charging_curve]
    assert len(set(all_powers)) == 1, "Charging curve must have a constant power draw"
    vehicle_max_power = all_powers[0]

    charging_process = [
        p
        for p in event.area.processes
        if p.electric_power is not None and p.duration is None
    ]
    if len(charging_process) != 1:
        raise ValueError("Area must have a process with electric power and no duration")

    power_at_depot = charging_process[0].electric_power

    return min(vehicle_max_power, power_at_depot)


def event_has_space_for_smart_charging(event: Event) -> bool:
    """
    Check if an event has space for smart charging.

    It needs to - if charged with full power - have at least
    2*TIME_STEP time left to charge

    :param event: The event to check
    :return: Whether the event has space for smart charging
    """

    duration = event.time_end - event.time_start
    energy_transferred = event.vehicle.vehicle_type.battery_capacity * (
        event.soc_end - event.soc_start
    )  # kWh

    max_power = max_charging_power_for_event(event)  # kW

    duration_at_max_power = timedelta(
        seconds=3600 * (energy_transferred / max_power)
    )  # seconds
    slack_duration = duration - duration_at_max_power

    return slack_duration >= 2 * TIME_STEP_DURATION


@dataclass
class SmartChargingEvent:
    original_event: Event
    """The original event that is being optimized."""

    vehicle_present: npt.NDArray[np.bool]
    """Array of booleans indicating whether a vehicle is present at each time step."""

    energy_packets_needed: int
    """The number of energy packets needed to transfer the energy."""

    energy_packets_per_time_step: int
    """How many energy packets can be transferred per time step (quantized max power)."""

    energy_packets_transferred: npt.NDArray[np.int64]
    """The number of energy packets transferred at each time step (this is the result)."""

    @classmethod
    def from_event(
        cls, event: Event, time_step_starts: List[datetime]
    ) -> "SmartChargingEvent":
        """
        Create a SmartChargingEvent from an event.

        :param event: The event to create the SmartChargingEvent from
        :param time_step_starts: An Array of the start times of the time steps (sorted)
                                 This will be used to discretize the event
        :return: A SmartChargingEvent
        """
        logger = logging.getLogger(__name__)

        # Find the Number of the first discrete time step after the event starts
        start_idx = [
            i for i, time in enumerate(time_step_starts) if time >= event.time_start
        ][0]

        # Find the number of the last discrete time step that still fully covers the event
        end_idx = [
            i for i, time in enumerate(time_step_starts) if time <= event.time_end
        ][-2]

        # Create the vehicle_present array
        vehicle_present = np.zeros(len(time_step_starts), dtype=bool)
        vehicle_present[start_idx : end_idx + 1] = True

        # Calculate the energy packets needed
        energy_transferred = event.vehicle.vehicle_type.battery_capacity * (
            event.soc_end - event.soc_start
        )
        energy_packets_needed = int(
            np.floor(energy_transferred / ENERGY_PER_PACKET)
        )  # Rounded up, wo we may have to increase the energy trasferred later

        # Calculate the energy packets per time step
        max_power = max_charging_power_for_event(event)
        energy_packets_per_time_step = max_power / POWER_QUANTIZATION
        assert energy_packets_per_time_step == int(
            energy_packets_per_time_step
        ), "Max power must be a multiple of the power quantization"
        energy_packets_per_time_step = int(energy_packets_per_time_step)

        # Sanity check: The energy packets per time step must be at least 1
        assert (
            energy_packets_per_time_step >= 1
        ), "Energy packets per time step must be at least 1"

        # Sanity check: The energy packets needed must be <= number of time steps * energy packets per time step
        if sum(vehicle_present) * energy_packets_per_time_step < energy_packets_needed:
            logger.warning(
                f"Energy packets needed ({energy_packets_needed}) has no flexibility. Scaling down."
            )
            energy_packets_needed = sum(vehicle_present) * energy_packets_per_time_step

        return cls(
            original_event=event,
            vehicle_present=vehicle_present,
            energy_packets_needed=energy_packets_needed,
            energy_packets_per_time_step=energy_packets_per_time_step,
            energy_packets_transferred=np.zeros(len(time_step_starts), dtype=int),
        )

    def update_original_event(self, time_step_starts: List[datetime]) -> None:
        """
        Update the original event's timeseries with the optimized charging schedule.

        This method creates a timeseries for the original Event showing how the
        state of charge changes over time based on the optimized charging schedule.

        Args:
            time_step_starts: Array of times at which each time step begins
        """
        event = self.original_event

        # Create the error of times and powers
        time_step_starts_unix = np.array([t.timestamp() for t in time_step_starts])
        times = time_step_starts_unix[np.where(self.vehicle_present)[0]]
        powers = (
            self.energy_packets_transferred[np.where(self.vehicle_present)[0]]
            * POWER_QUANTIZATION
        )
        energies = np.cumsum(powers) * TIME_STEP_DURATION.total_seconds() / 3600
        # Prepend 0, since the energy is 0 at the beginning
        energies = np.insert(energies, 0, 0)
        # Append one more value to the times, since we are integrating over the time steps
        times = np.append(times, times[-1] + TIME_STEP_DURATION.total_seconds())

        # Scale to socs
        socs = energies / event.vehicle.vehicle_type.battery_capacity

        delta_soc_from_event = event.soc_end - event.soc_start
        delta_soc_from_optimization = socs[-1] - socs[0]

        # Scale down
        assert delta_soc_from_optimization <= delta_soc_from_event
        scale_factor = delta_soc_from_event / delta_soc_from_optimization
        socs *= scale_factor

        # Add the initial SoC
        socs += event.soc_start

        # Insert the start and end times
        times = np.insert(times, 0, event.time_start.timestamp())
        times = np.append(times, event.time_end.timestamp())

        socs = np.insert(socs, 0, event.soc_start)
        socs = np.append(socs, event.soc_end)

        # Make sure socs is a list of floats
        socs_list = list(socs)
        socs_float_list = [float(s) for s in socs]

        # Create timeseries dictionary and update event
        tz = time_step_starts[0].tzinfo
        times_datetime = [datetime.fromtimestamp(t, tz) for t in times]
        event.timeseries = None  # type: ignore
        event.timeseries = {
            "time": [t.isoformat() for t in times_datetime],
            "soc": socs_float_list,  # type: ignore
        }


def optimize_charging_events_even(charging_events: List[Event]) -> None:
    """
    This function optimizes the power draw of a list of charging events.

    The power draw is optimized such that the total
    power draw is minimized, while the energy transferred remains constant.
    :param charging_events: The list of charging events to optimize
    :return: Nothing, the charging events are updated in place
    """
    logger = logging.getLogger(__name__)

    assert all(
        [event.event_type == EventType.CHARGING_DEPOT for event in charging_events]
    )
    start_time = min([event.time_start for event in charging_events])
    end_time = max([event.time_end for event in charging_events])

    # Create the time steps (start times of the time steps)
    time_steps = [
        start_time + i * TIME_STEP_DURATION
        for i in range(int((end_time - start_time) / TIME_STEP_DURATION))
    ]

    # Create the SmartChargingEvents
    smart_charging_events = [
        SmartChargingEvent.from_event(event, time_steps) for event in charging_events
    ]

    # Dicard the ones that need 0 energy
    smart_charging_events = [
        event for event in smart_charging_events if event.energy_packets_needed > 0
    ]

    # Solve the peak shaving problem
    try:
        updated_events, peak_power = solve_peak_shaving(
            smart_charging_events, time_steps
        )

        logger.info(f"Optimization successful. Peak power: {peak_power:.2f} kW")

    except ValueError as e:
        logger.error(f"Optimization failed: {e}")

    # Update the original events
    for smart_event in updated_events:
        smart_event.update_original_event(time_steps)


def solve_peak_shaving(
    charging_events: List[SmartChargingEvent],
    time_steps: List[datetime],
) -> Tuple[List[SmartChargingEvent], float]:
    """
    Solves the peak shaving problem for electric vehicles using integer linear programming.

    The problem:
    - Vehicles are present during some discrete timesteps
    - In each timestep where a vehicle is present, decide how much charging power to provide
    - Constraints:

          1. Charging power must not exceed the maximum in any timestep
          2. Each vehicle must receive its required total energy

    - Objective: Minimize the peak sum of all vehicles' charging powers

    Args:
        charging_events: List of smart charging events for each vehicle
        time_steps: Array of all timesteps in the scheduling horizon

    Returns:
        A tuple containing:
        - Updated list of smart charging events with optimized charging schedules
        - The optimal peak power value in kW

    Raises:
        ValueError: If the problem is infeasible or no solution could be found
    """
    logger = logging.getLogger(__name__)

    # Create a Pyomo model
    model = pyo.ConcreteModel(name="EV_Peak_Shaving")

    # Define indices
    num_vehicles = len(charging_events)
    num_timesteps = len(time_steps)

    # Define sets
    model.V = pyo.Set(initialize=range(num_vehicles), doc="Set of vehicles")
    model.T = pyo.Set(initialize=range(num_timesteps), doc="Set of timesteps")

    # Create sparse set of vehicle-timestep pairs where vehicle is present
    vehicle_present_pairs = []
    for v in range(num_vehicles):
        for t in range(num_timesteps):
            if (
                t < len(charging_events[v].vehicle_present)
                and charging_events[v].vehicle_present[t]
            ):
                vehicle_present_pairs.append((v, t))

    # Create a set from the list of pairs
    model.VT_present = pyo.Set(
        initialize=vehicle_present_pairs,
        doc="Set of (vehicle, timestep) pairs where the vehicle is present",
    )

    # Define parameters - only for relevant data
    # Maximum energy packets per timestep for each vehicle
    model.max_rate = pyo.Param(
        model.V,
        initialize=lambda model, v: charging_events[v].energy_packets_per_time_step,
        doc="Maximum energy packets per timestep for each vehicle",
    )

    # Total energy packets needed for each vehicle
    model.energy_req = pyo.Param(
        model.V,
        initialize=lambda model, v: charging_events[v].energy_packets_needed,
        doc="Total energy packets needed for each vehicle",
    )

    # Define decision variables - only for relevant pairs
    # Energy packets to transfer to vehicle v at timestep t (only when present)
    model.x = pyo.Var(
        model.VT_present,
        domain=pyo.NonNegativeIntegers,
        doc="Energy packets to transfer to vehicle v at timestep t when present",
    )

    # Peak power across all timesteps (in energy packets) - now integer
    # This variable is necessary because we're solving a "minimax" problem
    # (minimizing the maximum power at any timestep)
    model.peak = pyo.Var(
        domain=pyo.NonNegativeIntegers,
        doc="Peak total energy packets across all timesteps",
    )

    # Define constraints

    # Constraint 1: Charging limit - only need to constrain when vehicle is present
    def charging_limit_rule(model, v, t):  # type: ignore
        return model.x[v, t] <= model.max_rate[v]

    model.charging_limit = pyo.Constraint(
        model.VT_present,
        rule=charging_limit_rule,
        doc="Limit charging based on maximum rate when vehicle is present",
    )

    # Constraint 2: Energy requirement - each vehicle must receive its required energy
    def energy_requirement_rule(model, v):  # type: ignore
        return (
            sum(model.x[v, t] for (v_idx, t) in model.VT_present if v_idx == v)
            == model.energy_req[v]
        )

    model.energy_requirement = pyo.Constraint(
        model.V,
        rule=energy_requirement_rule,
        doc="Ensure each vehicle receives its required energy",
    )

    # Precompute vehicle presence by timestep for faster lookup
    vehicles_by_timestep: Dict[int, List[int]] = {}
    for v, t in model.VT_present:
        if t not in vehicles_by_timestep:
            vehicles_by_timestep[t] = []
        vehicles_by_timestep[t].append(v)

    # Only create constraints for timesteps with at least one vehicle present
    active_timesteps = sorted(vehicles_by_timestep.keys())
    model.active_T = pyo.Set(
        initialize=active_timesteps, doc="Timesteps with at least one vehicle present"
    )

    # Create timestep power expressions (pre-calculated sums)
    model.timestep_power = pyo.Expression(
        model.active_T,
        rule=lambda model, t: sum(model.x[v, t] for v in vehicles_by_timestep[t]),
        doc="Total power at each active timestep",
    )

    # Constraint 3: Peak power definition using expressions
    def peak_power_rule(model, t):  # type: ignore
        return model.timestep_power[t] <= model.peak

    model.peak_power = pyo.Constraint(
        model.active_T,
        rule=peak_power_rule,
        doc="Define peak power across active timesteps",
    )

    # Define objective: minimize peak power
    model.objective = pyo.Objective(
        expr=model.peak, sense=pyo.minimize, doc="Minimize peak power"
    )

    # Check if gurobi is available
    if not pyo.SolverFactory("gurobi_direct").available():
        warnings.warn("Gurobi is not available. Using GLPK instead.")
        if not pyo.SolverFactory("glpk").available():
            raise ValueError(
                "GLPK is not available. Install it using your package manager."
            )
        solver = pyo.SolverFactory("glpk")
    else:
        solver = pyo.SolverFactory("gurobi_direct")

    # Solve the model
    logger.info("Solving the peak shaving problem...")
    result = solver.solve(model)
    logger.info(f"Solver status: {result.solver.status}")

    # Check if an optimal solution was found
    if (
        result.solver.status == pyo.SolverStatus.ok
        and result.solver.termination_condition == pyo.TerminationCondition.optimal
    ):
        # Update charging schedules with the optimal solution
        for v in model.V:
            # Initialize all timesteps to zero
            for t in range(len(charging_events[v].energy_packets_transferred)):
                charging_events[v].energy_packets_transferred[t] = 0

            # Update only the timesteps where the vehicle is present
            for v_idx, t in model.VT_present:
                if v_idx == v and t < len(
                    charging_events[v].energy_packets_transferred
                ):
                    charging_events[v].energy_packets_transferred[t] = int(
                        model.x[v_idx, t].value
                    )

        # Calculate the actual peak power in kW
        peak_power = model.peak.value * POWER_QUANTIZATION
        return charging_events, peak_power
    else:
        # No optimal solution found
        error_msg = f"Failed to find an optimal solution. Status: {result.solver.status}, Termination: {result.solver.termination_condition}"
        raise ValueError(error_msg)


def add_slack_time_to_events_of_depot(
    depot: Depot,
    session: sqlalchemy.orm.session.Session,
    standby_departure_duration: timedelta = timedelta(minutes=5),
) -> None:
    logger = logging.getLogger(__name__)

    # Load all the charging events at this depot
    charging_events = (
        session.query(Event)
        .join(Area)
        .filter(Area.depot_id == depot.id)
        .filter(Event.event_type == EventType.CHARGING_DEPOT)
        .all()
    )

    # For each event, take the subsequent STANDBY_DEPARTURE event of the same vehicle
    # Reduce the STANDBY_DEPARTURE events duration to 5 minutes
    # Move the end time of the charging event to the start time of the STANDBY_DEPARTURE event
    for charging_event in charging_events:
        next_event = (
            session.query(Event)
            .filter(Event.time_start >= charging_event.time_end)
            .filter(Event.vehicle_id == charging_event.vehicle_id)
            .order_by(Event.time_start)
            .first()
        )

        if next_event is None or next_event.event_type != EventType.STANDBY_DEPARTURE:
            logger.info(
                f"Event {charging_event.id} has no STANDBY_DEPARTURE event after a CHARGING_DEPOT "
                f"event. No room for smart charging."
            )
            continue

        assert next_event.time_start == charging_event.time_end

        if (next_event.time_end - next_event.time_start) > standby_departure_duration:
            next_event.time_start = next_event.time_end - standby_departure_duration
            session.flush()
            # Add a timeseries to the charging event
            assert charging_event.timeseries is None
            charging_event.timeseries = {
                "time": [
                    charging_event.time_start.isoformat(),
                    charging_event.time_end.isoformat(),
                    next_event.time_start.isoformat(),
                ],
                "soc": [
                    charging_event.soc_start,
                    charging_event.soc_end,
                    charging_event.soc_end,
                ],
            }
            charging_event.time_end = next_event.time_start
            session.flush()
