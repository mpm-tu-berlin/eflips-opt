# eflips-opt

---

Part of the [eFLIPS/simBA](https://github.com/stars/ludgerheide/lists/ebus2030) list of projects.

---

This repository contains code for optimizing (electric) bus networks. The following optimization problems are considered:

- [Depot-Rotation Matching](#depot-rotation-matching): Given a set of depots and a set of rotations, find the optimal assignment of rotations to depots.
- [Vehicle Scheduling](#vehicle-scheduling): Given a set of trips, find the set of rotations (vehicle blocks) that serves all trips with the minimum number of vehicles, optionally respecting battery and duration limits.
- Smart Charging (`eflips.opt.smart_charging`): After a simulation, smooth the depot charging events to reduce peak power.

## Installation
1. Clone the repository
    ```bash 
    git clone git@github.com:mpm-tu-berlin/eflips-opt.git
    ```

2. Install the packages listed in `poetry.lock` and `pyproject.toml` into your Python environment. Notes:
    - Python 3.11 to 3.15 is supported.
    - Using the [poetry](https://python-poetry.org/) package manager is recommended. It can be installed according to the
      instructions listed [here](https://python-poetry.org/docs/#installing-with-the-official-installer).

3. In order to use the [Depot-Rotation Matching](#depot-rotation-matching), it is recommended to install the
   [Gurobi](https://www.gurobi.com/) solver. The solver can be installed by following the instructions listed [here](https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer). 
   The solver can be used with a license for academic purposes. If you do not have a license, you can request one [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Usage

### Depot-Rotation Matching

Example scripts for the Depot-Rotation Matching problem can be found in the `examples/depot_rotation_matching` directory.
The general usage is as follows:

```python
from eflips.opt.depot_rotation_matching import DepotRotationOptimizer

# Initialize the optimizer
optimizer = DepotRotationOptimizer(session, SCENARIO_ID)

# Create a depot input. Both existing and new depots can be added.
user_input_depot = [
   {
       "depot_station": 103281393,
       "capacity": 300,
       "vehicle_type": [84, 86, 87, 90],
   },  # Indira-Gandhi-Str
   {
       "name": "Suedost",
       "depot_station": (13.497371828836501, 52.46541010322369),
       "capacity": 260,
       "vehicle_type": [82, 84, 85, 86, 87, 90],
   },  # Südost
]
optimizer.get_depot_from_input(user_input_depot)
optimizer.data_preparation()
optimizer.optimize(time_report=True)
optimizer.write_optimization_results(delete_original_data=True)
```

### Vehicle Scheduling

The scheduling problem is formulated as a directed acyclic graph: nodes are trips, edges are feasible connections
between trips, weighted by the waiting time. The minimum number of rotations is a
[minimum path cover](https://en.wikipedia.org/wiki/Path_cover) of this graph, which is computed by the Rust solver
[eflips-schedule-rust](https://pypi.org/project/eflips-schedule-rust/). Per-trip weights (e.g. the state-of-charge
used by each trip) let the solver keep each rotation within the battery capacity and a maximum duration.

```python
from datetime import timedelta
from eflips.opt.scheduling import (
    passenger_trips_by_vehicle_type,
    create_graph,
    solve,
    write_back_rotation_plan,
)

# Create one schedule per vehicle type, so trips of e.g. 12m and 18m buses are not mixed
trips_by_vt = passenger_trips_by_vehicle_type(scenario, session)
for vehicle_type, trips in trips_by_vt.items():
    if not trips:
        continue
    # delta_socs: optional {trip_id: fraction of the battery used by the trip}
    graph = create_graph(trips, maximum_schedule_duration=timedelta(hours=24))
    rotation_graph = solve(graph, maximum_schedule_duration=timedelta(hours=24))

    # Replace the scenario's original rotations with the new ones
    write_back_rotation_plan(rotation_graph, session)
session.commit()
```

`create_graph()` has further parameters for break times and for preferring connections on the same line.
`visualize_with_dash_cytoscape(graph)` shows a graph in the browser.

## Development

We utilize the [GitHub Flow](https://docs.github.com/get-started/quickstart/github-flow) branching structure. This means
that the `main` branch is always deployable and that all development happens in feature branches. The feature branches
are merged into `main` via pull requests.

We use [black](https://black.readthedocs.io/en/stable/) for code formatting. You can use 
[pre-commit](https://pre-commit.com/) to ensure the code is formatted correctly before committing. You are also free to
use other methods to format the code, but please ensure that the code is formatted correctly before committing.

Please make sure that your `poetry.lock` and `pyproject.toml` files are consistent before committing. You can use `poetry check` to check this. This is also checked by pre-commit.

Each pull request must increase the version number in `pyproject.toml` (CI checks it against PyPI), following
[Semantic Versioning](https://semver.org/): patch for changes without API changes, minor for backwards-compatible
API changes such as new features, major for breaking changes. Every merge to `main` is published to PyPI.

Tests need a PostgreSQL/PostGIS database, given in `DATABASE_URL`. **All data in it is deleted.**

```bash
DATABASE_URL=postgresql://user:password@localhost/database PYTHONPATH=tests:. poetry run pytest
```

## License

This project is licensed under the AGPLv3 license - see the [LICENSE](LICENSE) file for details.

## Funding Notice

This code was developed as part of the project [eBus2030+](https://www.now-gmbh.de/projektfinder/e-bus-2030/) funded by the Federal German Ministry for Digital and Transport (BMDV) under grant number 03EMF0402.
