# eflips-opt

---

Part of the [eFLIPS/simBA](https://github.com/stars/ludgerheide/lists/ebus2030) list of projects.

---

This repository contains code for optimizing (electric) bus networks. The following optimization problems are considered:

- [Depot-Rotation Matching](#depot-rotation-matching): Given a set of depots and a set of rotations, find the optimal assignment of rotations to depots.
- [Simplified Electric Vehicle Scheduling Problem](#simplified-electric-vehicle-scheduling-problem): Given a set of trips, find the rotation of electric buses that minimizes the total number of vehicles needed to serve all trips.


## Installation
1. Clone the repository
    ```bash 
    git clone git@github.com:mpm-tu-berlin/eflips-opt.git
    ```

2. Install the packages listed in `poetry.lock` and `pyproject.toml` into your Python environment. Notes:
    - The suggested Python version is 3.11.*, it may work with other versions, but this is not tested.
    - Using the [poetry](https://python-poetry.org/) package manager is recommended. It can be installed according to the
      instructions listed [here](https://python-poetry.org/docs/#installing-with-the-official-installer).

3. In order to use the [Depot-Rotation Matching](#depot-rotation-matching), it is recommended to install the 
4. [Gurobi](https://www.gurobi.com/) solver. The solver can be installed by following the instructions listed [here](https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer). 
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

## Simplified Electric Vehicle Scheduling Problem

---

This section is not current, it will be updated soon.

---

Example scripts for the Simplified Electric Vehicle Scheduling Problem can be found in the
`examples/simplified_electric_vehicle_scheduling_problem` directory. It creates a graph of all possible trips and
their possible connections. The minimum number of rotations is now a 
[minimum path cover](https://en.wikipedia.org/wiki/Path_cover) of the graph. This is what the 
`minimum_path_cover_rotation_plan()` function does. Its result however tries to make rotations as long as possible,
ignoring the fact that the bus might need to be recharged (this is useful for identifying terminus stations to 
electrify though. 

In order to find a rotation plan that considers the battery state of the bus, the `soc_aware_rotation_plan()` function
can be used. It iterates over the solution of the `minimum_path_cover_rotation_plan()` function and heuristically finds
a good rotation plan that considers the battery state of the bus.

### Known issues

- The result is not optimal in terms of minimum waiting time at each terminus. This could be improved by fine-tuning
the candidate selection process in the `create_graph_of_possible_connections()` function.
- The result is not stable, changing with each new python process. This is due to there being multiple minimum path c
covers of the graph. This could possibly be fixed by iterating over all minimum path covers and selecting the one that
minimizes the dwell durations or the energy consumption of each bus. [See here for a possible implementation (but be careful of the combination explosion)](https://github.com/Xunius/bipartite_matching)

### Usage

```python
# You probably want to create one schedule for each vehicle type, e.g. not connect 12m buses' trips with 18m buses' trips
trips_by_vt = passenger_trips_by_vehicle_type(scenario, session)
for vehicle_type, trips in trips_by_vt.items():
   # The connection graph is what the schedule will be based on
   graph = create_graph_of_possible_connections(trips)
   
   # The minimum path cover rotation plan just makes rotations as long as possible
   rotation_graph = minimum_path_cover_rotation_plan(graph)
   
   # The soc aware rotation plan tries to make rotations that consider the battery state of the bus
   soc_aware_graph = soc_aware_rotation_plan(graph, soc_reserve=SOC_RESERVE)
   
   # The rotation plan can be visualized
   visualize_with_dash_cytoscape(soc_aware_graph)
 ```

## Development

We utilize the [GitHub Flow](https://docs.github.com/get-started/quickstart/github-flow) branching structure. This means
that the `main` branch is always deployable and that all development happens in feature branches. The feature branches
are merged into `main` via pull requests.

We use [black](https://black.readthedocs.io/en/stable/) for code formatting. You can use 
[pre-commit](https://pre-commit.com/) to ensure the code is formatted correctly before committing. You are also free to
use other methods to format the code, but please ensure that the code is formatted correctly before committing.

Please make sure that your `poetry.lock` and `pyproject.toml` files are consistent before committing. You can use `poetry check` to check this. This is also checked by pre-commit.

## License

This project is licensed under the AGPLv3 license - see the [LICENSE](LICENSE.md) file for details.

## Funding Notice

This code was developed as part of the project [eBus2030+](https://www.now-gmbh.de/projektfinder/e-bus-2030/) funded by the Federal German Ministry for Digital and Transport (BMDV) under grant number 03EMF0402.
