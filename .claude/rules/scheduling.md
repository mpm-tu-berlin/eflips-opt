---
paths: ["eflips/opt/scheduling/**"]
---
- The solver is the Rust extension `eflips_schedule_rust`: `solve()` serializes the graph to JSON (`scheduling/util.py`).
  Node weights are tuples of floats in [0, 1]; edge weights are ints in [0, 1e6]; infeasible connections must be absent, not high-weight.
- `examples/simplified_electric_vehicle_scheduling_problem/` uses a removed API (`create_graph_of_possible_connections`); don't copy from it.
