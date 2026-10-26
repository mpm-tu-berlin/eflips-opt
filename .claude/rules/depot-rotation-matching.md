---
paths: ["eflips/opt/depot_rotation_matching.py", "eflips/opt/util.py"]
---
- `DepotRotationOptimizer` is a stateful pipeline; methods must run in order:
  `get_depot_from_input` → `data_preparation` → `optimize` → `write_optimization_results`. State lives in `self.data` (DataFrames).
- Deadhead costs come from OpenRouteService (`BASE_URL`, optional `OPENROUTESERVICE_API_KEY`), pickled to
  `$DEPOT_ROTATION_MATCHING_ORS_CACHE` or a per-user tempdir. Delete the cache when changing what is cached.
- Coordinates: user input and shapely use `(lon, lat)`; `eflips.model.util.get_altitudes` takes `(lat, lon)`.
- Altitudes are fetched in one batched call (`_collect_altitudes`); don't add per-point lookups.
