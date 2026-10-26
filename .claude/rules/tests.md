---
paths: ["tests/**"]
---
- Test fixtures drop all tables in `DATABASE_URL`; never point it at a real database.
- `test_scheduling.py` shells out to `psql` to load `tests/data/eflips_one_day.sql.xz`; `psql` must be on PATH.
- ORS-dependent tests skip unless `BASE_URL` is set. Set `ELEVATION_DUMMY_MODE=True` to avoid network altitude lookups.
- `TestAltitudeBatching` runs without a database.
