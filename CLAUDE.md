# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

eflips-opt: optimization algorithms (depot assignment, vehicle scheduling, smart charging) that read and write an
[eflips-model](https://github.com/mpm-tu-berlin/eflips-model) PostGIS database. Namespace package `eflips.opt`
(`eflips/` has no `__init__.py`). Published to PyPI on every push to `main`.

<!-- black and strict mypy are enforced by hooks in .claude/settings.json; module/test details live in .claude/rules/. -->

## Commands
- Deps via poetry only (`poetry add`); never hand-edit `poetry.lock`. If poetry is unavailable, say so rather than editing the lock.
- Types (CI): `poetry run mypy eflips --explicit-package-bases --strict`
- Tests: `PYTHONPATH=tests:. poetry run pytest tests/<file>::<Class>::<test>` — needs `DATABASE_URL` (PostGIS, wiped by tests).

## PRs
- Bump `version` in `pyproject.toml` in every PR (CI rejects versions not above PyPI), using SemVer:
  patch = no API change, minor = backwards-compatible API change (e.g. new feature), major = breaking change.
- Put the new version in the commit subject, e.g. `Skip zero-length deadhead trips (v1.0.11)`.

## Working rules
- Make the smallest change that satisfies the request; no new abstractions or options unless asked.
- Never edit or delete tests to make them pass; if a test looks wrong, say so.
- New runtime dependency: propose it first (what it replaces, alternatives incl. existing deps/stdlib, license) and wait for approval.
