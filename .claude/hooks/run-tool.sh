#!/bin/bash
# Run a dev tool (black, mypy, ...) via poetry if available, else from PATH.
# Exits 127 if the tool cannot be found, so callers can skip instead of fail.
tool="$1"; shift
if command -v poetry >/dev/null 2>&1 && poetry run which "$tool" >/dev/null 2>&1; then
  exec poetry run "$tool" "$@"
elif command -v "$tool" >/dev/null 2>&1; then
  exec "$tool" "$@"
fi
exit 127
