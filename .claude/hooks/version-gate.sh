#!/bin/bash
# PreToolUse(Bash): before `git push` or `gh pr create`, require the pyproject.toml
# version to be higher than on origin/main (CI enforces the same against PyPI).
cmd=$(jq -r '.tool_input.command // empty')
grep -qE '(^|[;&| ])(git push|gh pr create)' <<<"$cmd" || exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0
[ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] && exit 0

ver() { grep -m1 -E '^version *=' | sed -E 's/.*"(.*)".*/\1/'; }
local_v=$(ver < pyproject.toml)
main_v=$(git show origin/main:pyproject.toml 2>/dev/null | ver)
[ -z "$main_v" ] && exit 0

higher=$(printf '%s\n%s\n' "$main_v" "$local_v" | sort -V | tail -1)
if [ "$local_v" = "$main_v" ] || [ "$higher" != "$local_v" ]; then
  echo "pyproject.toml version ($local_v) is not higher than origin/main ($main_v). Bump it per SemVer (patch: no API change, minor: backwards-compatible API change, major: breaking) before pushing." >&2
  exit 2
fi
exit 0
