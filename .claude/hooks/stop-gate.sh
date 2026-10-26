#!/bin/bash
# Stop: block finishing while black or strict mypy (the CI checks) fail.
# Only runs when Python files differ from HEAD. Tools that aren't installed are skipped.
input=$(cat)
[ "$(jq -r '.stop_hook_active' <<<"$input")" = "true" ] && exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0

changed=$( (git diff --name-only HEAD; git ls-files --others --exclude-standard) | grep '\.py$')
[ -z "$changed" ] && exit 0

run="$CLAUDE_PROJECT_DIR/.claude/hooks/run-tool.sh"
fail=""
out=$("$run" black --check -q . 2>&1); rc=$?
[ $rc -ne 0 ] && [ $rc -ne 127 ] && fail+="black --check failed:\n$out\n"
out=$("$run" mypy eflips --explicit-package-bases --strict 2>&1); rc=$?
[ $rc -ne 0 ] && [ $rc -ne 127 ] && fail+="mypy --strict failed:\n$(tail -40 <<<"$out")\n"

if [ -n "$fail" ]; then
  jq -n --arg r "$(printf "%b" "$fail")" '{decision:"block", reason:$r}'
fi
exit 0
