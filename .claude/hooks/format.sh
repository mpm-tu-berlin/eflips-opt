#!/bin/bash
# PostToolUse(Edit|Write): format edited Python files with black.
f=$(jq -r '.tool_input.file_path // empty')
case "$f" in
  *.py) "$CLAUDE_PROJECT_DIR/.claude/hooks/run-tool.sh" black -q "$f" ;;
esac
exit 0
