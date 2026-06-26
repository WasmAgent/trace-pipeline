#!/usr/bin/env bash
#
# install.sh — enable .githooks for this clone.
#
# Run once after cloning:
#   bash .githooks/install.sh

set -e
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

git config core.hooksPath .githooks
chmod +x .githooks/pre-push

echo "✓ Git hooks installed for trace-pipeline:"
echo "    core.hooksPath = $(git config --get core.hooksPath)"
echo
echo "  Pre-push will now run:"
echo "    - ruff check eval_trust/ evomerge/ scripts/ tests/ run_audit.py"
echo "    - python3 scripts/check_no_control_bytes.py"
echo "    - python3 -m pytest tests/ -x -q"
echo
echo "  Emergency bypass:  git push --no-verify"
