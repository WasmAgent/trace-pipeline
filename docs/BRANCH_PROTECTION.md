# Branch Protection

The branch-protection policy and pre-push hook setup for **all three
WasmAgent repos** (`wasmagent-js`, `bscode`, `trace-pipeline`) lives in
the wasmagent-js repo:

→ <https://github.com/WasmAgent/wasmagent-js/blob/main/docs/BRANCH_PROTECTION.md>

For trace-pipeline specifically:
- The required CI check name is whatever the active workflow renders
  in `gh run list --limit 1` — typically `CI / lint-and-test`.
- The local pre-push hook is at `.githooks/pre-push`; install once per
  clone with `bash .githooks/install.sh`. It runs
  `ruff check`, `python3 scripts/check_no_control_bytes.py`, and
  `python3 -m pytest tests/ -x -q`.

When the protocol changes, update the wasmagent-js copy only — this
file just points there.
