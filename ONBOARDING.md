# Onboarding

New contributor? The full onboarding procedure for **all three
WasmAgent repos** (`wasmagent-js`, `bscode`, `trace-pipeline`) lives
in the wasmagent-js repo:

→ <https://github.com/WasmAgent/wasmagent-js/blob/main/ONBOARDING.md>

trace-pipeline-specific quickstart (after `mise install` and
`cp .env.example .env.local`):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
bash .githooks/install.sh    # one-time, mirrors CI to pre-push
pytest tests/ -x -q          # 514 tests, ~2s
make test                    # broader suite (pytest + reproducer + examples)
```

If you want `validate-aep --require-signature` to actually verify
signatures, set `WASMAGENT_AEP_PUBKEY_<key_id>` env vars (see
`.env.example`). The dev/CI public keys live in the 1Password vault
`wasmagent-dev`.

When the onboarding procedure changes, update the wasmagent-js copy
only — this file just points there.
