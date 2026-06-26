"""Ed25519 public-key keystore for AEP signature verification.

Keys are loaded from environment variables in the form:
    WASMAGENT_AEP_PUBKEY_<KEY_ID>=<base64-encoded DER or raw 32-byte key>

The KEY_ID in the env var name is the upper-cased, hyphen-stripped version of
the key_id found in the AEP signature block, e.g.:
    key_id "aep-signer-v1"  →  env var  WASMAGENT_AEP_PUBKEY_AEP_SIGNER_V1

Design rules
- An unknown key_id (no matching env var) is always a verification failure.
- Keys must be 32-byte raw Ed25519 public keys, base64url or standard base64 encoded.
- This module never falls back silently; callers must handle KeyNotFoundError.
"""
from __future__ import annotations

import base64
import os
import re

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


class KeyNotFoundError(Exception):
    """Raised when a key_id has no corresponding env var."""


def _key_id_to_env_var(key_id: str) -> str:
    """Convert a key_id to the canonical env var name.

    Examples:
        "aep-signer-v1"    → "WASMAGENT_AEP_PUBKEY_AEP_SIGNER_V1"
        "prod.key/2024"    → "WASMAGENT_AEP_PUBKEY_PROD_KEY_2024"
    """
    # Replace any non-alphanumeric character with underscore, then upper-case
    sanitized = re.sub(r"[^A-Za-z0-9]", "_", key_id).upper()
    return f"WASMAGENT_AEP_PUBKEY_{sanitized}"


def _decode_pubkey_bytes(raw_b64: str) -> bytes:
    """Decode a base64 (standard or URL-safe) encoded public key to raw bytes."""
    raw_b64 = raw_b64.strip()
    # Add padding if needed
    padding = (4 - len(raw_b64) % 4) % 4
    padded = raw_b64 + "=" * padding
    try:
        return base64.urlsafe_b64decode(padded)
    except Exception:
        return base64.b64decode(padded)


def load_public_key(key_id: str) -> Ed25519PublicKey:
    """Load the Ed25519 public key for *key_id* from the environment.

    Raises:
        KeyNotFoundError: if no env var for this key_id is set.
        ValueError: if the env var value cannot be decoded as a valid Ed25519 key.
    """
    env_var = _key_id_to_env_var(key_id)
    raw_b64 = os.environ.get(env_var)
    if raw_b64 is None:
        raise KeyNotFoundError(
            f"No public key found for key_id={key_id!r}. "
            f"Set env var {env_var} to the base64-encoded Ed25519 public key."
        )
    try:
        key_bytes = _decode_pubkey_bytes(raw_b64)
    except Exception as exc:
        raise ValueError(
            f"Cannot base64-decode value of {env_var}: {exc}"
        ) from exc

    if len(key_bytes) == 32:
        # Raw 32-byte Ed25519 public key
        return Ed25519PublicKey.from_public_bytes(key_bytes)
    else:
        raise ValueError(
            f"Ed25519 public key for {env_var} must be 32 bytes (got {len(key_bytes)} after decode). "
            "Encode the raw 32-byte key as base64."
        )
