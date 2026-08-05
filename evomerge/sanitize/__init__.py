"""evomerge.sanitize — payload sanitization for training-data compilation.

Strips PII, API keys, and credentials from tool-call arguments and model
responses *before* they are compiled into SFT/DPO training records, so that
secrets captured in raw traces never leak into a training corpus.

The entry point is :class:`~evomerge.sanitize.scrubber.Scrubber`, plus the
convenience helpers :func:`scrub_text` and :func:`scrub_payload`.
"""
from evomerge.sanitize.scrubber import (
    DEFAULT_PATTERNS,
    ScrubPattern,
    ScrubReport,
    Scrubber,
    scrub_payload,
    scrub_text,
)

__all__ = [
    "DEFAULT_PATTERNS",
    "ScrubPattern",
    "ScrubReport",
    "Scrubber",
    "scrub_payload",
    "scrub_text",
]
