"""evomerge.pipeline.cascade — small-first cascade orchestrator.

Implements plan Section 11: "小模型优先 → 失败后 repair → 仍失败升级大模型"

The cascade runs three tiers in order:
  1. small_model  — fast, cheap; checked by router first
  2. repair       — compliance engine repair loop; triggered on constraint failure
  3. large_model  — fallback when repair exhausted or router says escalate

Each tier is a callable the caller supplies.  The cascade itself is
model-agnostic: swap in real model calls or test stubs.

Typical usage:

    from evomerge.pipeline.cascade import CascadeConfig, CascadeRunner, TierResult

    def call_small(task: str) -> str:
        return small_model.generate(task)

    def call_repair(task: str, artifact: str, hints: list[str]) -> str:
        return compliance_engine.repair(task, artifact, hints)

    def call_large(task: str) -> str:
        return large_model.generate(task)

    def verify(task: str, artifact: str) -> tuple[bool, list[str]]:
        result = compliance_engine.verify(task, artifact)
        return result.final_pass, [v.hint for v in result.violations]

    runner = CascadeRunner(
        config=CascadeConfig(max_repair_rounds=3),
        small_fn=call_small,
        repair_fn=call_repair,
        large_fn=call_large,
        verify_fn=verify,
        router=RouterRuleClassifier(),
    )
    outcome = runner.run(task_id="t1", task="Write a report...", spec=task_spec)
    print(outcome.tier_used, outcome.final_pass, outcome.repair_rounds)
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from evomerge.router.classifier import RouterConfig, RouterRuleClassifier
from evomerge.router.labels import RouterLabel
from evomerge.schemas.compliance import TaskSpec

# Type aliases for the three tier callables
SmallFn  = Callable[[str], str]                              # task → artifact
RepairFn = Callable[[str, str, list[str]], str]              # task, artifact, hints → patched
LargeFn  = Callable[[str], str]                              # task → artifact
VerifyFn = Callable[[str, str], tuple[bool, list[str]]]      # task, artifact → (pass, hints)


@dataclass
class CascadeConfig:
    """Tunable thresholds for the small-first cascade.

    Attributes:
        max_repair_rounds: maximum repair attempts before escalating to large model.
        router_config: threshold config forwarded to RouterRuleClassifier.
        skip_router_precheck: if True, always try small model regardless of router.
    """
    max_repair_rounds: int = 3
    router_config: RouterConfig = field(default_factory=RouterConfig)
    skip_router_precheck: bool = False


@dataclass
class CascadeOutcome:
    """Result of one cascade run.

    Attributes:
        task_id: identifier of the task that was run.
        final_pass: True if the final artifact satisfied all constraints.
        artifact: the final output text.
        tier_used: which tier produced the accepted artifact
                   ('small', 'repair', 'large', or 'failed').
        repair_rounds: number of repair rounds attempted.
        escalated: True if the large model was invoked.
        violation_hints: constraint hints from the last verification, if failed.
        router_label: the pre-run routing decision (or None if skipped).
    """
    task_id: str
    final_pass: bool
    artifact: str
    tier_used: str
    repair_rounds: int = 0
    escalated: bool = False
    violation_hints: list[str] = field(default_factory=list)
    router_label: RouterLabel | None = None


class CascadeRunner:
    """Orchestrate the small-first cascade for a single task.

    Args:
        config: CascadeConfig with thresholds.
        small_fn: callable that invokes the small model.
        repair_fn: callable that invokes the compliance repair step.
        large_fn: callable that invokes the large fallback model.
        verify_fn: callable that runs constraint verification.
        router: RouterRuleClassifier instance (optional; uses defaults if None).
        spec: default TaskSpec used for router feature extraction.
    """

    def __init__(
        self,
        config: CascadeConfig | None = None,
        *,
        small_fn: SmallFn,
        repair_fn: RepairFn,
        large_fn: LargeFn,
        verify_fn: VerifyFn,
        router: RouterRuleClassifier | None = None,
        spec: TaskSpec | None = None,
    ):
        self.config    = config or CascadeConfig()
        self.small_fn  = small_fn
        self.repair_fn = repair_fn
        self.large_fn  = large_fn
        self.verify_fn = verify_fn
        self.router    = router or RouterRuleClassifier(self.config.router_config)
        self.spec      = spec

    def run(
        self,
        task_id: str,
        task: str,
        *,
        spec: TaskSpec | None = None,
    ) -> CascadeOutcome:
        """Execute the cascade for one task.

        Steps:
          1. Router pre-check (skip if config.skip_router_precheck=True).
             If router says need_large_model, skip straight to tier 3.
          2. Tier 1: call small model, verify.
             If pass → return immediately.
          3. Tier 2: repair loop (up to max_repair_rounds).
             If any round passes verification → return.
          4. Tier 3: call large model, verify, return regardless.

        Returns:
            CascadeOutcome with the final artifact and metadata.
        """
        from evomerge.router.features import feature_from_record

        effective_spec = spec or self.spec

        # --- Step 1: router pre-check ---
        router_label: RouterLabel | None = None
        if not self.config.skip_router_precheck and effective_spec is not None:
            # Pre-run features (no EvalRecord yet — all eval fields default to 0)
            features = feature_from_record(effective_spec)
            router_label, _ = self.router.predict_with_reason(features)
            if router_label == RouterLabel.need_large_model:
                artifact = self.large_fn(task)
                passed, hints = self.verify_fn(task, artifact)
                return CascadeOutcome(
                    task_id=task_id, final_pass=passed, artifact=artifact,
                    tier_used="large", escalated=True,
                    violation_hints=hints, router_label=router_label,
                )

        # --- Step 2: small model ---
        artifact  = self.small_fn(task)
        passed, hints = self.verify_fn(task, artifact)
        if passed:
            return CascadeOutcome(
                task_id=task_id, final_pass=True, artifact=artifact,
                tier_used="small", router_label=router_label,
            )

        # --- Step 3: repair loop ---
        repair_rounds = 0
        for _ in range(self.config.max_repair_rounds):
            repair_rounds += 1
            artifact  = self.repair_fn(task, artifact, hints)
            passed, hints = self.verify_fn(task, artifact)
            if passed:
                return CascadeOutcome(
                    task_id=task_id, final_pass=True, artifact=artifact,
                    tier_used="repair", repair_rounds=repair_rounds,
                    router_label=router_label,
                )

        # --- Step 4: large model fallback ---
        artifact  = self.large_fn(task)
        passed, hints = self.verify_fn(task, artifact)
        return CascadeOutcome(
            task_id=task_id, final_pass=passed, artifact=artifact,
            tier_used="large" if passed else "failed",
            repair_rounds=repair_rounds, escalated=True,
            violation_hints=hints, router_label=router_label,
        )

    def run_batch(
        self,
        tasks: list[tuple[str, str]],
        *,
        spec: TaskSpec | None = None,
    ) -> list[CascadeOutcome]:
        """Run the cascade for a list of (task_id, task) pairs."""
        return [self.run(tid, task, spec=spec) for tid, task in tasks]


__all__ = ["CascadeConfig", "CascadeOutcome", "CascadeRunner"]
