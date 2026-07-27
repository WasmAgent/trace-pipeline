"""Tests for evomerge.router (features, labels, classifier)."""
from __future__ import annotations

from evomerge.eval.metrics import EvalRecord
from evomerge.router.classifier import RouterConfig, RouterRuleClassifier
from evomerge.router.features import RouterFeatures, feature_from_record
from evomerge.router.labels import (
    RouterLabel,
    RouterRecord,
    build_router_records,
    label_from_record,
)
from evomerge.schemas.compliance import TaskSpec
from evomerge.schemas.training import Provenance
from evomerge.synthesize.templates import TaskType, make_task_spec


def _rec(task_id="t1", group="C", final_pass=True, repair_rounds=0,
         escalated=False, tool_calls=0, valid_calls=0, latency=200.0) -> EvalRecord:
    return EvalRecord(
        task_id=task_id,
        group=group,
        final_pass=final_pass,
        repair_rounds=repair_rounds,
        repair_rounds_ok=repair_rounds if final_pass else 0,
        escalated=escalated,
        tool_calls_total=tool_calls,
        tool_calls_valid=valid_calls,
        latency_ms=latency,
    )


def _spec(n_hard=3, has_tools=True, max_repair=3) -> TaskSpec:
    return make_task_spec(
        TaskType.tool_call if has_tools else TaskType.repair,
        intent="test task",
        allowed_tools=["web_search"] if has_tools else None,
    )


class TestRouterFeatures:
    def test_no_record_defaults(self):
        spec = make_task_spec(TaskType.markdown_report, intent="test")
        f = feature_from_record(spec)
        assert f.eval_repair_rounds == 0
        assert f.eval_escalated == 0
        assert f.eval_tool_validity_rate == 1.0

    def test_with_record(self):
        spec = make_task_spec(TaskType.tool_call, intent="test")
        rec = _rec(tool_calls=4, valid_calls=3, repair_rounds=1)
        f = feature_from_record(spec, rec)
        assert f.eval_tool_calls_total == 4
        assert f.eval_tool_calls_valid == 3
        assert abs(f.eval_tool_validity_rate - 0.75) < 0.01
        assert f.eval_repair_rounds == 1

    def test_no_tool_calls_validity_rate_one(self):
        spec = make_task_spec(TaskType.repair, intent="test")
        rec = _rec(tool_calls=0, valid_calls=0)
        f = feature_from_record(spec, rec)
        assert f.eval_tool_validity_rate == 1.0

    def test_to_list_is_numeric(self):
        spec = make_task_spec(TaskType.markdown_report, intent="test")
        f = feature_from_record(spec)
        lst = f.to_list()
        assert all(isinstance(v, float) for v in lst)

    def test_to_list_length_stable(self):
        spec = make_task_spec(TaskType.markdown_report, intent="test")
        f1 = feature_from_record(spec)
        f2 = feature_from_record(spec, _rec())
        assert len(f1.to_list()) == len(f2.to_list())

    def test_has_tools_flag(self):
        spec_tools = make_task_spec(TaskType.tool_call, intent="test")
        spec_no_tools = make_task_spec(TaskType.repair, intent="test")
        f_tools = feature_from_record(spec_tools)
        f_no = feature_from_record(spec_no_tools)
        assert f_tools.taskspec_has_tools == 1
        assert f_no.taskspec_has_tools == 0


class TestRouterLabels:
    def test_passing_no_repair(self):
        rec = _rec(final_pass=True, repair_rounds=0)
        assert label_from_record(rec) == RouterLabel.small_model_can_handle

    def test_passing_with_repair(self):
        rec = _rec(final_pass=True, repair_rounds=2)
        assert label_from_record(rec) == RouterLabel.need_repair

    def test_failing_few_rounds(self):
        rec = _rec(final_pass=False, repair_rounds=1)
        assert label_from_record(rec) == RouterLabel.need_repair

    def test_failing_many_rounds_escalates(self):
        rec = _rec(final_pass=False, repair_rounds=2, escalated=False)
        assert label_from_record(rec, escalate_on_repeated_failure=2) == RouterLabel.need_large_model

    def test_already_escalated(self):
        rec = _rec(escalated=True)
        assert label_from_record(rec) == RouterLabel.need_large_model

    def test_too_many_repair_rounds_escalates(self):
        rec = _rec(repair_rounds=5)
        assert label_from_record(rec, max_repair_before_escalate=3) == RouterLabel.need_large_model

    def test_build_router_records_skips_missing_spec(self):
        specs = {"t1": make_task_spec(TaskType.repair, intent="test")}
        records = [_rec("t1"), _rec("t2")]
        recs = build_router_records(specs, records)
        assert len(recs) == 1
        assert recs[0].task_id == "t1"

    def test_router_record_to_dict(self):
        import json
        spec = make_task_spec(TaskType.repair, intent="test")
        rec = _rec()
        f = feature_from_record(spec, rec)
        rr = RouterRecord(
            task_id="t1",
            features=f,
            label=RouterLabel.small_model_can_handle,
            provenance=Provenance(source="test"),
        )
        json.dumps(rr.to_dict())  # must not raise


class TestRouterRuleClassifier:
    def setup_method(self):
        self.clf = RouterRuleClassifier()
        self.spec = make_task_spec(TaskType.tool_call, intent="test")

    def _features(self, **overrides) -> RouterFeatures:
        f = feature_from_record(self.spec, _rec())
        for k, v in overrides.items():
            object.__setattr__(f, k, v)
        return f

    def test_clean_run_small_model(self):
        f = feature_from_record(self.spec)  # no record → all zeros
        assert self.clf.predict(f) == RouterLabel.small_model_can_handle

    def test_escalated_flag(self):
        f = self._features(eval_escalated=1)
        assert self.clf.predict(f) == RouterLabel.need_large_model

    def test_too_many_repair_rounds(self):
        f = self._features(eval_repair_rounds=5)
        assert self.clf.predict(f) == RouterLabel.need_large_model

    def test_low_tool_validity_needs_repair(self):
        f = self._features(
            eval_tool_calls_total=5,
            eval_tool_validity_rate=0.5,
        )
        assert self.clf.predict(f) == RouterLabel.need_repair

    def test_high_latency_escalates(self):
        cfg = RouterConfig(max_latency_ms=1000.0)
        clf = RouterRuleClassifier(config=cfg)
        f = self._features(eval_latency_ms=5000.0)
        assert clf.predict(f) == RouterLabel.need_large_model

    def test_many_hard_constraints_escalates(self):
        cfg = RouterConfig(hard_constraint_limit=5)
        clf = RouterRuleClassifier(config=cfg)
        f = self._features(taskspec_n_hard=8)
        assert clf.predict(f) == RouterLabel.need_large_model

    def test_one_repair_round_needs_repair(self):
        f = self._features(eval_repair_rounds=1)
        assert self.clf.predict(f) == RouterLabel.need_repair

    def test_predict_with_reason_returns_string(self):
        f = feature_from_record(self.spec)
        label, reason = self.clf.predict_with_reason(f)
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_predict_batch(self):
        features_list = [feature_from_record(self.spec) for _ in range(5)]
        labels = self.clf.predict_batch(features_list)
        assert len(labels) == 5
        assert all(isinstance(l, RouterLabel) for l in labels)

    def test_custom_config(self):
        cfg = RouterConfig(max_repair_rounds=1)
        clf = RouterRuleClassifier(config=cfg)
        f = self._features(eval_repair_rounds=2)
        assert clf.predict(f) == RouterLabel.need_large_model
