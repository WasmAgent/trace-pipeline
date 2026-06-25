"""evomerge.router — router / escalation model feature extraction and training.

Implements plan Section 6 Phase 3:
  - RouterFeatures: structured feature vector extracted from TaskSpec + EvalRecord
  - RouterLabel: target label for training (small_model / need_repair /
                  need_large_model / need_human_review)
  - RouterRecord: (features, label) pair for classifier training
  - RouterRuleClassifier: deterministic rule-based router for baseline and
                          online use before an ML classifier is available
  - feature_from_record: factory function
"""
from evomerge.router.features import RouterFeatures, feature_from_record
from evomerge.router.labels import RouterLabel, RouterRecord, label_from_record
from evomerge.router.classifier import RouterRuleClassifier, RouterEvalReport

__all__ = [
    "RouterFeatures",
    "RouterLabel",
    "RouterRecord",
    "RouterRuleClassifier",
    "RouterEvalReport",
    "feature_from_record",
    "label_from_record",
]
