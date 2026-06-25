"""evomerge.validate — contamination and schema validation helpers."""
from evomerge.validate.contamination import check_contamination
from evomerge.validate.schema_check import validate_training_record
from evomerge.validate.quality_gate import QualityReport, run_quality_gate

__all__ = ["check_contamination", "run_quality_gate",
           "validate_training_record", "QualityReport"]
