"""Phase 1 Oxford-IIIT Pet classification and frozen-probe experiments."""

from .timing_matrix import BATCH_SIZES, STUDENT_VARIANTS, TimingTask, build_tasks

__all__ = ["BATCH_SIZES", "STUDENT_VARIANTS", "TimingTask", "build_tasks"]
