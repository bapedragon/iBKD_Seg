"""The result-independent Phase 1 H200 timing matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BATCH_SIZES = (64, 128)
STUDENT_VARIANTS = (
    ("vanilla", None),
    ("kd", None),
    ("lg", None),
    ("alg", None),
    ("ibkd", 0.25),
    ("ibkd", 0.5),
)


def lambda_tag(value: float | None) -> str:
    if value is None:
        return "none"
    return str(value).replace(".", "p")


@dataclass(frozen=True)
class TimingTask:
    index: int
    method: str
    batch_size: int
    fusion_ratio: float | None
    run_name: str
    summary_path: Path

    @property
    def label(self) -> str:
        suffix = "" if self.fusion_ratio is None else f"/lambda={self.fusion_ratio:g}"
        return f"{self.method.upper()}{suffix}/batch={self.batch_size}"


def build_tasks(output_dir: Path) -> list[TimingTask]:
    tasks: list[TimingTask] = []
    index = 1
    for method, fusion_ratio in STUDENT_VARIANTS:
        for batch_size in BATCH_SIZES:
            method_tag = method
            if fusion_ratio is not None:
                method_tag += f"_lambda{lambda_tag(fusion_ratio)}"
            run_name = f"pet_{method_tag}_b{batch_size}_timing_2ep_seed1"
            tasks.append(
                TimingTask(
                    index=index,
                    method=method,
                    batch_size=batch_size,
                    fusion_ratio=fusion_ratio,
                    run_name=run_name,
                    summary_path=output_dir / "students" / run_name / "summary.json",
                )
            )
            index += 1
    if len(tasks) != 12:
        raise AssertionError(f"Expected 12 student timing tasks, built {len(tasks)}")
    return tasks
