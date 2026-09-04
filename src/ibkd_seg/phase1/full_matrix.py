"""Prespecified Phase 1 full-classification task matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .timing_matrix import STUDENT_VARIANTS, lambda_tag


ENCODER_SEEDS = (1, 2, 3)


@dataclass(frozen=True)
class FullTask:
    index: int
    method: str
    batch_size: int
    fusion_ratio: float | None
    seed: int
    run_name: str
    summary_path: Path

    @property
    def variant(self) -> str:
        if self.fusion_ratio is None:
            return self.method
        return f"{self.method}_lambda{lambda_tag(self.fusion_ratio)}"

    @property
    def label(self) -> str:
        suffix = "" if self.fusion_ratio is None else f"/lambda={self.fusion_ratio:g}"
        return f"{self.method.upper()}{suffix}/batch={self.batch_size}/seed={self.seed}"


def build_full_tasks(output_dir: Path, *, batch_size: int) -> list[FullTask]:
    if batch_size not in {64, 128}:
        raise ValueError("Full classification batch must be 64 or 128")
    tasks: list[FullTask] = []
    for method, fusion_ratio in STUDENT_VARIANTS:
        for seed in ENCODER_SEEDS:
            method_tag = method
            if fusion_ratio is not None:
                method_tag += f"_lambda{lambda_tag(fusion_ratio)}"
            run_name = f"pet_{method_tag}_b{batch_size}_full_300ep_seed{seed}"
            tasks.append(
                FullTask(
                    index=len(tasks) + 1,
                    method=method,
                    batch_size=batch_size,
                    fusion_ratio=fusion_ratio,
                    seed=seed,
                    run_name=run_name,
                    summary_path=output_dir / "students" / run_name / "summary.json",
                )
            )
    if len(tasks) != 18:
        raise AssertionError(f"Expected 18 student tasks, built {len(tasks)}")
    return tasks
