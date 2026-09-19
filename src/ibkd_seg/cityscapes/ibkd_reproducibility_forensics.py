"""Paired one-batch forensics for deterministic Cityscapes iBKD execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import torch

from .ibkd_deterministic import DiagnosticCBAM, configure_variant
from .official_api import bootstrap


PROTOCOL_ID = "cityscapes_segmenter_l16_crop512_ibkd_forensics1_v10"
VARIANTS = (
    "alignment_only",
    "attention_only",
    "adaptive_regular",
    "flatmax_regular",
    "flatmax_gpu_deform",
    "adaptive_cpu_deform",
    "full_original",
    "full_deterministic_candidate",
)
CASES = tuple(
    (f"{variant}_{suffix}", variant)
    for variant in VARIANTS
    for suffix in ("a", "b")
)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _sequence_sha256(values) -> str:
    return hashlib.sha256(json.dumps(list(values), separators=(",", ":")).encode()).hexdigest()


def _operator_names(messages: list[str]) -> list[str]:
    names = set()
    pattern = re.compile(r"([A-Za-z0-9_:.]+) does not have a deterministic implementation")
    for message in messages:
        match = pattern.search(message)
        if match:
            names.add(match.group(1))
    return sorted(names)


def gradient_report(named_parameters, gradients) -> dict:
    digest = hashlib.sha256()
    hashes = {}
    none_names = []
    nonfinite_names = []
    for (name, parameter), gradient in zip(named_parameters, gradients, strict=True):
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        if gradient is None:
            hashes[name] = None
            none_names.append(name)
            digest.update(b"none")
            continue
        value = gradient.detach().cpu().contiguous()
        item_hash = _tensor_sha256(value)
        hashes[name] = item_hash
        digest.update(item_hash.encode())
        if not bool(torch.isfinite(value).all()):
            nonfinite_names.append(name)
    return {
        "sha256": digest.hexdigest(),
        "parameter_sha256": hashes,
        "parameter_count": len(named_parameters),
        "gradient_tensor_count": len(named_parameters) - len(none_names),
        "none_gradient_names": none_names,
        "nonfinite_gradient_names": nonfinite_names,
    }


def environment_controls_ok(run: dict) -> bool:
    environment = run.get("environment", {})
    return (
        environment.get("deterministic_algorithms") is True
        and environment.get("deterministic_warn_only") is True
        and environment.get("cublas_workspace_config") in {":4096:8", ":16:8"}
        and environment.get("cudnn_benchmark") is False
        and environment.get("cudnn_deterministic") is True
        and environment.get("cuda_matmul_allow_tf32") is False
        and environment.get("cudnn_allow_tf32") is False
        and environment.get("cpu_threads") == 1
    )


def compare_pair(left: dict, right: dict) -> dict:
    left_student = (left.get("gradients") or {}).get("student", {})
    right_student = (right.get("gradients") or {}).get("student", {})
    left_guide = (left.get("gradients") or {}).get("guidance", {})
    right_guide = (right.get("gradients") or {}).get("guidance", {})

    def mismatched_names(left_group, right_group):
        left_hashes = left_group.get("parameter_sha256", {})
        right_hashes = right_group.get("parameter_sha256", {})
        return sorted({*left_hashes, *right_hashes} - {
            name
            for name in {*left_hashes, *right_hashes}
            if left_hashes.get(name) == right_hashes.get(name)
        })

    student_mismatches = mismatched_names(left_student, right_student)
    guide_mismatches = mismatched_names(left_guide, right_guide)
    checks = {
        "both_completed": left.get("status") == right.get("status") == "completed",
        "input_exact": left.get("input_sha256") == right.get("input_sha256"),
        "sample_ids_exact": left.get("sample_ids") == right.get("sample_ids"),
        "student_initial_state_exact": (
            left.get("student_initial_state_sha256")
            == right.get("student_initial_state_sha256")
        ),
        "base_guidance_initial_state_exact": (
            left.get("base_guidance_initial_state_sha256")
            == right.get("base_guidance_initial_state_sha256")
        ),
        "variant_guidance_state_exact": (
            left.get("variant_guidance_state_sha256")
            == right.get("variant_guidance_state_sha256")
        ),
        "teacher_state_exact": left.get("teacher_state_sha256") == right.get("teacher_state_sha256"),
        "logits_exact": left.get("logits_sha256") == right.get("logits_sha256"),
        "student_forward_exact": left.get("student_forward_sha256") == right.get("student_forward_sha256"),
        "teacher_forward_exact": left.get("teacher_forward_sha256") == right.get("teacher_forward_sha256"),
        "alignment_scalar_exact": left.get("alignment_sha256") == right.get("alignment_sha256"),
        "fusion_scalar_exact": left.get("fusion_sha256") == right.get("fusion_sha256"),
        "objective_scalar_exact": left.get("objective_sha256") == right.get("objective_sha256"),
        "student_gradients_exact": left_student.get("sha256") == right_student.get("sha256"),
        "guidance_gradients_exact": left_guide.get("sha256") == right_guide.get("sha256"),
        "student_gradient_hashes_present": bool(left_student.get("sha256"))
        and bool(right_student.get("sha256")),
        "guidance_gradient_hashes_present": bool(left_guide.get("sha256"))
        and bool(right_guide.get("sha256")),
        "student_gradients_finite": not left_student.get("nonfinite_gradient_names")
        and not right_student.get("nonfinite_gradient_names"),
        "guidance_gradients_finite": not left_guide.get("nonfinite_gradient_names")
        and not right_guide.get("nonfinite_gradient_names"),
        "environment_controls_verified": environment_controls_ok(left)
        and environment_controls_ok(right),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "student_gradient_mismatch_count": len(student_mismatches),
        "guidance_gradient_mismatch_count": len(guide_mismatches),
        "student_gradient_mismatch_names": student_mismatches[:50],
        "guidance_gradient_mismatch_names": guide_mismatches[:50],
        "left_student_gradient_sha256": left_student.get("sha256"),
        "right_student_gradient_sha256": right_student.get("sha256"),
        "left_guidance_gradient_sha256": left_guide.get("sha256"),
        "right_guidance_gradient_sha256": right_guide.get("sha256"),
        "left_operators": left.get("nondeterministic_operators", []),
        "right_operators": right.get("nondeterministic_operators", []),
        "elapsed_seconds": [left.get("elapsed_seconds"), right.get("elapsed_seconds")],
    }


def diagnose(comparisons: dict, all_completed: bool) -> dict:
    passed = {name: item.get("passed", False) for name, item in comparisons.items()}
    candidate = comparisons.get("full_deterministic_candidate", {})
    candidate_operators = sorted({
        *candidate.get("left_operators", []),
        *candidate.get("right_operators", []),
    })
    baselines_passed = all(
        passed.get(name, False)
        for name in ("alignment_only", "attention_only", "flatmax_regular")
    )
    candidate_passed = (
        all_completed
        and baselines_passed
        and passed.get("full_deterministic_candidate", False)
        and not candidate_operators
    )
    if not all_completed:
        code = "incomplete_ibkd_forensics"
        conclusion = f"{len(CASES)}개 component 실행 중 하나 이상이 완료되지 않았습니다."
        next_step = "누락된 case의 runtime error를 먼저 수정합니다."
    elif not baselines_passed:
        code = "ibkd_base_or_attention_nondeterminism"
        conclusion = "정렬 또는 CBAM을 제외한 attention 경로부터 gradient가 일치하지 않습니다."
        next_step = "실패한 baseline의 mismatch parameter를 더 작은 연산으로 분리합니다."
    elif candidate_passed:
        code = "deterministic_candidate_passed"
        conclusion = (
            "1×1 adaptive max를 flattened max로 계산하고 deformable spatial branch만 CPU에서 실행한 "
            "iBKD 후보가 동일 gradient를 재현했습니다."
        )
        next_step = "이 후보로 25-step optimizer A/B 재현성과 step 시간 증가를 확인합니다."
    else:
        code = "deterministic_candidate_failed"
        conclusion = "결정성 후보에서도 gradient 또는 forward가 반복 간 일치하지 않았습니다."
        next_step = "후보 comparison의 mismatch parameter와 남은 경고 연산을 분리합니다."
    return {
        "code": code,
        "passed": candidate_passed,
        "variant_passed": passed,
        "original_full_reproducible": passed.get("full_original", False),
        "adaptive_max_pool_reproducible": passed.get("adaptive_regular", False),
        "flattened_max_pool_reproducible": passed.get("flatmax_regular", False),
        "gpu_deform_reproducible": passed.get("flatmax_gpu_deform", False),
        "adaptive_max_with_cpu_deform_reproducible": passed.get(
            "adaptive_cpu_deform", False
        ),
        "candidate_nondeterministic_operators": candidate_operators,
        "conclusion_ko": conclusion,
        "next_step_ko": next_step,
    }


def _validate_config(config: dict) -> None:
    expected = {
        "protocol_id": PROTOCOL_ID,
        "run_kind": "engineering_ibkd_reproducibility_forensics",
        "seed": 1,
        "train_samples": 2975,
        "val_samples": 1,
        "batch_size": 8,
        "image_size": 1024,
        "crop_size": 512,
        "decoder_layers": 1,
        "data_workers": 4,
        "gradient_checkpointing": True,
        "attention_query_chunk": 256,
        "guidance_beta": 0.5,
        "ibkd_fusion_ratio": 0.25,
        "warn_only_determinism": True,
        "test_used": False,
        "scientific_result": False,
        "full_training_authorized": False,
        "method": "ibkd",
        "automatic_hyperparameter_changes": False,
        "diagnostic_variants": list(VARIANTS),
        "independent_repeats_per_variant": 2,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"iBKD forensics config mismatch: {mismatches}")


def run_case(args, config: dict) -> int:
    import timm
    import torchvision
    import segm.utils.torch as ptu
    from . import official_api as api
    from .data import save_json
    from .full_data import FullDataset, batch_hash, train_loader
    from .runtime import seed_all, state_hash

    variant = dict(CASES)[args.case]
    device = torch.device("cuda")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("iBKD forensics requires exactly one CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ptu.device = device
    torch.set_num_threads(1)

    manifest = json.loads(args.manifest.read_text())
    dataset = FullDataset(args.data_dir, manifest, config, "train")
    seed_all(config["seed"], strict_determinism=False)
    model = api.student(
        args.cache_root,
        image_size=config["crop_size"],
        decoder_layers=config["decoder_layers"],
    ).to(device).train()
    student_initial = state_hash(model)
    seed_all(config["seed"] + 1000, strict_determinism=False)
    guide = api.guidance("ibkd", config).to(device).train()
    base_guidance_initial = state_hash(guide)
    configure_variant(guide, variant)
    variant_guidance_state = state_hash(guide)
    teacher = api.teacher(args.cache_root).to(device)
    teacher_state = state_hash(teacher)
    capture = api.FeatureCapture(model)
    seed_all(config["seed"] + 2000, strict_determinism=False)
    torch.use_deterministic_algorithms(True, warn_only=True)

    loader = train_loader(dataset, 1, 0, device)
    image, target, ids = next(iter(loader))
    input_digest = batch_hash(image, target, ids)
    image = image.to(device, non_blocking=True)
    student_named = [
        (name, parameter) for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    guidance_named = [
        (name, parameter) for name, parameter in guide.named_parameters()
        if parameter.requires_grad
    ]

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        logits, student_features = capture.forward(model, image)
        with torch.no_grad():
            teacher_features = teacher.extract_feat(api.teacher_input(image))[1:]
        alignment, fusion = guide(student_features, teacher_features)
        if variant == "alignment_only":
            raw_objective = alignment
        elif variant in {"full_original", "full_deterministic_candidate"}:
            ratio = config["ibkd_fusion_ratio"]
            raw_objective = (1.0 - ratio) * alignment + ratio * fusion
        else:
            raw_objective = fusion
        objective = config["guidance_beta"] * raw_objective
        all_named = student_named + guidance_named
        gradients = torch.autograd.grad(
            objective,
            [parameter for _, parameter in all_named],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        student_gradients = gradients[:len(student_named)]
        guidance_gradients = gradients[len(student_named):]
        torch.cuda.synchronize()

    warning_messages = sorted({str(item.message) for item in caught})
    student_features_sha = [_tensor_sha256(value) for value in student_features]
    teacher_features_sha = [_tensor_sha256(value) for value in teacher_features]
    result = {
        "status": "completed",
        "case": args.case,
        "variant": variant,
        "sample_ids": list(ids),
        "input_sha256": input_digest,
        "student_initial_state_sha256": student_initial,
        "base_guidance_initial_state_sha256": base_guidance_initial,
        "variant_guidance_state_sha256": variant_guidance_state,
        "teacher_state_sha256": teacher_state,
        "logits_sha256": _tensor_sha256(logits),
        "student_forward_sha256": _sequence_sha256(student_features_sha),
        "teacher_forward_sha256": _sequence_sha256(teacher_features_sha),
        "alignment": float(alignment.detach()),
        "alignment_sha256": _tensor_sha256(alignment),
        "fusion": float(fusion.detach()),
        "fusion_sha256": _tensor_sha256(fusion),
        "raw_objective": float(raw_objective.detach()),
        "objective": float(objective.detach()),
        "objective_sha256": _tensor_sha256(objective),
        "gradients": {
            "student": gradient_report(student_named, student_gradients),
            "guidance": gradient_report(guidance_named, guidance_gradients),
        },
        "nondeterministic_operators": _operator_names(warning_messages),
        "determinism_warning_count": len(caught),
        "determinism_warning_messages": warning_messages,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "runtime_error": None,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cpu_threads": torch.get_num_threads(),
        },
    }
    save_json(args.output_dir / "summary.json", result)
    print(
        f"[IBKD_FORENSICS_CASE_DONE] case={args.case} variant={variant} "
        f"objective={result['objective']:.9g} "
        f"student_grad={result['gradients']['student']['sha256']} "
        f"guidance_grad={result['gradients']['guidance']['sha256']} "
        f"operators={json.dumps(result['nondeterministic_operators'])}",
        flush=True,
    )
    return 0


def compact_run(run: dict, returncode: int | None) -> dict:
    return {
        key: run.get(key)
        for key in (
            "case",
            "variant",
            "status",
            "sample_ids",
            "input_sha256",
            "student_initial_state_sha256",
            "base_guidance_initial_state_sha256",
            "variant_guidance_state_sha256",
            "teacher_state_sha256",
            "logits_sha256",
            "student_forward_sha256",
            "teacher_forward_sha256",
            "alignment",
            "alignment_sha256",
            "fusion",
            "fusion_sha256",
            "raw_objective",
            "objective",
            "objective_sha256",
            "nondeterministic_operators",
            "determinism_warning_count",
            "elapsed_seconds",
            "peak_cuda_allocated_bytes",
            "runtime_error",
        )
    } | {
        "student_gradient_sha256": (run.get("gradients") or {}).get("student", {}).get("sha256"),
        "guidance_gradient_sha256": (run.get("gradients") or {}).get("guidance", {}).get("sha256"),
        "student_nonfinite_gradient_names": (
            (run.get("gradients") or {}).get("student", {}).get("nonfinite_gradient_names")
        ),
        "guidance_nonfinite_gradient_names": (
            (run.get("gradients") or {}).get("guidance", {}).get("nonfinite_gradient_names")
        ),
        "environment_controls_verified": environment_controls_ok(run),
        "returncode": returncode,
    }


def run_suite(args, config: dict) -> int:
    from .data import save_json, verify_manifest
    from .full_data import prepare_labels
    from .official_assets import verify

    verify(args.cache_root)
    manifest = json.loads(args.manifest.read_text())
    verify_manifest(args.data_dir, manifest)
    prepare_labels(
        args.data_dir,
        args.manifest,
        {"train": config["train_samples"], "val": config["val_samples"]},
        args.output_dir,
    )
    save_json(args.output_dir / "config.json", config)
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["PYTHONHASHSEED"] = str(config["seed"])
    runs, returncodes = {}, {}
    for case, variant in CASES:
        print(f"[IBKD_FORENSICS_CASE_START] case={case} variant={variant}", flush=True)
        command = [
            sys.executable,
            "-u",
            "-m",
            "ibkd_seg.cityscapes.ibkd_reproducibility_forensics",
            "--cache-root",
            str(args.cache_root),
            "--data-dir",
            str(args.data_dir),
            "--manifest",
            str(args.manifest),
            "--output-dir",
            str(args.output_dir / case),
            "--config",
            str(args.config),
            "--case",
            case,
        ]
        completed = subprocess.run(command, check=False, env=environment)
        returncodes[case] = completed.returncode
        summary_path = args.output_dir / case / "summary.json"
        runs[case] = (
            json.loads(summary_path.read_text())
            if summary_path.exists()
            else {
                "status": "missing_summary",
                "case": case,
                "variant": variant,
                "runtime_error": "summary.json missing",
            }
        )

    comparisons = {
        variant: compare_pair(runs[f"{variant}_a"], runs[f"{variant}_b"])
        for variant in VARIANTS
    }
    all_completed = all(
        returncodes.get(case) == 0 and runs[case].get("status") == "completed"
        for case, _ in CASES
    )
    diagnosis = diagnose(comparisons, all_completed)
    terminal = {
        "status": "completed" if all_completed else "failed",
        "protocol_id": PROTOCOL_ID,
        "all_cases_completed": all_completed,
        "cases_completed": sum(run.get("status") == "completed" for run in runs.values()),
        "total_cases": len(CASES),
        "seed": config["seed"],
        "method": config["method"],
        "variants": list(VARIANTS),
        "independent_repeats_per_variant": config[
            "independent_repeats_per_variant"
        ],
        "crop_size": config["crop_size"],
        "batch_size": config["batch_size"],
        "guidance_beta": config["guidance_beta"],
        "ibkd_fusion_ratio": config["ibkd_fusion_ratio"],
        "test_used": False,
        "scientific_result": False,
        "comparisons": comparisons,
        "diagnosis": diagnosis,
        "runs": {
            case: compact_run(runs[case], returncodes.get(case))
            for case, _ in CASES
        },
        "deterministic_candidate": {
            "adaptive_max_replacement": (
                "torch.max over the flattened spatial grid; same forward value and "
                "first-index tie gradient as AdaptiveMaxPool2d(1)"
            ),
            "deformable_spatial_execution": "same torchvision deform_conv2d equation on one CPU thread",
            "requires_all_ibkd_candidates_to_be_rerun": True,
        },
        "evaluation_metrics": None,
        "evaluation_metrics_reason": "one-batch backward diagnostic; no validation evaluation",
        "next_25step_optimizer_audit_authorized": diagnosis["passed"],
        "full_2000_reproducibility_audit_authorized": False,
        "full_training_authorized": False,
    }
    save_json(
        args.output_dir / "ibkd_forensics_summary.json",
        {"terminal_result": terminal, "runs": runs},
    )
    print(
        "[CITYSCAPES_IBKD_REPRO_FORENSICS_DONE] "
        + json.dumps(terminal, sort_keys=True, allow_nan=False, ensure_ascii=False),
        flush=True,
    )
    return 0 if all_completed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("cache-root", "data-dir", "manifest", "output-dir", "config"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--case", choices=[case for case, _ in CASES], help=argparse.SUPPRESS)
    args = parser.parse_args()
    for key in ("cache_root", "data_dir", "manifest", "output_dir", "config"):
        setattr(args, key, getattr(args, key).resolve())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Use a new empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    _validate_config(config)
    if args.case is not None:
        bootstrap(args.cache_root)
        return run_case(args, config)
    return run_suite(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
