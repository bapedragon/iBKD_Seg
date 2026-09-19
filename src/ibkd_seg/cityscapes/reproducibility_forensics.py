"""One-batch forward/backward forensics for Cityscapes LG reproducibility."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

from .official_api import bootstrap


PROTOCOL_ID = "cityscapes_segmenter_l16_crop512_repro_forensics1_v8"
CASES = (
    ("vanilla_a", "vanilla"),
    ("vanilla_b", "vanilla"),
    ("lg_a", "lg"),
    ("lg_b", "lg"),
)


def _digest_bytes(*values: bytes) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value)
    return digest.hexdigest()


def _tensor_sha256(tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return _digest_bytes(
        str(value.dtype).encode(),
        str(tuple(value.shape)).encode(),
        value.numpy().tobytes(),
    )


def _sequence_sha256(values: list[str]) -> str:
    return _digest_bytes(json.dumps(values, separators=(",", ":")).encode())


def _rng_snapshot(torch, np, random) -> dict:
    return {
        "python_sha256": _digest_bytes(pickle.dumps(random.getstate())),
        "numpy_sha256": _digest_bytes(pickle.dumps(np.random.get_state())),
        "torch_cpu_sha256": _tensor_sha256(torch.get_rng_state()),
        "torch_cuda_sha256": _tensor_sha256(torch.cuda.get_rng_state()),
    }


def _gradient_group(named_parameters, gradients) -> dict:
    digest = hashlib.sha256()
    none_names = []
    tensor_count = 0
    for (name, parameter), gradient in zip(named_parameters, gradients, strict=True):
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        if gradient is None:
            digest.update(b"none")
            none_names.append(name)
            continue
        tensor_count += 1
        value = gradient.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return {
        "sha256": digest.hexdigest(),
        "parameter_count": len(named_parameters),
        "gradient_tensor_count": tensor_count,
        "none_gradient_count": len(none_names),
        "none_gradient_names_sample": none_names[:10],
    }


def _component_gradients(torch, loss, groups: dict[str, list], *, retain_graph: bool) -> dict:
    ordered_groups = list(groups.items())
    flat = [item for _, named in ordered_groups for item in named]
    gradients = torch.autograd.grad(
        loss,
        [parameter for _, parameter in flat],
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    result = {}
    offset = 0
    for group_name, named in ordered_groups:
        count = len(named)
        result[group_name] = _gradient_group(named, gradients[offset:offset + count])
        offset += count
    del gradients
    return result


def _operator_names(messages: list[str]) -> list[str]:
    names = set()
    pattern = re.compile(r"([A-Za-z0-9_:.]+) does not have a deterministic implementation")
    for message in messages:
        match = pattern.search(message)
        if match:
            names.add(match.group(1))
    return sorted(names)


def _nested(value: dict, path: str):
    current = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def compare_cases(left: dict, right: dict, fields: list[str]) -> dict:
    checks = {field: _nested(left, field) == _nested(right, field) for field in fields}
    mismatched_indices_by_field = {}
    for field, exact in checks.items():
        left_value, right_value = _nested(left, field), _nested(right, field)
        if not exact and isinstance(left_value, list) and isinstance(right_value, list):
            mismatched_indices_by_field[field] = [
                index
                for index, (left_item, right_item) in enumerate(
                    zip(left_value, right_value, strict=False)
                )
                if left_item != right_item
            ]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "mismatched_fields": [field for field, exact in checks.items() if not exact],
        "mismatched_indices_by_field": mismatched_indices_by_field,
    }


def diagnose(runs: dict, comparisons: dict) -> dict:
    if not all(run.get("status") == "completed" for run in runs.values()):
        return {
            "code": "incomplete_forensics",
            "conclusion_ko": "네 실행 중 하나 이상이 완료되지 않았습니다.",
            "next_step_ko": "runtime_error가 발생한 component를 먼저 수정합니다.",
        }
    all_inputs = len({run["input_sha256"] for run in runs.values()}) == 1
    all_initial_students = len({run["student_initial_state_sha256"] for run in runs.values()}) == 1
    all_rng = len({run["rng"]["before_forward"]["torch_cuda_sha256"] for run in runs.values()}) == 1
    vanilla_within_exact = all(
        runs[case]["within_run"]["ce_backward_repeat_exact"] is True
        and runs[case]["within_run"]["vanilla_ce_vs_total_gradient_exact"] is True
        for case in ("vanilla_a", "vanilla_b")
    )
    lg_ce_within_exact = all(
        runs[case]["within_run"]["ce_backward_repeat_exact"] is True
        for case in ("lg_a", "lg_b")
    )
    lg_guidance_within_exact = all(
        runs[case]["within_run"]["guidance_student_backward_repeat_exact"] is True
        and runs[case]["within_run"]["guidance_module_backward_repeat_exact"] is True
        for case in ("lg_a", "lg_b")
    )
    if not all_inputs:
        code = "input_mismatch"
        conclusion = "한 배치의 이미지, label 또는 augmentation 결과가 실행 간 다릅니다."
        next_step = "DataLoader와 augmentation seed를 수정합니다."
    elif not all_initial_students:
        code = "student_initialization_mismatch"
        conclusion = "Student 초기 파라미터가 실행 간 다릅니다."
        next_step = "모델 초기화 seed와 가중치 로딩 순서를 수정합니다."
    elif not all_rng:
        code = "cuda_rng_mismatch_before_forward"
        conclusion = "첫 forward 직전 CUDA RNG 상태가 실행 간 다릅니다."
        next_step = "모델과 guidance 생성 뒤 CUDA seed 재설정을 수정합니다."
    elif not comparisons["vanilla_repeat_forward"]["passed"]:
        code = "student_forward_nondeterminism"
        conclusion = "Vanilla 반복에서 student logits 또는 feature가 forward 단계부터 달라집니다."
        next_step = "DropPath와 student forward CUDA 연산을 분리 검사합니다."
    elif not vanilla_within_exact or not comparisons["vanilla_repeat_ce_backward"]["passed"]:
        code = "base_ce_or_student_backward_nondeterminism"
        conclusion = "Vanilla forward는 같지만 CE/student backward gradient가 달라집니다."
        next_step = "CE와 student backward의 경고 연산을 deterministic 대체 연산으로 좁힙니다."
    elif not comparisons["vanilla_vs_lg_student_forward"]["passed"]:
        code = "method_setup_changes_student_forward"
        conclusion = "같은 초기값과 RNG인데 LG 설정 여부가 student forward를 바꿉니다."
        next_step = "LG 생성과 teacher 로딩 이후 RNG 및 forward 호출 순서를 점검합니다."
    elif not comparisons["lg_repeat_forward"]["passed"]:
        code = "lg_or_teacher_forward_nondeterminism"
        conclusion = "LG 반복에서 student, teacher 또는 guidance forward 값이 달라집니다."
        next_step = "mismatched_fields에 나온 forward component를 단독 실행합니다."
    elif not lg_ce_within_exact or not comparisons["lg_repeat_ce_backward"]["passed"]:
        code = "ce_backward_nondeterminism_under_lg"
        conclusion = "LG 실행에서도 CE backward gradient가 반복 간 달라집니다."
        next_step = "CE/student backward 경로를 deterministic 연산으로 교체해 확인합니다."
    elif not lg_guidance_within_exact or not comparisons["lg_repeat_guidance_backward"]["passed"]:
        code = "lg_guidance_backward_nondeterminism"
        conclusion = "Forward와 CE gradient는 같지만 LG guidance backward gradient가 달라집니다."
        next_step = "LG의 bilinear resize, 1×1 projection, MSE backward를 각각 분리 검사합니다."
    elif not comparisons["lg_repeat_total_backward"]["passed"]:
        code = "total_backward_composition_nondeterminism"
        conclusion = "개별 gradient는 같지만 합산한 total loss backward가 달라집니다."
        next_step = "CE와 β·guidance gradient 합산 및 backward 호출 방식을 점검합니다."
    else:
        code = "one_batch_components_exact"
        conclusion = "한 배치의 forward와 component별 backward가 모두 정확히 일치했습니다."
        next_step = "optimizer.step과 두 번째 학습 step을 추가해 최초 불일치 위치를 찾습니다."
    operators = sorted({name for run in runs.values() for name in run.get("nondeterministic_operators", [])})
    return {
        "code": code,
        "input_exact": all_inputs,
        "student_initialization_exact": all_initial_students,
        "cuda_rng_before_forward_exact": all_rng,
        "vanilla_within_run_backward_exact": vanilla_within_exact,
        "lg_ce_within_run_backward_exact": lg_ce_within_exact,
        "lg_guidance_within_run_backward_exact": lg_guidance_within_exact,
        "nondeterministic_operators": operators,
        "conclusion_ko": conclusion,
        "next_step_ko": next_step,
    }


def _validate_config(config: dict) -> None:
    expected = {
        "protocol_id": PROTOCOL_ID,
        "seed": 1,
        "train_samples": 2975,
        "val_samples": 1,
        "batch_size": 8,
        "image_size": 1024,
        "crop_size": 512,
        "decoder_layers": 1,
        "data_workers": 4,
        "guidance_beta": 0.02,
        "test_used": False,
        "warn_only_determinism": True,
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"Forensics config mismatch: {mismatches}")


def run_case(args, config: dict) -> int:
    import random
    import numpy as np
    import torch
    import torchvision
    import timm
    from torch.nn import functional as F
    import segm.utils.torch as ptu
    from . import official_api as api
    from .data import save_json
    from .full_data import FullDataset, batch_hash, train_loader
    from .runtime import seed_all, state_hash

    method = dict(CASES)[args.case]
    device = torch.device("cuda")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Forensics requires exactly one CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ptu.device = device
    torch.set_num_threads(4)

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
    guide = api.guidance(method, config)
    teacher = None
    if guide is not None:
        guide = guide.to(device).train()
        teacher = api.teacher(args.cache_root).to(device)
    guide_initial = None if guide is None else state_hash(guide)
    teacher_state = None if teacher is None else state_hash(teacher)
    capture = api.FeatureCapture(model)
    seed_all(config["seed"] + 2000, strict_determinism=False)
    torch.use_deterministic_algorithms(True, warn_only=True)
    loader = train_loader(dataset, 1, 0, device)
    image, target, ids = next(iter(loader))
    input_digest = batch_hash(image, target, ids)
    image = image.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    student_named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    guide_named = [] if guide is None else [
        (name, parameter) for name, parameter in guide.named_parameters() if parameter.requires_grad
    ]
    rng = {"before_forward": _rng_snapshot(torch, np, random)}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        logits, student_features = capture.forward(model, image)
        rng["after_student_forward"] = _rng_snapshot(torch, np, random)
        ce = F.cross_entropy(logits, target, ignore_index=255)
        teacher_features = []
        guided = ce.new_zeros(())
        if guide is not None:
            with torch.no_grad():
                teacher_features = teacher.extract_feat(api.teacher_input(image))[1:]
            guided = guide(student_features, teacher_features)
        total = ce + config["guidance_beta"] * guided
        rng["after_all_forward"] = _rng_snapshot(torch, np, random)

        print(f"[L16_FORENSICS_COMPONENT] case={args.case} component=ce_backward_a", flush=True)
        ce_a = _component_gradients(torch, ce, {"student": student_named}, retain_graph=True)
        rng["after_ce_backward_a"] = _rng_snapshot(torch, np, random)
        print(f"[L16_FORENSICS_COMPONENT] case={args.case} component=ce_backward_b", flush=True)
        ce_b = _component_gradients(torch, ce, {"student": student_named}, retain_graph=True)
        rng["after_ce_backward_b"] = _rng_snapshot(torch, np, random)

        guidance_a = guidance_b = None
        if guide is not None:
            groups = {"student": student_named, "guidance": guide_named}
            print(f"[L16_FORENSICS_COMPONENT] case={args.case} component=guidance_backward_a", flush=True)
            guidance_a = _component_gradients(torch, guided, groups, retain_graph=True)
            rng["after_guidance_backward_a"] = _rng_snapshot(torch, np, random)
            print(f"[L16_FORENSICS_COMPONENT] case={args.case} component=guidance_backward_b", flush=True)
            guidance_b = _component_gradients(torch, guided, groups, retain_graph=True)
            rng["after_guidance_backward_b"] = _rng_snapshot(torch, np, random)

        print(f"[L16_FORENSICS_COMPONENT] case={args.case} component=total_backward", flush=True)
        total_groups = {"student": student_named}
        if guide_named:
            total_groups["guidance"] = guide_named
        total_gradients = _component_gradients(torch, total, total_groups, retain_graph=False)
        rng["after_total_backward"] = _rng_snapshot(torch, np, random)
        torch.cuda.synchronize()

    warning_messages = sorted({str(item.message) for item in caught})
    feature_hashes = [_tensor_sha256(feature) for feature in student_features]
    teacher_feature_hashes = [_tensor_sha256(feature) for feature in teacher_features]
    within_run = {
        "ce_backward_repeat_exact": ce_a["student"]["sha256"] == ce_b["student"]["sha256"],
        "guidance_student_backward_repeat_exact": None,
        "guidance_module_backward_repeat_exact": None,
        "vanilla_ce_vs_total_gradient_exact": None,
    }
    if guide is None:
        within_run["vanilla_ce_vs_total_gradient_exact"] = (
            ce_a["student"]["sha256"] == total_gradients["student"]["sha256"]
        )
    else:
        within_run["guidance_student_backward_repeat_exact"] = (
            guidance_a["student"]["sha256"] == guidance_b["student"]["sha256"]
        )
        within_run["guidance_module_backward_repeat_exact"] = (
            guidance_a["guidance"]["sha256"] == guidance_b["guidance"]["sha256"]
        )
    result = {
        "status": "completed",
        "case": args.case,
        "method": method,
        "input_sha256": input_digest,
        "sample_ids": list(ids),
        "student_initial_state_sha256": student_initial,
        "guidance_initial_state_sha256": guide_initial,
        "teacher_state_sha256": teacher_state,
        "rng": rng,
        "forward": {
            "logits_sha256": _tensor_sha256(logits),
            "student_feature_sha256": feature_hashes,
            "student_feature_sequence_sha256": _sequence_sha256(feature_hashes),
            "teacher_feature_sha256": teacher_feature_hashes,
            "teacher_feature_sequence_sha256": _sequence_sha256(teacher_feature_hashes),
            "ce_sha256": _tensor_sha256(ce),
            "ce": float(ce.detach()),
            "guidance_sha256": _tensor_sha256(guided),
            "guidance": float(guided.detach()),
            "total_sha256": _tensor_sha256(total),
            "total": float(total.detach()),
        },
        "gradients": {
            "ce_a": ce_a,
            "ce_b": ce_b,
            "guidance_a": guidance_a,
            "guidance_b": guidance_b,
            "total": total_gradients,
        },
        "within_run": within_run,
        "nondeterministic_operators": _operator_names(warning_messages),
        "determinism_warning_count": len(caught),
        "determinism_warning_messages": warning_messages,
        "runtime_error": None,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
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
        },
    }
    save_json(args.output_dir / "summary.json", result)
    print(
        f"[L16_FORENSICS_CASE_DONE] case={args.case} method={method} "
        f"ce_repeat_exact={within_run['ce_backward_repeat_exact']} "
        f"operators={json.dumps(result['nondeterministic_operators'])}",
        flush=True,
    )
    return 0


def _compact_run(run: dict, returncode: int | None) -> dict:
    return {
        "case": run.get("case"),
        "method": run.get("method"),
        "status": run.get("status"),
        "returncode": returncode,
        "runtime_error": run.get("runtime_error"),
        "input_sha256": run.get("input_sha256"),
        "student_initial_state_sha256": run.get("student_initial_state_sha256"),
        "guidance_initial_state_sha256": run.get("guidance_initial_state_sha256"),
        "teacher_state_sha256": run.get("teacher_state_sha256"),
        "rng": run.get("rng"),
        "forward": run.get("forward"),
        "gradient_hashes": {
            component: {
                group: details.get("sha256")
                for group, details in (groups or {}).items()
            }
            for component, groups in (run.get("gradients") or {}).items()
        },
        "within_run": run.get("within_run"),
        "nondeterministic_operators": run.get("nondeterministic_operators"),
        "determinism_warning_count": run.get("determinism_warning_count"),
        "environment": run.get("environment"),
        "elapsed_seconds": run.get("elapsed_seconds"),
        "peak_cuda_allocated_bytes": run.get("peak_cuda_allocated_bytes"),
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
    for case, method in CASES:
        print(f"[L16_FORENSICS_CASE_START] case={case} method={method}", flush=True)
        command = [
            sys.executable,
            "-u",
            "-m",
            "ibkd_seg.cityscapes.reproducibility_forensics",
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
        if summary_path.exists():
            runs[case] = json.loads(summary_path.read_text())
        else:
            runs[case] = {
                "status": "missing_summary",
                "case": case,
                "method": method,
                "runtime_error": "summary.json missing",
            }

    forward_fields = [
        "input_sha256",
        "student_initial_state_sha256",
        "rng.before_forward",
        "forward.logits_sha256",
        "forward.student_feature_sha256",
        "forward.student_feature_sequence_sha256",
        "forward.ce_sha256",
    ]
    comparisons = {
        "vanilla_repeat_forward": compare_cases(
            runs["vanilla_a"], runs["vanilla_b"], forward_fields
        ),
        "vanilla_repeat_ce_backward": compare_cases(
            runs["vanilla_a"],
            runs["vanilla_b"],
            forward_fields
            + [
                "gradients.ce_a.student.sha256",
                "gradients.ce_b.student.sha256",
                "gradients.total.student.sha256",
            ],
        ),
        "vanilla_vs_lg_student_forward": compare_cases(
            runs["vanilla_a"], runs["lg_a"], forward_fields
        ),
        "lg_repeat_forward": compare_cases(
            runs["lg_a"],
            runs["lg_b"],
            forward_fields
            + [
                "guidance_initial_state_sha256",
                "teacher_state_sha256",
                "forward.teacher_feature_sha256",
                "forward.teacher_feature_sequence_sha256",
                "forward.guidance_sha256",
                "forward.total_sha256",
            ],
        ),
        "lg_repeat_ce_backward": compare_cases(
            runs["lg_a"],
            runs["lg_b"],
            forward_fields
            + [
                "gradients.ce_a.student.sha256",
                "gradients.ce_b.student.sha256",
            ],
        ),
        "lg_repeat_guidance_backward": compare_cases(
            runs["lg_a"],
            runs["lg_b"],
            [
                "gradients.guidance_a.student.sha256",
                "gradients.guidance_b.student.sha256",
                "gradients.guidance_a.guidance.sha256",
                "gradients.guidance_b.guidance.sha256",
            ],
        ),
        "lg_repeat_total_backward": compare_cases(
            runs["lg_a"],
            runs["lg_b"],
            [
                "gradients.total.student.sha256",
                "gradients.total.guidance.sha256",
            ],
        ),
    }
    diagnosis = diagnose(runs, comparisons)
    all_completed = all(
        returncodes.get(case) == 0 and runs[case].get("status") == "completed"
        for case, _ in CASES
    )
    terminal = {
        "status": "completed" if all_completed else "failed",
        "protocol_id": PROTOCOL_ID,
        "all_cases_completed": all_completed,
        "cases_completed": sum(run.get("status") == "completed" for run in runs.values()),
        "total_cases": len(CASES),
        "seed": config["seed"],
        "guidance_beta": config["guidance_beta"],
        "crop_size": config["crop_size"],
        "batch_size": config["batch_size"],
        "test_used": False,
        "scientific_result": False,
        "full_training_authorized": False,
        "next_optimizer_step_forensics_authorized": (
            all_completed and diagnosis["code"] == "one_batch_components_exact"
        ),
        "full_2000_reproducibility_audit_authorized": False,
        "runs": {
            case: _compact_run(runs[case], returncodes.get(case)) for case, _ in CASES
        },
        "comparisons": comparisons,
        "diagnosis": diagnosis,
    }
    save_json(
        args.output_dir / "forensics_summary.json",
        {"terminal_result": terminal, "runs": runs},
    )
    print(
        "[CITYSCAPES_L16_REPRO_FORENSICS_DONE] "
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
