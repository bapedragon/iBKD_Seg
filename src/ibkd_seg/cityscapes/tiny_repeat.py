"""Bounded LG/LG/ALG diagnostic: report every comparison without early exit."""
from __future__ import annotations

import hashlib
import math
from collections import Counter


def tensor_hashes(items):
    digest = hashlib.sha256()
    for name, value in sorted(items, key=lambda pair: pair[0]):
        digest.update(name.encode())
        if value is None:
            digest.update(b"none")
        else:
            tensor = value.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def gradient_hash(module):
    return tensor_hashes((name, p.grad) for name, p in module.named_parameters())


def rng_hash():
    import random
    import numpy as np
    import torch
    digest = hashlib.sha256()
    digest.update(repr(random.getstate()).encode())
    state = np.random.get_state()
    digest.update(repr((state[0], *state[2:])).encode())
    digest.update(state[1].tobytes())
    digest.update(torch.get_rng_state().numpy().tobytes())
    for value in torch.cuda.get_rng_state_all():
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def warning_summary(records):
    counts = Counter((str(w.message), w.category.__name__, w.filename, w.lineno) for w in records)
    return [{"message": message, "category": category, "filename": filename,
             "lineno": line, "count": count}
            for (message, category, filename, line), count in sorted(counts.items())]


def fixed_calibration(rows, beta):
    import statistics
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("Invalid fixed diagnostic beta")
    return {"ce_median": statistics.median(r["ce"] for r in rows),
            "guidance_median": statistics.median(r["guidance"] for r in rows),
            "beta_candidates": [], "pilot_beta": beta, "optimizer_updates": 0,
            "validation_used": False, "status": "shared_fixed_beta_not_reestimated"}


SCALARS = ("loss", "ce", "guidance", "grad_norm_unclipped")
HASHES = ("input_sha256", "rng_before", "rng_after", "logits_sha256",
          "student_gradient_sha256", "guidance_gradient_sha256",
          "student_after_sha256", "guidance_after_sha256")


def pair_report(left, right, *, steps, beta, rtol, atol):
    """Scalar tolerance is the original smoke rule; exact hashes are diagnostic."""
    checks = {}
    for key in ("student_initial_state_sha256", "guidance_initial_state_sha256",
                "teacher_state_sha256", "input_sha256", "calibration_input_sha256", "source_sha256", "config_sha256"):
        checks[key] = left.get(key) is not None and left.get(key) == right.get(key)
    a, b = left.get("losses", []), right.get("losses", [])
    complete = len(a) == len(b) == steps and all(
        x.get("step") == y.get("step") == i for i, (x, y) in enumerate(zip(a, b), 1))
    checks["complete_trajectory"] = complete
    checks["exact_shared_beta_all_steps"] = complete and all(
        x.get("beta") == y.get("beta") == beta for x, y in zip(a, b))
    first, maxima = None, {}
    for key in SCALARS:
        differences = []
        for x, y in zip(a, b):
            v, w = x[key], y[key]
            if not math.isfinite(v) or not math.isfinite(w):
                raise ValueError("Nonfinite diagnostic trajectory")
            delta = abs(v - w)
            differences.append((delta, delta / max(abs(v), abs(w), 1e-30), x["step"]))
            if not math.isclose(v, w, rel_tol=rtol, abs_tol=atol):
                event = {"step": x["step"], "key": key, "left": v, "right": w,
                         "absolute_difference": delta,
                         "allowed_difference": max(atol, rtol * max(abs(v), abs(w)))}
                if first is None or event["step"] < first["step"]:
                    first = event
        if differences:
            largest = max(differences)
            maxima[key] = {"max_absolute_difference": largest[0], "step_at_max_absolute": largest[2],
                           "max_relative_difference": max(item[1] for item in differences)}
    hash_report = {}
    for key in HASHES:
        missing = any(key not in x.get("trace", {}) or key not in y.get("trace", {}) for x, y in zip(a, b))
        mismatch = next((x["step"] for x, y in zip(a, b)
                         if x.get("trace", {}).get(key) != y.get("trace", {}).get(key)), None)
        hash_report[key] = {"complete": complete and not missing, "first_different_step": mismatch,
                            "all_equal": complete and not missing and mismatch is None}
    checks["full_input_trace_matches"] = hash_report["input_sha256"]["all_equal"]
    checks["rng_sequence_matches"] = all(hash_report[k]["all_equal"] for k in ("rng_before", "rng_after"))
    checks["all_hash_fields_present"] = all(r["complete"] for r in hash_report.values())
    return {"left": left["run_id"], "right": right["run_id"], "identity_checks": checks,
            "numerical_trajectory_match": complete and first is None,
            "first_scalar_mismatch": first, "scalar_differences": maxima,
            "exact_hash_comparison": hash_report,
            "passed": all(checks.values()) and first is None and
                      left["status"] == right["status"] == "passed"}


def compare_repeats(rows, config):
    by_id = {row["run_id"]: row for row in rows}
    pairs = {}
    for name, a, b in (("lg_repeat", "lg_a", "lg_b"), ("lg_a_vs_alg", "lg_a", "alg"),
                       ("lg_b_vs_alg", "lg_b", "alg")):
        if a not in by_id or b not in by_id:
            pairs[name] = {"passed": False, "error": "missing_run"}
        else:
            pairs[name] = pair_report(by_id[a], by_id[b], steps=config["steps"],
                                     beta=config["diagnostic_fixed_beta"],
                                     rtol=config["resume_rtol"], atol=config["resume_atol"])
    completed = len(rows) == 3 and set(by_id) == {"lg_a", "lg_b", "alg"} and all(r["status"] == "passed" for r in rows)
    passed = completed and all(p["passed"] for p in pairs.values())
    if not completed:
        interpretation = "execution_or_resume_failure_review_each_run"
    elif not all(all(p.get("identity_checks", {}).values()) for p in pairs.values()):
        interpretation = "input_initialization_beta_or_rng_mismatch"
    elif not pairs["lg_repeat"]["numerical_trajectory_match"]:
        interpretation = "same_LG_repeat_varies_not_evidence_of_ALG_controller_effect"
    elif not passed:
        interpretation = "LG_repeat_matches_but_cross_method_differs_requires_investigation"
    else:
        interpretation = "all_three_match_within_original_scalar_tolerance_not_proof_of_long_run_reproducibility"
    return {"status": "passed" if passed else "failed", "all_runs_completed": completed,
            "fixed_beta": config["diagnostic_fixed_beta"],
            "tolerance": {"rtol": config["resume_rtol"], "atol": config["resume_atol"]},
            "pairs": pairs, "interpretation": interpretation,
            "automatic_tolerance_changes": False, "natural_controller_off_tested": False}
