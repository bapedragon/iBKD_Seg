"""Cityscapes audit, smoke, training matrix, and standalone validation CLI."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

import torch

from .data import audit, json_hash, save_json, sha256, verify_manifest
from .runtime import DEFAULT_CONFIG, METHODS, evaluate_checkpoint, load_config, train


def build_commands(args, config):
    seeds = args.seeds or config["student_seeds"]
    if len(seeds) != len(set(seeds)) or any(s not in config["student_seeds"] for s in seeds):
        raise ValueError("Matrix seeds must be a unique subset of the configured seeds")
    base = [sys.executable, "-m", "ibkd_seg.cityscapes.run", "train",
            "--config", str(args.config.resolve()), "--data-dir", str(args.data_dir.resolve()),
            "--manifest", str(args.manifest.resolve()), "--device", args.device]
    teacher_dir = args.output_dir.resolve() / f"teacher_seed{config['teacher_seed']}"
    jobs = [("teacher", base + ["--kind", "teacher", "--seed", str(config["teacher_seed"]),
                                "--output-dir", str(teacher_dir)], teacher_dir)]
    for seed in seeds:
        for method in METHODS:
            output = args.output_dir.resolve() / f"{method}_seed{seed}"
            command = base + ["--kind", "student", "--method", method, "--seed", str(seed), "--output-dir", str(output)]
            if method != "vanilla":
                command += ["--teacher-checkpoint", str(teacher_dir / "best.pt")]
            jobs.append((f"{method}_seed{seed}", command, output))
    if args.resume:
        for _, command, output in jobs:
            if (output / "latest.pt").is_file():
                command.append("--resume")
    return jobs


def aggregate(jobs, output):
    rows, groups = [], []
    shared_teacher_hash = None
    teacher_identity = None
    initial_by_seed = {}
    for name, _, directory in jobs:
        path = directory / "summary.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text())
        if summary["status"] != "complete":
            continue
        identity = summary["identity"]
        if sha256(directory / "best.pt") != summary["best_checkpoint_sha256"]:
            raise ValueError(f"Selected checkpoint changed: {name}")
        if identity["kind"] == "teacher":
            shared_teacher_hash = summary["best_checkpoint_sha256"]
            teacher_identity = identity
        else:
            if teacher_identity is None:
                raise ValueError("Completed shared teacher must precede student results")
            for key in ("config_sha256", "manifest_sha256", "source_sha256"):
                if identity[key] != teacher_identity[key]:
                    raise ValueError(f"Mixed result contracts: {key}")
            seed = identity["seed"]
            initial = identity["initial_model_sha256"]
            if seed in initial_by_seed and initial_by_seed[seed] != initial:
                raise ValueError("Student initialization differs across methods")
            initial_by_seed[seed] = initial
            if identity["method"] != "vanilla" and identity["teacher_checkpoint_sha256"] != shared_teacher_hash:
                raise ValueError("Students used different teachers")
        rows.append({"run": name, "kind": identity["kind"], "method": identity["method"],
                     "seed": identity["seed"], "selected_epoch": summary["selected_epoch"],
                     "selection_metric": summary["selection_metric"],
                     "val_pixel_accuracy_percent": 100 * summary["selected_val_pixel_accuracy"],
                     "val_miou_percent": 100 * summary["selected_val_miou"],
                     "guidance_stop_epoch": summary["guidance_stop_epoch"],
                     "checkpoint_sha256": summary["best_checkpoint_sha256"]})
    for method in METHODS:
        matched = [r for r in rows if r["method"] == method]
        accuracy = [r["val_pixel_accuracy_percent"] for r in matched]
        iou = [r["val_miou_percent"] for r in matched]
        groups.append({"method": method, "seeds": [r["seed"] for r in matched],
                       "mean_val_pixel_accuracy_percent": statistics.mean(accuracy) if accuracy else None,
                       "sample_sd_pixel_accuracy_percent": statistics.stdev(accuracy) if len(accuracy) > 1 else None,
                       "mean_val_miou_percent": statistics.mean(iou) if iou else None,
                       "sample_sd_miou_percent": statistics.stdev(iou) if len(iou) > 1 else None})
    save_json(output / "matrix_summary.json", {
        "status": "complete" if len(rows) == len(jobs) else "partial",
        "scientific_result": False,
        "selection_metric": None if teacher_identity is None else teacher_identity["selection_metric"],
        "metric": "Pixel accuracy and mIoU at the same selected checkpoint; not independent test",
        "completed_runs": len(rows), "planned_runs": len(jobs), "rows": rows, "aggregates": groups,
    })
    if rows:
        with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_matrix(args):
    config = load_config(args.config)
    jobs = build_commands(args, config)
    for name, command, _ in jobs:
        print(f"{name}: {shlex.join(command)}", flush=True)
    if not args.execute:
        print("계획만 출력했습니다. 실제 학습은 동일 명령에 --execute를 추가합니다.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Fail before starting the teacher if any selected run would be overwritten.
    for name, command, directory in jobs:
        if directory.exists() and any(directory.iterdir()) and "--resume" not in command:
            raise FileExistsError(f"Existing output cannot be overwritten: {name}")
    logs = args.output_dir / "logs"
    logs.mkdir(exist_ok=True)
    save_json(args.output_dir / "matrix_plan.json", {
        "config_sha256": json_hash(config), "jobs": [{"name": n, "argv": c} for n, c, _ in jobs],
    })
    for name, command, _ in jobs:
        with (logs / f"{name}.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                result = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        aggregate(jobs, args.output_dir)
        if result:
            raise RuntimeError(f"{name} failed with exit {result}; see {logs / (name + '.log')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit", help="Hash and validate official fine train/val files")
    audit_parser.add_argument("--data-dir", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    smoke = sub.add_parser("smoke", help="Synthetic-only end-to-end check; no scientific scores")
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--threads", type=int, default=2)
    preflight = sub.add_parser("preflight", help="Synthetic train steps at the configured crop/batch/precision")
    preflight.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    preflight.add_argument("--device", default="cuda")
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--steps", type=int, default=3)
    for name in ("train", "matrix", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--data-dir", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--device", default="cuda")
        if name == "evaluate":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
            command.add_argument("--output-dir", type=Path, required=True)
            command.add_argument("--resume", action="store_true")
            if name == "matrix":
                command.add_argument("--seeds", type=int, nargs="+")
                command.add_argument("--execute", action="store_true", help="Without this flag, print commands only")
            else:
                command.add_argument("--kind", choices=("teacher", "student"), required=True)
                command.add_argument("--method", choices=METHODS)
                command.add_argument("--seed", type=int, required=True)
                command.add_argument("--teacher-checkpoint", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        manifest = audit(args.data_dir)
        save_json(args.output, manifest)
        print(json.dumps({"manifest": str(args.output), "sha256": sha256(args.output),
                          "train": len(manifest["splits"]["train"]), "val": len(manifest["splits"]["val"])}))
    elif args.command == "smoke":
        from .smoke import run_smoke
        if args.threads <= 0:
            parser.error("--threads must be positive")
        torch.set_num_threads(args.threads)
        run_smoke(args.output_dir, torch.device(args.device))
    elif args.command == "matrix":
        run_matrix(args)
    elif args.command == "preflight":
        from .preflight import run_preflight
        run_preflight(load_config(args.config), torch.device(args.device), args.output, steps=args.steps)
    else:
        manifest = json.loads(args.manifest.read_text())
        print("Verifying dataset byte sizes and SHA-256 hashes...", flush=True)
        verify_manifest(args.data_dir, manifest)
        if args.command == "train":
            result = train(config=load_config(args.config), data_root=args.data_dir, manifest=manifest,
                           output=args.output_dir, kind=args.kind, method=args.method, seed=args.seed,
                           device=torch.device(args.device), teacher_checkpoint=args.teacher_checkpoint, resume=args.resume)
            print(json.dumps(result), flush=True)
        else:
            result = evaluate_checkpoint(args.checkpoint, args.data_dir, manifest, torch.device(args.device))
            save_json(args.output, result)
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
