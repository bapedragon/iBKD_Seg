#!/usr/bin/env python3
"""25-batch β 후보 생성만 수행하는 H200 이슈 진입점."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO=Path(__file__).resolve().parents[3]


def main():
    os.chdir(REPO)
    output=Path(os.environ.get("B0_CALIBRATION_OUTPUT","/app/output/cityscapes_b0_beta_calibration_v1")).resolve()
    cache=Path(os.environ.get("B0_CALIBRATION_CACHE","/app/scratch/cityscapes_b0_smoke_v1/assets")).resolve()
    data=Path(os.environ.get("B0_CALIBRATION_DATA","/app/scratch/cityscapes_b0_smoke_v1/cityscapes")).resolve()
    zip_root=Path(os.environ.get("CITYSCAPES_ZIP_DIR","/app/data/chaoyang")).resolve()
    if output.exists() and any(p.name!="run.log" for p in output.iterdir()):
        raise FileExistsError(f"새 B0_CALIBRATION_OUTPUT 경로가 필요합니다: {output}")
    output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    report={"status":"running","calibration_spec_id":"cityscapes_segformer_b0_beta_calibration_v1",
            "families":[],"methods":[],"optimizer_updates":0,"backward_calls":0,"selected_epoch":None,
            "metrics":None,"metrics_reason":"Calibration only; no evaluation",
            "selection_performed":False,"beta_candidates_frozen":False,"test_used":False,
            "validation_samples":0,"output":str(output)}
    os.environ.update(PYTHONPATH=str(REPO/"src"),PYTHONUNBUFFERED="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    env=os.environ.copy()
    stage="initialization"
    def run(command,*,check=True):
        remaining=3*3600-(time.monotonic()-start)
        if remaining<=0:raise TimeoutError("3시간 calibration 작업 제한 도달")
        return subprocess.run(command,env=env,check=check,timeout=remaining).returncode
    def save():
        (output/"calibration_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    def set_stage(name):
        nonlocal stage
        stage=name
        report["stage"]=name
        print("[B0_CALIBRATION] stage="+name,flush=True)
        save()
    try:
        set_stage("install")
        run([sys.executable,"-m","pip","install","--disable-pip-version-check","-e",".",
             "opencv-python-headless==4.13.0.92","gdown==5.2.0","pytest==8.4.2"])
        with (output/"pip_freeze.txt").open("w") as target:
            subprocess.run([sys.executable,"-m","pip","freeze"],stdout=target,env=env,check=True)
        sys.path.insert(0,str(REPO/"src"))
        from ibkd_seg.cityscapes.b0.calibration import CONFIG,verify_spec,merge_families
        from ibkd_seg.cityscapes.b0.calibration_data import verify_preparation,prepare_inputs
        from ibkd_seg.cityscapes.b0.assets import prepare as prepare_assets
        from ibkd_seg.cityscapes.b0.reproducibility import configure_runtime
        from ibkd_seg.cityscapes.data import sha256,save_json
        from ibkd_seg.cityscapes.runtime import source_hash
        spec=verify_spec();configure_runtime()
        report["calibration_spec"]=spec
        report["calibration_spec_sha256"]=sha256(CONFIG)
        report["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
        report["source_sha256"]={str(p.relative_to(REPO)):sha256(p) for p in sorted((REPO/"src/ibkd_seg/cityscapes/b0").glob("*.py"))}
        report["shared_cityscapes_phase1_source_sha256"]=source_hash()
        set_stage("focused_unit_checks")
        run([sys.executable,"-m","pytest","-q","tests/test_cityscapes_b0_calibration.py"])
        set_stage("data")
        metadata=[(data/name).exists() for name in ("manifest.json","preparation.json")]
        if any(metadata) and not all(metadata):raise ValueError("불완전한 기존 data provenance; 새 데이터 경로가 필요합니다")
        report["prepared_data_cache_reused"]=all(metadata)
        if not all(metadata):
            run([sys.executable,str(REPO/"phase4/phase4_cityscapes/scripts/check_cityscapes_upload.py"),
                 "--search-root",str(zip_root),"--output",str(output/"zip_audit.json")])
            records=json.loads((output/"zip_audit.json").read_text())["files"]
            links=cache/"zip_links";links.mkdir(parents=True,exist_ok=True)
            for record in records:
                source=Path(record["path"]).resolve();target=links/source.name
                if target.is_symlink() or target.exists():
                    if target.resolve()!=source:raise ValueError("ZIP link collision")
                else:target.symlink_to(source)
            run([sys.executable,"-m","ibkd_seg.cityscapes.prepare","--zip-dir",str(links),"--data-dir",str(data)])
        report["data_verification"]=verify_preparation(data,spec["expected_archives"])
        set_stage("assets")
        report["asset_manifest"]=prepare_assets(cache,weight_names=("cirkd_teacher.pth","nvidia_mit_b0.bin"),
                                                manifest_name="calibration_asset_manifest.json")
        set_stage("shared_train_inputs")
        report["input_identity"]=prepare_inputs(cache,data,output)
        for family in ("lg_alg","ibkd"):
            set_stage(family)
            target=output/family
            code=run([sys.executable,"-m","ibkd_seg.cityscapes.b0.calibration","--family",family,
                      "--cache",str(cache),"--inputs",str(output/"inputs"),"--output",str(target)],check=False)
            if (target/"summary.json").exists():record=json.loads((target/"summary.json").read_text())
            else:record={"status":"failed","family":family,"error":"Worker exited without summary"}
            record["exit_code"]=code
            if code:record["status"]="failed"
            report["families"].append(record)
            save()
        set_stage("freeze_candidates")
        report.update(merge_families(report["families"]))
        candidates={key:report[key] for key in ("calibration_spec_id","calibration_spec_sha256","git_commit",
                    "source_sha256","shared_cityscapes_phase1_source_sha256","asset_manifest","data_verification",
                    "input_identity","methods","optimizer_updates","selected_epoch","metrics","metrics_reason",
                    "selection_performed","beta_candidates_frozen")}
        candidates["initial_states"]={r["family"]:{k:v for k,v in r.items() if k.endswith("state_sha256")} for r in report["families"]}
        save_json(output/"beta_candidates.json",candidates)
        report["beta_candidates_sha256"]=sha256(output/"beta_candidates.json")
        report["stage"]="completed"
    except Exception as error:
        import traceback
        traceback.print_exc()
        report.update(status="failed",failed_stage=stage,error=repr(error),beta_candidates_frozen=False)
    report["elapsed_seconds"]=time.monotonic()-start
    save()
    print(json.dumps(report,ensure_ascii=False),flush=True)
    raise SystemExit(0 if report["status"]=="passed" else 1)


if __name__=="__main__":main()
