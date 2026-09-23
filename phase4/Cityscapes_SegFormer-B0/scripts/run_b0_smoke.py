#!/usr/bin/env python3
"""H200 이슈 진입점: 설치·자료 검사·7개 smoke를 실행하고 마지막 줄에 전체 JSON을 출력한다."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO=Path(__file__).resolve().parents[3]
METHODS=("vanilla","lg","alg","ibkd_lambda025","ibkd_lambda050","fskd","c2vkd_clip_pool")


def main():
    os.chdir(REPO)
    output=Path(os.environ.get("B0_SMOKE_OUTPUT","/app/output/cityscapes_b0_smoke_v1")).resolve()
    cache=Path(os.environ.get("B0_SMOKE_CACHE","/app/scratch/cityscapes_b0_smoke_v1/assets")).resolve()
    data=Path(os.environ.get("B0_SMOKE_DATA","/app/scratch/cityscapes_b0_smoke_v1/cityscapes")).resolve()
    zip_root=Path(os.environ.get("CITYSCAPES_ZIP_DIR","/app/data/chaoyang")).resolve()
    if output.exists() and any(p.name!="run.log" for p in output.iterdir()):
        raise FileExistsError(f"새 출력 경로가 필요합니다: {output}")
    output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    report={"status":"running","scientific_result":False,"methods":[],"expected_methods":list(METHODS),
            "primary_methods":6,"supplementary_methods":1,"selected_epoch":None,"test_used":False,
            "full_validation":False,"output":str(output)}
    env=os.environ.copy()
    env.update(PYTHONPATH=str(REPO/"src"),MAX_JOBS="2",TORCH_CUDA_ARCH_LIST="9.0",PYTHONUNBUFFERED="1")
    stage="initialization"
    def run(command,*,check=True):
        remaining=9*3600-(time.monotonic()-start)
        if remaining<=0:raise TimeoutError("9시간 smoke 작업 제한 도달")
        return subprocess.run(command,env=env,check=check,timeout=remaining).returncode
    def save():
        (output/"smoke_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    try:
        stage="install"
        print("[B0_PIPELINE] stage="+stage,flush=True)
        run([sys.executable,"-m","pip","install","--disable-pip-version-check","-e",".",
             "opencv-python-headless==4.13.0.92","gdown==5.2.0","ninja==1.13.0","pytest==8.4.2"])
        run([sys.executable,"-m","pip","install","--disable-pip-version-check","--no-build-isolation","torchsort==0.1.10"])
        with (output/"pip_freeze.txt").open("w") as target:
            subprocess.run([sys.executable,"-m","pip","freeze"],stdout=target,env=env,check=True)
        stage="focused_unit_checks"
        run([sys.executable,"-m","pytest","-q","tests/test_cityscapes_b0.py"])
        stage="zip_audit"
        print("[B0_PIPELINE] stage="+stage,flush=True)
        audit=REPO/"phase4/phase4_cityscapes/scripts/check_cityscapes_upload.py"
        run([sys.executable,str(audit),"--search-root",str(zip_root),"--output",str(output/"zip_audit.json")])
        records=json.loads((output/"zip_audit.json").read_text())["files"]
        links=cache/"zip_links";links.mkdir(parents=True,exist_ok=True)
        for record in records:
            source=Path(record["path"])
            target=links/source.name
            if target.exists() and target.resolve()!=source.resolve():raise ValueError("ZIP link collision")
            if not target.exists():target.symlink_to(source)
        stage="prepare_data"
        if not (data/"manifest.json").exists():
            run([sys.executable,"-m","ibkd_seg.cityscapes.prepare","--zip-dir",str(links),"--data-dir",str(data)])
        stage="assets"
        run([sys.executable,"-m","ibkd_seg.cityscapes.b0.assets","--cache",str(cache)])
        stage="shared_inputs"
        # Import after installation, inside this process, using the same explicit source path.
        sys.path.insert(0,str(REPO/"src"))
        from ibkd_seg.cityscapes.b0.data import prepare
        from ibkd_seg.cityscapes.data import sha256
        identity=prepare(cache,data,output)
        report["input_identity"]=identity
        report["asset_manifest"]=json.loads((cache/"asset_manifest.json").read_text())
        report["source_sha256"]={str(p.relative_to(REPO)):sha256(p) for p in sorted((REPO/"src/ibkd_seg/cityscapes/b0").glob("*.py"))}
        from ibkd_seg.cityscapes.runtime import source_hash
        report["shared_cityscapes_phase1_source_sha256"]=source_hash()
        report["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
        report["spec_sha256"]={p.name:sha256(p) for p in sorted((REPO/"phase4/Cityscapes_SegFormer-B0/configs").glob("*.json"))}
        save()
        for method in METHODS:
            stage=method
            print("[B0_PIPELINE] method="+method,flush=True)
            path=output/method
            code=run([sys.executable,"-m","ibkd_seg.cityscapes.b0.smoke","--cache",str(cache),
                      "--inputs",str(output/"inputs.pt"),"--output",str(path),"--method",method],check=False)
            summary=path/"summary.json"
            row=json.loads(summary.read_text()) if summary.exists() else {
                "method":method,"status":"failed","error":"process exited before summary","returncode":code}
            if code!=0:row["status"]="failed"
            report["methods"].append(row);save()
        stage="cross_method_checks"
        passed=[r for r in report["methods"] if r["status"]=="passed"]
        if len({r["student_initial_state_sha256"] for r in passed})>1:raise ValueError("Student initialization differs")
        if len({r["teacher_state_sha256"] for r in passed if r["teacher_state_sha256"] is not None})>1:
            raise ValueError("Teacher state differs")
        if len({r["input_sha256"] for r in passed})>1:raise ValueError("Inputs differ")
        by_method={r["method"]:r for r in passed}
        if "lg" in by_method and "alg" in by_method:
            a,b=by_method["lg"],by_method["alg"]
            if a["smoke_beta"]!=b["smoke_beta"]:raise ValueError("LG/ALG beta differs")
            import math
            for x,y in zip(a["losses"],b["losses"]):
                if not math.isclose(x["loss"],y["loss"],rel_tol=2e-5,abs_tol=2e-6):
                    raise ValueError("LG/ALG differ before controller observation")
        report["passed_methods"]=len(passed)
        report["primary_status"]="passed" if all(by_method.get(m,{}).get("status")=="passed" for m in METHODS[:-1]) else "failed"
        report["c2vkd_supplementary_status"]=by_method.get(METHODS[-1],{}).get("status","failed")
        report["status"]="passed" if len(passed)==7 else "failed"
    except Exception as error:
        report.update(status="failed",failed_stage=stage,error=repr(error))
    report["elapsed_seconds"]=time.monotonic()-start
    report["not_run_methods"]=[m for m in METHODS if m not in {r["method"] for r in report["methods"]}]
    save()
    print("[CITYSCAPES_B0_SMOKE_DONE] status="+report["status"],flush=True)
    # Last line must contain the actual results, not only a path or success marker.
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return 0 if report["status"]=="passed" else 1


if __name__=="__main__":raise SystemExit(main())
