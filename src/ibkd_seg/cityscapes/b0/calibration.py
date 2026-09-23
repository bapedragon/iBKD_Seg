"""No-update beta calibration: 25 shared batches, four predeclared ratios."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import torch

from .assets import SPEC,modules,verify_protocols
from .models import Student,Teacher,Locality,AllBlockIBKD
from .smoke import compute,cpu_tree,rng_state,restore_rng,assert_tree
from .reproducibility import configure_runtime
from .calibration_data import load_batches
from ..data import save_json,sha256
from ..runtime import seed_all,state_hash

RATIOS=(.03,.07,.15,.30)
METHODS=("lg","alg","ibkd_lambda025","ibkd_lambda050")
CONFIG=SPEC/"b0_beta_calibration_v1.json"


def verify_spec():
    verify_protocols()
    spec=json.loads(CONFIG.read_text())
    for name,expected in spec["protocol_sha256"].items():
        if sha256(SPEC/name)!=expected:raise ValueError(f"Calibration protocol changed: {name}")
    if tuple(spec["target_ratios"])!=RATIOS or spec["batches"]!=25:
        raise ValueError("Calibration constants differ from the frozen specification")
    return spec


def spread(values):
    return {"min":min(values),"median":statistics.median(values),"max":max(values)}


def candidate_grid(rows,method):
    if method not in METHODS:raise ValueError(method)
    ce=[r["ce"] for r in rows]
    if method in ("lg","alg"):
        guidance=[r["locality"] for r in rows]
    else:
        ratio=.25 if method.endswith("025") else .5
        guidance=[(1-ratio)*r["alignment"]+ratio*r["fusion"] for r in rows]
    if not ce or any(not math.isfinite(x) or x<=0 for x in ce+guidance):
        raise ValueError("Calibration requires finite positive CE and guidance")
    c,g=statistics.median(ce),statistics.median(guidance)
    if g<=1e-12:raise ValueError("Guidance too small for a meaningful beta grid")
    candidates=[]
    for ratio in RATIOS:
        raw=ratio*c/g
        beta=float(f"{raw:.6g}")
        if not math.isfinite(beta) or beta<=0:raise ValueError("Invalid beta candidate")
        candidates.append({"candidate_id":f"beta_r{round(100*ratio):03d}","target_ratio":ratio,
                           "beta_unrounded":raw,"beta":beta,
                           "measured_weighted_guidance_to_ce":spread([beta*y/x for x,y in zip(ce,guidance)])})
    if len({r["beta"] for r in candidates})!=4:raise ValueError("Rounded beta candidates overlap")
    return {"method":method,"median_ce":c,"median_raw_guidance":g,"ce":spread(ce),
            "raw_guidance":spread(guidance),"candidates":candidates,"selected_beta":None,
            "optimizer_updates":0,"selected_epoch":None,"metrics":None,
            "last_loss_components":{k:v for k,v in rows[-1].items() if k not in ("batch","seconds")}}


def observe_batches(family,student,teacher,guide,batches,device,progress,*,expected_batches=25,emit=None):
    if family not in ("lg_alg","ibkd"):raise ValueError(family)
    method="lg" if family=="lg_alg" else "ibkd_lambda025"
    initial=cpu_tree(student.state_dict());guide_initial=cpu_tree(guide.state_dict())
    before_rng=rng_state();teacher_hash=state_hash(teacher)
    progress.update(losses=[],stage="measure_no_grad",optimizer_updates=0,backward_calls=0)
    try:
        with torch.no_grad():
            for index,(x,y) in enumerate(batches,1):
                if index>expected_batches:raise ValueError("More calibration batches than specified")
                if device.type=="cuda":torch.cuda.synchronize()
                start=time.perf_counter()
                _,terms,_=compute(method,student,teacher,guide,x.to(device),y.to(device),1.)
                values={name:float(value) for name,value in terms.items()}
                if any(not math.isfinite(x) for x in values.values()):raise ValueError("Nonfinite calibration loss")
                if device.type=="cuda":torch.cuda.synchronize()
                row={"batch":index,**values,"seconds":time.perf_counter()-start}
                progress["losses"].append(row)
                progress["completed_batches"]=index
                if emit:emit(row)
        if len(progress["losses"])!=expected_batches:raise ValueError("Incomplete calibration batches")
        # BN buffers may change in train mode, but no trainable parameter may change.
        assert_tree(dict(student.named_parameters()),{n:initial[n] for n,_ in student.named_parameters()},rtol=0,atol=0,path="student_parameters")
        assert_tree(dict(guide.named_parameters()),{n:guide_initial[n] for n,_ in guide.named_parameters()},rtol=0,atol=0,path="guide_parameters")
        if any(p.grad is not None for model in (student,teacher,guide) for p in model.parameters()):
            raise ValueError("Calibration unexpectedly accumulated gradients")
        if state_hash(teacher)!=teacher_hash:raise ValueError("Frozen teacher changed")
        progress.update(parameters_unchanged=True,gradients_absent=True,teacher_frozen_verified=True)
    finally:
        student.load_state_dict(initial,strict=True);guide.load_state_dict(guide_initial,strict=True)
        restore_rng(before_rng)
        assert_tree(student.state_dict(),initial,rtol=0,atol=0,path="restored_student")
        assert_tree(guide.state_dict(),guide_initial,rtol=0,atol=0,path="restored_guide")
        assert_tree(rng_state(),before_rng,rtol=0,atol=0,path="restored_rng")
        progress["student_guide_buffers_and_rng_restored"]=True
    methods=("lg","alg") if family=="lg_alg" else ("ibkd_lambda025","ibkd_lambda050")
    return [candidate_grid(progress["losses"],name) for name in methods]


def measure(args,progress):
    verify_spec();execution=configure_runtime();device=torch.device("cuda")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError("Exactly one CUDA GPU is required")
    progress.update(stage="models",execution=execution)
    upstream,teacher_source,_=modules(args.cache)
    seed_all(1);student=Student(upstream,args.cache/"weights/nvidia_mit_b0.bin").to(device).train()
    seed_all(100001)
    guide=(Locality() if args.family=="lg_alg" else AllBlockIBKD(deterministic=True)).to(device).train()
    teacher=Teacher(teacher_source,args.cache/"weights/cirkd_teacher.pth").to(device)
    progress.update(student_initial_state_sha256=state_hash(student),guide_initial_state_sha256=state_hash(guide),
                    teacher_state_sha256=state_hash(teacher),input_identity_sha256=sha256(args.inputs/"identity.json"),
                    calibration_spec_sha256=sha256(CONFIG),environment={"python":platform.python_version(),
                    "torch":str(torch.__version__),"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name()},
                    ibkd_deterministic_execution=getattr(guide,"deterministic_contract",None))
    seed_all(200001);torch.cuda.reset_peak_memory_stats()
    def emit(row):
        save_json(args.output/"summary.json",progress)
        print("[B0_CALIBRATION_BATCH] "+json.dumps({"family":args.family,**row}),flush=True)
    grids=observe_batches(args.family,student,teacher,guide,load_batches(args.inputs),device,progress,emit=emit)
    progress.update(status="passed",stage="completed",methods=grids,calibration_peak_cuda_bytes=torch.cuda.max_memory_allocated())
    return progress


def merge_families(records):
    if len(records)!=2 or {r["family"] for r in records}!={"lg_alg","ibkd"}:raise ValueError("Missing calibration family")
    if any(r["status"]!="passed" or r["completed_batches"]!=25 for r in records):raise ValueError("Calibration incomplete")
    for key in ("student_initial_state_sha256","teacher_state_sha256","input_identity_sha256","calibration_spec_sha256"):
        if len({r[key] for r in records})!=1:raise ValueError(f"Families differ: {key}")
    for r in records:
        if r["optimizer_updates"]!=0 or r["backward_calls"]!=0 or not all(r[k] for k in ("parameters_unchanged","gradients_absent","teacher_frozen_verified","student_guide_buffers_and_rng_restored")):
            raise ValueError("No-update calibration contract failed")
    first,second=records
    if [r["ce"] for r in first["losses"]]!=[r["ce"] for r in second["losses"]]:
        raise ValueError("Shared initial student produced different CE measurements")
    by_method={g["method"]:g for r in records for g in r["methods"]}
    if set(by_method)!=set(METHODS):raise ValueError("Missing candidate grid")
    if by_method["lg"]["candidates"]!=by_method["alg"]["candidates"]:raise ValueError("LG/ALG grids differ")
    return {"status":"passed","beta_candidates_frozen":True,"selection_performed":False,
            "methods":[by_method[m] for m in METHODS],"optimizer_updates":0,"selected_epoch":None,
            "metrics":None,"metrics_reason":"No validation or model selection during calibration"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("cache","inputs","output"):parser.add_argument("--"+name,required=True,type=Path)
    parser.add_argument("--family",required=True,choices=("lg_alg","ibkd"))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    result={"status":"running","family":args.family,"completed_batches":0,"optimizer_updates":0,
            "selected_epoch":None,"metrics":None,"selection_performed":False,"losses":[]}
    started=time.perf_counter()
    try:result=measure(args,result)
    except Exception as error:
        import traceback
        traceback.print_exc();result.update(status="failed",error=repr(error),failed_stage=result.get("stage","initialization"))
    result["elapsed_seconds"]=time.perf_counter()-started
    save_json(args.output/"summary.json",result);print(json.dumps(result),flush=True)
    raise SystemExit(0 if result["status"]=="passed" else 1)


if __name__=="__main__":main()
