"""Three real updates per method plus exact-state replay; no model selection."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import statistics
import time
from pathlib import Path

# Also supports direct module invocation; must precede any CUDA context creation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from .assets import SPEC,modules,verify_protocols
from .models import Student,Teacher,Locality,AllBlockIBKD
from .losses import FSKD,C2VKD
from .control import StepController,boundary_checks
from .data import stitch_logits,scores
from .reproducibility import configure_runtime,pixel_cross_entropy
from ..data import save_json,sha256
from ..runtime import seed_all,state_hash

METHODS=("vanilla","lg","alg","ibkd_lambda025","ibkd_lambda050","fskd","c2vkd_clip_pool")


def rng_state():
    return {"torch":torch.get_rng_state(),"numpy":np.random.get_state(),"python":random.getstate(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    torch.set_rng_state(state["torch"]);np.random.set_state(state["numpy"]);random.setstate(state["python"])
    if state["cuda"]: torch.cuda.set_rng_state_all(state["cuda"])


def cpu_tree(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def assert_tree(a,b,*,rtol,atol,path="state"):
    if isinstance(a,torch.Tensor):
        torch.testing.assert_close(a.detach().cpu(),b.detach().cpu(),rtol=rtol,atol=atol,
                                   msg=lambda message:f"{path}: {message}")
    elif isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b,err_msg=path)
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a: assert_tree(a[k],b[k],rtol=rtol,atol=atol,path=f"{path}.{k}")
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)): assert_tree(x,y,rtol=rtol,atol=atol,path=f"{path}[{i}]")
    elif isinstance(a,float):
        assert math.isclose(a,b,rel_tol=rtol,abs_tol=atol),(path,a,b)
    else: assert a==b,(path,a,b)


def compare_components(actual,expected,*,rtol,atol):
    checks={}
    for name in expected:
        try:
            assert_tree(actual[name],expected[name],rtol=rtol,atol=atol,path=name)
            checks[name]={"status":"passed"}
        except AssertionError as error:
            checks[name]={"status":"failed","error":str(error)}
    return {"status":"passed" if all(r["status"]=="passed" for r in checks.values()) else "failed",
            "checks":checks}


def compute(method,student,teacher,guide,image,label,beta):
    logits,raw,stages,attn=student(image,features=True)
    ce=pixel_cross_entropy(logits,label)
    losses={"ce":ce}
    if method=="vanilla": return ce,losses,ce.new_zeros(())
    tl,tf=teacher(image)
    if method in ("lg","alg"):
        losses["locality"]=guide(raw,tf)
        guidance=losses["locality"]
    elif method.startswith("ibkd"):
        losses["alignment"],losses["fusion"]=guide(raw,tf)
        ratio=.25 if method.endswith("025") else .5
        guidance=(1-ratio)*losses["alignment"]+ratio*losses["fusion"]
    elif method=="fskd":
        losses.update(guide(stages,attn,tf,logits,tl,label))
        guidance=losses["logit_kd"]+losses["global"]+losses["patch"]+40000*losses["attention"]
        return ce+guidance,losses,guidance
    else:
        losses.update(guide(stages,tf,logits,tl,label))
        guidance=losses["pdd"]+.1*losses["global"]+.1*losses["patch"]+.5*losses["linguistic"]
        return guidance,losses,guidance  # CE is diagnostic only for C2VKD.
    return ce+beta*guidance,losses,guidance


def gradient_norm(parameters):
    values=[p.grad.detach().double().square().sum() for p in parameters if p.grad is not None]
    if not values: return 0.
    norm=torch.stack(values).sum().sqrt()
    if not torch.isfinite(norm): raise ValueError("Nonfinite gradient")
    return float(norm)


def parameter_groups(student,guide):
    result={"encoder":[p for n,p in student.net.named_parameters() if not n.startswith("linear_")],
            "decoder":[p for n,p in student.net.named_parameters() if n.startswith("linear_")]}
    if guide is not None:
        for name,child in guide.named_children():
            params=[p for p in child.parameters() if p.requires_grad]
            if params: result["guidance."+name]=params
    return result


def measure(args,progress=None):
    if progress is None:progress={}
    progress["stage"]="protocol_and_device"
    verify_protocols()
    execution=configure_runtime()
    progress["execution"]=execution
    device=torch.device(args.device)
    if device.type!="cuda" or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError("This public smoke requires exactly one CUDA GPU; unit tests are separate")
    progress["stage"]="models_and_inputs"
    seed_all(1)
    us,ut,_=modules(args.cache)
    student=Student(us,args.cache/"weights/nvidia_mit_b0.bin").to(device).train()
    initial=cpu_tree(student.state_dict())
    initial_hash=state_hash(student)
    seed_all(100001)
    guide=None
    if args.method in ("lg","alg"): guide=Locality()
    elif args.method.startswith("ibkd"): guide=AllBlockIBKD(deterministic=True)
    elif args.method=="fskd": guide=FSKD()
    elif args.method=="c2vkd_clip_pool": guide=C2VKD(args.cache/"weights/clip_rn101.pt")
    if guide is not None: guide=guide.to(device).train()
    teacher=None if guide is None else Teacher(ut,args.cache/"weights/cirkd_teacher.pth").to(device)
    teacher_hash=None if teacher is None else state_hash(teacher)
    pool_hash=state_hash(guide.pool) if isinstance(guide,C2VKD) else None
    progress.update(student_initial_state_sha256=initial_hash,teacher_state_sha256=teacher_hash,
                    pool_state_sha256=pool_hash,ibkd_deterministic_execution=
                    guide.deterministic_contract if isinstance(guide,AllBlockIBKD) else None)
    # Produced by this job after data audit; weights_only=True never loads arbitrary external data objects.
    data=torch.load(args.inputs,map_location="cpu",weights_only=True)
    if len(data["train"])!=3 or any(tuple(x.shape)!=(16,3,512,512) for x,y in data["train"]):
        raise ValueError("Actual batch16/crop512 input contract violated")
    seed_all(200001)
    progress["stage"]="calibration"
    training_rng=rng_state()
    beta=0.
    calibration=[]
    if args.method in ("lg","alg","ibkd_lambda025","ibkd_lambda050"):
        with torch.no_grad():
            for image,label in data["train"]:
                _,terms,g=compute(args.method,student,teacher,guide,image.to(device),label.to(device),1.)
                calibration.append({"ce":float(terms["ce"]),"raw_guidance":float(g)})
        c=statistics.median(r["ce"] for r in calibration)
        g=statistics.median(r["raw_guidance"] for r in calibration)
        if not math.isfinite(c+g) or g<=1e-12: raise ValueError("Invalid calibration loss")
        beta=.07*c/g
        student.load_state_dict(initial,strict=True)
        restore_rng(training_rng)
    if state_hash(student)!=initial_hash:
        raise ValueError("Student state changed before the first optimizer update")
    del initial
    kind="ibkd" if args.method.startswith("ibkd") else args.method
    controller=StepController(kind,beta) if kind in ("lg","alg","ibkd") else None
    groups=parameter_groups(student,guide)
    parameters=[p for ps in groups.values() for p in ps if p.requires_grad]
    if len(parameters)!=len({id(p) for p in parameters}): raise ValueError("Duplicate optimizer parameter")
    optimizer=torch.optim.AdamW(parameters,lr=6e-5,betas=(.9,.999),eps=1e-8,weight_decay=1e-4,fused=False)
    protocol_hashes={p.name:sha256(p) for p in sorted(SPEC.glob("*.json"))}
    input_hash=sha256(args.inputs)
    rows=[]
    progress.update(losses=rows,input_sha256=input_hash,protocol_hashes=protocol_hashes,
                    smoke_beta=beta,beta_is_final_candidate=False,calibration=calibration,
                    completed_steps=0,selection_performed=False,diagnostic_metrics=None)
    completed=0
    component_grads={}
    initial_guide_hash=None if guide is None else state_hash(guide)
    torch.cuda.reset_peak_memory_stats()

    def step(batch, *, diagnose=False):
        nonlocal completed,component_grads
        image,label=(x.to(device) for x in batch)
        optimizer.zero_grad(set_to_none=True)
        lr=6e-5*(1-completed/80000)**.9
        for group in optimizer.param_groups: group["lr"]=lr
        current_beta=controller.beta if controller else 0.
        loss,terms,g=compute(args.method,student,teacher,guide,image,label,current_beta)
        if not all(torch.isfinite(v) for v in terms.values()) or not torch.isfinite(loss):
            raise ValueError("Nonfinite objective/component")
        if diagnose:
            target=student.net.patch_embed1.proj.weight
            for name,value in terms.items():
                grad=torch.autograd.grad(value,target,retain_graph=True,allow_unused=True)[0]
                norm=0. if grad is None else float(grad.detach().double().square().sum().sqrt())
                component_grads[name]=norm
                if not math.isfinite(norm) or norm<=0:
                    raise ValueError(f"No finite nonzero student gradient from component {name}")
        loss.backward()
        norms={name:gradient_norm(ps) for name,ps in groups.items()}
        if any(v<=0 for v in norms.values()): raise ValueError(f"Disconnected parameter group: {norms}")
        if teacher is not None and any(p.grad is not None for p in teacher.parameters()):
            raise ValueError("Frozen teacher has gradients")
        if isinstance(guide,C2VKD) and any(p.grad is not None for p in guide.pool.parameters()):
            raise ValueError("Frozen attention pool has gradients")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in parameters): raise ValueError("Nonfinite updated parameter")
        completed+=1
        if controller: controller.observe_step(float(g.detach()),image.shape[0])
        return {"step":completed,"loss":float(loss.detach()),"loss_components":{k:float(v.detach()) for k,v in terms.items()},
                "raw_guidance":float(g.detach()),"weighted_guidance":float(g.detach())*(current_beta if controller else 1.),
                "beta":current_beta,"lr":lr,"gradient_norms":norms}

    path=args.output/"resume_step2.pt"
    progress["stage"]="training"
    for i,batch in enumerate(data["train"]):
        if i==2:
            torch.save({"student":cpu_tree(student.state_dict()),"guide":None if guide is None else cpu_tree(guide.state_dict()),
                        "optimizer":cpu_tree(optimizer.state_dict()),"completed":completed,"rng":rng_state(),
                        "controller":None if controller is None else controller.state_dict(),"sampler_position":32,
                        "protocol_hashes":protocol_hashes,"input_hash":input_hash},path)
        torch.cuda.synchronize();start=time.perf_counter()
        row=step(batch,diagnose=i==0)
        torch.cuda.synchronize();row["seconds"]=time.perf_counter()-start
        rows.append(row)
        progress.update(completed_steps=completed,last_loss=row,component_encoder_gradient_norms=component_grads)
        save_json(args.output/"summary.json",progress)
        print("[B0_STEP] "+json.dumps({"method":args.method,**row}),flush=True)
    train_peak=torch.cuda.max_memory_allocated()
    progress.update(stage="resume_restore",train_peak_cuda_bytes=train_peak)
    expected={"student":cpu_tree(student.state_dict()),"guide":None if guide is None else cpu_tree(guide.state_dict()),
              "optimizer":cpu_tree(optimizer.state_dict()),"controller":None if controller is None else copy.deepcopy(controller.state_dict())}
    # This checkpoint was just created locally by this process and includes Python/NumPy RNG tuples.
    saved=torch.load(path,map_location="cpu",weights_only=False)
    if saved["protocol_hashes"]!=protocol_hashes or saved["input_hash"]!=input_hash or saved["sampler_position"]!=32:
        raise ValueError("Resume provenance changed")
    student.load_state_dict(saved["student"],strict=True)
    if guide is not None: guide.load_state_dict(saved["guide"],strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    if controller: controller.load_state_dict(saved["controller"])
    completed=saved["completed"];restore_rng(saved["rng"])
    restored={"student":student.state_dict(),"guide":None if guide is None else guide.state_dict(),
              "optimizer":optimizer.state_dict(),"controller":None if controller is None else controller.state_dict(),
              "completed":completed,"rng":rng_state()}
    restore_checks=compare_components(restored,{k:saved[k] for k in restored},rtol=0,atol=0)
    progress["restore_before_replay"]=restore_checks
    save_json(args.output/"summary.json",progress)
    if restore_checks["status"]!="passed":raise ValueError("Checkpoint did not restore the exact pre-update state")
    del saved,restored
    progress["stage"]="resume_replay_compare"
    replay=step(data["train"][2])
    tolerance={"rtol":2e-5,"atol":2e-6}
    expected["loss_components"]=rows[-1]["loss_components"]
    comparison=compare_components({"student":student.state_dict(),"guide":None if guide is None else guide.state_dict(),
                                   "optimizer":optimizer.state_dict(),"controller":None if controller is None else controller.state_dict(),
                                   "loss_components":replay["loss_components"]},expected,**tolerance)
    resume={**comparison,"replayed_update":3,"tolerance":tolerance,"checkpoint_sha256":sha256(path),
            "restore_before_replay":restore_checks,"replay_loss":replay}
    progress["resume"]=resume
    save_json(args.output/"summary.json",progress)
    # Evaluation always uses the original continuous trajectory, even if replay differs.
    student.load_state_dict(expected["student"],strict=True)
    if guide is not None:guide.load_state_dict(expected["guide"],strict=True)
    optimizer.load_state_dict(expected["optimizer"])
    if controller:controller.load_state_dict(expected["controller"])
    del expected
    progress["stage"]="frozen_state_checks"
    if state_hash(student)==initial_hash: raise ValueError("Student parameters did not update")
    if guide is not None and state_hash(guide)==initial_guide_hash: raise ValueError("Guidance did not update")
    if teacher is not None and state_hash(teacher)!=teacher_hash: raise ValueError("Teacher/BN changed")
    if pool_hash is not None and state_hash(guide.pool)!=pool_hash: raise ValueError("Pool changed")
    progress.update(teacher_frozen_verified=teacher is not None,pool_frozen_verified=pool_hash is not None)
    progress["stage"]="validation"
    student.eval()
    matrix=torch.zeros(19,19,dtype=torch.int64)
    valid_count=0
    with torch.inference_mode():
        for image,label in data["val"]:
            prediction=stitch_logits(student,image.to(device)).argmax(1).cpu()
            valid=label!=-1
            matrix+=torch.bincount(19*label[valid]+prediction[valid],minlength=361).reshape(19,19)
            valid_count+=int(valid.sum())
    if int(matrix.sum())!=valid_count: raise ValueError("Ignore mask / evaluation mismatch")
    result={**progress,"status":comparison["status"],"stage":"completed","method":args.method,"scientific_result":False,"selected_epoch":None,
            "selection_performed":False,"completed_steps":completed,"target_steps":3,"schedule_total_steps":80000,
            "actual_batch":16,"crop_hw":[512,512],"precision":"fp32","tf32":False,
            "validation_samples":2,"full_validation":False,"diagnostic_metrics":scores(matrix),
            "student_initial_state_sha256":initial_hash,"teacher_state_sha256":teacher_hash,
            "pool_state_sha256":pool_hash,"teacher_frozen_verified":teacher is not None,
            "pool_frozen_verified":pool_hash is not None,"input_sha256":input_hash,"protocol_hashes":protocol_hashes,
            "smoke_beta":beta,"beta_is_final_candidate":False,"beta_calibration_batches":len(calibration),
            "calibration_student_reset_verified":True,
            "calibration":calibration,"losses":rows,"last_loss":rows[-1],"component_encoder_gradient_norms":component_grads,
            "resume":resume,"validation_model_state":"original_continuous_update3",
            "controller_state":None if controller is None else controller.state_dict(),
            "controller_synthetic_boundary_checks":boundary_checks(),"train_peak_cuda_bytes":train_peak,
            "test_used":False,"environment":{"python":platform.python_version(),"torch":str(torch.__version__),
            "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"gpu_total_bytes":torch.cuda.get_device_properties(0).total_memory}}
    if comparison["status"]!="passed":result["failed_stage"]="resume_replay_compare"
    save_json(args.output/"summary.json",result)
    return result


def run_and_report(args,measure_fn=measure):
    result={"status":"running","method":args.method,"scientific_result":False,
            "smoke_spec_id":"cityscapes_segformer_b0_smoke_v2",
            "selected_epoch":None,"diagnostic_metrics":None,"losses":[],"completed_steps":0}
    try:
        result=measure_fn(args,result)
    except Exception as error:
        import traceback
        traceback.print_exc()
        result.update(status="failed",error=repr(error),failed_stage=result.get("stage","initialization"))
    save_json(args.output/"summary.json",result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result["status"]=="passed" else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("cache","inputs","output"):p.add_argument("--"+flag,type=Path,required=True)
    p.add_argument("--method",choices=METHODS,required=True)
    p.add_argument("--device",default="cuda",choices=("cuda",))
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    raise SystemExit(run_and_report(args))


if __name__=="__main__": main()
