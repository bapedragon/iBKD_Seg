"""Three real updates per method plus exact-state replay; no model selection."""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .assets import SPEC,modules,verify_protocols
from .models import Student,Teacher,Locality,AllBlockIBKD
from .losses import FSKD,C2VKD
from .control import StepController,boundary_checks
from .data import stitch_logits,scores
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


def assert_tree(a,b,*,rtol,atol):
    if isinstance(a,torch.Tensor):
        torch.testing.assert_close(a.detach().cpu(),b.detach().cpu(),rtol=rtol,atol=atol)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: assert_tree(a[k],b[k],rtol=rtol,atol=atol)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): assert_tree(x,y,rtol=rtol,atol=atol)
    elif isinstance(a,float):
        assert math.isclose(a,b,rel_tol=rtol,abs_tol=atol),(a,b)
    else: assert a==b,(a,b)


def compute(method,student,teacher,guide,image,label,beta):
    logits,raw,stages,attn=student(image,features=True)
    ce=F.cross_entropy(F.interpolate(logits,size=label.shape[-2:],mode="bilinear",align_corners=True),label,ignore_index=-1)
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


def measure(args):
    verify_protocols()
    device=torch.device(args.device)
    if device.type!="cuda" or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError("This public smoke requires exactly one CUDA GPU; unit tests are separate")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    seed_all(1)
    us,ut,_=modules(args.cache)
    student=Student(us,args.cache/"weights/nvidia_mit_b0.bin").to(device).train()
    initial=cpu_tree(student.state_dict())
    initial_hash=state_hash(student)
    seed_all(100001)
    guide=None
    if args.method in ("lg","alg"): guide=Locality()
    elif args.method.startswith("ibkd"): guide=AllBlockIBKD()
    elif args.method=="fskd": guide=FSKD()
    elif args.method=="c2vkd_clip_pool": guide=C2VKD(args.cache/"weights/clip_rn101.pt")
    if guide is not None: guide=guide.to(device).train()
    teacher=None if guide is None else Teacher(ut,args.cache/"weights/cirkd_teacher.pth").to(device)
    teacher_hash=None if teacher is None else state_hash(teacher)
    pool_hash=state_hash(guide.pool) if isinstance(guide,C2VKD) else None
    # Produced by this job after data audit; weights_only=True never loads arbitrary external data objects.
    data=torch.load(args.inputs,map_location="cpu",weights_only=True)
    if len(data["train"])!=3 or any(tuple(x.shape)!=(16,3,512,512) for x,y in data["train"]):
        raise ValueError("Actual batch16/crop512 input contract violated")
    seed_all(200001)
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
        print("[B0_STEP] "+json.dumps({"method":args.method,**row}),flush=True)
    train_peak=torch.cuda.max_memory_allocated()
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
    completed=saved["completed"];restore_rng(saved["rng"]);del saved
    replay=step(data["train"][2])
    tolerance={"rtol":2e-5,"atol":2e-6}
    assert_tree(student.state_dict(),expected["student"],**tolerance)
    if guide is not None: assert_tree(guide.state_dict(),expected["guide"],**tolerance)
    assert_tree(optimizer.state_dict(),expected["optimizer"],**tolerance)
    if controller: assert_tree(controller.state_dict(),expected["controller"],**tolerance)
    assert_tree(replay["loss_components"],rows[-1]["loss_components"],**tolerance)
    del expected
    if state_hash(student)==initial_hash: raise ValueError("Student parameters did not update")
    if guide is not None and state_hash(guide)==initial_guide_hash: raise ValueError("Guidance did not update")
    if teacher is not None and state_hash(teacher)!=teacher_hash: raise ValueError("Teacher/BN changed")
    if pool_hash is not None and state_hash(guide.pool)!=pool_hash: raise ValueError("Pool changed")
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
    result={"status":"passed","method":args.method,"scientific_result":False,"selected_epoch":None,
            "selection_performed":False,"completed_steps":completed,"target_steps":3,"schedule_total_steps":80000,
            "actual_batch":16,"crop_hw":[512,512],"precision":"fp32","tf32":False,
            "validation_samples":2,"full_validation":False,"diagnostic_metrics":scores(matrix),
            "student_initial_state_sha256":initial_hash,"teacher_state_sha256":teacher_hash,
            "pool_state_sha256":pool_hash,"teacher_frozen_verified":teacher is not None,
            "pool_frozen_verified":pool_hash is not None,"input_sha256":input_hash,"protocol_hashes":protocol_hashes,
            "smoke_beta":beta,"beta_is_final_candidate":False,"beta_calibration_batches":len(calibration),
            "calibration_student_reset_verified":True,
            "calibration":calibration,"losses":rows,"last_loss":rows[-1],"component_encoder_gradient_norms":component_grads,
            "resume":{"status":"passed","replayed_update":3,"tolerance":tolerance,"checkpoint_sha256":sha256(path)},
            "controller_state":None if controller is None else controller.state_dict(),
            "controller_synthetic_boundary_checks":boundary_checks(),"train_peak_cuda_bytes":train_peak,
            "test_used":False,"environment":{"python":platform.python_version(),"torch":str(torch.__version__),
            "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"gpu_total_bytes":torch.cuda.get_device_properties(0).total_memory}}
    save_json(args.output/"summary.json",result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("cache","inputs","output"):p.add_argument("--"+flag,type=Path,required=True)
    p.add_argument("--method",choices=METHODS,required=True)
    p.add_argument("--device",default="cuda",choices=("cuda",))
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        result=measure(args)
    except Exception as error:
        import traceback
        traceback.print_exc()
        result={"status":"failed","method":args.method,"error":repr(error),"scientific_result":False,
                "selected_epoch":None,"metrics":None,"reason":"execution_failed_before_complete_validation"}
        save_json(args.output/"summary.json",result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        raise SystemExit(1)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__": main()
