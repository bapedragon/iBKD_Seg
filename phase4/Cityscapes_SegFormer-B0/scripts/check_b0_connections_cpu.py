#!/usr/bin/env python3
"""실제 가중치 + 합성 64×64 입력의 CPU 연결 검사. H200/Cityscapes smoke를 대체하지 않는다."""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"src"))
from ibkd_seg.cityscapes.b0.assets import modules
from ibkd_seg.cityscapes.b0.models import Student,Teacher,Locality,AllBlockIBKD
from ibkd_seg.cityscapes.b0.losses import FSKD,C2VKD
from ibkd_seg.cityscapes.b0.control import StepController,boundary_checks
from ibkd_seg.cityscapes.b0.smoke import (
    METHODS,compute,cpu_tree,assert_tree,rng_state,restore_rng,parameter_groups,gradient_norm,
)
from ibkd_seg.cityscapes.runtime import seed_all,state_hash
from ibkd_seg.cityscapes.b0.reproducibility import configure_runtime


def check(cache,output):
    execution=configure_runtime()
    upstream,teacher_source,dataset_source=modules(cache)
    teacher=Teacher(teacher_source,cache/"weights/cirkd_teacher.pth")
    teacher_hash=state_hash(teacher)
    seed_all(99)
    batches=[(torch.randn(2,3,64,64)*40,torch.randint(0,19,(2,64,64))) for _ in range(2)]
    for _,y in batches:y[:,:3,:]=-1
    report={"status":"running","scientific_result":False,"device":"cpu","input":"synthetic",
            "batch":2,"crop_hw":[64,64],"selected_epoch":None,"metrics":None,
            "metrics_reason":"No Cityscapes data used","methods":[],"gpu_validated":False,"execution":execution}
    for method in METHODS:
        seed_all(1)
        student=Student(upstream,cache/"weights/nvidia_mit_b0.bin").train()
        initial_hash=state_hash(student)
        seed_all(100001)
        guide=None
        if method in ("lg","alg"):guide=Locality()
        elif method.startswith("ibkd"):guide=AllBlockIBKD(deterministic=True)
        elif method=="fskd":guide=FSKD(crop=64)
        elif method=="c2vkd_clip_pool":guide=C2VKD(cache/"weights/clip_rn101.pt")
        groups=parameter_groups(student,guide)
        optimizer=torch.optim.AdamW([p for group in groups.values() for p in group],lr=6e-5,weight_decay=1e-4)
        kind="ibkd" if method.startswith("ibkd") else method
        controller=StepController(kind,.001) if kind in ("lg","alg","ibkd") else None
        seed_all(200001)
        def update(batch,diagnose=False):
            optimizer.zero_grad(set_to_none=True)
            loss,terms,raw=compute(method,student,teacher,guide,*batch,.001)
            components={}
            if diagnose:
                for name,value in terms.items():
                    grad=torch.autograd.grad(value,student.net.patch_embed1.proj.weight,retain_graph=True)[0]
                    components[name]=float(grad.double().square().sum().sqrt())
                    assert torch.isfinite(grad).all() and components[name]>0,name
            loss.backward()
            norms={name:gradient_norm(params) for name,params in groups.items()}
            assert all(value>0 for value in norms.values()),norms
            optimizer.step()
            if controller:controller.observe_step(float(raw.detach()),2)
            return {"loss":float(loss.detach()),"loss_components":{k:float(v.detach()) for k,v in terms.items()},
                    "group_gradient_norms":norms,"component_gradient_norms":components}
        def state():
            return cpu_tree({"student":student.state_dict(),"guide":None if guide is None else guide.state_dict(),
                             "optimizer":optimizer.state_dict(),"controller":None if controller is None else controller.state_dict()})
        first=update(batches[0],diagnose=True)
        saved=state();saved["rng"]=rng_state()
        checkpoint=output/f"{method}_step1.pt"
        torch.save(saved,checkpoint);del saved
        expected_row=update(batches[1]);expected=state()
        saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
        student.load_state_dict(saved["student"])
        if guide is not None:guide.load_state_dict(saved["guide"])
        optimizer.load_state_dict(saved["optimizer"])
        if controller:controller.load_state_dict(saved["controller"])
        restore_rng(saved["rng"]);del saved
        replay=update(batches[1])
        assert_tree(state(),expected,rtol=0,atol=0)
        assert_tree(replay,expected_row,rtol=0,atol=0)
        assert state_hash(student)!=initial_hash
        assert not any(p.grad is not None for p in teacher.parameters())
        if isinstance(guide,C2VKD):assert not any(p.grad is not None for p in guide.pool.parameters())
        report["methods"].append({"method":method,"status":"passed","student_initial_sha256":initial_hash,
                                  "updates":2,"resume_replay_update":2,"bitwise_replay":True,
                                  "first_loss":first,"last_loss":replay})
        print("[CPU_CONNECTION] "+method+" passed",flush=True)
        del student,guide,optimizer,groups,expected
        gc.collect()
    assert len({r["student_initial_sha256"] for r in report["methods"]})==1
    assert state_hash(teacher)==teacher_hash
    rows={r["method"]:r for r in report["methods"]}
    assert_tree(rows["lg"]["last_loss"],rows["alg"]["last_loss"],rtol=0,atol=0)
    report.update(status="passed",teacher_frozen_verified=True,controller_checks=boundary_checks())
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        result=check(args.cache,args.output)
    except Exception as error:
        import traceback
        traceback.print_exc()
        result={"status":"failed","error":repr(error),"scientific_result":False,"selected_epoch":None,"metrics":None}
    (args.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if result["status"]=="passed" else 1)
