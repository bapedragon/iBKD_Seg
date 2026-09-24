#!/usr/bin/env python3
"""실제 가중치와 합성 입력으로 iBKD warm-up20 초기 경로를 점검한다. GPU 결과가 아니다."""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
import torch
from ibkd_seg.cityscapes.b0.assets import modules
from ibkd_seg.cityscapes.b0.models import Student,Teacher,AllBlockIBKD
from ibkd_seg.cityscapes.b0.reproducibility import configure_runtime
from ibkd_seg.cityscapes.b0.training import Engine,fit,evaluate
from ibkd_seg.cityscapes.b0_warmup20.screen import configure_engine,check_protection
from ibkd_seg.cityscapes.runtime import seed_all
from ibkd_seg.cityscapes.data import save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',scientific_result=False,input='synthetic',batch=2,crop_hw=[64,64],
                last_loss=None,selected_epoch=None,metrics=None,methods=[],h200_verified=False,
                ignore_only_train_samples=1,ignore_only_val_samples=1,guidance_warmup_epochs=20,training_scope='first_six_updates_only')
    started=time.perf_counter()
    try:
        configure_runtime();upstream,teacher_source,_=modules(args.cache)
        teacher=Teacher(teacher_source,args.cache/'weights/cirkd_teacher.pth')
        seed_all(99)
        data=[(torch.randn(2,3,64,64)*40,torch.randint(0,19,(2,64,64)),['synthetic_a','synthetic_b']) for _ in range(6)]
        val=[(torch.randn(1,3,64,128)*40,torch.randint(0,19,(1,64,128)),['synthetic_val']) for _ in range(2)]
        data[2][1][0].fill_(-1)
        val[0][1].fill_(-1)
        for method in ('ibkd_lambda025','ibkd_lambda050'):
            seed_all(1);student=Student(upstream,args.cache/'weights/nvidia_mit_b0.bin')
            seed_all(100001);guide=AllBlockIBKD(deterministic=True)
            engine=Engine(student,teacher,guide,method,.1,torch.device('cpu'),
                          dict(batch_size=2,synthetic=True,method=method,guidance_warmup_epochs=20),args.output/method)
            configure_engine(engine)
            seed_all(200001)
            status=fit(engine,iter(data),lambda:evaluate(student,iter(val),torch.device('cpu'),expected_count=2,shape=(64,128)),
                       target=6,validation_every=2,checkpoint_every=2,stability_step=5)
            check_protection(engine)
            assert engine.controller.controller.active and engine.controller.controller.warmup_epochs==20
            assert status=='completed' and engine.progress['inline_replay']['status']=='passed'
            assert [r['step'] for r in engine.progress['history']]==[2,4,6]
            row=dict(method=method,status=status,completed_steps=6,last_loss=engine.progress['last_loss'],
                     selected_epoch=None,selected_step=engine.progress['best']['epoch'],metrics=engine.progress['best']['metrics'],
                     inline_replay=engine.progress['inline_replay'],teacher_frozen_verified=True)
            report['methods'].append(row);print('[CPU_TRAINING_CHECK] '+json.dumps(row),flush=True)
            del engine,student,guide;gc.collect()
        report['status']='passed'
    except Exception as error:
        import traceback
        traceback.print_exc();report.update(status='failed',error=repr(error))
    report['elapsed_seconds']=time.perf_counter()-started;save_json(args.output/'summary.json',report)
    print(json.dumps(report),flush=True)
    raise SystemExit(0 if report['status']=='passed' else 1)


if __name__=='__main__':main()
