"""One frozen B0 candidate through step 2,000, with safe interruption/resume."""
import argparse
import hashlib
import json
import os
import platform
import signal
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import cv2
import numpy as np
import torch

from .assets import SPEC,REPO,modules,verify_protocols
from .models import Student,Teacher,Locality,AllBlockIBKD
from .losses import FSKD
from .reproducibility import configure_runtime
from .training import Engine,evaluate,fit
from .training_data import PlannedDataset,batches,plan_hash
from ..data import sha256,save_json
from ..runtime import seed_all,state_hash,source_hash

CONFIG=SPEC/'b0_screen2000_v2.json'
GRID=SPEC/'b0_beta_grid_frozen_v1.json'
SCREEN_SHA256='1bdfa6c19fcda642cd69e91c21871a4aaa20d8ee911ae19352c86aa2feabe08c'


def specification():
    verify_protocols()
    if sha256(CONFIG)!=SCREEN_SHA256:raise ValueError('Screening specification changed after freezing')
    config=json.loads(CONFIG.read_text());grid=json.loads(GRID.read_text())
    for name,digest in config['protocol_sha256'].items():
        if sha256(SPEC/name)!=digest:raise ValueError(f'Frozen protocol changed: {name}')
    if config['target_steps']!=2000 or config['schedule_horizon']!=80000 or config['batch_size']!=16:
        raise ValueError('Unexpected training budget')
    return config,grid


def candidates(config,grid,pack):
    result=[]
    for method in config['groups'][pack]:
        entries=[dict(candidate_id='fixed',beta=0.)] if method in ('vanilla','fskd') else grid['methods'][method]['candidates']
        for candidate in entries:
            result.append(dict(method=method,**candidate,run_id=method+'_'+candidate['candidate_id']))
    return result


def code_identity():
    paths=sorted((REPO/'src/ibkd_seg/cityscapes/b0').glob('*.py'))
    return dict(b0={str(p.relative_to(REPO)):sha256(p) for p in paths},shared=source_hash())


def build_engine(cache,candidate,device,signature,output,grid):
    upstream,teacher_source,_=modules(cache)
    seed_all(1);student=Student(upstream,cache/'weights/nvidia_mit_b0.bin').to(device).train()
    if state_hash(student)!=grid['student_initial_state_sha256']:raise ValueError('Initial student differs from calibration')
    seed_all(100001);method=candidate['method'];guide=None
    if method in ('lg','alg'):guide=Locality()
    elif method.startswith('ibkd'):guide=AllBlockIBKD(deterministic=True)
    elif method=='fskd':guide=FSKD()
    if guide is not None:guide=guide.to(device).train()
    family='lg_alg' if method in ('lg','alg') else 'ibkd' if method.startswith('ibkd') else None
    if family and state_hash(guide)!=grid['guide_initial_state_sha256'][family]:raise ValueError('Initial guide differs from calibration')
    teacher=None if guide is None else Teacher(teacher_source,cache/'weights/cirkd_teacher.pth').to(device)
    if teacher is not None and state_hash(teacher)!=grid['teacher_state_sha256']:raise ValueError('Teacher differs from calibration')
    seed_all(200001)
    return Engine(student,teacher,guide,method,candidate['beta'],device,signature,output)


def result(engine,candidate,status,error=None,*,target=2000):
    p=engine.progress;best=p['best'];controller=None if engine.controller is None else engine.controller.state_dict()
    stop_epoch=None if controller is None else controller['controller']['stop_epoch']
    return dict(status=status,**candidate,completed_steps=p['global_step'],target_steps=target,schedule_horizon=80000,
                last_loss=p['last_loss'],selected_epoch=None,selected_epoch_reason='Iteration-based training; see selected_step',
                selected_step=None if best is None else best['epoch'],metrics=None if best is None else best['metrics'],
                best=best,last_metrics=p['last_metrics'],last_eval_step=p['last_eval_step'],validation_history=p['history'],
                controller=controller,guidance_stop_after_step=None if stop_epoch is None else stop_epoch*186,
                inline_replay=p['inline_replay'],stability_500_passed=p['stability_500_passed'],
                calibration_input_verified=p['calibration_input_verified'],first_25_batch_hashes=p['first_25_batch_hashes'],
                last_input_hash=p['last_input_hash'],test_used=False,final_80k_result=target==80000 and status=='completed',signature=engine.signature,
                training_seconds=p['training_seconds'],validation_seconds=p['validation_seconds'],
                checkpoint_seconds=p['checkpoint_seconds'],input_wait_seconds=p['input_wait_seconds'],
                error=error,resume_pointer=str(engine.output/'resume.json'),teacher_frozen_verified=engine.teacher is not None,
                peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated() if engine.device.type=='cuda' else None)


def run(args,candidate,report,stop):
    config,grid=specification();execution=configure_runtime()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Exactly one CUDA GPU is required')
    device=torch.device('cuda')
    preflight=json.loads(args.preflight.read_text());plan=np.load(args.plan,allow_pickle=False)
    if plan.shape!=(1280000,5) or plan.dtype!=np.int32 or plan_hash(plan)!=preflight['plan_sha256']:
        raise ValueError('Augmentation plan differs')
    if preflight['status']!='passed' or preflight['calibration_tensor_sha256']!=grid['calibration_tensor_sha256']:
        raise ValueError('Calibration input preflight missing')
    if preflight.get('checked_batches')!=config['input']['preflight_batches'] or preflight.get('samples')!=32*16:
        raise ValueError('Extended input preflight missing')
    manifest=sha256(args.data/'manifest.json')
    if manifest!=grid['manifest_sha256'] or manifest!=preflight['manifest_sha256']:raise ValueError('Dataset differs from calibration')
    lists=args.cache/'cirkd/dataset/list/cityscapes'
    train=PlannedDataset(args.data,lists/'train.lst',plan);val=PlannedDataset(args.data,lists/'val.lst')
    if len(train.rows)!=2975 or len(val)!=500:raise ValueError('Incomplete train/val')
    signature=dict(protocol_id=config['id'],protocol_sha256=sha256(CONFIG),grid_sha256=sha256(GRID),
                   candidate=candidate,batch_size=16,seed=1,manifest_sha256=manifest,plan_sha256=preflight['plan_sha256'],
                   source=code_identity(),execution=execution,
                   environment=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
                                    numpy=np.__version__,opencv=cv2.__version__,gpu=torch.cuda.get_device_name()))
    engine=build_engine(args.cache,candidate,device,signature,args.output,grid)
    if args.resume:
        info=engine.load(args.resume);print('[B0_RESUME] '+json.dumps(info),flush=True)
    engine.progress['calibration_input_verified']=True
    torch.cuda.reset_peak_memory_stats();status='running';error=None
    def remaining():return args.deadline-time.time()
    def should_stop():return stop['requested'] or remaining()<90
    def persist_summary(state):
        report.update(result(engine,candidate,state,target=args.target_steps));save_json(args.output/'summary.json',report)
    try:
        loader=batches(train,engine.progress['global_step']*16,args.target_steps*16,16,workers=4,prefetch=2)
        def validation():
            vloader=batches(val,0,500,1,workers=4,prefetch=2)
            try:
                return evaluate(engine.student,vloader,device,stop=should_stop,
                    emit=lambda n:print(f'[B0_VAL] {candidate["run_id"]} step={engine.progress["global_step"]} images={n}/500',flush=True))
            finally:vloader.close()
        def emit(kind,row):
            report['stage']='validation' if kind.startswith('validation') else 'training'
            if kind=='train':
                with (args.output/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if row['step']>3 and row['step']%20!=0:return
            print('[B0_'+kind.upper()+'] '+json.dumps(dict(run_id=candidate['run_id'],**row)),flush=True)
            persist_summary('running')
        status=fit(engine,loader,validation,target=args.target_steps,stop=should_stop,emit=emit)
    except Exception as exc:
        import traceback
        traceback.print_exc();status='failed';error=repr(exc)
        # Keep the last atomic checkpoint; never overwrite it with a potentially damaged step.
    report.update(result(engine,candidate,status,error,target=args.target_steps));report['stage']=status
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','data','output','plan','preflight'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--run-id',required=True);parser.add_argument('--deadline',required=True,type=float)
    parser.add_argument('--target-steps',type=int,choices=(2000,10000,80000),default=2000)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()) and not args.resume:raise ValueError('Nonempty output requires explicit resume')
    config,grid=specification();inventory=[r for pack in config['groups'] for r in candidates(config,grid,pack)]
    candidate=next(r for r in inventory if r['run_id']==args.run_id)
    stop={'requested':False}
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.update(requested=True))
    report=dict(status='running',**candidate,completed_steps=0,last_loss=None,selected_epoch=None,metrics=None,test_used=False)
    started=time.perf_counter()
    try:report=run(args,candidate,report,stop)
    except Exception as exc:
        import traceback
        traceback.print_exc();report.update(status='failed',error=repr(exc))
    report['elapsed_seconds']=time.perf_counter()-started
    save_json(args.output/'summary.json',report);print(json.dumps(report),flush=True)
    raise SystemExit(0 if report['status'] in ('completed','paused') else 1)


if __name__=='__main__':main()
