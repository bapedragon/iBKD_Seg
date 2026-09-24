"""iBKD-only warm-up-20 revision with all four beta candidates through 10k."""
import argparse
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

from ..b0.assets import SPEC,REPO,verify_protocols
from ..b0.reproducibility import configure_runtime
from ..b0.training import evaluate,fit
from ..b0.training_data import PlannedDataset,batches,plan_hash
from ..data import sha256,save_json
from ..b0.screen import build_engine as base_build_engine, result as base_result, code_identity as base_code_identity

CONFIG=SPEC/'b0_ibkd_warmup20_v1.json'
GRID=SPEC/'b0_beta_grid_frozen_v1.json'
SCREEN_SHA256='748660679663782b5190e614218b918e4f2f6549216f548ec3d84d94799e8901'


def specification():
    verify_protocols()
    if sha256(CONFIG)!=SCREEN_SHA256:raise ValueError('Screening specification changed after freezing')
    config=json.loads(CONFIG.read_text());grid=json.loads(GRID.read_text())
    for name,digest in config['protocol_sha256'].items():
        if sha256(SPEC/name)!=digest:raise ValueError(f'Frozen protocol changed: {name}')
    if config['target_steps']!=10000 or config['schedule_horizon']!=80000 or config['batch_size']!=16:
        raise ValueError('Unexpected training budget')
    if config['controller']['warmup']!=20 or config['controller']['interval']!=186:raise ValueError('Warm-up contract differs')
    return config,grid


def candidates(config,grid,pack):
    result=[]
    for method in config['groups'][pack]:
        for candidate in grid['methods'][method]['candidates']:
            result.append(dict(method=method,**candidate,run_id=method+'_'+candidate['candidate_id']))
    return result


def code_identity():
    identity=base_code_identity()
    identity['warmup20']={str(p.relative_to(REPO)):sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    return identity


def configure_engine(engine):
    if engine.method not in ('ibkd_lambda025','ibkd_lambda050'):
        raise ValueError('Warm-up-20 revision is iBKD-only; ALG remains warm-up 0')
    if engine.progress['global_step'] or engine.controller.intervals or engine.controller.steps:
        raise ValueError('Set the warm-up policy before training or loading a matching checkpoint')
    engine.controller.controller.warmup_epochs=20
    return engine


def check_protection(engine):
    c=engine.controller.controller
    if c.warmup_epochs!=20 or c.threshold!=-.02 or c.window!=50:raise ValueError('Warm-up controller changed')
    if any(beta!=c.beta for beta in c.beta_history[:20]):raise ValueError('Guidance disabled during warm-up')
    if c.stop_epoch is not None and c.stop_epoch<20:raise ValueError('Guidance stopped before interval 20')


def build_engine(cache,candidate,device,signature,output,grid):
    return configure_engine(base_build_engine(cache,candidate,device,signature,output,grid))


def result(engine,candidate,status,error=None,*,target=10000):
    row=base_result(engine,candidate,status,error,target=target)
    row.update(guidance_warmup_epochs=20,minimum_guidance_steps=3720,
               interim_2000_metrics=next((h['metrics'] for h in engine.progress['history'] if h['step']==2000),None),
               warmup_protection_passed=None if status=='failed' else engine.progress['global_step']>=3720,
               warmup0_checkpoint_reused=False)
    return row


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
                   source=code_identity(),execution=execution,guidance_warmup_epochs=20,
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
            check_protection(engine)
            report['stage']='validation' if kind.startswith('validation') else 'training'
            if kind=='train':
                with (args.output/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if row['step']>3 and row['step']%20!=0 and row['step'] not in (2000,3719,3721):return
            print('[B0_'+kind.upper()+'] '+json.dumps(dict(run_id=candidate['run_id'],**row)),flush=True)
            persist_summary('running')
        status=fit(engine,loader,validation,target=args.target_steps,stop=should_stop,emit=emit)
    except Exception as exc:
        import traceback
        traceback.print_exc();status='failed';error=repr(exc)
        # Keep the last atomic checkpoint; never overwrite it with a potentially damaged step.
    if status!='failed':check_protection(engine)
    report.update(result(engine,candidate,status,error,target=args.target_steps));report['stage']=status
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','data','output','plan','preflight'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--run-id',required=True);parser.add_argument('--deadline',required=True,type=float)
    parser.add_argument('--target-steps',type=int,choices=(2000,10000,80000),default=10000)
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
