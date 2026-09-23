"""B0 training engine: 80k schedule, inline replay, full-state checkpoints."""
import copy
import math
import time

import numpy as np
import torch

from .control import StepController
from .smoke import compute,cpu_tree,rng_state,restore_rng,assert_tree,parameter_groups,gradient_norm
from .reproducibility import pixel_cross_entropy
from .data import scores
from ..runtime import state_hash
from ..full_checkpoint import save_checkpoint,load_checkpoint,save_best


def packed_rng():
    value=rng_state();name,keys,position,gauss,cached=value['numpy']
    value['numpy']=[name,keys.tolist(),position,gauss,cached]
    return value


def unpack_rng(value):
    value=copy.deepcopy(value);value['numpy'][1]=np.asarray(value['numpy'][1],dtype=np.uint32)
    value['numpy']=tuple(value['numpy']);restore_rng(value)


def learning_rate(completed):
    if not 0<=completed<80000:raise ValueError('Invalid schedule cursor')
    return 6e-5*(1-completed/80000)**.9


def initial_progress():
    return dict(global_step=0,best=None,last_metrics=None,last_eval_step=0,history=[],last_loss=None,
                training_seconds=0.,validation_seconds=0.,checkpoint_seconds=0.,input_wait_seconds=0.,
                inline_replay=None,stability_500_passed=False,calibration_input_verified=False,
                first_25_batch_hashes=[],last_input_hash=None)


class Engine:
    def __init__(self,student,teacher,guide,method,beta,device,signature,output):
        self.student=student.train();self.teacher=teacher;self.guide=guide
        if guide is not None:guide.train()
        self.method=method;self.device=device;self.signature=signature;self.output=output
        self.groups=parameter_groups(student,guide)
        self.parameters=[p for params in self.groups.values() for p in params]
        expected=[p for model in (student,guide) if model is not None for p in model.parameters() if p.requires_grad]
        if len(self.parameters)!=len({id(p) for p in expected}) or {id(p) for p in self.parameters}!={id(p) for p in expected}:
            raise ValueError('Optimizer does not own every trainable parameter exactly once')
        self.optimizer=torch.optim.AdamW(self.parameters,lr=6e-5,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
        kind='ibkd' if method.startswith('ibkd') else method
        self.controller=StepController(kind,beta) if kind in ('lg','alg','ibkd') else None
        self.teacher_hash=None if teacher is None else state_hash(teacher)
        self.progress=initial_progress()

    def capture(self):
        return cpu_tree(dict(signature=self.signature,progress=self.progress,student=self.student.state_dict(),
                             guide=None if self.guide is None else self.guide.state_dict(),
                             optimizer=self.optimizer.state_dict(),controller=None if self.controller is None else self.controller.state_dict(),
                             rng=packed_rng(),sampler_next_position=self.progress['global_step']*self.signature['batch_size']))

    def restore(self,payload):
        if payload['signature']!=self.signature:raise ValueError('Checkpoint signature differs')
        if payload['sampler_next_position']!=payload['progress']['global_step']*self.signature['batch_size']:
            raise ValueError('Checkpoint sampler cursor differs')
        self.student.load_state_dict(payload['student'],strict=True)
        if self.guide is not None:self.guide.load_state_dict(payload['guide'],strict=True)
        elif payload['guide'] is not None:raise ValueError('Unexpected guide state')
        # Optimizer.load_state_dict may reuse CPU tensors (including CUDA Adam's
        # CPU step counter). Replaying must never mutate the saved snapshot.
        self.optimizer.load_state_dict(cpu_tree(payload['optimizer']))
        if self.controller is not None:self.controller.load_state_dict(payload['controller'])
        elif payload['controller'] is not None:raise ValueError('Unexpected controller')
        self.progress=copy.deepcopy(payload['progress']);unpack_rng(payload['rng'])
        self.student.train()
        if self.guide is not None:self.guide.train()
        assert_tree(self.capture(),payload,rtol=0,atol=0,path='restored_full_state')

    def save(self):
        started=time.perf_counter()
        pointer=save_checkpoint(self.output,self.capture())
        self.progress['checkpoint_seconds']+=time.perf_counter()-started
        return pointer

    def load(self,pointer):
        payload,info=load_checkpoint(pointer,self.signature,self.output)
        self.restore(payload)
        return info

    def step(self,images,labels):
        step=self.progress['global_step'];lr=learning_rate(step)
        for group in self.optimizer.param_groups:group['lr']=lr
        self.optimizer.zero_grad(set_to_none=True)
        beta=0. if self.controller is None else self.controller.beta
        enabled=self.guide is not None and (self.controller is None or beta>0)
        images=images.to(self.device);labels=labels.to(self.device)
        if enabled:loss,terms,raw=compute(self.method,self.student,self.teacher,self.guide,images,labels,beta)
        else:
            loss=pixel_cross_entropy(self.student(images),labels);terms={'ce':loss};raw=None
        if not bool(torch.isfinite(loss)) or not all(bool(torch.isfinite(x)) for x in terms.values()):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        norms={name:gradient_norm(params) for name,params in self.groups.items()}
        if not all(norms[name]>0 for name in ('encoder','decoder')):raise ValueError('Missing student gradient')
        if enabled and not all(v>0 for k,v in norms.items() if k.startswith('guidance.')):raise ValueError('Missing guide gradient')
        if self.teacher is not None and any(p.grad is not None for p in self.teacher.parameters()):raise ValueError('Teacher gradient')
        self.optimizer.step()
        if not bool(torch.stack([torch.isfinite(p).all() for p in self.parameters]).all()):raise ValueError('Nonfinite parameter')
        raw_value=None if raw is None else float(raw.detach())
        if self.controller:self.controller.observe_step(raw_value or 0.,images.shape[0])
        row=dict(step=step+1,loss=float(loss.detach()),loss_components={k:float(v.detach()) for k,v in terms.items()},
                 raw_guidance=raw_value,weighted_guidance=0. if raw_value is None else raw_value*(beta if self.controller else 1.),
                 beta=beta,lr=lr,guidance_on=enabled,gradient_norms=norms)
        self.progress.update(global_step=step+1,last_loss=row)
        return row

    def replay_next_step(self,images,labels):
        """One update is replayed, then the original continuous trajectory is restored."""
        self.save()
        before,_=load_checkpoint(self.output/'resume.json',self.signature,self.output)
        self.restore(before)  # Verify the actual serialized checkpoint before the first path.
        first=self.step(images,labels);expected=self.capture()
        self.restore(before)
        replay=self.step(images,labels);actual=self.capture()
        try:
            assert_tree(replay,first,rtol=2e-5,atol=2e-6,path='replay_loss_gradient')
            assert_tree(actual,expected,rtol=2e-5,atol=2e-6,path='replay_all_state')
        finally:
            self.restore(expected)
        self.progress['inline_replay']=dict(status='passed',checkpoint_step=before['progress']['global_step'],
                                             replay_step=self.progress['global_step'],restore='bitwise',rtol=2e-5,atol=2e-6)
        return first

    def check_teacher(self):
        if self.teacher is not None and state_hash(self.teacher)!=self.teacher_hash:raise ValueError('Teacher state changed')


@torch.inference_mode()
def evaluate(student,loader,device,*,expected_count=500,shape=(1024,2048),stop=lambda:False,emit=None):
    previous_mode=student.training;before_rng=packed_rng();student.eval()
    matrix=torch.zeros(19,19,dtype=torch.int64);count=0
    try:
        for images,labels,_ in loader:
            if stop():return None
            if tuple(images.shape)!=(1,3,*shape) or tuple(labels.shape)!=(1,*shape):raise ValueError('Invalid evaluation shape')
            if shape[1]!=2*shape[0]:raise ValueError('Expected two square halves')
            images=images.to(device);side=shape[0]
            parts=[student(images[...,left:left+side]) for left in (0,side)]
            logits=torch.nn.functional.interpolate(torch.cat(parts,dim=-1),size=shape,mode='bilinear',align_corners=True)
            prediction=logits.argmax(1).cpu();valid=labels!=-1
            matrix+=torch.bincount(19*labels[valid]+prediction[valid],minlength=361).reshape(19,19)
            count+=1
            if emit and (count%100==0 or count==expected_count):emit(count)
        if count!=expected_count or matrix.sum()==0:raise ValueError('Incomplete validation inventory')
        result=scores(matrix);result.update(validation_samples=count,full_validation=expected_count==500)
        return result
    finally:
        student.train(previous_mode);unpack_rng(before_rng)


def record_validation(engine,metrics):
    progress=engine.progress;step=progress['global_step'];old=progress['best']
    if old is None or metrics['miou']>old['metrics']['miou']:
        progress['best']=save_best(engine.output,engine.student,engine.signature,step,metrics)
    progress.update(last_metrics=metrics,last_eval_step=step)
    progress['history'].append(dict(step=step,metrics=metrics))


def fit(engine,loader,evaluate_fn,*,target=2000,validation_every=400,checkpoint_every=100,
        stability_step=500,stop=lambda:False,emit=lambda kind,value:None):
    """Shared production/test loop; a paused full validation is retried on resume."""
    from .training_data import batch_digest
    engine.save();iterator=iter(loader)
    try:
        while True:
            p=engine.progress;step=p['global_step']
            if step==target and p['last_eval_step']==target:status='completed';break
            if step>target:raise ValueError('Checkpoint exceeds target')
            if stop():status='paused';break
            if step>0 and step%validation_every==0 and p['last_eval_step']<step:
                emit('validation_start',dict(step=step));started=time.perf_counter()
                metrics=evaluate_fn()
                p['validation_seconds']+=time.perf_counter()-started
                if metrics is None:status='paused';break
                record_validation(engine,metrics);engine.check_teacher();engine.save()
                emit('validation',dict(step=step,metrics=metrics))
                continue
            if step==target:raise ValueError('Target must coincide with full validation')
            began=time.perf_counter();images,labels,names=next(iterator)
            p['input_wait_seconds']+=time.perf_counter()-began;digest=batch_digest(images,labels,names)
            if engine.device.type=='cuda':torch.cuda.synchronize()
            began=time.perf_counter()
            row=engine.replay_next_step(images,labels) if step==2 and p['inline_replay'] is None else engine.step(images,labels)
            if engine.device.type=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-began;p=engine.progress
            p['training_seconds']+=elapsed;p['last_input_hash']=digest
            if step<25:p['first_25_batch_hashes'].append(digest)
            if p['global_step']==stability_step:
                if p['inline_replay'] is None or p['last_eval_step']<validation_every:raise ValueError('Initial training checks incomplete')
                p['stability_500_passed']=True
            emit('train',dict(row,seconds=elapsed,input_sha256=digest))
            if p['global_step']%checkpoint_every==0 or p['global_step'] in (3,stability_step):engine.save()
    finally:
        if hasattr(loader,'close'):loader.close()
    engine.check_teacher();engine.save()
    return status


def rank_candidates(results,expected_ids,*,target=2000,keep=2):
    by_id={r['run_id']:r for r in results}
    if len(by_id)!=len(results) or set(by_id)!=set(expected_ids):raise ValueError('Incomplete candidate inventory')
    if any(r['status']!='completed' or r['completed_steps']!=target or r['last_eval_step']!=target or not r['best'] for r in results):
        return dict(status='pending',selected_run_ids=[],reason='All four complete candidates are required')
    ordered=sorted(results,key=lambda r:(-r['best']['metrics']['miou'],r['best']['epoch'],r['candidate_id']))
    return dict(status='selected',selected_run_ids=[r['run_id'] for r in ordered[:keep]],
                ranking=[dict(run_id=r['run_id'],best_step=r['best']['epoch'],best_miou=r['best']['metrics']['miou']) for r in ordered])
