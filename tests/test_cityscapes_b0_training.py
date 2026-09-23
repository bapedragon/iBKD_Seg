"""Meaningful checks for the new data stream, full-state replay and selection."""
import copy
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from ibkd_seg.cityscapes.b0 import training as tr
from ibkd_seg.cityscapes.b0.training_data import make_plan,PlannedDataset,batches,batch_digest
from ibkd_seg.cityscapes.b0.screen import specification,candidates
from ibkd_seg.cityscapes.b0.smoke import assert_tree,cpu_tree
from ibkd_seg.cityscapes.b0.data import sample_tensors
from ibkd_seg.cityscapes.runtime import seed_all


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Module()
        self.net.encoder=nn.Sequential(nn.Conv2d(3,4,1),nn.BatchNorm2d(4),nn.Dropout(.2))
        self.net.linear_pred=nn.Conv2d(4,19,1)

    def forward(self,x):return self.net.linear_pred(self.net.encoder(x))


def toy_compute(method,student,teacher,guide,images,labels,beta):
    logits=student(images);ce=tr.pixel_cross_entropy(logits,labels)
    with torch.no_grad():target=teacher(images)
    raw=(guide(logits)-target).square().mean()
    return ce+beta*raw,dict(ce=ce,locality=raw),raw


def engine(tmp_path,monkeypatch):
    monkeypatch.setattr(tr,'compute',toy_compute)
    seed_all(7);student=TinyStudent();guide=nn.Sequential(nn.Conv2d(19,3,1));teacher=nn.Conv2d(3,3,1).eval().requires_grad_(False)
    return tr.Engine(student,teacher,guide,'lg',.4,torch.device('cpu'),dict(batch_size=2,identity='test'),tmp_path)


def inputs():
    generator=torch.Generator().manual_seed(123)
    return [(torch.randn(2,3,8,8,generator=generator),torch.randint(0,19,(2,8,8),generator=generator)) for _ in range(6)]


def test_checkpoint_replay_and_new_process_equivalent_resume(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    first=engine(tmp_path/'first',monkeypatch);stream=inputs()
    for x,y in stream[:2]:first.step(x,y)
    first.replay_next_step(*stream[2])
    assert first.progress['global_step']==3 and first.progress['inline_replay']['status']=='passed'
    first.save();checkpoint=tmp_path/'first/resume.json'
    for batch in stream[3:]:first.step(*batch)
    expected=first.capture()
    resumed=engine(tmp_path/'resumed',monkeypatch);info=resumed.load(checkpoint)
    assert info['global_step']==3
    for batch in stream[3:]:resumed.step(*batch)
    actual=resumed.capture()
    # save() reports its wall time after serializing, which is not scientific state.
    expected['progress']['checkpoint_seconds']=actual['progress']['checkpoint_seconds']
    assert_tree(actual,expected,rtol=0,atol=0)
    resumed.check_teacher()


def test_identity_change_rejected(tmp_path,monkeypatch):
    e=engine(tmp_path,monkeypatch);e.step(*inputs()[0]);payload=e.capture()
    payload['signature']['identity']='different'
    with pytest.raises(ValueError,match='signature'):e.restore(payload)


def test_interrupted_validation_resumes_before_next_training_step(tmp_path,monkeypatch):
    e=engine(tmp_path/'first',monkeypatch);stream=[(x,y,['a','b']) for x,y in inputs()]
    events=[]
    def interrupted():return None
    assert tr.fit(e,iter(stream),interrupted,target=6,validation_every=2,stability_step=5,checkpoint_every=2,
                  emit=lambda kind,value:events.append((kind,value['step'])))=='paused'
    assert e.progress['global_step']==2 and e.progress['last_eval_step']==0
    later=engine(tmp_path/'later',monkeypatch);later.load(tmp_path/'first/resume.json')
    resumed_events=[]
    def validation():return dict(miou=.2,pixel_accuracy=.5)
    assert tr.fit(later,iter(stream[2:]),validation,target=6,validation_every=2,stability_step=5,checkpoint_every=2,
                  emit=lambda kind,value:resumed_events.append((kind,value['step'])))=='completed'
    assert resumed_events[:3]==[('validation_start',2),('validation',2),('train',3)]
    assert [r['step'] for r in later.progress['history']]==[2,4,6]
    assert later.progress['inline_replay']['status']=='passed' and later.progress['stability_500_passed']
    extended=engine(tmp_path/'extended',monkeypatch);extended.load(tmp_path/'later/resume.json')
    before_counter=extended.optimizer.state[next(iter(extended.student.parameters()))]['step'].item()
    assert tr.fit(extended,iter(stream[:2]),validation,target=8,validation_every=2,checkpoint_every=2)=='completed'
    assert extended.progress['global_step']==8 and [r['step'] for r in extended.progress['history']]==[2,4,6,8]
    assert extended.optimizer.state[next(iter(extended.student.parameters()))]['step'].item()==before_counter+2


def test_guidance_off_skips_teacher_guide_and_weight_decay(tmp_path,monkeypatch):
    e=engine(tmp_path,monkeypatch);e.step(*inputs()[0]);e.controller.beta=0.
    before=cpu_tree(e.guide.state_dict())
    def forbidden(*args):raise AssertionError('Guidance executed after off')
    monkeypatch.setattr(e.teacher,'forward',forbidden);monkeypatch.setattr(e.guide,'forward',forbidden)
    row=e.step(*inputs()[1])
    assert not row['guidance_on'] and row['raw_guidance'] is None and row['weighted_guidance']==0
    assert_tree(e.guide.state_dict(),before,rtol=0,atol=0)
    assert all(p.grad is None for p in e.guide.parameters())


def test_poly_schedule_keeps_80000_horizon():
    assert tr.learning_rate(0)==6e-5
    assert tr.learning_rate(1999)==pytest.approx(6e-5*(1-1999/80000)**.9)
    assert tr.learning_rate(2000)>5.8e-5


def test_eval_two_halves_ignore_and_rng_mode_restore():
    class Perfect(nn.Module):
        def forward(self,x):
            logits=x.new_zeros((1,19,*x.shape[-2:]));logits[:,2]=1
            return logits
    model=Perfect().train();images=torch.zeros(1,3,8,16);labels=torch.full((1,8,16),2);labels[:,:,0]=-1
    before=tr.packed_rng()
    result=tr.evaluate(model,[(images,labels,['one']),(images,labels,['two'])],torch.device('cpu'),expected_count=2,shape=(8,16))
    assert result['valid_pixels']==240 and result['pixel_accuracy']==1 and result['class_iou'][2]==1
    assert result['miou']==pytest.approx(1/19)
    assert model.training;assert_tree(tr.packed_rng(),before,rtol=0,atol=0)
    assert tr.evaluate(model,[(images,labels,['one'])],torch.device('cpu'),expected_count=2,shape=(8,16),stop=lambda:True) is None
    with pytest.raises(ValueError,match='Incomplete'):tr.evaluate(model,[(images,labels,['one'])],torch.device('cpu'),expected_count=2,shape=(8,16))


def test_earliest_best_step_preserved_on_miou_tie(tmp_path,monkeypatch):
    e=engine(tmp_path,monkeypatch)
    e.progress['global_step']=400;tr.record_validation(e,dict(miou=.3,pixel_accuracy=.6))
    first=copy.deepcopy(e.progress['best'])
    e.progress['global_step']=800;tr.record_validation(e,dict(miou=.3,pixel_accuracy=.7))
    assert e.progress['best']==first and e.progress['last_eval_step']==800


def test_rank_requires_four_and_uses_miou_step_candidate_id():
    rows=[dict(run_id=f'lg_{i}',candidate_id=f'beta_{i}',status='completed',completed_steps=2000,last_eval_step=2000,
               best=dict(epoch=step,metrics=dict(miou=score))) for i,step,score in ((3,800,.5),(2,400,.5),(1,400,.5),(4,400,.4))]
    expected=[r['run_id'] for r in rows]
    assert tr.rank_candidates(rows,expected)['selected_run_ids']==['lg_1','lg_2']
    rows[0]['status']='paused'
    assert tr.rank_candidates(rows,expected)['status']=='pending'
    with pytest.raises(ValueError,match='inventory'):tr.rank_candidates(rows[:-1],expected)


def test_three_user_packs_and_frozen_betas():
    config,grid=specification();packs=[candidates(config,grid,name) for name in ('pack1','pack2','pack3')]
    assert list(map(len,packs))==[6,8,4]
    assert len({r['run_id'] for pack in packs for r in pack})==18
    assert {r['method'] for r in packs[0]}=={'vanilla','fskd','lg'}
    assert {r['method'] for r in packs[1]}=={'alg','ibkd_lambda025'}
    assert {r['method'] for r in packs[2]}=={'ibkd_lambda050'}
    assert [r['beta'] for r in packs[2]]==[.4303,1.00403,2.1515,4.303]


class Indexed:
    def __len__(self):return 32
    def __getitem__(self,i):return torch.full((3,2,2),float(i)),torch.full((2,2),i,dtype=torch.long),str(i)


def test_prefetch_threads_resume_exact_order_without_rng_changes():
    seed_all(9);before=tr.packed_rng()
    full=list(batches(Indexed(),0,32,4,workers=4));part=list(batches(Indexed(),12,32,4,workers=2))
    assert [batch_digest(*b) for b in part]==[batch_digest(*b) for b in full[3:]]
    assert_tree(tr.packed_rng(),before,rtol=0,atol=0)
    with pytest.raises(ValueError,match='Incomplete'):list(batches(Indexed(),0,31,4,workers=4))


def test_plan_does_not_consume_training_rng():
    seed_all(3);before=tr.packed_rng();p=make_plan(32)
    assert p.shape==(32,5)
    assert_tree(tr.packed_rng(),before,rtol=0,atol=0)
    np.testing.assert_array_equal(p,make_plan(32))


def test_planned_transforms_equal_actual_pinned_cirkd(tmp_path):
    cache=Path(os.environ.get('B0_SCREEN_CACHE','/private/tmp/b0_smoke_assets'))
    if not (cache/'cirkd/dataset/cityscapes.py').exists():pytest.skip('Pinned upstream cache unavailable; H200 also verifies real calibration tensor hash')
    from ibkd_seg.cityscapes.b0.assets import modules
    _,_,upstream=modules(cache)
    yy,xx=np.mgrid[:1024,:2048]
    image=np.stack((xx%256,yy%256,(xx+yy)%256),axis=-1).astype(np.uint8)
    labels=np.asarray([7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33],dtype=np.uint8)[(xx//31+yy//17)%19]
    cv2.imwrite(str(tmp_path/'image.png'),image);cv2.imwrite(str(tmp_path/'label.png'),labels)
    listing=tmp_path/'train.lst';listing.write_text('image.png label.png\n')
    original=upstream.CSTrainValSet(str(tmp_path),str(listing),crop_size=(512,512))
    plan=make_plan(25);planned=PlannedDataset(tmp_path,listing,plan)
    seed_all(1)
    for i in range(25):
        expected=sample_tensors(original[0]);actual=planned[i]
        assert_tree(actual,expected,rtol=0,atol=0)


@pytest.mark.parametrize('workers',[0,2])
def test_ignore_only_crop_in_batch26_keeps_stream_and_training(tmp_path,monkeypatch,workers):
    # One image has an ignored left half and a valid right half. Batch 26 is
    # the first mixed batch, matching the boundary missed by 25-batch preflight.
    image=np.full((8,16,3),128,dtype=np.uint8)
    raw=np.zeros((8,16),dtype=np.uint8);raw[:,8:]=7
    cv2.imwrite(str(tmp_path/'image.png'),image);cv2.imwrite(str(tmp_path/'label.png'),raw)
    listing=tmp_path/'train.lst';listing.write_text('image.png label.png\n')
    valid=[0,5,0,8,1];ignored=[0,5,0,0,1]
    plan=np.asarray([valid]*50+[ignored,valid]+[valid]*2,dtype=np.int32)
    dataset=PlannedDataset(tmp_path,listing,plan,crop=8)
    stream=list(batches(dataset,0,len(plan),2,workers=workers))
    assert len(stream)==27
    assert all((y!=-1).all() for _,y,_ in stream[:25])
    assert (stream[25][1][0]==-1).all() and (stream[25][1][1]==0).all()
    resumed=list(batches(dataset,50,len(plan),2,workers=workers))
    assert [batch_digest(*b) for b in resumed]==[batch_digest(*b) for b in stream[25:]]
    model=engine(tmp_path/'training',monkeypatch)
    assert tr.fit(model,iter(stream),lambda:dict(miou=.2,pixel_accuracy=.5),target=27,
                  validation_every=27,checkpoint_every=100)=='completed'
    assert model.progress['global_step']==27 and model.progress['inline_replay']['status']=='passed'
    assert np.isfinite(model.progress['last_loss']['loss'])


def test_ignore_only_samples_use_batch_valid_pixels_for_ce_and_kd():
    from ibkd_seg.cityscapes.b0.losses import logit_kd
    generator=torch.Generator().manual_seed(8)
    student=torch.randn(2,19,4,4,generator=generator,requires_grad=True)
    teacher=torch.randn(2,19,4,4,generator=generator)
    labels=torch.randint(0,19,(2,4,4),generator=generator);labels[0]=-1
    ce=tr.pixel_cross_entropy(student,labels);kd=logit_kd(student,teacher,labels)
    torch.testing.assert_close(ce,tr.pixel_cross_entropy(student[1:],labels[1:]),rtol=0,atol=0)
    torch.testing.assert_close(kd,logit_kd(student[1:],teacher[1:],labels[1:]),rtol=0,atol=0)
    (ce+kd).backward()
    assert torch.isfinite(student.grad).all() and student.grad[0].count_nonzero()==0
    assert student.grad[1].count_nonzero()>0
    # A whole batch without labels still has no supervised mean; fail explicitly.
    with pytest.raises(ValueError,match='No valid labels'):tr.pixel_cross_entropy(student,torch.full_like(labels,-1))
    with pytest.raises(ValueError,match='No valid labels'):logit_kd(student,teacher,torch.full_like(labels,-1))


def test_ignore_only_samples_match_pinned_cirkd_and_validation(tmp_path):
    cache=Path(os.environ.get('B0_SCREEN_CACHE','/private/tmp/b0_smoke_assets'))
    if not (cache/'cirkd/dataset/cityscapes.py').exists():pytest.skip('Pinned upstream cache unavailable')
    from ibkd_seg.cityscapes.b0.assets import modules
    _,_,upstream=modules(cache)
    cv2.imwrite(str(tmp_path/'image.png'),np.full((16,32,3),128,dtype=np.uint8))
    cv2.imwrite(str(tmp_path/'label.png'),np.zeros((16,32),dtype=np.uint8))
    listing=tmp_path/'train.lst';listing.write_text('image.png label.png\n')
    original=upstream.CSTrainValSet(str(tmp_path),str(listing),crop_size=(8,8))
    planned=PlannedDataset(tmp_path,listing,make_plan(1,crop=8,shape=(16,32)),crop=8)
    seed_all(1);expected=sample_tensors(original[0])
    assert_tree(planned[0],expected,rtol=0,atol=0)
    assert (planned[0][1]==-1).all()
    assert (PlannedDataset(tmp_path,listing)[0][1]==-1).all()
