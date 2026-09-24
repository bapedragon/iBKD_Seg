"""Protection boundary, real optimizer resume, and isolation from warm-up zero."""
import json

import pytest
import torch

from ibkd_seg.cityscapes.b0.control import StepController
from ibkd_seg.cityscapes.b0.screen import code_identity as original_identity, specification as original_spec
from ibkd_seg.cityscapes.b0.smoke import assert_tree,cpu_tree
from ibkd_seg.cityscapes.b0.training import rank_candidates,fit,evaluate
from ibkd_seg.cityscapes.b0_warmup20.screen import (
    configure_engine,check_protection,specification,candidates,REPO,run_scope,apply_scope,result,
)
from test_cityscapes_b0_training import engine as toy_engine,inputs


def warm_engine(path,monkeypatch,*,warmup=20,beta=.4):
    e=toy_engine(path,monkeypatch)
    e.method='ibkd_lambda025';e.controller=StepController('ibkd',beta)
    e.signature['guidance_warmup_epochs']=warmup
    return configure_engine(e) if warmup==20 else e


def test_scope_grid_and_frozen_warmup0_source_identity():
    config,grid=specification();old,_=original_spec()
    assert config['controller']['warmup']==20 and old['controller']['warmup']==0
    assert config['target_steps']==10000 and config['selection']['keep_per_method']==1
    assert config['selection']['interim_2000']=='observe_only_no_candidate_elimination'
    rows=[r for pack in config['groups'] for r in candidates(config,grid,pack)]
    assert len(rows)==8 and {r['method'] for r in rows}=={'ibkd_lambda025','ibkd_lambda050'}
    recorded=json.loads((REPO/'phase4/Cityscapes_SegFormer-B0/reports/h200_screen2000_v2_pack3/log_audit.json').read_text())
    assert original_identity()==recorded['r030_signature']['source']


def test_exact_step3720_boundary_and_original_alg_unchanged(tmp_path,monkeypatch):
    e=warm_engine(tmp_path,monkeypatch)
    alg=StepController('alg',.4)
    for step in range(1,3721):
        assert e.controller.beta==.4
        e.controller.observe_step(1.,2);alg.observe_step(1.,2)
        e.progress['global_step']=step;check_protection(e)
        if step in (2000,3719):assert e.controller.controller.active
    assert e.controller.controller.stop_epoch==20 and e.controller.beta==0
    assert alg.controller.warmup_epochs==0 and alg.controller.stop_epoch==2
    e.controller.observe_step(0.,2)
    assert e.controller.beta==0 and e.controller.steps==1


def test_minimum_period_does_not_force_stop_at20(tmp_path,monkeypatch):
    e=warm_engine(tmp_path,monkeypatch)
    for epoch in range(1,22):
        for _ in range(186):e.controller.observe_step(1000.-20*epoch,2)
    check_protection(e)
    assert e.controller.controller.stop_epoch is None and e.controller.beta==.4


def test_full_checkpoint_resume_across_warmup_boundary(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    e=warm_engine(tmp_path/'continuous',monkeypatch)
    # Construct a boundary fixture: the parameter/optimizer state is a tiny
    # model, and prior interval losses are synthetic. No GPU result is claimed.
    for _ in range(3719):e.controller.observe_step(1.,2)
    e.progress['global_step']=3719;e.save()
    batch=inputs()[0];row=e.step(*batch);check_protection(e)
    assert row['step']==3720 and row['guidance_on'] and e.controller.controller.stop_epoch==20
    guide_before=cpu_tree(e.guide.state_dict())
    row=e.step(*batch);check_protection(e)
    assert row['step']==3721 and not row['guidance_on'] and row['raw_guidance'] is None
    assert all(p.grad is None for p in e.guide.parameters())
    assert_tree(e.guide.state_dict(),guide_before,rtol=0,atol=0)
    expected=e.capture()
    resumed=warm_engine(tmp_path/'resumed',monkeypatch)
    resumed.load(tmp_path/'continuous/resume.json')
    for _ in range(2):resumed.step(*batch)
    check_protection(resumed);actual=resumed.capture()
    expected['progress']['checkpoint_seconds']=actual['progress']['checkpoint_seconds']
    assert_tree(actual,expected,rtol=0,atol=0)


def test_warmup0_checkpoint_and_non_ibkd_override_rejected(tmp_path,monkeypatch):
    old=warm_engine(tmp_path/'old',monkeypatch,warmup=0)
    new=warm_engine(tmp_path/'new',monkeypatch)
    with pytest.raises(ValueError,match='signature'):new.restore(old.capture())
    original_alg=toy_engine(tmp_path/'alg',monkeypatch);original_alg.method='alg'
    with pytest.raises(ValueError,match='iBKD-only'):configure_engine(original_alg)
    new.progress['global_step']=1
    with pytest.raises(ValueError,match='before training'):configure_engine(new)


def test_protection_rejects_early_off(tmp_path,monkeypatch):
    e=warm_engine(tmp_path,monkeypatch)
    e.controller.controller.beta_history=[.4,0.]
    with pytest.raises(ValueError,match='disabled during warm-up'):check_protection(e)


def test_selection_waits_for_all_four_at10k():
    config,grid=specification();rows=candidates(config,grid,'lambda025')
    ids=[r['run_id'] for r in rows]
    for i,row in enumerate(rows):
        row.update(status='completed',completed_steps=2000,last_eval_step=2000,
                   best=dict(epoch=1600,metrics=dict(miou=40.+i)))
    def rank():
        return rank_candidates(rows,ids,target=config['target_steps'],keep=config['selection']['keep_per_method'])
    assert rank()['status']=='pending' and not rank()['selected_run_ids']
    for row in rows[:-1]:row.update(completed_steps=10000,last_eval_step=10000)
    assert rank()['status']=='pending'
    rows[-1].update(completed_steps=10000,last_eval_step=10000)
    assert rank()['selected_run_ids']==[ids[-1]]


@pytest.mark.parametrize('candidate_index',range(4))
def test_smoke32_all_betas_replay_val2_and_checkpoint_isolation(tmp_path,monkeypatch,candidate_index):
    torch.set_num_threads(1)
    config,grid=specification();candidate=candidates(config,grid,'lambda025')[candidate_index]
    e=warm_engine(tmp_path/'smoke',monkeypatch,beta=candidate['beta'])
    e.signature=apply_scope({**e.signature,'candidate':candidate},32)
    stream=inputs()
    train=[(*stream[i%len(stream)],['a','b']) for i in range(32)]
    val=[(torch.cat([x[:1],x[:1]],-1),torch.cat([y[:1],y[:1]],-1),['val']) for x,y in stream[:2]]
    observed=[];scope=run_scope(32)
    def emit(kind,row):
        check_protection(e)
        if kind=='train':observed.append(row)
    status=fit(e,iter(train),lambda:evaluate(e.student,iter(val),e.device,expected_count=scope['validation_samples'],shape=(8,16)),
               target=scope['target_steps'],validation_every=scope['validation_every'],emit=emit)
    row=result(e,candidate,status,target=32)
    assert status=='completed' and len(observed)==32
    assert all(r['guidance_on'] and r['beta']==candidate['beta'] for r in observed)
    assert row['inline_replay']['status']=='passed'
    assert row['diagnostic_metrics']['validation_samples']==2 and not row['diagnostic_metrics']['full_validation']
    assert row['selected_epoch'] is None and row['selected_step'] is None and not row['selection_performed']
    assert row['guidance_protection_status']=='protected_so_far_boundary_not_reached'
    assert row['warmup_protection_passed'] is None and not row['minimum_period_reached']
    same=warm_engine(tmp_path/'smoke_resume',monkeypatch,beta=candidate['beta']);same.signature=e.signature
    same.load(tmp_path/'smoke/resume.json')
    assert same.progress['global_step']==32 and same.controller.controller.warmup_epochs==20
    full=warm_engine(tmp_path/'full',monkeypatch,beta=candidate['beta'])
    full.signature=apply_scope({**full.signature,'candidate':candidate},2000)
    with pytest.raises(ValueError,match='signature'):full.restore(e.capture())
    assert apply_scope(full.signature,10000)==full.signature
    assert run_scope(2000)['validation_every']==400 and run_scope(2000)['validation_samples']==500
