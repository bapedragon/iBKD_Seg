"""Calibration must preserve training state and freeze only valid, comparable grids."""
import copy
import statistics

import pytest
import torch
from torch import nn

from ibkd_seg.cityscapes.b0 import calibration as cal
from ibkd_seg.cityscapes.b0.calibration_data import sample_positions
from ibkd_seg.cityscapes.runtime import seed_all


def test_mix_each_batch_before_median():
    rows=[{"ce":10.,"alignment":a,"fusion":f} for a,f in ((1.,100.),(10.,1.),(100.,10.))]
    grid=cal.candidate_grid(rows,"ibkd_lambda025")
    assert grid["median_raw_guidance"]==25.75
    assert grid["median_raw_guidance"]!=.75*statistics.median([1,10,100])+.25*statistics.median([100,1,10])
    for candidate,ratio in zip(grid["candidates"],cal.RATIOS):
        assert candidate["beta_unrounded"]==ratio*10/25.75
        assert candidate["beta"]==float(f"{ratio*10/25.75:.6g}")
        measured=[candidate["beta"]*(.75*r["alignment"]+.25*r["fusion"])/r["ce"] for r in rows]
        assert candidate["measured_weighted_guidance_to_ce"]==cal.spread(measured)
    assert grid["selected_beta"] is None


@pytest.mark.parametrize("guidance",[0.,-1.,float("nan"),float("inf"),1e-13])
def test_invalid_guidance_never_becomes_large_candidate(guidance):
    with pytest.raises(ValueError):cal.candidate_grid([{"ce":3.,"locality":guidance}],"lg")


@pytest.mark.parametrize("fail_on_second",[False,True])
def test_measurement_restores_buffers_rng_and_never_builds_gradient(monkeypatch,fail_on_second):
    seed_all(10)
    student=nn.Sequential(nn.Linear(4,4),nn.BatchNorm1d(4),nn.Dropout(.5)).train()
    guide=nn.Sequential(nn.Linear(4,4),nn.BatchNorm1d(4)).train()
    teacher=nn.Linear(4,4).eval().requires_grad_(False)
    batches=[(torch.randn(3,4),torch.zeros(3,dtype=torch.long)) for _ in range(3)]
    initial=[cal.cpu_tree(m.state_dict()) for m in (student,guide,teacher)]
    initial_rng=cal.rng_state();calls=[]
    def fake_compute(method,s,t,g,x,y,beta):
        assert not torch.is_grad_enabled()
        z=s(x);q=g(z);t(x)
        calls.append(1)
        assert s[1].num_batches_tracked>0
        if fail_on_second and len(calls)==2:raise RuntimeError("injected measurement failure")
        terms={"ce":z.square().mean()+1,"locality":q.square().mean()+1}
        return sum(terms.values()),terms,terms["locality"]
    monkeypatch.setattr(cal,"compute",fake_compute)
    progress={}
    if fail_on_second:
        with pytest.raises(RuntimeError,match="injected"):
            cal.observe_batches("lg_alg",student,teacher,guide,batches,torch.device("cpu"),progress,expected_batches=3)
        assert len(progress["losses"])==1
    else:
        grids=cal.observe_batches("lg_alg",student,teacher,guide,batches,torch.device("cpu"),progress,expected_batches=3)
        assert grids[0]["candidates"]==grids[1]["candidates"]
        assert progress["parameters_unchanged"] and progress["teacher_frozen_verified"]
        assert progress["optimizer_updates"]==progress["backward_calls"]==0
    for m,expected in zip((student,guide,teacher),initial):
        cal.assert_tree(m.state_dict(),expected,rtol=0,atol=0)
        assert all(p.grad is None for p in m.parameters())
    cal.assert_tree(cal.rng_state(),initial_rng,rtol=0,atol=0)
    assert progress["student_guide_buffers_and_rng_restored"]


def records():
    result=[]
    for family,methods in (("lg_alg",("lg","alg")),("ibkd",("ibkd_lambda025","ibkd_lambda050"))):
        rows=[{"ce":3.+i/100,"locality":2.,"alignment":4.,"fusion":1.} for i in range(25)]
        result.append({"family":family,"status":"passed","completed_batches":25,"optimizer_updates":0,"backward_calls":0,
                       **{key:True for key in ("parameters_unchanged","gradients_absent","teacher_frozen_verified","student_guide_buffers_and_rng_restored")},
                       **{key:"same" for key in ("student_initial_state_sha256","teacher_state_sha256","input_identity_sha256","calibration_spec_sha256")},
                       "losses":rows,"methods":[cal.candidate_grid(rows,m) for m in methods]})
    return result


def test_merge_only_freezes_candidates_not_selected_model():
    report=cal.merge_families(records())
    assert report["beta_candidates_frozen"] and not report["selection_performed"]
    assert len(report["methods"])==4 and report["metrics"] is None
    assert all(len(g["candidates"])==4 and g["selected_beta"] is None for g in report["methods"])


@pytest.mark.parametrize("change",["input","ce","update","restore","failed","backward"])
def test_incomparable_measurements_never_freeze(change):
    items=copy.deepcopy(records());other=items[1]
    if change=="input":other["input_identity_sha256"]="different"
    if change=="ce":other["losses"][0]["ce"]+=.01
    if change=="update":other["optimizer_updates"]=1
    if change=="restore":other["student_guide_buffers_and_rng_restored"]=False
    if change=="failed":other["status"]="failed"
    if change=="backward":other["backward_calls"]=1
    with pytest.raises(ValueError):cal.merge_families(items)


def test_calibration_extends_same_seed1_smoke_sampler():
    extended=sample_positions(400)
    assert extended[:48]==sample_positions(48)
    assert len(extended)==400 and len(set(extended))==400
    assert all(0<=i<1280000 for i in extended)


def test_frozen_spec_matches_parent_and_implementation():
    spec=cal.verify_spec()
    assert spec["actual_batch"]==16 and spec["crop_hw"]==[512,512]
    assert spec["beta_rounding_significant_digits"]==6
    assert spec["backward_calls"]==spec["optimizer_updates"]==spec["validation_samples"]==0
