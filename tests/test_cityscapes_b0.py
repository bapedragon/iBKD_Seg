"""Small deterministic checks of reductions, gradients and controller boundaries."""
import math

import pytest
import torch
from torch.nn import functional as F

from ibkd_seg.cityscapes.b0.assets import hf_to_cirkd
from ibkd_seg.cityscapes.b0.losses import gram_mse,pdd,logit_kd,FSKD
from ibkd_seg.cityscapes.b0.control import StepController,boundary_checks
from ibkd_seg.cityscapes.b0.data import stitch_logits,scores
from ibkd_seg.cityscapes.b0.smoke import assert_tree


@pytest.mark.parametrize("shape",[(2,7,5),(7,5)])
def test_gram_chunk_forward_backward(shape):
    torch.manual_seed(3)
    s=torch.randn(*shape,dtype=torch.double,requires_grad=True)
    t=torch.randn(*shape,dtype=torch.double)
    actual=gram_mse(s,t,denominator=4,chunk=3)
    expected=F.mse_loss(s@s.transpose(-1,-2)/4,t@t.transpose(-1,-2)/4)
    torch.testing.assert_close(actual,expected)
    ga=torch.autograd.grad(actual,s,retain_graph=True)[0]
    gb=torch.autograd.grad(expected,s)[0]
    torch.testing.assert_close(ga,gb)


def test_pdd_constant_shift_gradient_and_ignore():
    torch.manual_seed(4)
    s=torch.randn(2,19,3,4,dtype=torch.double,requires_grad=True)
    t=torch.randn_like(s)
    y=torch.randint(0,19,(2,3,4));y[:,0,0]=-1
    a=pdd(s,t,y);b=pdd(s,t,y,normalize_target=False)
    torch.testing.assert_close(a-b,a.new_tensor(.5*math.log(2)))
    ga=torch.autograd.grad(a,s,retain_graph=True)[0];gb=torch.autograd.grad(b,s)[0]
    torch.testing.assert_close(ga,gb)
    assert (ga[:,:,0,0]==0).all()
    with pytest.raises(ValueError):pdd(s,t,torch.full_like(y,-1))


def test_pixel_kd_has_no_image_area_multiplier():
    s=torch.tensor([[[[.5]],[[1.2]]]],requires_grad=True)
    t=torch.tensor([[[[1.]],[[.2]]]])
    a=logit_kd(s,t,torch.zeros(1,1,1,dtype=torch.long))
    b=logit_kd(s.expand(-1,-1,4,4),t.expand(-1,-1,4,4),torch.zeros(1,4,4,dtype=torch.long))
    torch.testing.assert_close(a,b)


def test_controller_partial_resume_and_boundaries():
    assert len(boundary_checks())==3
    a=StepController("ibkd",.3)
    for _ in range(17):a.observe_step(2.,16)
    b=StepController("ibkd",.3);b.load_state_dict(a.state_dict())
    assert b.steps==17 and b.samples==272 and b.controller.losses==[]
    for _ in range(169):
        a.observe_step(3.,16);b.observe_step(3.,16)
    assert a.state_dict()==b.state_dict()


def test_hf_kv_merge_preserves_values():
    state={"segformer.encoder.block.0.0.attention.self.key.weight":torch.ones(2,2),
           "segformer.encoder.block.0.0.attention.self.value.weight":torch.full((2,2),2.),
           "classifier.weight":torch.randn(3,2)}
    x=hf_to_cirkd(state)
    assert set(x)=={"block1.0.attn.kv.weight"}
    torch.testing.assert_close(x["block1.0.attn.kv.weight"],torch.tensor([[1.,1.],[1.,1.],[2.,2.],[2.,2.]]))
    with pytest.raises(ValueError):hf_to_cirkd({"garbage":torch.ones(1)})


def test_fskd_rank_reaches_student_attention():
    torch.manual_seed(3)
    # Real channel dimensions with small spatial grids; no pretrained/GPU claim.
    stages=[None,None,torch.randn(2,160,4,4,requires_grad=True),torch.randn(2,256,2,2,requires_grad=True)]
    teacher=[None,None,torch.randn(2,1024,8,8),torch.randn(2,2048,8,8)]
    logits=torch.randn(2,19,16,16,requires_grad=True)
    tlogits=torch.randn_like(logits)
    labels=torch.randint(0,19,(2,16,16))
    pre=torch.randn(2,8,4,4,requires_grad=True)
    losses=FSKD(crop=64)(stages,pre.softmax(-1),teacher,logits,tlogits,labels)
    grad=torch.autograd.grad(losses["attention"],pre)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0


def test_stitch_low_resolution_then_resize():
    x=torch.zeros(1,3,1024,2048);x[:,:,:,1024:]=1
    def model(tile):return F.adaptive_avg_pool2d(tile[:,:1],(2,2)).expand(-1,19,-1,-1)
    expected=F.interpolate(torch.cat((model(x[...,:1024]),model(x[...,1024:])),dim=-1),size=(1024,2048),mode="bilinear",align_corners=True)
    torch.testing.assert_close(stitch_logits(model,x),expected)


def test_metrics_average_all_19_classes_and_resume_tree():
    matrix=torch.zeros(19,19,dtype=torch.int64);matrix[0,0]=10
    assert scores(matrix)["miou"]==pytest.approx(1/19)
    with pytest.raises(AssertionError):assert_tree({"step":3},{"step":2},rtol=0,atol=0)
def test_cirkd_train_and_validation_record_contract():
    import numpy as np
    from ibkd_seg.cityscapes.b0.data import sample_tensors
    x=np.zeros((3,2,2),dtype=np.float32)
    y=np.array([[0,18],[255,-1]],dtype=np.int32)
    _,a,name=sample_tensors((x,y,"train_name"))
    _,b,val_name=sample_tensors((x,y,(2,2,3),("image_path","val_name")))
    assert a.tolist()==[[0,18],[-1,-1]] and torch.equal(a,b)
    assert name=="train_name" and val_name==("image_path","val_name")
