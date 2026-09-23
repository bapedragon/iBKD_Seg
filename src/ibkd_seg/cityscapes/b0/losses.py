"""Explicit reductions for the selected FSKD and C2VKD reconstructions."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def resize(x, shape):
    return F.interpolate(x, size=shape, mode="bilinear", align_corners=False)


def valid_logits(student, teacher, labels):
    shape = labels.shape[-2:]
    s = F.interpolate(student,size=shape,mode="bilinear",align_corners=True).permute(0,2,3,1)
    t = F.interpolate(teacher,size=shape,mode="bilinear",align_corners=True).permute(0,2,3,1)
    valid = labels != -1
    if not valid.any():
        raise ValueError("No valid labels")
    return s[valid],t[valid].detach(),labels[valid]


def logit_kd(student, teacher, labels):
    s,t,_ = valid_logits(student,teacher,labels)
    return F.kl_div(s.log_softmax(-1),t.softmax(-1),reduction="batchmean")


def pdd(student, teacher, labels, *, normalize_target=True):
    s,t,y = valid_logits(student,teacher,labels)
    s,t = s.softmax(-1),t.softmax(-1)
    mask = F.one_hot(y.long(),s.shape[-1]).bool()
    sb = torch.stack((s.masked_select(mask),s.masked_fill(mask,0).sum(-1)),dim=-1)
    tb = torch.stack((t.masked_select(mask),t.masked_fill(mask,0).sum(-1)),dim=-1)
    q = tb + tb.new_tensor([1.,0.])
    if normalize_target:
        q = (q/2).clamp_min(1e-8)
        q = q/q.sum(-1,keepdim=True)
    return (0.5*sb*(sb.clamp_min(1e-8).log()-q.clamp_min(1e-8).log())).sum(-1).mean()


def gram_mse(s, t, *, denominator=1, chunk=256):
    """Exact all-pairs loss, supporting either [B,N,C] or [N,C]."""
    n = s.shape[-2]
    if s.shape[:-1] != t.shape[:-1]:
        raise ValueError("Relation grids differ")
    total = s.new_zeros(())
    for start in range(0,n,chunk):
        def compute(a,b,*,start=start):
            aa = a[...,start:start+chunk,:] @ a.transpose(-1,-2) / denominator
            bb = b[...,start:start+chunk,:] @ b.transpose(-1,-2) / denominator
            return (aa-bb).square().sum()
        total = total + (checkpoint(compute,s,t,use_reentrant=False) if torch.is_grad_enabled() else compute(s,t))
    batch = s.shape[0] if s.ndim==3 else 1
    return total/(batch*n*n)


class Alignment(nn.Module):
    def __init__(self,cs,ct,ns,nt):
        super().__init__()
        self.spatial = nn.Sequential(nn.Linear(ns,nt),nn.GELU())
        self.channel = nn.Sequential(nn.Conv2d(cs,ct,3,padding=1),nn.GELU())
        nn.init.trunc_normal_(self.spatial[0].weight,std=.02)
        nn.init.zeros_(self.spatial[0].bias)
        nn.init.kaiming_normal_(self.channel[0].weight,mode="fan_out",nonlinearity="relu")

    def forward(self,s,t):
        x = self.spatial(s.flatten(2)).reshape(s.shape[0],s.shape[1],*t.shape[-2:])
        return self.channel(x)


class FSKD(nn.Module):
    def __init__(self, *, crop=512):
        super().__init__()
        self.align = nn.ModuleList([Alignment(160,1024,(crop//16)**2,(crop//8)**2),
                                   Alignment(256,2048,(crop//32)**2,(crop//8)**2)])

    def forward(self, stages, attention, teacher, slogits, tlogits, labels):
        import torchsort
        global_loss,patch_loss = stages[2].new_zeros(()),stages[2].new_zeros(())
        for j,adapter in zip((2,3),self.align):
            s,t = stages[j],teacher[j].detach()
            global_loss = global_loss + F.mse_loss(adapter(s,t),t)/s.shape[0]
            st = F.normalize(s.flatten(2).transpose(1,2),dim=-1,eps=1e-8)
            tt = F.normalize(resize(t,s.shape[-2:]).flatten(2).transpose(1,2),dim=-1,eps=1e-8)
            patch_loss = patch_loss + gram_mse(st,tt)
        importance = attention.mean(dim=(1,2))
        target = resize(teacher[3],stages[3].shape[-2:]).square().mean(1).flatten(1).softmax(-1)
        n = importance.shape[-1]
        if n<2:
            raise ValueError("FSKD attention requires at least two spatial tokens")
        rs = torchsort.soft_rank(importance,regularization="l2",regularization_strength=1.)
        rt = torchsort.soft_rank(target,regularization="l2",regularization_strength=1.)
        attn = (6*(rs-rt).square().sum(-1)/(n*(n*n-1))).mean()
        return {"global":global_loss,"patch":patch_loss,"attention":attn,
                "logit_kd":logit_kd(slogits,tlogits,labels)}


class AttentionPool(nn.Module):
    """Frozen CLIP RN101 pool; the global query equals DenseCLIP's global output."""
    def __init__(self, path):
        super().__init__()
        model = torch.jit.load(str(path),map_location="cpu")
        original = {k.removeprefix("visual.attnpool."):v.float() for k,v in model.state_dict().items()
                    if k.startswith("visual.attnpool.")}
        del model
        if tuple(original["positional_embedding"].shape)!=(50,2048):
            raise ValueError("CLIP RN101 pool positional shape changed")
        for key in ("q_proj","k_proj","v_proj","c_proj"):
            setattr(self,key,nn.Linear(2048,512 if key=="c_proj" else 2048))
        pos = original.pop("positional_embedding")
        spatial = resize(pos[1:].T.reshape(1,2048,7,7),(16,16)).flatten(2)[0].T
        original["positional_embedding"] = torch.cat((pos[:1],spatial))
        self.positional_embedding = nn.Parameter(torch.empty(257,2048))
        self.load_state_dict(original,strict=True)
        self.eval().requires_grad_(False)

    def forward(self,x):
        x = resize(x,(16,16)).flatten(2).permute(2,0,1)
        x = torch.cat((x.mean(0,keepdim=True),x),dim=0)+self.positional_embedding[:,None,:]
        y,_ = F.multi_head_attention_forward(
            query=x[:1],key=x,value=x,embed_dim_to_check=2048,num_heads=32,
            in_proj_weight=None,in_proj_bias=torch.cat((self.q_proj.bias,self.k_proj.bias,self.v_proj.bias)),
            bias_k=None,bias_v=None,add_zero_attn=False,dropout_p=0.,
            out_proj_weight=self.c_proj.weight,out_proj_bias=self.c_proj.bias,
            training=False,need_weights=False,use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,k_proj_weight=self.k_proj.weight,v_proj_weight=self.v_proj.weight)
        return y[0]


class C2VKD(nn.Module):
    def __init__(self,pool_path):
        super().__init__()
        self.visual = nn.Conv2d(256,2048,1,bias=False)
        self.linguistic = nn.Conv2d(256,512,1,bias=False)
        self.pool = AttentionPool(pool_path)

    def train(self, mode=True):
        super().train(mode)
        self.pool.eval()
        return self

    def forward(self,stages,teacher,slogits,tlogits,labels):
        s = resize(self.visual(stages[3]),(32,32))
        t = resize(teacher[3].detach(),(32,32))
        sl = self.linguistic(F.adaptive_avg_pool2d(stages[3],1)).flatten(1)
        with torch.no_grad():
            tl = self.pool(teacher[3])
        def kl(s,t):
            ls,lt = s.log_softmax(1),t.log_softmax(1)
            return (lt.exp()*(lt-ls)).mean()
        def rows(x):
            x = F.normalize(x.permute(0,2,3,1).reshape(-1,2048),dim=-1,eps=1e-8)
            return x-x.mean(-1,keepdim=True)
        return {"pdd":pdd(slogits,tlogits,labels),"global":kl(s,t),
                "patch":gram_mse(rows(s),rows(t),denominator=2047),"linguistic":kl(sl,tl)}
