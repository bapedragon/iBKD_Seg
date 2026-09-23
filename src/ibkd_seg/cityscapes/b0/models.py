"""Expose CIRKD B0 blocks and teacher backbone features without replacing forwards."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .assets import hf_to_cirkd, modules
from ...phase1.models import IBKD
from ..models import ChunkedCrossAttention


class Student(nn.Module):
    def __init__(self, upstream, checkpoint=None):
        super().__init__()
        self.net = upstream.MiT_B0(None, (512,512), 19, nn.BatchNorm2d)
        if checkpoint is not None:
            weights = hf_to_cirkd(torch.load(checkpoint, map_location="cpu", weights_only=True))
            expected = {k for k in self.net.state_dict() if k.startswith(("patch_embed", "block", "norm"))}
            if set(weights) != expected:
                raise ValueError(f"Encoder keys differ: missing={expected-set(weights)}, extra={set(weights)-expected}")
            complete = self.net.state_dict()
            complete.update(weights)
            self.net.load_state_dict(complete, strict=True)
        self.capture = False
        self.raw, self.stages, self.hw, self.attention = [], [], {}, None
        for stage in range(1,5):
            def patch_hook(_m, _args, out, stage=stage):
                if self.capture:
                    self.hw[stage] = out[1:]
            getattr(self.net, f"patch_embed{stage}").register_forward_hook(patch_hook)
            for block in getattr(self.net, f"block{stage}"):
                def block_hook(_m, args, out):
                    if self.capture:
                        self.raw.append(out.transpose(1,2).reshape(out.shape[0],out.shape[2],args[1],args[2]))
                block.register_forward_hook(block_hook)
            def norm_hook(_m, _args, out, stage=stage):
                if self.capture:
                    self.stages.append(out.transpose(1,2).reshape(out.shape[0],out.shape[2],*self.hw[stage]))
            getattr(self.net, f"norm{stage}").register_forward_hook(norm_hook)
        def attention_hook(_m, args):
            if self.capture:
                self.attention = args[0]
        self.net.block4[-1].attn.attn_drop.register_forward_pre_hook(attention_hook)

    def forward(self, x, *, features=False):
        self.raw, self.stages, self.hw, self.attention = [], [], {}, None
        self.capture = features
        try:
            logits = self.net(x)[0]
            output = (logits, self.raw, self.stages, self.attention) if features else logits
        finally:
            self.capture = False
            self.raw, self.stages, self.hw, self.attention = [], [], {}, None
        return output


class Teacher(nn.Module):
    def __init__(self, upstream, checkpoint=None):
        super().__init__()
        self.net = upstream.DeepLabV3(19, backbone="resnet101", aux=True, pretrained_base="None")
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if "state_dict" in state:
                state = state["state_dict"]
            state = {k.removeprefix("module."):v for k,v in state.items()}
            self.net.load_state_dict(state, strict=True)
        self.eval().requires_grad_(False)

    def forward(self, x):
        with torch.no_grad():
            features = self.net.base_forward(x)
            logits, _ = self.net.head(features[-1])
        return logits, features


class Locality(nn.Module):
    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleList(nn.Conv2d(s,t,1) for s,t in zip((32,160,256),(512,1024,2048)))

    def forward(self, raw, teacher):
        loss = raw[0].new_zeros(())
        for i,p,t in zip((0,4,7), self.projections, teacher[1:]):
            s = p(raw[i])
            shape = tuple(max(a,b) for a,b in zip(s.shape[-2:],t.shape[-2:]))
            s = F.interpolate(s,size=shape,mode="bilinear",align_corners=False)
            t = F.interpolate(t,size=shape,mode="bilinear",align_corners=False)
            loss = loss + F.mse_loss(s,t)
        return loss


class AllBlockIBKD(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapters = nn.ModuleList(nn.Identity() if c==256 else nn.Conv2d(c,256,1)
                                     for c in (32,32,64,64,160,160,256,256))
        self.core = IBKD((512,1024,2048), student_channels=256, student_blocks=8)
        for i, original in enumerate(self.core.fusion):
            replacement = ChunkedCrossAttention((512,1024,2048)[i], chunk_size=256, recompute=True)
            replacement.load_state_dict(original.state_dict(), strict=True)
            self.core.fusion[i] = replacement

    def forward(self, raw, teacher):
        aligned = [F.interpolate(a(x),size=(16,16),mode="bilinear",align_corners=False)
                   for a,x in zip(self.adapters,raw,strict=True)]
        return self.core(aligned, teacher[1:])


def build(cache, device):
    s,t,_ = modules(cache)
    student = Student(s,cache / "weights/nvidia_mit_b0.bin").to(device)
    teacher = Teacher(t,cache / "weights/cirkd_teacher.pth").to(device)
    return student, teacher
