"""Declared FSKD DeiT-Tiny recipe transfer to our Segmenter Cityscapes track.

The classification recipe is public; its use for Cityscapes is a reconstruction,
not an author-verified segmentation recipe. No C2VKD/CLIP assets are used.
"""
from __future__ import annotations

from importlib.metadata import version

import torch
from torch import nn
from torch.nn import functional as F

from .b0.losses import Alignment, gram_mse, resize


class CLSAttentionCapture:
    """Read the existing last encoder attention without a second model forward."""

    def __init__(self, model):
        if model.encoder.distilled:
            raise ValueError("The Tiny protocol expects one CLS token and no DIST token")
        if model.encoder.blocks[-1].attn.attn_drop.p != 0:
            raise ValueError("Expected the common zero attention-dropout recipe")
        self.enabled = False
        self.value = None

        def capture(_module, _inputs, output):
            if self.enabled:
                attention = output[1]
                # Keep the native CLS query and remove only its CLS key.
                self.value = attention[:, :, 0, 1:]

        self.handle = model.encoder.blocks[-1].attn.register_forward_hook(capture)

    def forward(self, capture, model, images):
        self.enabled = True
        self.value = None
        try:
            logits, features = capture.forward(model, images)
            if self.value is None:
                raise RuntimeError("The encoder did not expose CLS attention")
            attention = self.value
        finally:
            self.enabled = False
            self.value = None
        return logits, features, attention


def pixel_logit_kd(student, teacher, labels):
    """T=1 KL(teacher || student), mean over valid pixels, ignore=255."""
    shape = labels.shape[-2:]
    student = F.interpolate(student, size=shape, mode="bilinear", align_corners=False)
    teacher = F.interpolate(teacher.detach(), size=shape, mode="bilinear", align_corners=False)
    valid = labels != 255
    if not valid.any():
        raise ValueError("No valid pixels for FSKD logit KD")
    s = student.permute(0, 2, 3, 1)[valid]
    t = teacher.permute(0, 2, 3, 1)[valid]
    return F.kl_div(s.log_softmax(-1), t.softmax(-1), reduction="batchmean")


def attention_rank_loss(cls_attention, teacher_last, grid):
    import torchsort
    # soft_rank keeps the gradient into the actual student attention.
    importance = cls_attention.mean(1)
    target = resize(teacher_last.detach(), grid).square().mean(1).flatten(1).softmax(-1)
    if importance.shape != target.shape or importance.shape[-1] < 2:
        raise ValueError("CLS attention and teacher spatial grid differ")
    ranks = torchsort.soft_rank(importance, regularization="l2", regularization_strength=1.)
    target_ranks = torchsort.soft_rank(target, regularization="l2", regularization_strength=1.)
    n = importance.shape[-1]
    # Algebraically 1-Spearman; avoid subtracting two near-one float32 values.
    return (6 * (ranks - target_ranks).square().sum(-1) / (n * (n * n - 1))).mean()


class TinyFSKD(nn.Module):
    def __init__(self, *, crop=512):
        super().__init__()
        if crop % 16 or crop < 32:
            raise ValueError("FSKD Tiny requires a patch16-compatible crop >=32")
        self.crop = crop
        # README's custom_model/deit.py stage_info (not the generic timm wrapper).
        self.student_blocks = (0, 3)
        self.teacher_blocks = (0, 1)
        self.align = nn.ModuleList([
            Alignment(192, 256, (crop // 16) ** 2, (crop // 4) ** 2),
            Alignment(192, 512, (crop // 16) ** 2, (crop // 8) ** 2),
        ])

    def forward(self, features, attention, teacher, student_logits, teacher_logits, labels):
        if len(features) != 12 or len(teacher) != 4:
            raise ValueError("FSKD expects 12 Tiny blocks and all four ResNet stages")
        glob, patch = features[0].new_zeros(()), features[0].new_zeros(())
        for si, ti, adapter in zip(self.student_blocks, self.teacher_blocks, self.align, strict=True):
            s, t = features[si], teacher[ti].detach()
            expected = self.crop // (4 if ti == 0 else 8)
            if (s.shape[1:] != (192, self.crop // 16, self.crop // 16) or
                    t.shape[1:] != ((256 if ti == 0 else 512), expected, expected)):
                raise ValueError("Unexpected Tiny/ResNet-D8 feature geometry")
            # Preserve the public code's additional /B after MSEmean.
            glob = glob + F.mse_loss(adapter(s, t), t) / s.shape[0]
            st = F.normalize(s.flatten(2).transpose(1, 2), dim=-1, eps=1e-8)
            tt = F.normalize(resize(t, s.shape[-2:]).flatten(2).transpose(1, 2), dim=-1, eps=1e-8)
            patch = patch + gram_mse(st, tt)
        return {"global": glob, "patch": patch,
                "attention": attention_rank_loss(attention, teacher[3], features[-1].shape[-2:]),
                "logit_kd": pixel_logit_kd(student_logits, teacher_logits, labels)}


def weighted_components(raw):
    # README's ImageNet DeiT-Ti example, not the PiT-Ti CIFAR example used for B0.
    weights = {"global": 100., "patch": 1., "attention": 1000000., "logit_kd": 1.}
    if set(raw) != set(weights):
        raise ValueError("Missing or unexpected FSKD loss components")
    return {name: raw[name] * weight for name, weight in weights.items()}


def verify_soft_rank(device):
    import torchsort
    if version("torchsort") != "0.1.10":
        raise RuntimeError("Expected torchsort 0.1.10")
    x = torch.tensor([[.1, .3, .2, .4]], device=device, requires_grad=True)
    rank = torchsort.soft_rank(x, regularization="l2", regularization_strength=1.)
    grad = torch.autograd.grad(rank.square().sum(), x)[0]
    if not torch.isfinite(grad).all() or not bool(grad.abs().max() > 0):
        raise RuntimeError("soft_rank has no finite nonzero student gradient")
    return {"version": version("torchsort"), "device": str(device), "forward_backward": "passed"}
