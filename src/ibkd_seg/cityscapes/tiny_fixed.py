"""Keep the smoke-verified FSKD*/C2VKD* recipes in the resumable Tiny runner."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .runtime import state_hash


class FixedRecipe:
    def __init__(self, method, model, teacher, config, cache_root, device):
        from . import official_api as api
        self.method, self.model, self.teacher = method, model, teacher
        self.protocol = config[method]
        self.soft_rank = None
        if method == 'fskd':
            from .tiny_fskd import TinyFSKD, CLSAttentionCapture, weighted_components, verify_soft_rank
            self.guide = TinyFSKD(crop=config['crop_size'], student_channels=config['encoder_channels'])
            self.soft_rank = verify_soft_rank(device)
            self.capture = api.FeatureCapture(model)
            self.attention = CLSAttentionCapture(model)
        elif method == 'c2vkd':
            from .tiny_c2vkd import TinyC2VKD, FinalFeatureCapture, weighted_components
            self.guide = TinyC2VKD(cache_root / 'weights/clip_rn101.pt',
                                  student_channels=config['encoder_channels'])
            self.capture = FinalFeatureCapture(model, student_channels=config['encoder_channels'])
        else:
            raise ValueError('Unsupported fixed Tiny recipe')
        self.weighted_components = weighted_components
        self.guide = self.guide.to(device).train()
        self.pool_initial_hash = state_hash(self.guide.pool) if method == 'c2vkd' else None

    def losses(self, images, target):
        from .official_api import teacher_input
        if self.method == 'fskd':
            logits, features, attention = self.attention.forward(self.capture, self.model, images)
        else:
            logits, features = self.capture.forward(self.model, images)
        ce = F.cross_entropy(logits, target, ignore_index=255)
        with torch.no_grad():
            teacher_features = self.teacher.extract_feat(teacher_input(images))
            teacher_logits = self.teacher.decode_head(teacher_features)
        if self.method == 'fskd':
            raw = self.guide(features, attention, teacher_features, logits, teacher_logits, target)
            if not raw['attention'].requires_grad:
                raise RuntimeError('FSKD attention detached from the student')
        else:
            raw = self.guide(features, teacher_features, logits, teacher_logits, target)
        weighted = self.weighted_components(raw)
        if not all(bool(torch.isfinite(x)) for x in raw.values()):
            raise FloatingPointError('Nonfinite fixed-recipe component')
        guided = sum(weighted.values())
        # C2VKD PDD already includes GT supervision; diagnostic CE is not added twice.
        loss = self.protocol['coefficients']['ce'] * ce + guided
        return loss, ce, guided, dict(
            components={k: float(v.detach()) for k, v in raw.items()},
            weighted_components={k: float(v.detach()) for k, v in weighted.items()},
            standalone_ce_coefficient=self.protocol['coefficients']['ce'], guidance_multiplier=1.)

    def check_gradients(self):
        modules = list(self.guide.align) if self.method == 'fskd' else [self.guide.visual, self.guide.linguistic]
        if any(not any(p.grad is not None and bool(p.grad.abs().max() > 0) for p in m.parameters()) for m in modules):
            raise RuntimeError('Missing fixed-recipe adapter gradient')

    def check_frozen(self, *, final=False):
        if self.method == 'c2vkd':
            pool = self.guide.pool
            if pool.training or any(p.requires_grad or p.grad is not None for p in pool.parameters()):
                raise RuntimeError('C2VKD CLIP attention pool must stay frozen/eval')
            if final and state_hash(pool) != self.pool_initial_hash:
                raise RuntimeError('C2VKD CLIP attention pool changed')

    def metadata(self, *, final=False):
        self.check_frozen(final=final)
        return dict(fixed_loss_coefficients=self.protocol['coefficients'],
                    method_provenance=self.protocol, controller=None,
                    soft_rank_execution=self.soft_rank,
                    clip_pool=None if self.method != 'c2vkd' else dict(
                        asset=self.guide.asset, derived_pool_state_sha256=self.pool_initial_hash,
                        frozen_no_grad_unchanged=True if final else None))
