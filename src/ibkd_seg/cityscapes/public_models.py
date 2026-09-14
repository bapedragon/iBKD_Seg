"""DeepLabV3-R101 and Segmenter-S/16 mask-decoder architecture smoke models.

No pretrained weights are loaded by this module. This is deliberately separate
from the older ResNet50/DeiT + custom convolution-decoder pilot.
Mask decoder adapted from rstrudel/segmenter, commit
20d1bfad354165ee45c3f65972a4d9c131f58d53 (MIT; see phase4 license notice).
"""
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import Block

from .models import TEACHER_CHANNELS, build_guidance


class DeepLabV3Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models.segmentation import deeplabv3_resnet101

        self.model = deeplabv3_resnet101(
            weights=None, weights_backbone=None, num_classes=19, aux_loss=False,
        )
        # IntermediateLayerGetter already contains all four ResNet stages.
        self.model.backbone.return_layers = {
            "layer1": "stage1", "layer2": "stage2",
            "layer3": "stage3", "layer4": "out",
        }

    def features(self, images):
        outputs = self.model.backbone(images)
        return [outputs[name] for name in ("stage1", "stage2", "stage3", "out")]

    def forward(self, images, *, return_features=False):
        features = self.features(images)
        logits = self.model.classifier(features[-1])
        return (logits, features) if return_features else logits


class MaskTransformer(nn.Module):
    """Segmenter class-token mask decoder with two Transformer blocks.

Uses timm's equivalent pre-norm blocks/SDPA and safe FP32 mask normalization.
This is an architecture adaptation, not an upstream checkpoint converter.
"""
    def __init__(self, width=384, classes=19):
        super().__init__()
        self.classes = classes
        self.proj_dec = nn.Linear(width, width)
        self.cls_emb = nn.Parameter(torch.empty(1, classes, width))
        self.blocks = nn.ModuleList([
            Block(width, width // 64, mlp_ratio=4, qkv_bias=True,
                  proj_drop=0.1, attn_drop=0.1,
                  norm_layer=partial(nn.LayerNorm, eps=1e-5)) for _ in range(2)
        ])
        self.decoder_norm = nn.LayerNorm(width)
        self.mask_norm = nn.LayerNorm(classes)
        self.proj_patch = nn.Parameter(torch.randn(width, width) * width ** -0.5)
        self.proj_classes = nn.Parameter(torch.randn(width, width) * width ** -0.5)
        self.apply(self._init)
        nn.init.trunc_normal_(self.cls_emb, std=0.02)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, grid):
        tokens = self.proj_dec(tokens)
        tokens = torch.cat([tokens, self.cls_emb.expand(tokens.shape[0], -1, -1)], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.decoder_norm(tokens)
        patches = tokens[:, :-self.classes] @ self.proj_patch
        classes = tokens[:, -self.classes:] @ self.proj_classes
        # Norm reductions in FP32 avoid BF16 underflow; retain the original
        # cosine-mask equation, with epsilon only for exactly zero vectors.
        with torch.autocast(tokens.device.type, enabled=False):
            masks = F.normalize(patches.float(), dim=-1) @ F.normalize(classes.float(), dim=-1).transpose(1, 2)
            masks = self.mask_norm(masks)
        return masks.transpose(1, 2).reshape(tokens.shape[0], self.classes, *grid)


class SegmenterStudent(nn.Module):
    def __init__(self, config):
        super().__init__()
        import timm

        self.encoder = timm.create_model(
            "vit_small_patch16_384", pretrained=False, num_classes=0,
            img_size=tuple(config["crop_size"]), dynamic_img_size=True,
            drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-5),
        )
        self.encoder.set_grad_checkpointing(True)
        self.decoder = MaskTransformer()

    def forward(self, images, *, return_features=False):
        height, width = images.shape[-2:]
        images = F.pad(images, (0, (-width) % 16, 0, (-height) % 16))
        tokens, features = self.encoder.forward_intermediates(
            images, indices=list(range(12)), norm=False,
            output_fmt="NCHW", intermediates_only=False,
        )
        logits = self.decoder(tokens[:, self.encoder.num_prefix_tokens:], features[-1].shape[-2:])
        # Remove patch padding before the shared native-resolution evaluator.
        logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        logits = logits[..., :height, :width]
        return (logits, list(features)) if return_features else logits


def public_guidance(method, config):
    module = build_guidance(method, config)
    if module is not None:
        # Existing LG/iBKD implementations assume DeiT-Tiny width 192.
        # Only the adapter input dimension changes for ViT-S width 384.
        module.projections = nn.ModuleList(nn.Conv2d(384, c, 1) for c in TEACHER_CHANNELS)
    return module
