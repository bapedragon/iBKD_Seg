"""Scratch segmentation backbones and exact, query-chunked iBKD attention."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ibkd_seg.phase1.models import IBKD, ConvCrossAttention, LocalityGuidance

TEACHER_CHANNELS = (512, 1024, 2048)
DECODER_BLOCKS = (2, 5, 8, 11)


class MultiLevelDecoder(nn.Module):
    """Four lateral projections followed by a common convolutional fusion head."""
    def __init__(self, input_channels, channels):
        super().__init__()
        def block(in_channels, out_channels, kernel):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False),
                nn.GroupNorm(8, out_channels), nn.GELU(),
            )
        self.lateral = nn.ModuleList(block(c, channels, 1) for c in input_channels)
        self.fuse = nn.Sequential(block(4 * channels, channels, 3), nn.Conv2d(channels, 19, 1))

    def forward(self, features):
        size = features[0].shape[-2:]
        maps = [F.interpolate(layer(x), size=size, mode="bilinear", align_corners=False)
                for layer, x in zip(self.lateral, features, strict=True)]
        return self.fuse(torch.cat(maps, dim=1))


class Segmenter(nn.Module):
    def __init__(self, kind: str, config: dict):
        super().__init__()
        self.kind = kind
        if kind == "teacher":
            from torchvision.models import resnet50
            self.encoder = resnet50(weights=None)
            self.encoder.fc = nn.Identity()
            channels = (256, 512, 1024, 2048)
        elif kind == "student":
            import timm
            self.encoder = timm.create_model(
                "deit_tiny_patch16_224", pretrained=False, num_classes=0,
                img_size=tuple(config["crop_size"]), dynamic_img_size=True,
                drop_path_rate=config["drop_path_rate"],
            )
            self.encoder.set_grad_checkpointing(config["gradient_checkpointing"])
            # The segmentation decoder consumes pre-norm spatial block outputs.
            # No unused classifier/final normalization parameters are optimized.
            self.encoder.norm = nn.Identity()
            channels = (192,) * 4
        else:
            raise ValueError(kind)
        self.decoder = MultiLevelDecoder(channels, config["decoder_channels"])

    def features(self, images):
        if self.kind == "student":
            return list(self.encoder.forward_intermediates(
                images, indices=list(range(12)), norm=False,
                output_fmt="NCHW", intermediates_only=True,
            ))
        net = self.encoder
        value = net.maxpool(net.relu(net.bn1(net.conv1(images))))
        features = []
        for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
            value = layer(value)
            features.append(value)
        return features

    def forward(self, images, *, return_features=False):
        features = self.features(images)
        decoder_features = features if self.kind == "teacher" else [features[i] for i in DECODER_BLOCKS]
        logits = self.decoder(decoder_features)
        return (logits, features) if return_features else logits


class ChunkedCrossAttention(ConvCrossAttention):
    """Identical all-key softmax per query; checkpoint chunks to bound saved attention."""
    def __init__(self, channels, *, chunk_size, recompute):
        super().__init__(channels, num_heads=4)
        self.chunk_size, self.recompute = chunk_size, recompute

    def forward(self, student_feature, teacher_feature):
        if student_feature.shape != teacher_feature.shape:
            raise ValueError("Student and teacher grids must match after alignment")
        batch, channels, height, width = student_feature.shape
        def split(value):
            return value.flatten(2).transpose(1, 2).reshape(
                batch, height * width, self.num_heads, self.head_dim
            ).permute(0, 2, 1, 3)
        query = split(self.query(self.cbam(student_feature)))
        key, value = split(self.key(teacher_feature)), split(self.value(teacher_feature))
        outputs = []
        for chunk in query.split(self.chunk_size, dim=2):
            if self.recompute and torch.is_grad_enabled():
                result = checkpoint(F.scaled_dot_product_attention, chunk, key, value, use_reentrant=False)
            else:
                result = F.scaled_dot_product_attention(chunk, key, value)
            outputs.append(result)
        output = torch.cat(outputs, dim=2).transpose(1, 2).reshape(batch, height * width, channels)
        return self.output(output.transpose(1, 2).reshape(batch, channels, height, width))


def build_guidance(method: str, config: dict):
    if method == "vanilla":
        return None
    if method in {"lg", "alg"}:
        return LocalityGuidance(TEACHER_CHANNELS)
    if method != "ibkd":
        raise ValueError(method)
    module = IBKD(TEACHER_CHANNELS)
    # Keep the existing modules/initialization/state_dict while changing evaluation order only.
    for index, original in enumerate(module.fusion):
        replacement = ChunkedCrossAttention(
            TEACHER_CHANNELS[index], chunk_size=config["attention_query_chunk"],
            recompute=config["gradient_checkpointing"],
        )
        replacement.load_state_dict(original.state_dict(), strict=True)
        module.fusion[index] = replacement
    return module
