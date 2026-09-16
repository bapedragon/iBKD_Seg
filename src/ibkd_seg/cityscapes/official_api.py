"""Compatibility boundary around the unmodified 2021 upstream implementation.

Only import wiring and activation recomputation are adapted. Model forward,
initial-weight conversion, augmentation, optimizer and inference are upstream.
"""
from __future__ import annotations

import collections.abc
import importlib
import sys
import types
from pathlib import Path


def bootstrap(root: Path):
    # Must precede imports of timm/mmcv/mmseg and the shared phase1 modules.
    if any(name in sys.modules for name in ("timm", "mmcv", "mmseg", "segm")):
        raise RuntimeError("Official runtime must be bootstrapped in a fresh process")
    sys.path[:0] = [str(root / name) for name in ("segmenter", "mmcv", "mmseg", "deps")]
    legacy = types.ModuleType("torch._six")
    legacy.container_abcs = collections.abc
    legacy.string_classes = (str, bytes)
    legacy.int_classes = (int,)
    sys.modules["torch._six"] = legacy
    import mmseg
    import mmcv
    import timm
    import torch
    from mmcv.utils.parrots_wrapper import SyncBatchNorm
    if not hasattr(torch.nn.SyncBatchNorm, "_specify_ddp_gpu_num"):
        # Removed setup-only API. Teacher SyncBN runs frozen in eval mode;
        # its normalization forward and checkpoint buffers are untouched.
        SyncBatchNorm._specify_ddp_gpu_num = lambda self, gpu_size: None
    if (mmcv.__version__, mmseg.__version__, timm.__version__) != ("1.3.8", "0.14.1", "0.4.12"):
        raise RuntimeError("Unexpected legacy runtime versions")
    # Avoid unrelated PointRend/DCN imports that require old compiled mmcv ops.
    # ResNetV1c/ASPP/FCN/EncoderDecoder themselves remain unmodified upstream.
    for part in ("models", "models.backbones", "models.decode_heads", "models.segmentors"):
        module = types.ModuleType("mmseg." + part)
        module.__path__ = [str(Path(mmseg.__file__).parent.joinpath(*part.split(".")))]
        sys.modules[module.__name__] = module
    for name in ("models.builder", "models.losses", "models.backbones.resnet",
                 "models.decode_heads.aspp_head", "models.decode_heads.fcn_head",
                 "models.segmentors.encoder_decoder"):
        importlib.import_module("mmseg." + name)


def recipe(root):
    import yaml
    original = yaml.safe_load((root / "segmenter/segm/config.yml").read_text())
    net = dict(original["model"]["vit_large_patch16_384"])
    data = original["dataset"]["cityscapes"]
    net.update(image_size=(data["crop_size"],) * 2, backbone="vit_large_patch16_384",
               n_cls=19, dropout=0.0, drop_path_rate=0.1,
               decoder=dict(original["decoder"]["mask_transformer"], name="mask_transformer"))
    return net, data


def student(root, *, recompute=True):
    from segm.model import factory
    from .official_assets import WEIGHTS
    net, _ = recipe(root)
    original_loader = factory.load_custom_pretrained

    def verified_loader(model, default_cfg):
        if default_cfg["url"] != WEIGHTS["vit_large_384.npz"]["url"]:
            raise RuntimeError("Upstream changed ViT initialization source")
        # Original timm 0.4.12 NPZ loader: every encoder block, cls, head,
        # norms and patch projection, with original bilinear pos interpolation.
        model.load_pretrained(str(root / "weights/vit_large_384.npz"))

    factory.load_custom_pretrained = verified_loader
    try:
        model = factory.create_segmenter(net)
    finally:
        factory.load_custom_pretrained = original_loader
    if model.encoder.n_layers != 24 or model.encoder.d_model != 1024:
        raise RuntimeError("Expected original L/16")
    if recompute:
        enable_recomputation(model)
    return model


def enable_recomputation(model):
    import torch
    from torch.utils.checkpoint import checkpoint
    for block in list(model.encoder.blocks) + list(model.decoder.blocks):
        original = block.forward

        def forward(*args, _original=original, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_original, *args, use_reentrant=False,
                                  preserve_rng_state=True, **kwargs)
            return _original(*args, **kwargs)

        block.forward = forward


class FeatureCapture:
    def __init__(self, model):
        self.features = [None] * len(model.encoder.blocks)
        self.enabled = False
        self.handles = []
        for index, block in enumerate(model.encoder.blocks):
            def capture(_module, _inputs, output, index=index):
                if self.enabled:
                    self.features[index] = output
            self.handles.append(block.register_forward_hook(capture))

    def forward(self, model, image):
        self.enabled = True
        try:
            logits = model(image)
        finally:
            self.enabled = False
        h, w = image.shape[-2] // 16, image.shape[-1] // 16
        features = [x[:, 1:].transpose(1, 2).reshape(image.shape[0], 1024, h, w)
                    for x in self.features]
        self.features = [None] * len(self.features)
        return logits, features


def teacher(root):
    import torch
    from mmcv import Config
    from mmseg.models.builder import build_segmentor
    path = root / "mmseg/configs/deeplabv3/deeplabv3_r101-d8_512x1024_80k_cityscapes.py"
    config = Config.fromfile(str(path))
    config.model.pretrained = None  # Entire Cityscapes checkpoint is loaded below.
    model = build_segmentor(config.model)
    # Legacy checkpoint metadata includes numpy objects. File is hash-verified
    # against the pinned official artifact before this trusted pickle load.
    checkpoint = torch.load(root / "weights/deeplabv3_r101.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().requires_grad_(False)


def teacher_input(student_image):
    rgb255 = student_image * 127.5 + 127.5
    mean = rgb255.new_tensor([123.675, 116.28, 103.53])[None, :, None, None]
    std = rgb255.new_tensor([58.395, 57.12, 57.375])[None, :, None, None]
    return (rgb255 - mean) / std


def guidance(method, config):
    from ibkd_seg.phase1.models import IBKD, LocalityGuidance
    from .models import ChunkedCrossAttention
    if method == "vanilla":
        return None
    channels = (512, 1024, 2048)
    kwargs = dict(student_channels=1024, student_blocks=24)
    if method in ("lg", "alg"):
        return LocalityGuidance(channels, **kwargs)
    module = IBKD(channels, **kwargs)
    for index, original in enumerate(module.fusion):
        replacement = ChunkedCrossAttention(channels[index], chunk_size=config["attention_query_chunk"], recompute=True)
        replacement.load_state_dict(original.state_dict(), strict=True)
        module.fusion[index] = replacement
    return module


def optimizer_scheduler(model, guidance_module, root):
    from types import SimpleNamespace
    import torch
    import math
    from segm.optim.factory import create_optimizer, create_scheduler
    _, data = recipe(root)
    args = SimpleNamespace(opt="sgd", lr=data["learning_rate"], weight_decay=0.0,
                           momentum=0.9, clip_grad=None, sched="polynomial",
                           epochs=data["epochs"], min_lr=1e-5, poly_power=0.9,
                           poly_step_size=1, iter_max=math.ceil(2975 / data["batch_size"]) * data["epochs"],
                           iter_warmup=0)
    bundle = torch.nn.ModuleList([model] + ([] if guidance_module is None else [guidance_module]))
    optimizer = create_optimizer(args, bundle)
    return optimizer, create_scheduler(args, optimizer)
