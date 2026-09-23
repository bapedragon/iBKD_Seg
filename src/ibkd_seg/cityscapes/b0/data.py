"""Identical cached smoke tensors from the first 48 positions of the 80k sampler."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .assets import modules
from ..data import sha256,save_json,verify_manifest
from ..runtime import seed_all
from ..real_smoke import validate_manifest


def sample_tensors(sample):
    # CIRKD train returns three fields; validation additionally returns image size.
    if len(sample) not in (3,4):
        raise ValueError("Unexpected CIRKD dataset record")
    image,label,name = sample[0],sample[1],sample[-1]
    x,y = torch.from_numpy(image).float(),torch.from_numpy(label).long()
    y[(y<0)|(y>=19)] = -1
    if not torch.isfinite(x).all() or not (y!=-1).any():
        raise ValueError("Invalid input/label")
    return x,y,name


def prepare(cache,root,output):
    _,_,upstream = modules(cache)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    verify_manifest(root,manifest)
    source = cache / "cirkd/dataset/list/cityscapes"
    train = upstream.CSTrainValSet(str(root),str(source/"train.lst"),crop_size=(512,512))
    val = upstream.CSValSet(str(root),str(source/"val.lst"))
    if len(train)!=2975 or len(val)!=500:
        raise ValueError("Unexpected split counts")
    # Repeat/truncate then shuffle is identical to indexing the base list modulo 2975.
    positions = torch.randperm(80000*16,generator=torch.Generator().manual_seed(1))[:48].tolist()
    seed_all(1)
    digest = hashlib.sha256()
    saved = {"train":[],"val":[],"train_names":[],"val_names":[],"expanded_sampler_positions":positions}
    batch=[]
    for split,indices,ds in (("train",[i%2975 for i in positions],train),("val",[0,1],val)):
        for index in indices:
            x,y,name = sample_tensors(ds[index])
            digest.update(str(name).encode())
            for value in (x,y): digest.update(value.contiguous().numpy().tobytes())
            saved[split+"_names"].append(name)
            if split=="train":
                batch.append((x,y))
                if len(batch)==16:
                    saved[split].append((torch.stack([r[0] for r in batch]),torch.stack([r[1] for r in batch])))
                    batch=[]
            else:
                saved[split].append((x[None],y[None]))
    if len(saved["train"])!=3 or len(saved["val"])!=2:
        raise ValueError("Smoke batch contract violated")
    path=output/"inputs.pt"
    torch.save(saved,path)
    identity={"input_file_sha256":sha256(path),"tensor_sha256":digest.hexdigest(),
              "manifest_sha256":sha256(manifest_path),"train_names":saved["train_names"],
              "val_names":saved["val_names"],"expanded_sampler_positions":positions,
              "full_validation":False,"validation_samples":2,"test_used":False,
              "data_workers_for_cached_tensor_generation":0}
    save_json(output/"input_identity.json",identity)
    return identity


def stitch_logits(model,image):
    if image.shape[0]!=1 or tuple(image.shape[-2:])!=(1024,2048):
        raise ValueError("Expected one full-resolution Cityscapes image")
    parts=[model(image[...,offset:offset+1024]) for offset in (0,1024)]
    logits=torch.cat(parts,dim=-1)
    return torch.nn.functional.interpolate(logits,size=(1024,2048),mode="bilinear",align_corners=True)


def scores(matrix):
    x=matrix.double()
    intersection=x.diag();union=x.sum(0)+x.sum(1)-intersection
    iou=intersection/union.clamp_min(1)
    return {"miou":float(iou.mean()),"pixel_accuracy":float(intersection.sum()/x.sum()),
            "class_iou":iou.tolist(),"valid_pixels":int(x.sum()),"metric_scale":"0..1",
            "miou_class_count":19,"absent_classes_have_zero_iou":True,
            "confusion_matrix":matrix.tolist()}
