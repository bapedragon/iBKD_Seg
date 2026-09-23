"""Shared 25-batch training-only input for B0 beta calibration."""
import hashlib
import json
from pathlib import Path

import torch

from .assets import modules
from .data import sample_tensors
from ..data import save_json,sha256,verify_manifest
from ..real_smoke import validate_manifest
from ..runtime import seed_all


def verify_preparation(root,expected_archives):
    manifest_path=root/"manifest.json"
    provenance=json.loads((root/"preparation.json").read_text())
    if provenance.get("status")!="passed" or provenance.get("manifest_sha256")!=sha256(manifest_path):
        raise ValueError("Prepared data provenance does not match its manifest")
    archives={r["name"]:r for r in provenance["archives"]}
    if set(archives)!=set(expected_archives):raise ValueError("Unexpected source archives")
    for name,expected in expected_archives.items():
        if any(archives[name][key]!=expected[key] for key in ("bytes","sha256")):
            raise ValueError(f"Prepared archive identity changed: {name}")
    manifest=json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    verify_manifest(root,manifest)
    return {"status":"passed","manifest_sha256":sha256(manifest_path),
            "preparation_sha256":sha256(root/"preparation.json"),"all_prepared_file_hashes_verified":True}


def sample_positions(count):
    return torch.randperm(80000*16,generator=torch.Generator().manual_seed(1))[:count].tolist()


def prepare_inputs(cache,root,output,*,batches=25,batch_size=16):
    _,_,upstream=modules(cache)
    train=upstream.CSTrainValSet(str(root),str(cache/"cirkd/dataset/list/cityscapes/train.lst"),crop_size=(512,512))
    if len(train)!=2975:raise ValueError("Unexpected training split")
    positions=sample_positions(batches*batch_size)
    seed_all(1)
    folder=output/"inputs";folder.mkdir(parents=True,exist_ok=False)
    records=[];tensor_hash=hashlib.sha256()
    for index in range(batches):
        samples=[sample_tensors(train[p%2975]) for p in positions[index*batch_size:(index+1)*batch_size]]
        x=torch.stack([r[0] for r in samples]);y=torch.stack([r[1] for r in samples])
        if tuple(x.shape)!=(16,3,512,512) or tuple(y.shape)!=(16,512,512):
            raise ValueError("Calibration must use actual batch16/crop512")
        names=[r[2] for r in samples]
        for name in names:tensor_hash.update(name.encode())
        for value in (x,y):tensor_hash.update(value.contiguous().numpy().tobytes())
        path=folder/f"batch_{index:02d}.pt"
        torch.save({"image":x,"label":y},path)
        records.append({"path":path.name,"bytes":path.stat().st_size,"sha256":sha256(path),"names":names})
        print(f"[B0_CALIBRATION_INPUT] batch={index+1}/{batches}",flush=True)
    identity={"batches":batches,"batch_size":batch_size,"crop_hw":[512,512],
              "tensor_sha256":tensor_hash.hexdigest(),"expanded_sampler_positions":positions,
              "manifest_sha256":sha256(root/"manifest.json"),"batch_files":records,
              "validation_samples":0,"test_used":False,"workers":0,"seed":1}
    save_json(folder/"identity.json",identity)
    return identity


def load_batches(folder):
    identity=json.loads((folder/"identity.json").read_text())
    if identity["batches"]!=25 or identity["batch_size"]!=16 or len(identity["batch_files"])!=25:
        raise ValueError("Expected exactly 25 batches of 16 training images")
    for record in identity["batch_files"]:
        relative=Path(record["path"])
        if relative.is_absolute() or len(relative.parts)!=1:raise ValueError("Invalid batch filename")
        path=folder/relative
        if path.stat().st_size!=record["bytes"] or sha256(path)!=record["sha256"]:
            raise ValueError(f"Calibration input changed: {path}")
        batch=torch.load(path,map_location="cpu",weights_only=True)
        x,y=batch["image"],batch["label"]
        if x.shape!=(16,3,512,512) or y.shape!=(16,512,512):raise ValueError("Batch shape changed")
        yield x,y
