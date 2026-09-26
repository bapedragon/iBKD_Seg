"""Data/initial-model setup for timing only; never access previous training outputs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .tiny_val_timing import save_json, sha

# User archives audited locally and on H200; these are identity hashes, not published official SHA values.
ARCHIVES = {
    'leftImg8bit_trainvaltest.zip': {
        'bytes': 11598567370, 'sha256': '1927eb6450e29ebde0d611d30b874abe96747f5d092f6ec0cbbb6a3e1f6ce09d'},
    'gtFine_trainvaltest.zip': {
        'bytes': 263041307, 'sha256': 'dbacde05ea136f3036aa24bfc87b5272f0d999e34380e10cbb919f7368448332'},
}


def preflight_initial(data_dir, zip_dir, config):
    """Stdlib preflight: either existing data or the two original archives must be available."""
    if data_dir.name != 'cityscapes':
        raise ValueError('Data directory must be named cityscapes for the upstream loader')
    manifest_path = data_dir / 'manifest.json'
    source = dict(training_config=config['dataset_config'], manifest_path=str(manifest_path),
                  data_dir=str(data_dir), zip_dir=str(zip_dir), needs_prepare=not manifest_path.is_file())
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if {k:len(v) for k,v in manifest['splits'].items()} != {'train':2975, 'val':500}:
            raise ValueError('Require official fine train2975/val500 manifest without test')
        for component in ('leftImg8bit', 'gtFine'):
            if not (data_dir / component / 'val').is_dir():
                raise FileNotFoundError(f'Missing extracted val data: {data_dir / component / "val"}')
        source.update(manifest=manifest, manifest_sha256=sha(manifest_path))
    else:
        for name, expected in ARCHIVES.items():
            path = zip_dir / name
            if not path.is_file():
                raise FileNotFoundError(f'No extracted manifest at {manifest_path}; required data ZIP missing: {path}')
            if path.stat().st_size != expected['bytes']:
                raise ValueError(f'Cityscapes ZIP byte size mismatch: {path}')
    return source


def prepare_initial_data(source, config, output):
    if source['needs_prepare']:
        records = {}
        for name, expected in ARCHIVES.items():
            path = Path(source['zip_dir']) / name
            print(f'[TI16_VAL500_DATA] verify_zip={name}', flush=True)
            actual = dict(bytes=path.stat().st_size, sha256=sha(path))
            if actual != expected:
                raise ValueError(f'Cityscapes ZIP byte size/SHA-256 mismatch: {path}')
            records[name] = actual
        save_json(output / 'source_archives.json', records)
        # Existing extractor rejects conflicting files, checks CRC, keeps train/val only and audits PNGs.
        subprocess.run([sys.executable, '-u', '-m', 'ibkd_seg.cityscapes.prepare',
                        '--zip-dir', source['zip_dir'], '--data-dir', source['data_dir']], check=True)
    ready = preflight_initial(Path(source['data_dir']), Path(source['zip_dir']), config)
    if ready['needs_prepare']:
        raise RuntimeError('Data preparation did not produce a manifest')
    return ready


def initial_student(root, config, factory, seed_all, state_hash):
    """No Cityscapes checkpoint loader or optimizer is involved in this path."""
    seed_all(config['seed'])
    model = factory(root, backbone=config['student_backbone'], image_size=config['window_size'],
                    decoder_layers=config['decoder_layers'], recompute=True)
    model = model.eval().requires_grad_(False)
    return model, state_hash(model)
