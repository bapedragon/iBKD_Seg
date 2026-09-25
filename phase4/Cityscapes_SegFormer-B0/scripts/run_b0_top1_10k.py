#!/usr/bin/env python3
"""ALG·iBKD λ=0.25의 2k 1위 후보만 각각 총 10k까지 실행한다."""
import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PLAN = REPO / 'phase4/Cityscapes_SegFormer-B0/configs/b0_top2_10k_v1.json'
MAX_JOB_SECONDS = 9 * 3600 + 40 * 60


def inventory_for(plan, profiles):
    rows = []
    for method in plan['execution_order']:
        spec = plan['retained'][method]
        module, config, grid = profiles[spec['profile']]
        candidates = module.candidates(config, grid, spec['pack'])
        retained = []
        for wanted in spec['candidates']:
            row = next(r for r in candidates if r['run_id'] == wanted['run_id'])
            if row['method'] != method or row['beta'] != wanted['beta']:
                raise ValueError('Selected candidate differs from frozen grid')
            retained.append(row)
        rows.append(dict(candidate=retained[0], profile=spec['profile'],
                         guidance_warmup_epochs=spec['guidance_warmup_epochs']))
    return rows


def validate_resume(source, row, module, config):
    source = Path(source)
    if not (source / 'resume.json').is_file():
        raise ValueError(f'Missing full-state resume.json: {source}')
    summary = json.loads((source / 'summary.json').read_text())
    candidate = row['candidate']
    signature = summary.get('signature', {})
    if any(summary.get(k) != v for k, v in candidate.items()):
        raise ValueError('Resume candidate differs')
    if (signature.get('candidate') != candidate or signature.get('protocol_id') != config['id']
            or signature.get('source') != module.code_identity()
            or signature.get('protocol_sha256') != hashlib.sha256(module.CONFIG.read_bytes()).hexdigest()):
        raise ValueError('Resume protocol/code/candidate signature differs')
    if 'diagnostic_scope' in signature or summary.get('full_validation') is False:
        raise ValueError('Diagnostic smoke cannot resume into full validation')
    warmup = summary.get('controller', {}).get('controller', {}).get('warmup_epochs')
    if warmup != row['guidance_warmup_epochs']:
        raise ValueError('Resume guidance warm-up differs')
    if not 0 <= summary['completed_steps'] <= 10000:
        raise ValueError('Resume exceeds 10k target')
    return summary


def discover_resume(row, module, config, roots):
    """Find extracted, compatible run bundles; never treat metrics as a checkpoint."""
    run_id = row['candidate']['run_id']
    found = set()
    pruned = {'checkpoints', 'weights', 'assets', 'cityscapes', 'leftImg8bit', 'gtFine',
              'gtCoarse', '.git', '__pycache__', 'node_modules', 'venv', '.venv'}
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            continue
        if root.name == run_id:
            found.add(root)
            continue
        for parent, directories, _ in os.walk(root, followlinks=False):
            parent = Path(parent)
            for name in directories:
                if name == run_id:
                    found.add((parent / name).resolve())
            depth = len(parent.relative_to(root).parts)
            directories[:] = [d for d in directories if d not in pruned and d != run_id] if depth < 8 else []
    eligible, ignored = [], []
    for source in sorted(found):
        summary_file = source / 'summary.json'
        if not summary_file.is_file():
            raise ValueError(f'Incomplete candidate folder (missing summary.json): {source}')
        summary = json.loads(summary_file.read_text())
        signature = summary.get('signature', {})
        warmup = summary.get('controller', {}).get('controller', {}).get('warmup_epochs')
        if (signature.get('protocol_id') != config['id'] or warmup != row['guidance_warmup_epochs']
                or 'diagnostic_scope' in signature or summary.get('full_validation') is False
                or summary.get('completed_steps', 0) > 10000):
            ignored.append(dict(path=str(source), reason='Different protocol/warmup, diagnostic smoke, or beyond 10k'))
            continue
        validate_resume(source, row, module, config)
        pointer = json.loads((source / 'resume.json').read_text())
        step = pointer.get('current', {}).get('global_step')
        if pointer.get('format') != 1 or not isinstance(step, int) or not 0 <= step <= 10000:
            raise ValueError(f'Invalid full-state checkpoint pointer: {source}')
        eligible.append((step, source))
    # Prefer the furthest matching full checkpoint; lexical path breaks ties.
    eligible.sort(key=lambda item: (-item[0], str(item[1])))
    selected = eligible[0][1] if eligible else None
    return selected, dict(search_roots=[str(p) for p in roots],
                          compatible=[dict(path=str(p), checkpoint_step=s) for s, p in eligible],
                          ignored=ignored, selected=None if selected is None else str(selected))


def worker_command(row, cache, data, output, plan_path, preflight, deadline, resume):
    command = [sys.executable, '-m', 'ibkd_seg.cityscapes.' + row['profile'] + '.screen',
               '--cache', str(cache), '--data', str(data), '--output', str(output),
               '--plan', str(plan_path), '--preflight', str(preflight),
               '--run-id', row['candidate']['run_id'], '--deadline', str(deadline),
               '--target-steps', '10000']
    if resume is not None:
        command.extend(['--resume', str(resume)])
    return command


def terminal_summary(report):
    result = {k: v for k, v in report.items() if k not in ('protocols', 'code', 'assets', 'runs')}
    result['runs'] = [{k: v for k, v in row.items()
                       if k not in ('validation_history', 'signature', 'first_25_batch_hashes', 'best')}
                      for row in report['runs']]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', choices=('auto', 'fresh', 'resume'), default='auto')
    args = parser.parse_args()
    os.chdir(REPO)
    output = Path(os.environ.get('B0_10K_OUTPUT', '/app/output/cityscapes_b0_top1_10k_v1')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = started + MAX_JOB_SECONDS - 180
    report = dict(status='running', job='rank1', runs=[], selection={}, last_loss=None,
                  selected_epoch=None, metrics=None, test_used=False, target_steps_per_run=10000,
                  output=str(output), start_mode=args.start,
                  job_runtime=dict(id='b0_job_runtime_9h40_v1', maximum_job_seconds=MAX_JOB_SECONDS,
                                   shutdown_reserve_seconds=180, shared_across_both_runs=True))
    can_write = False
    def save():
        if not can_write:
            return
        (output / 'group_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    def record_run(record):
        report['runs'] = [r for r in report['runs'] if r['run_id'] != record['run_id']] + [record]
    def stage(name):
        report['stage'] = name
        save()
        print('[B0_TOP1_10K] ' + json.dumps(dict(stage=name)), flush=True)
    def run(command, check=True):
        remaining = deadline + 90 - time.time()
        if remaining <= 0:
            raise TimeoutError('Shared job time budget exhausted')
        return subprocess.run(command, check=check, timeout=remaining, env=os.environ.copy()).returncode
    try:
        group_value = os.environ.get('B0_10K_RESUME_FROM')
        group = Path(group_value).resolve() if group_value else None
        individual = dict(alg=os.environ.get('B0_10K_ALG_RESUME_FROM'),
                          ibkd_lambda025=os.environ.get('B0_10K_IBKD_RESUME_FROM'))
        if args.start == 'fresh' and (group is not None or any(individual.values())):
            raise ValueError('Resume paths require --start auto or resume')
        if (args.start == 'auto' and group is None and not any(individual.values())
                and (output / 'group_summary.json').is_file()):
            group = output
        if args.start == 'resume' and group is None and not all(individual.values()):
            raise ValueError('Provide a previous combined group OR both candidate run directories')
        if group is not None and any(individual.values()):
            raise ValueError('Use combined group OR individual resume paths, not both')
        # Read previous metadata before overwriting an in-place group summary.
        previous = None if group is None else json.loads((group / 'group_summary.json').read_text())
        if any(p.name != 'run.log' for p in output.iterdir()) and group != output:
            raise ValueError('Nonempty output requires in-place B0_10K_RESUME_FROM')
        plan = json.loads(PLAN.read_text())
        if previous is not None and (previous.get('plan') != plan or previous.get('job') != 'rank1'):
            raise ValueError('Resume job plan differs')
        if previous is not None:
            report['runs'] = previous['runs']
        report.update(plan=plan, resume_from=None if group is None else str(group))
        can_write = True
        os.environ.update(PYTHONPATH=str(REPO / 'src'), PYTHONUNBUFFERED='1', PYTHONHASHSEED='1',
                          CUBLAS_WORKSPACE_CONFIG=':4096:8', MAX_JOBS='2', TORCH_CUDA_ARCH_LIST='9.0')
        stage('install')
        run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '-e', '.',
             'opencv-python-headless==4.13.0.92', 'gdown==5.2.0', 'pytest==8.4.2'])
        with (output / 'pip_freeze.txt').open('w') as f:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
        sys.path.insert(0, str(REPO / 'src'))
        import numpy as np
        from ibkd_seg.cityscapes.b0.assets import prepare as prepare_assets
        from ibkd_seg.cityscapes.b0.calibration_data import verify_preparation
        from ibkd_seg.cityscapes.b0.training_data import make_plan, plan_hash, PlannedDataset, batches, calibration_digest_update
        from ibkd_seg.cityscapes.b0.reproducibility import configure_runtime
        from ibkd_seg.cityscapes.data import save_json
        profiles = {}
        for name in ('b0', 'b0_warmup20'):
            module = importlib.import_module('ibkd_seg.cityscapes.' + name + '.screen')
            config, grid = module.specification()
            profiles[name] = (module, config, grid)
        configure_runtime()
        inventory = inventory_for(plan, profiles)
        report.update(expected_runs=inventory, protocols={n: p[1] for n, p in profiles.items()},
                      code={n: p[0].code_identity() for n, p in profiles.items()},
                      git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
        if previous is None:
            report['runs'] = [dict(**r['candidate'], status='pending', completed_steps=0,
                                  last_loss=None, selected_epoch=None, metrics=None) for r in inventory]
        for method, spec in plan['retained'].items():
            report['selection'][method] = dict(status='pending', selected_run_ids=[],
                retained_run_ids=[r['run_id'] for r in spec['candidates']],
                reason='Final beta selection waits for BOTH retained candidates at 10k; this job runs rank1 only')
        # Copy only these two full-state bundles; no silent restart for an invalid resume.
        stage('resume_preflight')
        for row in inventory:
            candidate = row['candidate']; target = output / 'runs' / candidate['run_id']
            source = None
            if group is not None:
                source = group / 'runs' / candidate['run_id']
                prior = next(r for r in previous['runs'] if r['run_id'] == candidate['run_id'])
                if not (source / 'resume.json').exists():
                    if prior.get('completed_steps', 0) != 0 or prior.get('status') != 'pending':
                        raise ValueError(f'Missing checkpoint for previously started candidate: {source}')
                    source = None
            elif args.start == 'resume':
                source = Path(individual[candidate['method']]).resolve()
            elif args.start == 'auto':
                explicit = individual[candidate['method']]
                if explicit:
                    source = Path(explicit).resolve()
                else:
                    roots = [Path(p) for p in os.environ.get('B0_10K_SEARCH_ROOTS', '/app/output:/app/data').split(os.pathsep) if p]
                    module, config, _ = profiles[row['profile']]
                    source, discovery = discover_resume(row, module, config, roots)
                    row['resume_discovery'] = discovery
            if source is not None:
                module, config, _ = profiles[row['profile']]
                row['resume_source_summary'] = validate_resume(source, row, module, config)
                row['resume_source'] = str(source)
                if source != target:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(source, target)
            row['starts_from'] = 'full_checkpoint' if source is not None else 'initialization'
            print('[B0_START] ' + json.dumps(dict(run_id=candidate['run_id'], starts_from=row['starts_from'],
                  source=None if source is None else str(source),
                  recorded_steps=None if source is None else row['resume_source_summary']['completed_steps'])), flush=True)
        # The shared input contract is identical in both frozen profiles.
        _, config, grid = profiles['b0']
        cache = Path(os.environ.get('B0_10K_CACHE', '/app/scratch/cityscapes_b0_smoke_v1/assets')).resolve()
        data = Path(os.environ.get('B0_10K_DATA', '/app/scratch/cityscapes_b0_smoke_v1/cityscapes')).resolve()
        zip_root = Path(os.environ.get('CITYSCAPES_ZIP_DIR', '/app/data/chaoyang')).resolve()
        stage('unit_checks')
        run([sys.executable,'-m','pytest','-q','tests/test_cityscapes_b0_training.py','tests/test_cityscapes_b0_calibration.py','tests/test_cityscapes_b0_warmup20.py','tests/test_cityscapes_b0_top1_10k.py'])
        stage('data')
        metadata=[(data/name).exists() for name in ('manifest.json','preparation.json')]
        if any(metadata) and not all(metadata):raise ValueError('Incomplete data cache provenance')
        if not all(metadata):
            run([sys.executable,str(REPO/'phase4/phase4_cityscapes/scripts/check_cityscapes_upload.py'),
                 '--search-root',str(zip_root),'--output',str(output/'zip_audit.json')])
            links=output/'zip_links';links.mkdir(exist_ok=True)
            for record in json.loads((output/'zip_audit.json').read_text())['files']:
                source=Path(record['path']).resolve();target=links/source.name
                if target.is_symlink() or target.exists():
                    if target.resolve()!=source:raise ValueError('ZIP link differs')
                else:target.symlink_to(source)
            run([sys.executable,'-m','ibkd_seg.cityscapes.prepare','--zip-dir',str(links),'--data-dir',str(data)])
        report['data_verification']=verify_preparation(data,config['expected_archives'])
        if report['data_verification']['manifest_sha256']!=grid['manifest_sha256']:raise ValueError('Data differs from beta calibration')
        stage('assets')
        report['assets']=prepare_assets(cache,weight_names=('cirkd_teacher.pth','nvidia_mit_b0.bin'),manifest_name='top1_10k_asset_manifest.json')
        stage('input_preflight')
        plan=make_plan(1280000);plan_path=output/'augmentation_plan.npy'
        with plan_path.with_suffix('.tmp').open('wb') as f:np.save(f,plan,allow_pickle=False)
        plan_path.with_suffix('.tmp').replace(plan_path)
        dataset=PlannedDataset(data,cache/'cirkd/dataset/list/cityscapes/train.lst',plan)
        digest=hashlib.sha256();checked_digest=hashlib.sha256();ignored=[];valid_counts=[]
        checked_batches=config['input']['preflight_batches']
        for index,(x,y,names) in enumerate(batches(dataset,0,checked_batches*16,16,workers=4),1):
            if index<=25:calibration_digest_update(digest,x,y,names)
            calibration_digest_update(checked_digest,x,y,names)
            valid=y!=-1;valid_counts.append(int(valid.sum()))
            if not valid.any():raise ValueError(f'No valid labels in preflight batch {index}: {names}')
            for sample in (~valid.flatten(1).any(1)).nonzero().flatten().tolist():
                ignored.append(dict(batch=index,sample_in_batch=sample+1,name=names[sample]))
            if index%5==0 or index==checked_batches:print(f'[B0_INPUT_CHECK] {index}/{checked_batches}',flush=True)
        if digest.hexdigest()!=grid['calibration_tensor_sha256']:raise ValueError('Training transforms differ from measured calibration inputs')
        preflight=dict(status='passed',plan_sha256=plan_hash(plan),calibration_tensor_sha256=digest.hexdigest(),
                       manifest_sha256=grid['manifest_sha256'],samples=checked_batches*16,optimizer_updates=0,
                       calibration_samples=400,checked_batches=checked_batches,checked_tensor_sha256=checked_digest.hexdigest(),
                       valid_pixels_per_batch=valid_counts,ignore_only_samples=ignored)
        save_json(output/'input_preflight.json',preflight);report['input_preflight']=preflight
        print('[B0_INPUT_PREFLIGHT] '+json.dumps(preflight),flush=True)
        del plan,dataset,x,y
        for row in inventory:
            candidate = row['candidate']; run_id = candidate['run_id']; target = output / 'runs' / run_id
            pointer = target / 'resume.json' if row['starts_from'] == 'full_checkpoint' else None
            if time.time() > deadline - 900:
                prior = row.get('resume_source_summary', dict(**candidate, completed_steps=0, last_loss=None,
                                                             metrics=None, selected_epoch=None))
                prior = {**prior, 'status': 'pending', 'reason': 'Insufficient shared job time; resume this group',
                         'target_steps': 10000, 'resume_available': pointer is not None}
                record_run(prior)
                continue
            stage(run_id)
            command = worker_command(row, cache, data, target, plan_path,
                                     output / 'input_preflight.json', deadline, pointer)
            code = run(command, check=False)
            record = (json.loads((target / 'summary.json').read_text()) if (target / 'summary.json').exists()
                      else dict(status='failed', **candidate, last_loss=None, metrics=None, selected_epoch=None,
                                error='Worker exited without summary'))
            if code:
                record.update(status='failed', exit_code=code)
            record['starts_from'] = row['starts_from']
            record['guidance_warmup_epochs'] = row['guidance_warmup_epochs']
            record_run(record)
            save()
        complete = [r for r in report['runs'] if r['status'] == 'completed']
        if len(complete) == 2:
            if len({json.dumps(r['first_25_batch_hashes']) for r in complete}) != 1:
                raise ValueError('Candidates used different initial training inputs')
            if len({json.dumps(r['signature']['environment'], sort_keys=True) for r in complete}) != 1:
                raise ValueError('Candidate runtime environments differ')
        report['status'] = ('failed' if any(r['status'] == 'failed' for r in report['runs'])
                            else 'completed' if len(complete) == 2 else 'paused')
        report['completed_runs'] = len(complete)
        save_json(output / 'selection.json', report['selection'])
    except Exception as error:
        import traceback
        traceback.print_exc()
        report.update(status='failed', error=repr(error), failed_stage=report.get('stage'))
    report['elapsed_seconds'] = time.time() - started
    # Avoid repeating the full imported summaries in expected_runs and the final log.
    for row in report.get('expected_runs', []):
        row.pop('resume_source_summary', None)
    save()
    print(json.dumps(terminal_summary(report), ensure_ascii=False), flush=True)
    raise SystemExit(1 if report['status'] == 'failed' else 0)


if __name__ == '__main__':
    main()
