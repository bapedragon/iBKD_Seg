"""Time full Cityscapes val: explicit initial-model v2 or verified checkpoint v1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
import traceback
import warnings
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[3] / 'phase4/Cityscapes_Segmenter-Ti16/configs'
CONFIG = CONFIG_DIR / 'val500_timing_v1.json'
INITIAL_CONFIG = CONFIG_DIR / 'val500_timing_initial_v2.json'
MARKER = '[CITYSCAPES_TI16_VAL500_TIMING_FINAL] '


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_config(path):
    config = json.loads(path.read_text())
    allowed = {p.name:p for p in (CONFIG, INITIAL_CONFIG)}
    if path.name not in allowed or config != json.loads(allowed[path.name].read_text()):
        raise ValueError('Use the committed full-val timing config')
    return config


def preflight(pointer, data_dir, config):
    """Stdlib-only checks run before pip installation; no alternate model fallback."""
    if not pointer.is_file():
        raise FileNotFoundError(f'500-step checkpoint pointer missing: {pointer}. Restore the lg_b1 run folder '
                                'or set CITYSCAPES_TI16_EVAL_RESUME to its relocated resume.json; no training was started.')
    run_dir = pointer.parent
    identity_path, summary_path = run_dir / 'identity.json', run_dir / 'summary.json'
    identity, summary = (json.loads(p.read_text()) for p in (identity_path, summary_path))
    training = json.loads((CONFIG_DIR / config['training_config']).read_text())
    plan = next(p for p in training['runs'] if p['id'] == config['run_id'])
    if identity['config_sha256'] != json_hash(training) or identity['plan'] != plan:
        raise ValueError('Checkpoint training config/candidate differs from the fixed LG beta1 run')
    if identity['source_sha256'] != config['training_source_sha256']:
        raise ValueError('Checkpoint training source differs from the verified grid commit')
    if (summary['status'] != 'passed' or summary['run_id'] != plan['id'] or
            summary['completed_steps'] != config['checkpoint_step'] or
            summary['selected_step'] != config['checkpoint_step'] or
            summary['selected_epoch'] != config['checkpoint_epoch']):
        raise ValueError('Require a passed 500-step endpoint; no fallback to earlier checkpoints')
    if not summary.get('final_student_sha256'):
        raise ValueError('Original final student state hash is missing')
    manifest_path = data_dir / 'manifest.json'
    if not manifest_path.is_file() or sha(manifest_path) != identity['manifest_sha256']:
        raise ValueError(f'Training manifest is missing or changed: {manifest_path}')
    manifest = json.loads(manifest_path.read_text())
    if {k: len(v) for k, v in manifest['splits'].items()} != {'train': 2975, 'val': 500}:
        raise ValueError('Require official train/val inventory without test')
    for component in ('leftImg8bit', 'gtFine'):
        if not (data_dir / component / 'val').is_dir():
            raise FileNotFoundError(f'Missing extracted val data: {data_dir / component / "val"}')
    index = json.loads(pointer.read_text())
    if index.get('format') != 1:
        raise ValueError('Unknown checkpoint pointer format')
    record = index['current']
    if record['global_step'] != 500 or record != summary['checkpoint']['current']:
        raise ValueError('Checkpoint pointer does not match the completed 500-step result')
    relative = Path(record['file'])
    if relative.is_absolute() or '..' in relative.parts or relative.parts[:1] != ('checkpoints',):
        raise ValueError('Unsafe checkpoint path')
    checkpoint = (run_dir / relative).resolve()
    if not checkpoint.is_relative_to(run_dir.resolve()):
        raise ValueError('Checkpoint path escapes its run folder')
    if checkpoint.stat().st_size != record['bytes'] or sha(checkpoint) != record['sha256']:
        raise ValueError('Checkpoint byte size/SHA-256 mismatch')
    return dict(pointer=str(pointer), checkpoint=str(checkpoint), bytes=record['bytes'], sha256=record['sha256'],
                identity=identity, source_summary=summary, training_config=training, manifest=manifest,
                manifest_path=str(manifest_path), signature_checks='passed')


def student_state(payload, source):
    """Accept model weights for evaluation only, never resume training across code versions."""
    if payload['signature'] != source['identity']:
        raise ValueError('Checkpoint payload signature differs from identity.json')
    progress = payload['progress']
    if (progress['global_step'] != 500 or progress['epoch'] != 2 or progress['next_batch'] != 128 or
            len(progress['rows']) != 500 or progress['rows'][-1]['step'] != 500):
        raise ValueError('Unexpected training endpoint inside checkpoint')
    if progress['rows'][-1] != source['source_summary']['last']:
        raise ValueError('Saved last training result differs from summary.json')
    return payload['model']


def evaluate(model, dataset, inference, config, synchronize, on_progress):
    import torch
    from .evaluation import confusion_update, metrics

    if len(dataset) != config['validation_samples']:
        raise ValueError('Validation subset is incomplete')
    matrix = torch.zeros(19, 19, dtype=torch.int64)
    timings, ids = [], []
    valid_pixels = 0
    synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        for index in range(len(dataset)):
            began = time.perf_counter()
            ims, metas, target, sample_id = dataset[index]
            if list(target.shape) != config['original_target_hw']:
                raise ValueError('Validation must use original-resolution ground truth')
            loaded = time.perf_counter()
            logits = inference(model, ims, metas, tuple(target.shape), config['window_size'],
                               config['window_stride'], batch_size=config['inference_window_batch_size'])
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite validation logits')
            prediction = logits.argmax(0).cpu()
            synchronize()
            inferred = time.perf_counter()
            confusion_update(matrix, prediction, target)
            valid_pixels += int((target != 255).sum())
            ids.append(sample_id)
            finished = time.perf_counter()
            timings.append(dict(id=sample_id, total_seconds=finished-began, load_seconds=loaded-began,
                                inference_seconds=inferred-loaded, metric_seconds=finished-inferred))
            del logits, prediction
            if (index + 1) % 25 == 0 or index == 0 or index + 1 == len(dataset):
                on_progress(index + 1, time.perf_counter() - started)
    synchronize()
    scores = metrics(matrix)
    wall_seconds = time.perf_counter() - started
    if len(set(ids)) != len(dataset) or scores['valid_pixels'] != valid_pixels:
        raise ValueError('Duplicate val IDs or incorrect void-pixel accounting')
    if scores['evaluated_classes'] != config['expected_evaluated_classes']:
        raise ValueError('Full validation did not cover all expected classes')
    return dict(metrics=scores, validation_seconds=wall_seconds, validation_samples=len(ids),
                images_per_second=len(ids)/wall_seconds, mean_image_seconds=wall_seconds/len(ids),
                median_image_seconds=statistics.median(r['total_seconds'] for r in timings),
                first_image_seconds=timings[0]['total_seconds'],
                input_loading_seconds=sum(r['load_seconds'] for r in timings),
                inference_and_transfer_seconds=sum(r['inference_seconds'] for r in timings),
                metric_accumulation_seconds=sum(r['metric_seconds'] for r in timings),
                validation_ids=ids, per_image_timings=timings)


def terminal_report(report):
    from .tiny_grid_report import bounded_text
    keys = ('status', 'protocol_id', 'phase', 'method', 'beta', 'lambda', 'selected_step', 'selected_epoch',
            'initialization', 'trained_checkpoint_loaded', 'pretrained_assets', 'initial_student_sha256',
            'checkpoint_selection', 'last_training_loss', 'last_training_ce', 'last_training_guidance',
            'checkpoint', 'student_state_unchanged', 'validation_samples', 'full_validation',
            'optimizer_updates', 'teacher_loaded', 'guidance_loaded', 'test_used', 'automatic_next_stage',
            'validation_seconds', 'images_per_second', 'mean_image_seconds', 'median_image_seconds',
            'first_image_seconds', 'input_loading_seconds', 'inference_and_transfer_seconds',
            'metric_accumulation_seconds', 'phase_seconds', 'invocation_seconds', 'total_job_seconds',
            'peak_allocated_bytes', 'environment', 'warning_count', 'deterministic_warning_count',
            'pipeline_exit_code', 'summary_path')
    result = {k: report.get(k) for k in keys}
    scores = report.get('metrics')
    result['metrics_percent'] = None if scores is None else dict(
        pixel_accuracy=100*scores['pixel_accuracy'], miou=100*scores['miou'],
        class_iou={k: None if v is None else 100*v for k, v in scores['class_iou'].items()},
        evaluated_classes=scores['evaluated_classes'], valid_pixels=scores['valid_pixels'])
    result['error'] = bounded_text(report.get('error'), 2048)
    result['timing_scope'] = 'one full val pass including first-image startup, input loading, inference and metrics; data audit warms OS cache'
    result['score_scope'] = report.get('score_scope', '500-step checkpoint full-val timing probe; not a 2000-step result or beta ranking')
    line = MARKER + json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(',', ':'))
    if len(line) > 50000:
        raise ValueError('Unexpected terminal report size')
    return line


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--zip-dir', type=Path, default=Path('/app/data/chaoyang'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--prepare-data-only', action='store_true')
    parser.add_argument('--failure-code', type=int)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / 'summary.json'
    started = time.perf_counter()
    report = dict(status='running', phase='preflight', optimizer_updates=0, teacher_loaded=False,
                  guidance_loaded=False, test_used=False, automatic_next_stage=False, full_validation=False,
                  validation_samples=0, phase_seconds={}, summary_path=str(summary_path))
    if args.config.name == INITIAL_CONFIG.name:
        initial_defaults = json.loads(INITIAL_CONFIG.read_text())
        report.update(protocol_id=initial_defaults['protocol_id'],
                      initialization=initial_defaults['initialization'], trained_checkpoint_loaded=False,
                      checkpoint_selection=initial_defaults['checkpoint_selection'],
                      score_scope=initial_defaults['score_scope'], method='initial_model_timing_only',
                      selected_step=0, selected_epoch=0)
    if args.failure_code is not None:
        if summary_path.is_file():
            report = json.loads(summary_path.read_text())
        report.update(status='failed', pipeline_exit_code=args.failure_code)
        report.setdefault('error', 'Pipeline setup failed; see run.log')
        began = float(os.environ.get('CITYSCAPES_VAL_TIMING_STARTED', time.time()))
        report['total_job_seconds'] = time.time() - began
        save_json(summary_path, report)
        line = terminal_report(report)
        (output / 'terminal_summary.log').write_text(line + '\n')
        print(line, flush=True)
        return
    failed = False
    try:
        config = load_config(args.config)
        report.update(protocol_id=config['protocol_id'], checkpoint_selection=config['checkpoint_selection'])
        initial = config.get('trained_checkpoint_loaded') is False
        if initial:
            from .tiny_val_initial import preflight_initial, prepare_initial_data
            report.update(initialization=config['initialization'], trained_checkpoint_loaded=False,
                          score_scope=config['score_scope'], method='initial_model_timing_only',
                          selected_step=0, selected_epoch=0)
            if args.resume is not None:
                raise ValueError('Initial-model timing does not accept --resume; use v1 for checkpoint evaluation')
            source = preflight_initial(args.data_dir.resolve(), args.zip_dir.resolve(), config)
            save_json(output / 'data_preflight.json', {k:v for k,v in source.items() if k != 'manifest'})
        else:
            if args.resume is None:
                raise ValueError('Checkpoint timing v1 requires --resume; select v2 explicitly for initial-model timing')
            source = preflight(args.resume.resolve(), args.data_dir.resolve(), config)
            save_json(output / 'checkpoint_preflight.json', {k:source[k] for k in
                      ('pointer', 'checkpoint', 'bytes', 'sha256', 'signature_checks')})
        report['phase_seconds']['preflight'] = time.perf_counter() - started
        if args.preflight_only:
            detail = f"checkpoint=none data_preparation_needed={source['needs_prepare']}" if initial else 'checkpoint=500 signature=passed bytes_sha256=passed data_manifest=passed'
            print('[TI16_VAL500_PREFLIGHT] ' + detail, flush=True)
            return
        if args.prepare_data_only:
            if not initial:
                raise ValueError('--prepare-data-only is available only for initial-model v2')
            began = time.perf_counter()
            report['phase'] = 'data_preparation'
            ready = prepare_initial_data(source, config, output)
            save_json(output / 'data_setup.json', dict(
                seconds=time.perf_counter()-began, extracted=source['needs_prepare'],
                manifest_sha256=ready['manifest_sha256'], data_dir=ready['data_dir']))
            print('[TI16_VAL500_DATA] ready train=2975 val=500', flush=True)
            return
        if initial and source['needs_prepare']:
            raise FileNotFoundError('Run --prepare-data-only before evaluation; the launcher does this automatically')
        from .official_api import bootstrap, student
        from .official_assets import verify
        report['phase'] = 'runtime_setup'
        began = time.perf_counter()
        provenance = verify(args.cache_root, student='tiny', include_teacher=not initial)
        save_json(output / 'asset_provenance.json', provenance)
        if not initial and provenance['weights'] != source['identity']['assets']:
            raise ValueError('Pinned upstream weights differ from the training assets')
        report['pretrained_assets'] = provenance['weights']
        if initial and (output / 'data_setup.json').is_file():
            report['data_setup'] = json.loads((output / 'data_setup.json').read_text())
            report['phase_seconds']['data_preparation'] = report['data_setup']['seconds']
        bootstrap(args.cache_root)
        import torch
        import segm.utils.torch as ptu
        from segm.model.utils import inference
        from .full_data import FullDataset
        from .data import verify_manifest
        from .real_smoke import validate_manifest
        from .runtime import seed_all, state_hash
        from .tiny_repeat import warning_summary
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'H200' not in torch.cuda.get_device_name():
            raise RuntimeError('Use exactly one H200 GPU, matching the training setup')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8')
        device = torch.device('cuda'); ptu.device = device
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(config['cpu_threads'])
        torch.use_deterministic_algorithms(True, warn_only=True)
        report['environment'] = dict(python=platform.python_version(), torch=str(torch.__version__),
                                     cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), precision='fp32',
                                     window_size=512, window_stride=512, window_batch_size=1, cpu_threads=4)
        report['phase_seconds']['runtime_setup'] = time.perf_counter()-began
        report['phase'] = 'val_data_audit'; began=time.perf_counter()
        validate_manifest(source['manifest'])
        verify_manifest(args.data_dir, dict(source['manifest'], splits={'val':source['manifest']['splits']['val']}))
        dataset = FullDataset(args.data_dir, source['manifest'], source['training_config'], 'val')
        report['phase_seconds']['val_data_audit'] = time.perf_counter()-began
        report['phase'] = 'student_load'; began=time.perf_counter()
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            try:
                if initial:
                    from .tiny_val_initial import initial_student
                    model, expected_hash = initial_student(args.cache_root, config, student, seed_all, state_hash)
                    report['initial_student_sha256'] = expected_hash
                else:
                    payload = torch.load(source['checkpoint'], map_location='cpu', weights_only=True)
                    weights = student_state(payload, source)
                    seed_all(source['training_config']['seed'])
                    model = student(args.cache_root, backbone='vit_tiny_patch16_384', image_size=512,
                                    decoder_layers=1, recompute=True)
                    model.load_state_dict(weights, strict=True)
                    del weights, payload
                    expected_hash = source['source_summary']['final_student_sha256']
                    if state_hash(model) != expected_hash:
                        raise ValueError('Loaded student differs from the recorded 500-step endpoint')
                    plan = source['identity']['plan']; last=source['source_summary']['last']
                    report.update(method=plan['method'], beta=plan['beta'], **{'lambda':None},
                                  trained_checkpoint_loaded=True,
                                  selected_step=500, selected_epoch=2, last_training_loss=last['loss'],
                                  last_training_ce=last['ce'], last_training_guidance=last['guidance'],
                                  checkpoint={k:source[k] for k in ('pointer','checkpoint','bytes','sha256')})
                model = model.to(device).eval().requires_grad_(False)
                report['phase_seconds']['student_load'] = time.perf_counter()-began
                report['phase'] = 'full_val500'
                torch.cuda.reset_peak_memory_stats()

                def progress(n, seconds):
                    report.update(validation_samples=n, partial_validation_seconds=seconds)
                    save_json(output / 'progress.json', report)
                    print(f'[TI16_VAL500_PROGRESS] samples={n}/500 elapsed_seconds={seconds:.3f} '
                          f'estimated_remaining_seconds={seconds/n*(500-n):.3f}', flush=True)

                result = evaluate(model, dataset, inference, config, torch.cuda.synchronize, progress)
                save_json(output / 'per_image_timings.json', result.pop('per_image_timings'))
                save_json(output / 'validation_ids.json', result.pop('validation_ids'))
                report.update(result, full_validation=True, peak_allocated_bytes=torch.cuda.max_memory_allocated())
                report['student_state_unchanged'] = state_hash(model) == expected_hash
                if not report['student_state_unchanged'] or any(p.grad is not None for p in model.parameters()):
                    raise RuntimeError('Evaluation modified student weights or produced gradients')
                if not initial and sha(Path(source['checkpoint'])) != source['sha256']:
                    raise RuntimeError('Source checkpoint changed during evaluation')
                report.update(status='passed', phase='complete')
            finally:
                warning_rows = warning_summary(records)
                save_json(output / 'warnings.json', warning_rows)
                report['warning_count'] = sum(w['count'] for w in warning_rows)
                report['deterministic_warning_count'] = sum('deterministic' in str(w.message) for w in records)
    except Exception as error:
        failed=True
        report.update(status='failed', error=repr(error))
        (output / 'traceback.txt').write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        if failed or not (args.preflight_only or args.prepare_data_only):
            report['invocation_seconds'] = time.perf_counter()-started
            began = float(os.environ.get('CITYSCAPES_VAL_TIMING_STARTED', time.time()-report['invocation_seconds']))
            report['total_job_seconds'] = time.time()-began
            save_json(summary_path, report)
            line=terminal_report(report)
            (output / 'terminal_summary.log').write_text(line+'\n')
            print(line, flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
