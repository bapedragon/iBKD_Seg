"""Fixed-beta Tiny screening with epoch-correct sampling and resumable state."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[3] / 'phase4/Cityscapes_Segmenter-Ti16/configs'
BASE_BETA = {'lg': 0.018658411532808426, 'alg': 0.018658411532808426,
             'ibkd_025': 0.04923668244316936, 'ibkd_050': 0.07336694459440872}
GRID_V5 = 'beta_grid500_both_lambdas_v5.json'
GRID_V6 = 'beta_grid500_shared_lg_8betas_v6.json'
HIGH_BETA100 = 'high_beta100_v1.json'
CLASSIFICATION_RATIOS = {'lg': 21.848531468328726, 'ibkd_l025': 21.761244776056202,
                         'ibkd_l050': 21.84707408786154}


def validate_high_beta(config):
    old = load_config(CONFIG_DIR / GRID_V6)
    changed = {'protocol_id', 'run_kind', 'steps', 'runs', 'beta_multipliers',
               'beta_status', 'checkpoint_every_steps'}
    for key, value in old.items():
        if key not in changed and config.get(key) != value:
            raise ValueError(f'High-beta common protocol drift: {key}')
    expected = []
    for method, lam, prefix, base_key in [('lg', None, 'lg', 'lg'),
            ('ibkd', .25, 'ibkd_l025', 'ibkd_025'), ('ibkd', .5, 'ibkd_l050', 'ibkd_050')]:
        base = BASE_BETA[base_key]
        matched = CLASSIFICATION_RATIOS[prefix] / .03 * base
        for i, (suffix, beta, reference) in enumerate([
                ('b2p5', 2.5, 'original_classification_numeric_beta'),
                ('ratio', matched, 'matched_initial_scalar_loss_ratio')], 1):
            row = dict(id=f'{prefix}_{suffix}', method=method, candidate=i, beta=beta,
                       initial_target_ratio=beta / base * .03, beta_reference=reference)
            if lam is not None:
                row['lambda'] = lam
            expected.append(row)
    if config['runs'] != expected or config['steps'] != 100 or config['beta_multipliers'] != []:
        raise ValueError('High-beta probe must use the six authorized 100-step conditions')
    if (config['job_budget_seconds'], config['save_reserve_seconds']) != (36000, 120):
        raise ValueError('High-beta probe must stop two minutes before ten hours')
    if not config['check_state_each_step'] or not config['record_failed_step_details']:
        raise ValueError('High-beta probe requires per-update finite-state and failure checks')
    if config['classification_reference']['ratios'] != CLASSIFICATION_RATIOS:
        raise ValueError('Classification calibration reference changed')
    return config


def diagnostic_scalar(value):
    value = float(value)
    return value if math.isfinite(value) else 'NaN' if math.isnan(value) else '+Inf' if value > 0 else '-Inf'


def load_config(path):
    config = json.loads(path.read_text())
    locked = CONFIG_DIR / path.name
    if path.name not in (GRID_V5, GRID_V6, HIGH_BETA100) or config != json.loads(locked.read_text()):
        raise ValueError('Use a committed Tiny beta-grid config')
    if path.name == HIGH_BETA100:
        return validate_high_beta(config)
    shared = path.name == GRID_V6
    multipliers = (.5, 1, 1.5, 2, 3, 4, 6, 8) if shared else (1, 2, 4, 8)
    if config['steps'] != 500 or len(config['runs']) != (24 if shared else 16):
        raise ValueError('Unexpected number of fixed 500-step conditions')
    old = json.loads((CONFIG_DIR / 'smoke25_v1.json').read_text())
    changed = {'protocol_id', 'run_kind', 'steps', 'runs', 'calibration_batches', 'beta_status', 'beta_multipliers'}
    for key, value in old.items():
        if key not in changed and config.get(key) != value:
            raise ValueError(f'Common Tiny protocol drift: {key}')
    if config['ibkd_lambdas'] != [.25, .5]:
        raise ValueError('Both iBKD lambdas are required by the user protocol')
    expected = []
    for method, ratio, prefix, base_key in (
            ('lg', None, 'lg', 'lg'), ('alg', None, 'alg', 'alg'),
            ('ibkd', .25, 'ibkd_l025', 'ibkd_025'), ('ibkd', .5, 'ibkd_l050', 'ibkd_050')):
        if shared and method == 'alg':
            continue
        for index, multiplier in enumerate(multipliers, 1):
            row = dict(id=f'{prefix}_b{index}', method=method, candidate=index,
                       beta=BASE_BETA[base_key] * multiplier, initial_target_ratio=.03 * multiplier)
            if method == 'ibkd':
                row['lambda'] = ratio
            expected.append(row)
    if config['runs'] != expected or config['calibration_batches'] != 0 or config['beta_multipliers'] != list(multipliers):
        raise ValueError('Candidate values or fixed-beta policy changed')
    if shared and (not config.get('lg_alg_shared_screen') or config['alg_earliest_off_step'] != 745):
        raise ValueError('LG/ALG shared screen must end before any possible ALG shutdown')
    return config


def initial_progress():
    return dict(global_step=0, epoch=1, next_batch=0, beta=None, samples=0,
                sums=dict(loss=0., ce=0., guidance=0.), best=None,
                completed_epochs=[], rows=[], input_hashes=[])


def record_step(progress, row, count, digest, controller, *, train_samples):
    """Observe only complete epochs; preserve the partial epoch for continuation."""
    progress['global_step'] += 1
    progress['next_batch'] += 1
    progress['samples'] += count
    row.update(step=progress['global_step'], epoch=progress['epoch'], beta=progress['beta'], batch_samples=count)
    progress['rows'].append(row)
    progress['input_hashes'].append(digest)
    for key in progress['sums']:
        progress['sums'][key] += row[key] * count
    if progress['samples'] > train_samples:
        raise RuntimeError('Epoch duplicated samples')
    if progress['samples'] == train_samples:
        means = {k: v / train_samples for k, v in progress['sums'].items()}
        controller.observe(progress['epoch'], means['guidance'], beta_used=progress['beta'])
        progress['completed_epochs'].append(dict(epoch=progress['epoch'], end_step=progress['global_step'],
                                                 samples=train_samples, beta=progress['beta'], **means))
        progress.update(epoch=progress['epoch'] + 1, next_batch=0, beta=None, samples=0,
                        sums=dict(loss=0., ce=0., guidance=0.))
        return True
    return False


def compare_grid(rows, config):
    by_id = {row['run_id']: row for row in rows}
    controls, issues = {}, []
    controls['all_planned_runs_present_once'] = (len(rows) == len(config['runs']) and
        len(by_id) == len(rows) and set(by_id) == {p['id'] for p in config['runs']})
    if not controls['all_planned_runs_present_once']:
        issues.append('run_inventory')
    available = [r for r in rows if r.get('student_initial_state_sha256')]
    for key in ('student_initial_state_sha256', 'teacher_state_sha256'):
        controls[key] = len(available) == len(rows) and len({r.get(key) for r in available}) == 1
        if not controls[key]:
            issues.append(key)
    for family, methods in (('lg_alg', ('lg', 'alg')), ('ibkd', ('ibkd',))):
        group = [r for r in rows if r.get('method') in methods]
        name = family + '_same_adapter_initialization'
        controls[name] = bool(group) and all(r.get('guidance_initial_state_sha256') for r in group) and len(
            {r.get('guidance_initial_state_sha256') for r in group}) == 1
        if not controls[name]:
            issues.append(name)
    # Compare every observed input prefix, including a candidate stopped early.
    reference = max(rows, key=lambda r: len(r.get('input_hashes', [])), default={}).get('input_hashes', [])
    controls['same_observed_input_prefixes'] = bool(reference) and all(
        r.get('input_hashes', []) == reference[:len(r.get('input_hashes', []))] for r in rows)
    if not controls['same_observed_input_prefixes']:
        issues.append('input_prefixes')
    pairs = []
    alg_present = any(p['method'] == 'alg' for p in config['runs'])
    for candidate in range(1, len(config['beta_multipliers']) + 1) if alg_present else ():
        a, b = by_id.get(f'lg_b{candidate}', {}), by_id.get(f'alg_b{candidate}', {})
        first = None
        for x, y in zip(a.get('losses', []), b.get('losses', [])):
            for key in ('loss', 'ce', 'guidance', 'grad_norm_unclipped'):
                if not math.isclose(x[key], y[key], rel_tol=config['resume_rtol'], abs_tol=config['resume_atol']):
                    first = dict(step=x['step'], key=key, lg=x[key], alg=y[key], absolute_difference=abs(x[key]-y[key]))
                    break
            if first:
                break
        complete = a.get('completed_steps') == b.get('completed_steps') == config['steps']
        shared = a.get('initial_beta') is not None and a.get('initial_beta') == b.get('initial_beta')
        same_adapter = a.get('guidance_initial_state_sha256') is not None and a.get('guidance_initial_state_sha256') == b.get('guidance_initial_state_sha256')
        pair = dict(candidate=candidate, complete=complete, same_beta=shared, same_adapter=same_adapter,
                    first_scalar_mismatch=first, matches=complete and shared and same_adapter and first is None)
        pairs.append(pair)
        if complete and not pair['matches']:
            issues.append(f'lg_alg_b{candidate}_review')
    return dict(identity_checks=controls, lg_alg_pairs=pairs, review_items=issues,
                lg_alg_scope='separate_runs' if alg_present else 'shared_lg_pre_shutdown_screen_no_independent_alg_result',
                tolerance=dict(rtol=config['resume_rtol'], atol=config['resume_atol']))


def trajectory_summary(rows):
    if not rows:
        return None
    result = {}
    for key in ('loss', 'ce', 'guidance', 'weighted_guidance_to_seg_loss', 'grad_norm_unclipped'):
        values = [r[key] for r in rows]
        result[key] = dict(first25_median=statistics.median(values[:25]),
                           last25_median=statistics.median(values[-25:]), maximum=max(values), minimum=min(values))
    return result


def run(args, config, plan, output, report):
    import random
    import numpy as np
    import torch
    import segm.utils.torch as ptu
    from torch.nn import functional as F
    from segm.model.utils import inference
    from . import official_api as api
    from .data import json_hash, sha256, save_json
    from .evaluation import confusion_update, metrics
    from .full_data import FullDataset, train_loader, batch_hash
    from .full_checkpoint import save_checkpoint, load_checkpoint, tree_hash
    from .ibkd_deterministic import apply_deterministic_candidate
    from .runtime import seed_all, state_hash, source_hash, controller_for, restore_controller
    from .tiny_smoke import assert_state_close

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible H200 GPU')
    if config.get('full_validation_at_endpoint') and 'H200' not in torch.cuda.get_device_name():
        raise RuntimeError('The 2000-step pack requires an H200, matching the timing estimate')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8')
    device = torch.device('cuda'); ptu.device = device
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(config['ibkd_cpu_threads'] if plan['method'] == 'ibkd' else 4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    manifest = json.loads(args.manifest.read_text())
    datasets = {s: FullDataset(args.data_dir, manifest, config, s) for s in ('train', 'val')}
    seed_all(config['seed'])
    model = api.student(args.cache_root, backbone=config['student_backbone'], image_size=config['crop_size'],
                        decoder_layers=config['decoder_layers'], recompute=config['gradient_checkpointing']).to(device).train()
    seed_all(config['seed'] + 1000)
    guide = api.guidance(plan['method'], config)
    candidate = apply_deterministic_candidate(guide) if plan['method'] == 'ibkd' else None
    if candidate is not None and not candidate['applied']:
        raise RuntimeError('iBKD deterministic candidate was not applied')
    guide = guide.to(device).train()
    teacher = api.teacher(args.cache_root).to(device)
    capture = api.FeatureCapture(model)
    initial_student, initial_guide, teacher_hash = state_hash(model), state_hash(guide), state_hash(teacher)
    optimizer, scheduler = api.optimizer_scheduler(model, guide, args.cache_root, total_steps=config['total_steps'])
    if scheduler.iter_max != 80000 or not optimizer.defaults['nesterov']:
        raise RuntimeError('Unexpected 80k Nesterov SGD schedule')
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    controller = controller_for(plan['method'], dict(config, guidance_beta=plan['beta']))
    environment = dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                       cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(),
                       precision='fp32', cpu_threads=torch.get_num_threads())
    identity = dict(config_sha256=json_hash(config), source_sha256=source_hash(),
                    manifest_sha256=sha256(args.manifest), assets=report['assets'], plan=plan,
                    initial_student_sha256=initial_student, initial_guide_sha256=initial_guide,
                    teacher_sha256=teacher_hash, environment=environment)
    seed_all(config['seed'] + 2000)
    progress = initial_progress()
    if args.resume:
        saved, resume_info = load_checkpoint(args.resume, identity, output)
        model.load_state_dict(saved['model'], strict=True); guide.load_state_dict(saved['guide'], strict=True)
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        restore_controller(controller, saved['controller']); progress = saved['progress']
        random.setstate(saved['python_rng'])
        rng = saved['numpy_rng']; np.random.set_state((rng[0], np.array(rng[1], dtype=np.uint32), *rng[2:]))
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        report['resume'] = resume_info
        del saved
    if not 0 <= progress['global_step'] <= config['steps']:
        raise ValueError('Checkpoint step is outside the configured endpoint')
    report.update(student_initial_state_sha256=initial_student, guidance_initial_state_sha256=initial_guide,
                  teacher_state_sha256=teacher_hash, environment=environment, ibkd_deterministic_candidate=candidate,
                  completed_steps=progress['global_step'], last=progress['rows'][-1] if progress['rows'] else None)
    save_json(output / 'identity.json', identity)
    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    last_checkpoint_step = -1
    should_stop = getattr(args, 'should_stop', lambda: False)
    paused = False

    def checkpoint():
        nonlocal last_checkpoint_step
        tensors = parameters + [v for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)]
        if not all(bool(torch.isfinite(t).all()) for t in tensors):
            raise FloatingPointError('Nonfinite parameter/optimizer state; previous recovery checkpoint retained')
        rng = np.random.get_state()
        payload = dict(signature=identity, model=model.state_dict(), guide=guide.state_dict(),
                       optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), controller=controller.state_dict(),
                       progress=progress, python_rng=random.getstate(), numpy_rng=(rng[0], rng[1].tolist(), *rng[2:]),
                       torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        pointer = save_checkpoint(output, payload)
        report['checkpoint'] = dict(pointer=str(output / 'resume.json'), saved_step=progress['global_step'],
                                    current=pointer['current'], retained=True,
                                    scope='last_verified_finite_recovery_state')
        last_checkpoint_step = progress['global_step']

    def step(batch):
        images, target = (x.to(device) for x in batch[:2])
        if not torch.isfinite(images).all() or not (target != 255).any():
            raise RuntimeError('Invalid/all-void input batch')
        optimizer.zero_grad(set_to_none=True)
        beta = progress['beta']
        logits, features = capture.forward(model, images) if beta > 0 else (model(images), None)
        ce = F.cross_entropy(logits, target, ignore_index=255)
        alignment = fusion = guided = ce.new_zeros(())
        if beta > 0:
            with torch.no_grad():
                tfeatures = teacher.extract_feat(api.teacher_input(images))[1:]
            if plan['method'] == 'ibkd':
                alignment, fusion = guide(features, tfeatures)
                guided = (1 - plan['lambda']) * alignment + plan['lambda'] * fusion
            else:
                guided = guide(features, tfeatures)
        loss = ce + beta * guided
        if config.get('record_failed_step_details'):
            report['step_observation'] = dict(
                step=report['attempted_step'], phase='forward',
                **{k: diagnostic_scalar(v.detach()) for k, v in
                   dict(loss=loss, ce=ce, guidance=guided, alignment=alignment, fusion=fusion,
                        weighted_guidance=beta * guided).items()})
        if not all(bool(torch.isfinite(v)) for v in (loss, ce, guided, alignment, fusion)):
            raise FloatingPointError('Nonfinite loss/CE/guidance before backward')
        if config.get('record_failed_step_details'):
            report['step_observation']['phase'] = 'backward'
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf'), error_if_nonfinite=False)
        if config.get('record_failed_step_details'):
            report['step_observation'].update(phase='gradient_check', gradient_norm=diagnostic_scalar(norm))
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite gradient norm before optimizer update')
        if progress['next_batch'] == 0:
            modules = [model.encoder, model.decoder] + ([guide] if beta > 0 else [])
            if any(not any(p.grad is not None and bool(p.grad.abs().max() > 0) for p in m.parameters()) for m in modules):
                raise RuntimeError('Missing active-module gradient')
        if any(p.grad is not None for p in teacher.parameters()):
            raise RuntimeError('Frozen teacher received gradients')
        row = dict(loss=float(loss.detach()), ce=float(ce.detach()), guidance=float(guided.detach()),
                   alignment=float(alignment.detach()), fusion=float(fusion.detach()),
                   weighted_guidance=beta * float(guided.detach()),
                   weighted_guidance_to_seg_loss=beta * float(guided.detach()) / max(float(ce.detach()), 1e-30),
                   grad_norm_unclipped=float(norm), lr=optimizer.param_groups[0]['lr'])
        if config.get('record_failed_step_details'):
            report['step_observation']['phase'] = 'optimizer_update'
        optimizer.step(); scheduler.step_update(scheduler.last_epoch + 1)
        if config.get('check_state_each_step'):
            tensors = parameters + [v for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)]
            finite = all(bool(torch.isfinite(t).all()) for t in tensors)
            report['step_observation'].update(phase='post_update_state_check', parameters_optimizer_finite=finite)
            if not finite:
                raise FloatingPointError('Nonfinite parameter/optimizer state after optimizer update')
        return row

    report['phase'] = 'training'
    checkpoint()
    with (output / 'steps.jsonl').open('w') as log:
        while progress['global_step'] < config['steps']:
            if should_stop():
                paused = True
                break
            epoch = progress['epoch']
            if progress['beta'] is None:
                progress['beta'] = controller.beta_for_epoch(epoch)
            for batch in train_loader(datasets['train'], epoch, progress['next_batch'], device):
                if should_stop():
                    paused = True
                    break
                digest = batch_hash(*batch)
                report['attempted_step'] = progress['global_step'] + 1
                start = time.perf_counter()
                try:
                    row = step(batch)
                except Exception:
                    if config.get('record_failed_step_details'):
                        report['failure_observation'] = dict(report.pop('step_observation', {}), input_sha256=digest)
                        report.update(input_hashes=progress['input_hashes'], losses=progress['rows'],
                                      trajectory=trajectory_summary(progress['rows']),
                                      teacher_frozen_verified=state_hash(teacher) == teacher_hash,
                                      controller=controller.state_dict(), train_wall_seconds=time.monotonic()-began)
                    raise
                report.pop('step_observation', None)
                torch.cuda.synchronize()
                row['seconds'] = time.perf_counter() - start
                ended_epoch = record_step(progress, row, len(batch[2]), digest, controller,
                                          train_samples=len(datasets['train']))
                log.write(json.dumps(dict(row, input_sha256=digest), allow_nan=False) + '\n'); log.flush()
                report.update(completed_steps=progress['global_step'], last=row,
                              completed_epochs=len(progress['completed_epochs']), controller=controller.state_dict())
                # This compact progress record remains available after an interrupted child.
                if (progress['global_step'] == 1 or ended_epoch or
                        progress['global_step'] % config.get('progress_every_steps', 1) == 0 or
                        progress['global_step'] == config['steps']):
                    save_json(output / 'progress.json', dict(report, input_hashes=progress['input_hashes'],
                                                             losses=progress['rows']))
                if progress['global_step'] == 1 or progress['global_step'] % config.get('console_every_steps', 25) == 0:
                    print(f"[TI16_GRID_STEP] run={plan['id']} step={progress['global_step']}/{config['steps']} "
                          f"epoch={row['epoch']} loss={row['loss']:.6g} ce={row['ce']:.6g} guidance={row['guidance']:.6g} "
                          f"beta={row['beta']:.9g} grad_norm={row['grad_norm_unclipped']:.6g} seconds={row['seconds']:.2f}", flush=True)
                if ended_epoch or progress['global_step'] % config['checkpoint_every_steps'] == 0:
                    checkpoint()
                if progress['global_step'] >= config['steps']:
                    break
            if paused:
                break
            if progress['epoch'] == epoch and progress['global_step'] < config['steps']:
                raise RuntimeError('Loader exhausted before a complete epoch')
    if last_checkpoint_step != progress['global_step']:
        checkpoint()
    if state_hash(teacher) != teacher_hash:
        raise RuntimeError('Teacher weights/BN changed')
    # Read verified saved tensors and compare every persisted optimizer/module state.
    report['phase'] = 'checkpoint_roundtrip'
    saved, reload_info = load_checkpoint(output / 'resume.json', identity, output)
    for actual, restored in ((model.state_dict(), saved['model']), (guide.state_dict(), saved['guide']),
                             (optimizer.state_dict(), saved['optimizer']), (scheduler.state_dict(), saved['scheduler'])):
        assert_state_close(actual, restored, rtol=0, atol=0)
    if tree_hash(saved['progress']) != tree_hash(progress) or saved['controller'] != controller.state_dict():
        raise RuntimeError('Checkpoint progress/controller mismatch')
    del saved
    report.update(checkpoint=dict(report['checkpoint'],
                                   strict_state_roundtrip='passed', retained=True, generation=reload_info['generation'],
                                   scope='complete_training_state_saved_not_a_long_run_resume_test'),
                  input_hashes=progress['input_hashes'], input_sha256=json_hash(progress['input_hashes']),
                  losses=progress['rows'], trajectory=trajectory_summary(progress['rows']),
                  teacher_frozen_verified=True, train_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  train_wall_seconds=time.monotonic() - began,
                  median_step_seconds=statistics.median(r['seconds'] for r in progress['rows']) if progress['rows'] else None,
                  final_student_sha256=state_hash(model), final_guide_sha256=state_hash(guide),
                  completed_epoch_records=progress['completed_epochs'], next_epoch=progress['epoch'],
                  next_batch=progress['next_batch'], partial_epoch_samples=progress['samples'],
                  controller=controller.state_dict(),
                  natural_guidance_off_tested=controller.stop_epoch is not None,
                  guidance_stop_epoch=controller.stop_epoch,
                  guidance_stop_step=(None if controller.stop_epoch is None else math.ceil(2975 / 8) * controller.stop_epoch + 1))
    if paused or should_stop():
        report.update(status='paused', phase='saved_for_resume', pause_reason='time_budget_or_signal',
                      selected_step=None, selected_epoch=None, full_validation=False, validation_samples=0)
        return
    if config.get('full_validation_at_endpoint'):
        from .tiny_val_timing import evaluate, ValidationInterrupted
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        torch.set_num_threads(config['evaluation_cpu_threads'])
        report['phase'] = 'full_validation'
        def val_progress(n, seconds):
            report.update(validation_samples=n, partial_validation_seconds=seconds)
            save_json(output / 'progress.json', report)
            print(f'[TI16_GRID2000_VAL] run={plan["id"]} samples={n}/500 seconds={seconds:.3f}', flush=True)
        try:
            result = evaluate(model, datasets['val'], inference, config, torch.cuda.synchronize,
                              val_progress, should_stop=should_stop)
        except ValidationInterrupted as error:
            report.update(status='paused', phase='validation_pending', pause_reason=str(error),
                          selected_step=None, selected_epoch=None, full_validation=False,
                          diagnostic_metrics=None)
            return
        save_json(output / 'per_image_timings.json', result.pop('per_image_timings'))
        scores = result.pop('metrics')
        report.update(result, diagnostic_metrics=scores, full_validation=True,
                      student_unchanged_during_validation=state_hash(model) == report['final_student_sha256'])
        if not report['student_unchanged_during_validation']:
            raise RuntimeError('Validation modified student weights')
        report.update(status='passed', phase='complete', stability='finite_2000_updates_completed_not_80k_guarantee',
                      selected_step=config['steps'], selected_epoch=progress['rows'][-1]['epoch'],
                      selection_rule='fixed_2000_endpoint_not_best_checkpoint')
        return
    report['phase'] = 'diagnostic_validation'
    model.eval()
    matrix = torch.zeros(19, 19, dtype=torch.int64)
    valid_pixels, val_ids = 0, []
    with torch.no_grad():
        for index in range(config['diagnostic_validation_samples']):
            ims, metas, target, sample_id = datasets['val'][index]
            pred = inference(model, ims, metas, tuple(target.shape), config['window_size'],
                             config['window_stride'], batch_size=1).argmax(0).cpu()
            confusion_update(matrix, pred, target)
            valid_pixels += int((target != 255).sum()); val_ids.append(sample_id)
    scores = metrics(matrix)
    if scores['valid_pixels'] != valid_pixels:
        raise RuntimeError('Validation void pixel accounting mismatch')
    report.update(status='passed', phase='complete', stability=f"finite_{config['steps']}_updates_completed_not_long_run_guarantee",
                  selected_step=config['steps'], selected_epoch=progress['rows'][-1]['epoch'],
                  selection_rule=f"fixed_{config['steps']}_endpoint_not_best_checkpoint", validation_ids=val_ids,
                  validation_samples=len(val_ids), diagnostic_metrics=scores, full_validation=False,
                  natural_guidance_off_tested=controller.stop_epoch is not None,
                  guidance_stop_epoch=controller.stop_epoch,
                  guidance_stop_step=(None if controller.stop_epoch is None else math.ceil(2975 / 8) * controller.stop_epoch + 1))


def compact(row):
    from .tiny_smoke import compact_run
    result = compact_run(row)
    for key in ('initial_beta', 'candidate', 'attempted_step', 'failure_stage', 'stability', 'trajectory',
                'completed_epochs', 'next_epoch', 'next_batch', 'partial_epoch_samples', 'controller',
                'guidance_stop_epoch', 'guidance_stop_step', 'train_wall_seconds', 'median_step_seconds',
                'completed_epoch_records', 'initial_target_ratio', 'failure_observation'):
        result[key] = row.get(key)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('cache-root', 'data-dir', 'manifest', 'output-dir', 'config'):
        parser.add_argument('--' + flag, required=True, type=Path)
    parser.add_argument('--run-id', help=argparse.SUPPRESS)
    parser.add_argument('--resume', type=Path, help='Verified resume.json for the same run/config; requires --run-id')
    args = parser.parse_args()
    if args.resume and not args.run_id:
        parser.error('--resume requires --run-id')
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    from .data import save_json
    from .tiny_repeat import warning_summary
    from .tiny_grid_report import final_line
    report = dict(status='running', runs=[], scientific_result=False, test_used=False,
                  automatic_next_stage=False, beta_ranking_performed=False)
    failed = False
    terminal_plans = []
    started = time.monotonic()
    stopped = {'requested': False}
    previous_signals = {}
    try:
        config = load_config(args.config)
        terminal_plans = [p for p in config['runs'] if not args.run_id or p['id'] == args.run_id]
        report.update(protocol_id=config['protocol_id'], expected_runs=len(config['runs']),
                      expected_steps=config['steps'],
                      ibkd_lambdas=config['ibkd_lambdas'],
                      lg_alg_shared_screen=config.get('lg_alg_shared_screen', False))
        if args.config.name == HIGH_BETA100:
            from .tiny_screen2000 import stopping_deadlines
            job_started = float(os.environ.get('CITYSCAPES_TI16_JOB_STARTED', time.time()))
            hard, stop_at = stopping_deadlines(job_started, config)
            def request_stop(signum, frame):
                stopped['requested'] = True
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous_signals[sig] = signal.getsignal(sig)
                signal.signal(sig, request_stop)
            args.should_stop = lambda: stopped['requested'] or time.time() >= stop_at
            report.update(job_budget_seconds=36000, save_reserve_seconds=120,
                          hard_deadline_unix=hard, stop_at_unix=stop_at,
                          classification_reference=config['classification_reference'])
        if args.run_id:
            if not terminal_plans:
                raise ValueError('Unknown candidate run ID')
            plan = terminal_plans[0]
            report.update(run_id=plan['id'], method=plan['method'], candidate=plan['candidate'],
                          initial_beta=plan['beta'], initial_target_ratio=plan['initial_target_ratio'],
                          **{'lambda': plan.get('lambda')}, completed_steps=0, selected_step=None,
                          selected_epoch=None, diagnostic_metrics=None, full_validation=False)
        from .official_api import bootstrap
        from .official_assets import verify
        bootstrap(args.cache_root)
        provenance = verify(args.cache_root, student='tiny')
        report['assets'] = provenance['weights']
        save_json(output / 'config.json', config); save_json(output / 'provenance.json', provenance)
        if args.run_id:
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter('always')
                try:
                    run(args, config, plan, output, report)
                finally:
                    report['warning_summary'] = warning_summary(records)
                    report['deterministic_warning_count'] = sum('deterministic' in str(w.message) for w in records)
                    save_json(output / 'warnings.json', report['warning_summary'])
        else:
            from .full_data import prepare_labels
            prepare_labels(args.data_dir, args.manifest, {'train': 2975, 'val': 500}, output)
            rows = []
            for plan in config['runs']:
                if getattr(args, 'should_stop', lambda: False)():
                    break
                destination = output / plan['id']
                command = [sys.executable, '-u', '-m', 'ibkd_seg.cityscapes.tiny_grid']
                for flag in ('cache-root', 'data-dir', 'manifest', 'config'):
                    command += ['--' + flag, str(getattr(args, flag.replace('-', '_')).resolve())]
                command += ['--output-dir', str(destination), '--run-id', plan['id']]
                if args.config.name == HIGH_BETA100:
                    from .tiny_screen2000 import launch_child
                    completed = launch_child(command, args.should_stop)
                else:
                    completed = subprocess.run(command, check=False)
                result_path = destination / 'summary.json'
                if result_path.exists():
                    row = json.loads(result_path.read_text())
                    if completed.returncode != 0 and row['status'] == 'passed':
                        row.update(status='runtime_failure', error=f'child_exit={completed.returncode}')
                else:
                    progress = destination / 'progress.json'
                    row = json.loads(progress.read_text()) if progress.exists() else {}
                    row.update(status='runtime_failure', run_id=plan['id'], method=plan['method'],
                               initial_beta=plan['beta'], candidate=plan['candidate'], **{'lambda': plan.get('lambda')},
                               error=f'child_exit={completed.returncode}; final summary missing', selected_step=None,
                               selected_epoch=None)
                rows.append(row)
                report['runs'] = [compact(r) for r in rows]
                save_json(output / 'grid_summary.json', report)
                if row.get('status') == 'paused':
                    break
            report['cross_checks'] = compare_grid(rows, config)
            report['finite_candidates'] = [r['run_id'] for r in rows if r['status'] == 'passed']
            report['failed_candidates'] = [r['run_id'] for r in rows if r['status'] != 'passed']
            failed = bool(report['failed_candidates'] or report['cross_checks']['review_items'])
            report['status'] = 'needs_review' if failed else 'passed'
            report['selection_note'] = f"{config['steps']}-step numerical screen only; no automatic permanent exclusion or top-beta ranking"
    except Exception as error:
        failed = True
        report.update(status='numerical_failure' if isinstance(error, FloatingPointError) else 'runtime_failure',
                      error=repr(error), failure_stage=report.get('phase', 'candidate' if args.run_id else 'suite_setup_or_report'))
        progress = output / 'progress.json'
        if args.run_id and progress.exists():
            partial = json.loads(progress.read_text())
            for key in ('losses', 'input_hashes'):
                report.setdefault(key, partial.get(key, []))
            report['trajectory'] = trajectory_summary(report.get('losses', []))
        (output / 'traceback.txt').write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
        name = 'summary.json' if args.run_id else 'grid_summary.json'
        report.update(summary_path=str(output / name), invocation_seconds=time.monotonic() - started)
        save_json(output / name, report)
        terminal = dict(report, runs=[compact(report)]) if args.run_id else report
        line = final_line(terminal, terminal_plans, child=bool(args.run_id))
        (output / 'terminal_summary.log').write_text(line + '\n', encoding='ascii')
        print(line, flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
